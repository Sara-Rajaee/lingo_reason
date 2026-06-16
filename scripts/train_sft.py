"""SFT/GRPO training for Qwen reasoning distillation.

Builds a chat-formatted dataset from distilled reasoning traces (optionally
mixed with OpenThoughts2 math) and runs TRL SFTTrainer or GRPOTrainer with either LoRA or
full fine-tuning.
"""
import argparse
import os
import sys
import math
import unicodedata
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
torch.backends.cuda.enable_cudnn_sdp(False)
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

from data_generation import ReasoningDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", choices=["sft", "grpo"], default="sft")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    p.add_argument("--distilled-paths", nargs="+", required=True)
    p.add_argument("--distilled-pct", type=float, default=100.0)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-frac", type=float, default=0.02)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--openthoughts-cache", default="data/openthoughts_math_subset.json")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Train on prompt→final_answer only (drop <think> reasoning trace)")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    ## RL training arguments
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--loss-type", choices=["grpo", "bnpo"], default="dapo")
    p.add_argument("--use-vllm", action="store_true")
    return p.parse_args()


def build_rows(args):
    ds = ReasoningDataset(
        distilled_path=args.distilled_paths,
        distilled_percentage=args.distilled_pct,
        max_samples=args.max_samples,
        openthoughts_subset_path=args.openthoughts_cache,
        seed=args.seed,
    )
    rows = []
    for row in ds:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        reasoning = (row.get("reasoning") or "").strip()
        final_answer = (row.get("final_answer") or "").strip()
        if not prompt:
            continue
        output_row = {
            "prompt": prompt,
            "reasoning": reasoning,
            "final_answer": final_answer,
        }
        if args.no_reasoning or args.trainer == "grpo":
            if not final_answer:
                continue
            output_row["reasoning"] = ""
            rows.append(output_row)
        else:
            if not reasoning:
                continue
            rows.append(output_row)
    return rows


def format_with_chat_template(rows, tokenizer, no_reasoning=False):
    out = []
    for r in rows:
        if no_reasoning:
            assistant = '['+r['final_answer']+']'
        else:
            assistant = f"<think>\n{r['reasoning']}\n</think>\n\n[{r['final_answer']}]"
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        out.append({"text": text})
    return out


def format_for_grpo(rows):
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "ground_truth": r["final_answer"],
        }
        for r in rows
        if r.get("prompt") and r.get("final_answer")
    ]


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        message = completion[0]
        if isinstance(message, dict):
            return str(message.get("content") or "")
    return str(completion or "")

def extract_final_answer(text):
    text = str(text or "").strip()
    lower_text = text.lower()
    if "</think>" in lower_text:
        think_end = lower_text.rfind("</think>")
        return text[think_end + len("</think>"):].strip()
    return text.splitlines()[-1].strip() if text else ""


def evaluate(predictions, references, eval_types=None, points=None):
    def normalize(text):
        text = unicodedata.normalize('NFKC', str(text))
        text = ' '.join(str(text).strip().lower().split())
        return text.strip(' \t\n\r"\'`.,;:!?')

    scores = []
    for prediction, reference in zip(predictions, references):
        match = int(normalize(extract_final_answer(prediction)) == normalize(reference))
        scores.append(match)

    return scores


def answer_reward(completions, ground_truth, **kwargs):
    predictions = [completion_text(completion) for completion in completions]
    results = evaluate(predictions, ground_truth)
    return results


def reasoning_format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_text(completion)
        match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            rewards.append(0.1)
            continue
        if "</think>" in text.lower() and text[:text.lower().find("</think>")].strip():
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset...")
    rows = build_rows(args)
    print(f"  Collected {len(rows)} rows")
    if args.trainer == "sft":
        formatted = format_with_chat_template(rows, tokenizer, no_reasoning=args.no_reasoning)
    else:
        formatted = format_for_grpo(rows)
    full_ds = Dataset.from_list(formatted)

    if args.eval_frac > 0 and len(full_ds) > 50:
        split = full_ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = full_ds, None
    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds) if eval_ds else 0}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = None
    if args.mode == "lora":
        from peft import LoraConfig
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    if args.trainer == "sft":
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=args.max_seq_length,
            packing=False,
            dataset_text_field="text",
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=args.save_steps if eval_ds is not None else None,
            report_to="none",
            seed=args.seed,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    else:
        train_config = GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            loss_type=args.loss_type,
            use_vllm=args.use_vllm,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=args.save_steps if eval_ds is not None else None,
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            args=train_config,
            reward_funcs=[answer_reward, reasoning_format_reward],
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    print(f"Starting {args.trainer.upper()} training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
