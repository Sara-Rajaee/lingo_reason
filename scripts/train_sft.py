"""SFT training for Qwen reasoning distillation.

Builds a chat-formatted dataset from distilled reasoning traces (optionally
mixed with OpenThoughts2 math) and runs TRL SFTTrainer with either LoRA or
full fine-tuning.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
torch.backends.cuda.enable_cudnn_sdp(False)
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_generation import ReasoningDataset


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--dedupe-by-prompt", action="store_true",
                   help="Keep one row per unique prompt (collapses k-sample distillation dups)")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
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
        if args.no_reasoning:
            if not final_answer:
                continue
            rows.append({"prompt": prompt, "reasoning": "", "final_answer": final_answer})
        else:
            if not reasoning:
                continue
            rows.append({"prompt": prompt, "reasoning": reasoning, "final_answer": final_answer})
    return rows


def format_with_chat_template(rows, tokenizer, no_reasoning=False):
    out = []
    for r in rows:
        if no_reasoning:
            assistant = r['final_answer']
        else:
            assistant = f"<think>\n{r['reasoning']}\n</think>\n\n{r['final_answer']}"
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": assistant},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        out.append({"text": text})
    return out


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
    if args.dedupe_by_prompt:
        seen = set()
        deduped = []
        for r in rows:
            if r["prompt"] in seen:
                continue
            seen.add(r["prompt"])
            deduped.append(r)
        print(f"  Deduped by prompt: {len(rows)} -> {len(deduped)}")
        rows = deduped
    formatted = format_with_chat_template(rows, tokenizer, no_reasoning=args.no_reasoning)
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
        max_seq_length=args.max_seq_length,
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
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
