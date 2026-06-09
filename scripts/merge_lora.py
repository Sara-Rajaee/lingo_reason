"""Merge a trained LoRA adapter into its base model for vLLM serving."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default=None,
                    help="Override base model; otherwise read from adapter_config.json")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.base_model is None:
        import json
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model = cfg["base_model_name_or_path"]
    else:
        base_model = args.base_model

    print(f"Base:    {base_model}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output:  {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
