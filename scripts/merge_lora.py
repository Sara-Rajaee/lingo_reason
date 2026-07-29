import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = sys.argv[1]
base_model = sys.argv[2]
out_dir = sys.argv[3]

print(f"Loading base: {base_model}")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

print(f"Loading adapter: {adapter_dir}")
model = PeftModel.from_pretrained(model, adapter_dir)

print("Merging...")
model = model.merge_and_unload()

print(f"Saving to: {out_dir}")
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("Done.")
