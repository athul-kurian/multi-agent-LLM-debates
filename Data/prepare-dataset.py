from datasets import load_dataset

data = load_dataset("openai/gsm8k", "main")

data["train"].to_csv("gsm8k.csv")