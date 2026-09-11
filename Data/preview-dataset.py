import pandas as pd

df = pd.read_csv("gsm8k.csv")

print("Questions:")
for question in df["question"].head(5):
    print(question+"\n")

print()

print("Answers:")
for answer in df["answer"].head(5):
    print(answer+"\n")

print()

for i in range(5):
    question, answer = df.loc[i, "question"], df.loc[i, "answer"]
    print(question+"\n")
    final_answer = answer.split("####")[-1].strip()
    print(final_answer+"\n")
