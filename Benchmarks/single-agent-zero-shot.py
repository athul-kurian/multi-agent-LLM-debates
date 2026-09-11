from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import csv
from time import sleep

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL = "meta-llama/llama-3.3-70b-instruct"
# MODEL = "meta-llama/llama-3.2-3b-instruct"
# MODEL = "meta-llama/llama-3.1-8b-instruct"
# MODEL = "mistralai/ministral-3b-2512"
# MODEL = "google/gemma-3-4b-it"

DATA_FILE_PATH = "./Data/gsm8k.csv"
RESULTS_FILE_PATH = "./Benchmarks/Results/single-agent-zero-shot-llama-70b-results.csv"
TEMPERATURE = 0.0
NUM_SAMPLES = 500
RANDOM_SEED = 42
RATE_LIMIT_SLEEP = 0.5 # used 2.0 for mistral server and 3.0 for google
MAX_TOKENS = 50 # restricting max tokens for non-cot response

SYSTEM_PROMPT = "Output ONLY the final numerical answer prefixed by ####."


def extract_final_answer(answer):
    if "####" not in answer:
        return None

    answer = answer.split("####", 1)[1].strip()

    try:
        int(answer)
        return answer
    except ValueError:
        return None


def get_llm_response(prompt, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=TEMPERATURE)
    return response


def unpack_llm_response(response):
    llm_answer = response.choices[0].message.content
    llm_answer_value = extract_final_answer(llm_answer)
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    inference_cost = response.usage.cost

    return llm_answer, llm_answer_value, input_tokens, output_tokens, total_tokens, inference_cost


def main():
    df = pd.read_csv(DATA_FILE_PATH)
    sample = df.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED)

    with open(RESULTS_FILE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "question", "answer", "llm_answer", "llm_answer_value", "correct_value", "input_tokens", "output_tokens", "total_tokens", "inference_cost"])
        successful = 0
        failed = 0

        for index, row in sample.iterrows():
            question, answer = row["question"], row["answer"]
            correct_value = extract_final_answer(answer)

            print(f"Processing question {successful + failed + 1}/{NUM_SAMPLES}")

            try:
                response = get_llm_response(question, MODEL)
                llm_answer, llm_answer_value, input_tokens, output_tokens, total_tokens, inference_cost = unpack_llm_response(response)
                writer.writerow([index, question, answer, llm_answer, llm_answer_value, correct_value, input_tokens, output_tokens, total_tokens, inference_cost])
                successful += 1
            except Exception as e:
                failed += 1
                print(f"Request failed: {e}")
                print("Skipping question.")

            f.flush()
            sleep(RATE_LIMIT_SLEEP)

    print("\nEvaluation complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()