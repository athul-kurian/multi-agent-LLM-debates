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

# MODEL = "meta-llama/llama-3.3-70b-instruct"
# MODEL = "meta-llama/llama-3.2-3b-instruct"
# MODEL = "meta-llama/llama-3.1-8b-instruct"
# MODEL = "mistralai/ministral-3b-2512"
MODEL = "google/gemma-3-4b-it"

DATA_FILE_PATH = "./Data/gsm8k.csv"
RESULTS_FILE_PATH = "./Benchmarks/Results/gemma-4b-results.csv" # change file path
TEMPERATURE = 0.0
NUM_SAMPLES = 500
RANDOM_SEED = 42
RATE_LIMIT_SLEEP = 0.5 # used 2.0 for mistral server and 3.0 for google

SYSTEM_PROMPT = (
    "Solve the math problem carefully and show the important solution steps. "
    "On the final line, output only the final numerical answer prefixed by ####."
)

FEW_SHOTS = [
    {
        "question": "Benny bought 2 soft drinks for$ 4 each and 5 candy bars. He spent a total of 28 dollars. How much did each candy bar cost?",
        "answer": (
            "Benny spent 2 * $4 = $<<2*4=8>>8 on soft drinks.\n"
            "Benny spent a total of $28 - $8 on soft drinks = $<<28-8=20>>20 on candy bars.\n"
            "Benny spent $20 / 5 candy bars = $<<20/5=4>>4 for each candy bar.\n"
            "#### 4"
        )
    },
    {
        "question": "Wickham is throwing a huge Christmas party... how many plates does he need in total for his guests?",
        "answer": (
            "The number of guests that bring a plus one is 30 / 2 = <<30/2=15>>15 guests\n"
            "The total number of guests at the party, including the plus ones, is 30 + 15 = <<30+15=45>>45 guests\n"
            "Since there are 3 courses, Wickham will need 45 * 3 = <<45*3=135>>135 plates\n"
            "#### 135"
        )
    },
    {
        "question": "Five coaster vans are used to transport students for their field trip. Each van carries 28 students, 60 of which are boys. How many are girls?",
        "answer": (
            "There are a total of 5 vans x 28 students = <<5*28=140>>140 students.\n"
            "If 60 are boys, then 140 - 60 = <<140-60=80>>80 of these students are girls.\n"
            "#### 80"
        )
    }
]

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
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for few_shot in FEW_SHOTS:
        messages.append({"role": "user", "content": few_shot["question"]})
        messages.append({"role": "assistant", "content": few_shot["answer"]})

    messages.append({"role": "user", "content": prompt})

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