import pandas as pd
import csv
import asyncio
from time import sleep

from helpers import *
from constants import *
from EfficientDebates import run_efficient_debate

async def main():
    df = pd.read_csv(DATA_FILE_PATH)
    sample = df.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED)

    with open(RESULTS_FILE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        fieldnames = [
            "index",
            "question",
            "answer",
            "correct_value",
            "final_llm_answer_value",
            "selected_agent",
            "stopped_round",
            "num_rounds",
            "num_calls",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_inference_cost",
        ]

        for round_number in [1, 2, 3]:
            for agent_name in ["agent_1", "agent_2", "agent_3"]:
                prefix = f"r{round_number}_{agent_name}"

                fieldnames.extend([
                    f"{prefix}_model",
                    f"{prefix}_answer",
                    f"{prefix}_answer_value",
                ])

        writer.writerow(fieldnames)

        successful = 0
        failed = 0

        for question_number, (index, row) in enumerate(sample.iterrows(), start=1):
            question = row["question"]
            answer = row["answer"]

            correct_value = extract_final_answer(answer)

            print()
            print(f"Processing question {question_number}/{NUM_SAMPLES}")
            print(f"Dataset index: {index}")

            try:
                debate_result = await run_efficient_debate(question)
                csv_row = flatten_debate_result(
                    debate_result=debate_result,
                    question_index=index,
                    question=question,
                    gold_answer=answer,
                    gold_value=correct_value
                )

                writer.writerow([csv_row.get(field) for field in fieldnames])

                # Immediately write this question's result to disk.
                f.flush()

                predicted_value = debate_result["final_answer"]

                if predicted_value == correct_value:
                    successful += 1
                else:
                    failed += 1

                print(f"  Gold answer: {correct_value}")
                print(f"  Debate answer: {predicted_value}")
                print(f"  Calls used: {debate_result['num_calls']}")
                print(f"  Rounds used: {debate_result['num_rounds']}")

            except Exception as e:
                failed += 1
                print(f"Question failed completely: {e}")

            # delay between questions.
            sleep(RATE_LIMIT_SLEEP)

    print("\nEvaluation complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    asyncio.run(main())