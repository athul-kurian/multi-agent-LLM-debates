from AsyncCaller import run_round
from helpers import *


# =======Main EfficientDebates algorithm=======

async def run_efficient_debate(question):
    """
    Implements the adaptive pruning strategy.

    Round 1:
        3 calls

    If majority:
        STOP

    Otherwise:
        Round 2 -> +3 calls

    If majority:
        STOP

    Otherwise:
        Round 3 -> +3 calls

    If still no majority:
        Select Agent 1.
    """

    all_rounds = {}

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost = 0.0

    # ---Round 1---

    print("  Round 1: querying 3 agents concurrently...")

    round_results = await run_round(question=question, round_number=1)
    all_rounds[1] = round_results

    for result in round_results.values():
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_tokens += result["total_tokens"]

        if result["inference_cost"] is not None:
            total_cost += float(result["inference_cost"])

    round_1_answers = {name: result["llm_answer_value"] for name, result in round_results.items()}

    majority_answer, has_majority = get_majority_answer(round_1_answers)

    print(f"  Round 1 answers: {round_1_answers}")

    if has_majority:
        print(f"  Majority found in round 1: {majority_answer}")

        return {
            "final_answer": majority_answer,
            "selected_agent": "majority",
            "stopped_round": 1,
            "num_rounds": 1,
            "num_calls": 3,
            "rounds": all_rounds,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }

    # ---Rounds 2 and 3---

    for round_number in [2, 3]:
        print(f"  No majority. Round {round_number}: debating...")

        round_results = await run_round(
            question=question,
            round_number=round_number,
            previous_results=round_results
        )

        all_rounds[round_number] = round_results

        for result in round_results.values():
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_tokens += result["total_tokens"]

            if result["inference_cost"] is not None:
                total_cost += float(result["inference_cost"])

        current_answers = {
            name: result["llm_answer_value"]
            for name, result in round_results.items()
        }

        majority_answer, has_majority = get_majority_answer(current_answers)

        print(f"  Round {round_number} answers: {current_answers}")

        if has_majority:
            print(f"  Majority found in round {round_number}: {majority_answer}")

            return {
                "final_answer": majority_answer,
                "selected_agent": "majority",
                "stopped_round": round_number,
                "num_rounds": round_number,
                "num_calls": round_number * 3,
                "rounds": all_rounds,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }

    # ---Hard stop after round 3---

    # Select Agent 1 if there is still no majority after three rounds.

    agent_1_result = all_rounds[3]["agent_1"]
    final_answer = agent_1_result["llm_answer_value"]

    print("  No majority after round 3. Selecting Agent 1: {final_answer}")

    return {
        "final_answer": final_answer,
        "selected_agent": "agent_1",
        "stopped_round": 3,
        "num_rounds": 3,
        "num_calls": 9,
        "rounds": all_rounds,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }