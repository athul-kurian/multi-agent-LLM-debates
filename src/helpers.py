from collections import Counter

from prompts import *

# =======HELPER FUNCTIONS=======

def extract_final_answer(answer):
    """
    Extract and normalize the final answer after ####.

    Returns:
        String representation of the integer answer, or None.
    """

    if answer is None or "####" not in answer:
        return None

    answer = answer.split("####", 1)[1].strip()

    # Remove an optional dollar sign.
    if answer.startswith("$"):
        answer = answer[1:].strip()

    # Remove commas used as thousands separators.
    answer = answer.replace(",", "")

    try:
        value = float(answer)
    except ValueError:
        return None

    # Reject non-whole-number floats.
    if not value.is_integer():
        return None

    # Convert whole-number floats to integer form.
    return str(int(value))


def get_majority_answer(agent_answers):
    """
    agent_answers:
        {
            "agent_1": "42",
            "agent_2": "42",
            "agent_3": "17"
        }

    Returns:
        majority_answer, has_majority
    """

    valid_answers = [answer for answer in agent_answers.values() if answer is not None]

    if len(valid_answers) < 2:
        return None, False

    counts = Counter(valid_answers)
    answer, count = counts.most_common(1)[0]

    # Majority means at least 2 out of 3.
    if count >= 2:
        return answer, True

    return None, False


def build_few_shot_messages():
    """
    Takes all messages in FEW_SHOTS and returns a constructed messages list

    """
    messages = []

    for few_shot in FEW_SHOTS:
        messages.append({
            "role": "user",
            "content": few_shot["question"]
        })

        messages.append({
            "role": "assistant",
            "content": few_shot["answer"]
        })

    return messages


def build_initial_messages(question):
    """
        Build the prompt for round 1.
    """
    
    messages = [
        {
            "role": "system",
            "content": INITIAL_SYSTEM_PROMPT
        }
    ]

    messages.extend(build_few_shot_messages())

    messages.append({
        "role": "user",
        "content": question
    })

    return messages


def build_debate_messages(question, previous_answer, other_agent_answers):
    """
    Build the prompt for rounds 2 and 3.

    Each agent sees:
      - the original question
      - its own previous solution
      - the other two agents' solutions
    """

    messages = [
        {
            "role": "system",
            "content": DEBATE_SYSTEM_PROMPT
        }
    ]

    messages.extend(build_few_shot_messages())

    debate_prompt = (
        f"Original problem:\n"
        f"{question}\n\n"

        f"Your previous solution:\n"
        f"{previous_answer}\n\n"

        f"Other agents' solutions:\n\n"

        f"Agent 1:\n"
        f"{other_agent_answers.get('agent_1', 'No solution available.')}\n\n"

        f"Agent 2:\n"
        f"{other_agent_answers.get('agent_2', 'No solution available.')}\n\n"

        f"Agent 3:\n"
        f"{other_agent_answers.get('agent_3', 'No solution available.')}"
    )

    messages.append({
        "role": "user",
        "content": debate_prompt
    })

    return messages

def flatten_debate_result(debate_result, question_index, question, gold_answer, gold_value):
    """
    Convert the nested debate structure into one CSV row.
    """

    row = {
        "index": question_index,
        "question": question,
        "answer": gold_answer,
        "correct_value": gold_value,
        "final_llm_answer_value": debate_result["final_answer"],
        "selected_agent": debate_result["selected_agent"],
        "stopped_round": debate_result["stopped_round"],
        "num_rounds": debate_result["num_rounds"],
        "num_calls": debate_result["num_calls"],
        "total_input_tokens": debate_result["total_input_tokens"],
        "total_output_tokens": debate_result["total_output_tokens"],
        "total_tokens": debate_result["total_tokens"],
        "total_inference_cost": debate_result["total_cost"],
    }

    #---Save every agent's result from every round---

    for round_number in [1, 2, 3]:
        round_results = debate_result["rounds"].get(round_number, {})

        for agent_name in ["agent_1", "agent_2", "agent_3"]:
            result = round_results.get(agent_name)
            prefix = f"r{round_number}_{agent_name}"

            if result is None:
                row[f"{prefix}_model"] = None
                row[f"{prefix}_answer"] = None
                row[f"{prefix}_answer_value"] = None
            else:
                row[f"{prefix}_model"] = result["model"]
                row[f"{prefix}_answer"] = result["llm_answer"]
                row[f"{prefix}_answer_value"] = result["llm_answer_value"]

    return row