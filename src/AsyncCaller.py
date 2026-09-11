from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv

from helpers import extract_final_answer
from constants import TEMPERATURE, BASE_URL
from agents import AGENTS
from helpers import *

load_dotenv()

# =======OPEN-AI CLIENT=======

async_client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# =======Async LLM call=======

async def call_agent(agent_name, model, messages):
    """
    Make one asynchronous LLM request.

    Returns a dictionary containing:
        agent
        model
        raw answer
        extracted answer
        token counts
        cost
        error
    """

    try:
        response = await async_client.chat.completions.create(model=model, messages=messages, temperature=TEMPERATURE)

        llm_answer = response.choices[0].message.content
        llm_answer_value = extract_final_answer(llm_answer)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        inference_cost = response.usage.cost

        return {
            "agent": agent_name,
            "model": model,
            "llm_answer": llm_answer,
            "llm_answer_value": llm_answer_value,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "inference_cost": inference_cost,
            "error": None
        }

    except Exception as e:
        return {
            "agent": agent_name,
            "model": model,
            "llm_answer": None,
            "llm_answer_value": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "inference_cost": None,
            "error": str(e)
        }

# =======Run one debate round asynchronously=======

async def run_round(question, round_number, previous_results=None):
    """
    Run all three agents concurrently.

    Round 1:
        Every agent receives the original question.

    Rounds 2/3:
        Every agent receives:
            - original question
            - its previous solution
            - the other two agents' solutions
    """

    tasks = []

    for agent_name, model in AGENTS.items():
        if round_number == 1:
            messages = build_initial_messages(question)
        else:
            previous_answer = previous_results[agent_name]["llm_answer"]

            other_agent_answers = {name: result["llm_answer"] for name, result in previous_results.items()}

            messages = build_debate_messages(
                question=question,
                previous_answer=previous_answer,
                other_agent_answers=other_agent_answers
            )

        tasks.append(call_agent(agent_name=agent_name, model=model, messages=messages))

    # execute the three requests concurrently.
    results = await asyncio.gather(*tasks)

    return {result["agent"]: result for result in results}