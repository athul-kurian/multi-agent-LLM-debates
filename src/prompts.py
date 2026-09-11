
# =======PROMPTS=======

INITIAL_SYSTEM_PROMPT = (
    "Solve the math problem carefully and show the important solution steps. "
    "On the final line, output only the final numerical answer prefixed by ####."
)


DEBATE_SYSTEM_PROMPT = (
    "You are participating in a multi-agent debate to solve a math problem"
    "Review your previous solution and the other agents' solutions"
    "and reconsider your answer"
    "On the final line, output only the final numerical answer prefixed by ####."
)

FEW_SHOTS = [
    {
        "question": (
            "Benny bought 2 soft drinks for $4 each and 5 candy bars. "
            "He spent a total of 28 dollars. How much did each candy bar cost?"
        ),
        "answer": (
            "Benny spent 2 * $4 = $<<2*4=8>>8 on soft drinks.\n"
            "Benny spent a total of $28 - $8 on soft drinks = "
            "$<<28-8=20>>20 on candy bars.\n"
            "Benny spent $20 / 5 candy bars = "
            "$<<20/5=4>>4 for each candy bar.\n"
            "#### 4"
        )
    },
    {
        "question": (
            "Wickham is throwing a huge Christmas party... "
            "how many plates does he need in total for his guests?"
        ),
        "answer": (
            "The number of guests that bring a plus one is "
            "30 / 2 = <<30/2=15>>15 guests\n"
            "The total number of guests at the party, including the "
            "plus ones, is 30 + 15 = <<30+15=45>>45 guests\n"
            "Since there are 3 courses, Wickham will need "
            "45 * 3 = <<45*3=135>>135 plates\n"
            "#### 135"
        )
    },
    {
        "question": (
            "Five coaster vans are used to transport students for their "
            "field trip. Each van carries 28 students, 60 of which are "
            "boys. How many are girls?"
        ),
        "answer": (
            "There are a total of 5 vans x 28 students = "
            "<<5*28=140>>140 students.\n"
            "If 60 are boys, then 140 - 60 = "
            "<<140-60=80>>80 of these students are girls.\n"
            "#### 80"
        )
    }
]
