"""
A minimal AI agent built from scratch to understand how tool-calling agents work.

An agent is an LLM in a loop: given a question and a set of tools, the model
decides whether to call a tool, reads the result, and decides again - continuing
until it produces a final text answer. No agent frameworks used, so the loop,
the tool-calling, and the safety limits are all visible.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
os.environ["GOOGLE_API_KEY"] = api_key  # some internal clients look for this name

CHAT_MODEL = "models/gemini-3.6-flash"
MAX_STEPS = 10          # backstop: hard limit on tool-calling rounds
MAX_REPEATS = 3         # loop detection: abort if the same call repeats this many times


# --- Tools: plain functions the agent can choose to call ---
def calculator(expression):
    """Evaluate a math expression like '15 * 24' or 'sqrt(144) + 95'. Supports basic math and common functions."""
    import math
    allowed = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


def get_planet_fact(planet):
    """Return a fact about a planet."""
    facts = {
        "mercury": "Mercury has no moons.",
        "jupiter": "Jupiter has at least 95 known moons.",
        "saturn": "Saturn has a spectacular ring system.",
    }
    return facts.get(planet.lower(), f"No fact available for {planet}.")


TOOLS = [calculator, get_planet_fact]
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "get_planet_fact": get_planet_fact,
}


def run_agent(question):
    """Run the agent loop until the model gives a final answer or a safety limit trips."""
    model = genai.GenerativeModel(model_name=CHAT_MODEL, tools=TOOLS)
    chat = model.start_chat()

    try:
        response = chat.send_message(question)
    except Exception as e:
        print(f"(Request failed: {e})")
        return

    call_history = []  # record of calls made, to detect repetition

    for step in range(MAX_STEPS):
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        # No tool requested -> the model has given its final answer.
        if not function_calls:
            print(f"Final answer: {response.text}")
            return

        results = []
        for call in function_calls:
            name = call.name
            args = dict(call.args)

            # Loop detection: if the exact same call keeps repeating, the agent is stuck.
            signature = f"{name}({args})"
            call_history.append(signature)
            if call_history.count(signature) >= MAX_REPEATS:
                print(f"(Aborting: '{signature}' called {MAX_REPEATS} times - agent is looping.)")
                return

            print(f"  Model called: {signature}")
            output = AVAILABLE_TOOLS[name](**args)
            print(f"  Tool returned: {output}")

            results.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": output},
                    )
                )
            )

        try:
            response = chat.send_message(results)
        except Exception as e:
            print(f"(Request failed: {e})")
            return

    print(f"(Stopped: reached the {MAX_STEPS}-step limit without a final answer.)")


def main():
    while True:
        question = input("\nAsk the agent (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        run_agent(question)


if __name__ == "__main__":
    main()