"""
Minimal multi-agent system: a loop where agents talk to each other.

This is the MVP — two agents passing messages back and forth through
a shared conversation, coordinated by a simple orchestrator.
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def create_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-key-here":
        print("Set your GOOGLE_API_KEY in .env first.")
        print("Get one at: https://aistudio.google.com/apikey")
        raise SystemExit(1)
    return genai.Client(api_key=api_key)


def agent_turn(client, agent_name, system_prompt, conversation):
    """One agent takes a turn: sees the conversation so far, produces a response."""
    messages = [{"role": "user", "parts": [{"text": msg}]} for msg in conversation]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config={"system_instruction": system_prompt},
    )
    return response.text


def run(rounds=3):
    client = create_client()

    agents = {
        "Proposer": "You are a creative problem-solver. Propose ideas and build on feedback. Be concise.",
        "Critic": "You are a sharp but constructive critic. Find weaknesses and suggest improvements. Be concise.",
    }

    # Seed the conversation
    conversation = [
        "Design a simple system where AI agents can collaborate to solve problems better than any single agent."
    ]

    print(f"Seed: {conversation[0]}\n{'='*60}\n")

    for i in range(rounds):
        for name, system_prompt in agents.items():
            response = agent_turn(client, name, system_prompt, conversation)
            conversation.append(f"[{name}]: {response}")
            print(f"--- {name} (round {i+1}) ---")
            print(response)
            print()

    print("="*60)
    print(f"Conversation had {len(conversation)} messages over {rounds} rounds.")


if __name__ == "__main__":
    run()
