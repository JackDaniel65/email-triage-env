"""
Baseline inference script for EmailTriageEnv
Uses OpenAI-compatible API via Groq.
Set your API key: export OPENAI_API_KEY=your_groq_key_here
"""

import os
import json
from groq import Groq
from env.environment import EmailTriageEnv
from utils.models import Action

# reads from OPENAI_API_KEY env variable as required
client = Groq(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an email triage AI assistant. You will be given an email and a task to perform.
Always respond with a valid JSON object and nothing else. No explanation, no markdown, just raw JSON."""

TASK_PROMPTS = {
    1: 'Classify this email as spam or not_spam. Respond ONLY with: {"label": "spam"} or {"label": "not_spam"}',
    2: 'Classify priority (urgent/normal/low) and category (billing/support/feedback/other). Respond ONLY with: {"priority": "...", "category": "..."}',
    3: 'Draft a professional 2-4 sentence reply. Respond ONLY with: {"reply": "your reply here"}',
}


def run_agent_on_task(task_id: int) -> dict:
    env = EmailTriageEnv(task_id=task_id, shuffle=False)
    obs = env.reset()
    rewards = []
    step = 0

    print(f"\n{'='*50}")
    print(f"Running Task {task_id}")
    print(f"{'='*50}")

    while not env.state()["done"]:
        if obs.email_id == "END":
            break

        user_msg = f"Subject: {obs.subject}\nFrom: {obs.sender}\nBody: {obs.body}\n\n{TASK_PROMPTS[task_id]}"

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
        except Exception as e:
            print(f"  [step {step}] Agent error: {e}")
            parsed = {}

        action = Action(
            label=parsed.get("label"),
            priority=parsed.get("priority"),
            category=parsed.get("category"),
            reply=parsed.get("reply"),
        )

        result = env.step(action)
        rewards.append(result.reward)
        print(f"  Email: {obs.email_id} | Reward: {result.reward:.2f}")
        obs = result.observation
        step += 1

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nTask {task_id} Complete | Avg Reward: {avg:.4f}")
    return {"task_id": task_id, "avg_reward": round(avg, 4), "rewards": rewards}


def main():
    print("EmailTriageEnv - Baseline Inference Script")
    print("Agent: llama-3.3-70b-versatile (OpenAI-compatible API)\n")

    results = []
    for task_id in [1, 2, 3]:
        result = run_agent_on_task(task_id)
        results.append(result)

    print(f"\n{'='*50}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*50}")
    for r in results:
        print(f"Task {r['task_id']}: {r['avg_reward']:.4f}")

    overall = sum(r["avg_reward"] for r in results) / len(results)
    print(f"\nOverall Score: {overall:.4f}")

    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "overall": round(overall, 4)}, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
