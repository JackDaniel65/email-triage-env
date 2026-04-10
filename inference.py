import os
import json
from typing import List, Optional
from openai import OpenAI
from env.environment import EmailTriageEnv
from utils.models import Action

# mandatory env variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "email-triage-env"
MAX_STEPS = 20


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = """You are an email triage AI assistant. You will be given an email and a task.
Always respond with a valid JSON object and nothing else. No explanation, no markdown, just raw JSON."""

TASK_PROMPTS = {
    1: 'Classify this email as spam or not_spam. Respond ONLY with: {"label": "spam"} or {"label": "not_spam"}',
    2: 'Classify priority (urgent/normal/low) and category (billing/support/feedback/other). Respond ONLY with: {"priority": "...", "category": "..."}',
    3: 'Draft a professional 2-4 sentence reply. Respond ONLY with: {"reply": "your reply here"}',
}

TASK_NAMES = {
    1: "spam_detection",
    2: "priority_categorization",
    3: "reply_drafting",
}


def get_model_action(client: OpenAI, obs, task_id: int) -> dict:
    user_msg = f"Subject: {obs.subject}\nFrom: {obs.sender}\nBody: {obs.body}\n\n{TASK_PROMPTS[task_id]}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = completion.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {}


def run_task(client: OpenAI, task_id: int) -> dict:
    task_name = TASK_NAMES[task_id]
    env = EmailTriageEnv(task_id=task_id, shuffle=False)
    obs = env.reset()

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.state()["done"] or obs.email_id == "END":
                break

            parsed = get_model_action(client, obs, task_id)

            action = Action(
                label=parsed.get("label"),
                priority=parsed.get("priority"),
                category=parsed.get("category"),
                reply=parsed.get("reply"),
            )

            result = env.step(action)
            reward = result.reward
            done = result.done
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(parsed)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            obs = result.observation

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "task_name": task_name, "score": score, "rewards": rewards}


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task_id in [1, 2, 3]:
        result = run_task(client, task_id)
        results.append(result)

    overall = sum(r["score"] for r in results) / len(results)

    print(f"\n[SUMMARY] overall={overall:.3f}", flush=True)
    for r in results:
        print(f"[SUMMARY] task={r['task_name']} score={r['score']:.3f}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "results": [{"task_id": r["task_id"], "avg_reward": r["score"], "rewards": r["rewards"]} for r in results],
            "overall": round(overall, 4)
        }, f, indent=2)


if __name__ == "__main__":
    main()
