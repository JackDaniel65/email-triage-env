from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from utils.models import Action

app = FastAPI()

# store environment sessions in memory
sessions = {}


def get_env(task_id: int = 1) -> EmailTriageEnv:
    if task_id not in sessions:
        sessions[task_id] = EmailTriageEnv(task_id=task_id, shuffle=False)
        sessions[task_id].reset()
    return sessions[task_id]


class ActionRequest(BaseModel):
    task_id: int = 1
    label: Optional[str] = None
    priority: Optional[str] = None
    category: Optional[str] = None
    reply: Optional[str] = None


@app.get("/")
def root():
    return {"name": "EmailTriageEnv", "version": "1.0.0", "status": "running"}


@app.get("/reset")
def reset(task_id: int = 1):
    env = EmailTriageEnv(task_id=task_id, shuffle=False)
    sessions[task_id] = env
    obs = env.reset()
    return {"observation": obs.model_dump(), "status": "reset complete"}


@app.post("/step")
def step(request: ActionRequest):
    env = get_env(request.task_id)
    action = Action(
        label=request.label,
        priority=request.priority,
        category=request.category,
        reply=request.reply,
    )
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state(task_id: int = 1):
    env = get_env(task_id)
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": 1,
                "name": "spam_detection",
                "difficulty": "easy",
                "description": "Classify email as spam or not_spam",
                "action_schema": {"label": "spam | not_spam"},
            },
            {
                "task_id": 2,
                "name": "priority_categorization",
                "difficulty": "medium",
                "description": "Assign priority and category to email",
                "action_schema": {
                    "priority": "urgent | normal | low",
                    "category": "billing | support | feedback | other",
                },
            },
            {
                "task_id": 3,
                "name": "reply_drafting",
                "difficulty": "hard",
                "description": "Draft a professional reply to the email",
                "action_schema": {"reply": "string"},
            },
        ]
    }


@app.get("/grader")
def grader(task_id: int = 1):
    env = get_env(task_id)
    s = env.state()
    return {
        "task_id": task_id,
        "avg_reward": s["avg_reward"],
        "total_reward": s["total_reward"],
        "episode_rewards": s["episode_rewards"],
        "done": s["done"],
    }


@app.get("/baseline")
def baseline():
    import json
    import os
    if os.path.exists("baseline_results.json"):
        with open("baseline_results.json") as f:
            return json.load(f)
    return {"error": "baseline_results.json not found, run baseline.py first"}
