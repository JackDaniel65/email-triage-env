from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env.environment import EmailTriageEnv
from utils.models import Action

app = FastAPI()

sessions = {}

BASELINE_RESULTS = {
    "results": [
        {"task_id": 1, "avg_reward": 1.0, "rewards": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},
        {"task_id": 2, "avg_reward": 0.95, "rewards": [1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,1.0,1.0]},
        {"task_id": 3, "avg_reward": 0.6058, "rewards": [0.4,0.4,0.65,0.3,0.875,0.5,0.4,0.8,0.9,0.833]}
    ],
    "overall": 0.8519
}

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
            {"task_id": 1, "name": "spam_detection", "difficulty": "easy", "description": "Classify email as spam or not_spam", "action_schema": {"label": "spam | not_spam"}},
            {"task_id": 2, "name": "priority_categorization", "difficulty": "medium", "description": "Assign priority and category to email", "action_schema": {"priority": "urgent | normal | low", "category": "billing | support | feedback | other"}},
            {"task_id": 3, "name": "reply_drafting", "difficulty": "hard", "description": "Draft a professional reply to the email", "action_schema": {"reply": "string"}},
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
    return BASELINE_RESULTS
