from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>EmailTriageEnv</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #e0e0e0; padding: 40px; }
        h1 { color: #00ff88; font-size: 2rem; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 40px; }
        .section { margin-bottom: 36px; }
        .section h2 { color: #00cc66; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; border-bottom: 1px solid #222; padding-bottom: 8px; }
        .card { background: #151515; border: 1px solid #222; border-radius: 8px; padding: 20px; margin-bottom: 12px; }
        .endpoint { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
        .method { background: #003322; color: #00ff88; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
        .method.post { background: #1a1a00; color: #ffcc00; }
        .url { color: #00aaff; font-size: 0.9rem; }
        .desc { color: #666; font-size: 0.82rem; margin-left: 60px; margin-bottom: 8px; }
        .task { display: flex; justify-content: space-between; align-items: center; padding: 14px 0; border-bottom: 1px solid #1a1a1a; }
        .task:last-child { border-bottom: none; }
        .task-name { font-size: 0.95rem; }
        .badge { padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; }
        .easy { background: #003322; color: #00ff88; }
        .medium { background: #1a1200; color: #ffaa00; }
        .hard { background: #1a0000; color: #ff4444; }
        .score { color: #00ff88; font-size: 1.1rem; font-weight: bold; }
        .overall { background: #0a1f0a; border: 1px solid #00ff88; border-radius: 8px; padding: 20px; text-align: center; }
        .overall-score { color: #00ff88; font-size: 2.5rem; font-weight: bold; }
        .overall-label { color: #888; font-size: 0.85rem; margin-top: 4px; }
        a { color: #00aaff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>📧 EmailTriageEnv</h1>
    <p class="subtitle">A real-world OpenEnv environment for AI email triage agents</p>

    <div class="section">
        <h2>Baseline Scores</h2>
        <div class="card">
            <div class="task">
                <span class="task-name">Task 1 — Spam Detection</span>
                <span class="badge easy">Easy</span>
                <span class="score">1.0000</span>
            </div>
            <div class="task">
                <span class="task-name">Task 2 — Priority + Category</span>
                <span class="badge medium">Medium</span>
                <span class="score">0.9500</span>
            </div>
            <div class="task">
                <span class="task-name">Task 3 — Reply Drafting</span>
                <span class="badge hard">Hard</span>
                <span class="score">0.6058</span>
            </div>
        </div>
        <div class="overall">
            <div class="overall-score">0.8519</div>
            <div class="overall-label">Overall Baseline Score</div>
        </div>
    </div>

    <div class="section">
        <h2>API Endpoints</h2>
        <div class="card">
            <div class="endpoint"><span class="method">GET</span><span class="url">/reset?task_id=1</span></div>
            <div class="desc">Reset environment and get first observation</div>
            <div class="endpoint"><span class="method post">POST</span><span class="url">/step</span></div>
            <div class="desc">Submit an action and get reward + next observation</div>
            <div class="endpoint"><span class="method">GET</span><span class="url">/state?task_id=1</span></div>
            <div class="desc">Get current episode state and running scores</div>
            <div class="endpoint"><span class="method">GET</span><span class="url">/tasks</span></div>
            <div class="desc">List all tasks and their action schemas</div>
            <div class="endpoint"><span class="method">GET</span><span class="url">/grader?task_id=1</span></div>
            <div class="desc">Get grader scores after episode completes</div>
            <div class="endpoint"><span class="method">GET</span><span class="url">/baseline</span></div>
            <div class="desc">Get baseline inference scores for all 3 tasks</div>
        </div>
    </div>

    <div class="section">
        <h2>Links</h2>
        <div class="card">
            <p>📦 <a href="https://github.com/JackDaniel65/email-triage-env" target="_blank">GitHub Repository</a></p>
            <br>
            <p>🤗 <a href="https://huggingface.co/spaces/darkdarkdark234/email-triage-env" target="_blank">Hugging Face Space</a></p>
            <br>
            <p>📄 <a href="/docs" target="_blank">API Documentation (Swagger)</a></p>
        </div>
    </div>
</body>
</html>
"""

@app.get("/health")
def health():
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
