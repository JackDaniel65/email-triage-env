from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from env.environment import EmailTriageEnv
from utils.models import Action

app = FastAPI(title="EmailTriageEnv", description="A real-world OpenEnv environment for AI email triage agents")

sessions = {}

BASELINE_RESULTS = {
    "results": [
        {"task_id": 1, "avg_reward": 0.95, "rewards": [0.95]*20},
        {"task_id": 2, "avg_reward": 0.905, "rewards": [0.95,0.95,0.95,0.95,0.50,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.50]},
        {"task_id": 3, "avg_reward": 0.677, "rewards": [0.46,0.61,0.77,0.33,0.77,0.61,0.46,0.88,0.87,0.78,0.38,0.68,0.88,0.41,0.80,0.83,0.88,0.89,0.49,0.75]}
    ],
    "overall": 0.844
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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EmailTriageEnv</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
  :root{--bg:#080c10;--surface:#0e1420;--border:#1e2d40;--green:#00e676;--blue:#29b6f6;--orange:#ffa726;--red:#ef5350;--text:#e0e6ed;--muted:#546e7a;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;}
  .hero{background:linear-gradient(135deg,#080c10 0%,#0a1628 50%,#080c10 100%);border-bottom:1px solid var(--border);padding:60px 40px 40px;text-align:center;position:relative;overflow:hidden;}
  .hero::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle at 50% 50%,rgba(0,230,118,0.03) 0%,transparent 60%);animation:pulse 4s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:0.5}50%{opacity:1}}
  .badge{display:inline-block;background:rgba(0,230,118,0.1);border:1px solid rgba(0,230,118,0.3);color:var(--green);padding:4px 14px;border-radius:20px;font-size:0.75rem;font-family:'JetBrains Mono',monospace;letter-spacing:1px;margin-bottom:20px;}
  h1{font-size:3rem;font-weight:700;background:linear-gradient(135deg,#fff 0%,#00e676 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;}
  .subtitle{color:var(--muted);font-size:1.1rem;font-weight:300;max-width:600px;margin:0 auto 40px;line-height:1.6;}
  .overall-score{display:inline-flex;flex-direction:column;align-items:center;background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.2);border-radius:16px;padding:24px 48px;margin-top:10px;}
  .score-num{font-size:3.5rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--green);line-height:1;}
  .score-label{color:var(--muted);font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin-top:6px;}
  .container{max-width:1100px;margin:0 auto;padding:40px;}
  .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:40px;}
  .task-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;transition:border-color 0.2s,transform 0.2s;}
  .task-card:hover{border-color:var(--green);transform:translateY(-2px);}
  .task-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;}
  .task-name{font-weight:600;font-size:1rem;}
  .diff-badge{padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
  .easy{background:rgba(0,230,118,0.1);color:var(--green);}
  .medium{background:rgba(255,167,38,0.1);color:var(--orange);}
  .hard{background:rgba(239,83,80,0.1);color:var(--red);}
  .score-bar-bg{background:var(--border);border-radius:4px;height:6px;margin:12px 0;}
  .score-bar{height:6px;border-radius:4px;background:linear-gradient(90deg,var(--green),var(--blue));transition:width 1s ease;}
  .task-score{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:var(--green);}
  .task-desc{color:var(--muted);font-size:0.85rem;margin-top:8px;line-height:1.5;}
  .section-title{font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;color:var(--muted);margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid var(--border);}
  .endpoints-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:40px;}
  .endpoint-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 20px;display:flex;align-items:center;gap:14px;text-decoration:none;color:inherit;transition:border-color 0.2s;}
  .endpoint-card:hover{border-color:var(--blue);}
  .method-badge{font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;padding:3px 8px;border-radius:4px;min-width:44px;text-align:center;}
  .get{background:rgba(41,182,246,0.1);color:var(--blue);}
  .post{background:rgba(255,167,38,0.1);color:var(--orange);}
  .ep-info{flex:1;}
  .ep-path{font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:var(--text);}
  .ep-desc{font-size:0.78rem;color:var(--muted);margin-top:2px;}
  .links-row{display:flex;gap:16px;margin-bottom:40px;}
  .link-btn{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 20px;color:var(--text);text-decoration:none;font-size:0.9rem;transition:border-color 0.2s,color 0.2s;}
  .link-btn:hover{border-color:var(--green);color:var(--green);}
  .info-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:40px;}
  .info-row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);font-size:0.9rem;}
  .info-row:last-child{border-bottom:none;}
  .info-key{color:var(--muted);}
  .info-val{font-family:'JetBrains Mono',monospace;color:var(--green);}
  @media(max-width:768px){h1{font-size:2rem;}.grid-3{grid-template-columns:1fr;}.endpoints-grid{grid-template-columns:1fr;}.links-row{flex-direction:column;}.container{padding:20px;}}
</style>
</head>
<body>
<div class="hero">
  <div class="badge">OPENENV COMPATIBLE</div>
  <h1>📧 EmailTriageEnv</h1>
  <p class="subtitle">A production-grade reinforcement learning environment where AI agents learn to triage customer emails — from spam detection to intelligent reply drafting.</p>
  <div class="overall-score">
    <div class="score-num">0.844</div>
    <div class="score-label">Overall Baseline Score</div>
  </div>
</div>
<div class="container">
  <p class="section-title">Task Performance</p>
  <div class="grid-3">
    <div class="task-card">
      <div class="task-header"><span class="task-name">Spam Detection</span><span class="diff-badge easy">Easy</span></div>
      <div class="task-score">0.950</div>
      <div class="score-bar-bg"><div class="score-bar" style="width:95%"></div></div>
      <div class="task-desc">Classify each email as spam or not_spam with binary reward signal.</div>
    </div>
    <div class="task-card">
      <div class="task-header"><span class="task-name">Priority + Category</span><span class="diff-badge medium">Medium</span></div>
      <div class="task-score">0.905</div>
      <div class="score-bar-bg"><div class="score-bar" style="width:90.5%"></div></div>
      <div class="task-desc">Assign priority and category with partial credit scoring.</div>
    </div>
    <div class="task-card">
      <div class="task-header"><span class="task-name">Reply Drafting</span><span class="diff-badge hard">Hard</span></div>
      <div class="task-score">0.677</div>
      <div class="score-bar-bg"><div class="score-bar" style="width:67.7%"></div></div>
      <div class="task-desc">Draft a professional reply scored on keywords, tone, and length.</div>
    </div>
  </div>
  <p class="section-title">API Endpoints</p>
  <div class="endpoints-grid">
    <a href="/reset" class="endpoint-card"><span class="method-badge get">GET</span><div class="ep-info"><div class="ep-path">/reset</div><div class="ep-desc">Reset environment, get first observation</div></div></a>
    <div class="endpoint-card"><span class="method-badge post">POST</span><div class="ep-info"><div class="ep-path">/step</div><div class="ep-desc">Submit action, receive reward + next obs</div></div></div>
    <a href="/state" class="endpoint-card"><span class="method-badge get">GET</span><div class="ep-info"><div class="ep-path">/state</div><div class="ep-desc">Current episode state and running scores</div></div></a>
    <a href="/tasks" class="endpoint-card"><span class="method-badge get">GET</span><div class="ep-info"><div class="ep-path">/tasks</div><div class="ep-desc">List all tasks and action schemas</div></div></a>
    <a href="/grader" class="endpoint-card"><span class="method-badge get">GET</span><div class="ep-info"><div class="ep-path">/grader</div><div class="ep-desc">Grader scores after episode completion</div></div></a>
    <a href="/baseline" class="endpoint-card"><span class="method-badge get">GET</span><div class="ep-info"><div class="ep-path">/baseline</div><div class="ep-desc">Baseline inference scores for all 3 tasks</div></div></a>
  </div>
  <p class="section-title">Environment Info</p>
  <div class="info-box">
    <div class="info-row"><span class="info-key">Dataset Size</span><span class="info-val">20 emails</span></div>
    <div class="info-row"><span class="info-key">Tasks</span><span class="info-val">3 (easy → medium → hard)</span></div>
    <div class="info-row"><span class="info-key">Reward Range</span><span class="info-val">(0.05, 0.95) exclusive</span></div>
    <div class="info-row"><span class="info-key">Baseline Model</span><span class="info-val">llama-3.3-70b-versatile</span></div>
    <div class="info-row"><span class="info-key">API</span><span class="info-val">OpenAI-compatible</span></div>
    <div class="info-row"><span class="info-key">Framework</span><span class="info-val">OpenEnv + FastAPI</span></div>
  </div>
  <p class="section-title">Links</p>
  <div class="links-row">
    <a href="https://github.com/JackDaniel65/email-triage-env" target="_blank" class="link-btn">⭐ GitHub Repository</a>
    <a href="https://huggingface.co/spaces/darkdarkdark234/email-triage-env" target="_blank" class="link-btn">🤗 Hugging Face Space</a>
    <a href="/docs" target="_blank" class="link-btn">📄 API Docs (Swagger)</a>
  </div>
</div>
</body>
</html>"""

@app.get("/health")
def health():
    return {"name": "EmailTriageEnv", "version": "1.0.0", "status": "running"}

@app.api_route("/reset", methods=["GET", "POST"])
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
    return {"observation": result.observation.model_dump(), "reward": result.reward, "done": result.done, "info": result.info}

@app.get("/state")
def state(task_id: int = 1):
    env = get_env(task_id)
    return env.state()

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"task_id": 1, "name": "spam_detection", "difficulty": "easy", "description": "Classify email as spam or not_spam", "action_schema": {"label": "spam | not_spam"}},
        {"task_id": 2, "name": "priority_categorization", "difficulty": "medium", "description": "Assign priority and category to email", "action_schema": {"priority": "urgent | normal | low", "category": "billing | support | feedback | other"}},
        {"task_id": 3, "name": "reply_drafting", "difficulty": "hard", "description": "Draft a professional reply to the email", "action_schema": {"reply": "string"}},
    ]}

@app.get("/grader")
def grader(task_id: int = 1):
    env = get_env(task_id)
    s = env.state()
    return {"task_id": task_id, "avg_reward": s["avg_reward"], "total_reward": s["total_reward"], "episode_rewards": s["episode_rewards"], "done": s["done"]}

@app.get("/baseline")
def baseline():
    return BASELINE_RESULTS

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
