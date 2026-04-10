---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📧 EmailTriageEnv

> A production-grade OpenEnv environment where AI agents learn to triage customer emails.

## Baseline Scores

| Task | Difficulty | Score |
|------|-----------|-------|
| Spam Detection | Easy | 0.950 |
| Priority + Category | Medium | 0.905 |
| Reply Drafting | Hard | 0.677 |
| **Overall** | | **0.844** |

## Tasks

**Task 1 — Spam Detection (Easy)**
Classify each email as spam or not_spam. Binary reward signal.

**Task 2 — Priority + Category (Medium)**
Assign priority (urgent/normal/low) and category (billing/support/feedback/other). Partial credit scoring.

**Task 3 — Reply Drafting (Hard)**
Draft a professional reply. Scored on keywords, tone, professionalism, and length.

## Action Space

```python
class Action(BaseModel):
    label: Optional[str]      # spam | not_spam
    priority: Optional[str]   # urgent | normal | low
    category: Optional[str]   # billing | support | feedback | other
    reply: Optional[str]      # reply text
```

## Observation Space

```python
class Observation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    task_id: int
    step_number: int
```

## Setup

```bash
pip install -r requirements.txt
python main.py

export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_groq_key
python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/reset` | Reset environment |
| POST | `/step` | Submit action |
| GET | `/state` | Current state |
| GET | `/tasks` | List tasks |
| GET | `/grader` | Episode scores |
| GET | `/baseline` | Baseline scores |
