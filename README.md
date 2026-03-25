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

# EmailTriageEnv

A real-world OpenEnv environment where an AI agent learns to triage customer emails — classifying spam, assigning priorities, and drafting professional replies.

---

## Motivation

Email triage is one of the most common real-world tasks handled by AI assistants today. This environment lets an agent practice the full spectrum — from simple spam detection to nuanced reply drafting — with graded feedback at every step.

---

## Environment Description

The agent is presented with a stream of emails one at a time. Depending on the active task, it must:

- **Task 1 (Easy):** Classify each email as `spam` or `not_spam`
- **Task 2 (Medium):** Assign a `priority` (urgent/normal/low) and `category` (billing/support/feedback/other)
- **Task 3 (Hard):** Draft a short, professional reply to the email

---

## Action Space
```python
class Action(BaseModel):
    label: Optional[str]      # "spam" or "not_spam" — Task 1
    priority: Optional[str]   # "urgent", "normal", "low" — Task 2
    category: Optional[str]   # "billing", "support", "feedback", "other" — Task 2
    reply: Optional[str]      # reply text — Task 3
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

---

## Reward Function

| Task | Reward Logic |
|------|-------------|
| Task 1 | 1.0 for correct label, 0.0 for wrong |
| Task 2 | 0.5 for correct priority + 0.5 for correct category |
| Task 3 | Partial credit based on keyword match, tone, and length |

---

## Setup & Usage
```bash
pip install -r requirements.txt
python main.py
```

### Baseline agent (requires Groq API key)
```bash
export GROQ_API_KEY=your_key_here
python baseline.py
```

### Docker
```bash
docker build -t email-triage-env .
docker run email-triage-env
```

---

## Baseline Scores

| Task | Difficulty | Score |
|------|-----------|-------|
| 1 | Easy | ~0.85 |
| 2 | Medium | ~0.65 |
| 3 | Hard | ~0.45 |
