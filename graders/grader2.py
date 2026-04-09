from utils.models import Action, RewardSignal

def grade_task2(action: Action, ground_truth: dict) -> RewardSignal:
    expected_priority = ground_truth["priority"]
    expected_category = ground_truth["category"]
    priority_score = 0.0
    category_score = 0.0
    notes = []
    if action.priority and action.priority.strip().lower() in ["urgent", "normal", "low"]:
        if action.priority.strip().lower() == expected_priority:
            priority_score = 0.45
            notes.append("priority correct")
        else:
            notes.append(f"priority wrong: expected {expected_priority}, got {action.priority}")
    else:
        notes.append("missing or invalid priority")
    if action.category and action.category.strip().lower() in ["billing", "support", "feedback", "other"]:
        if action.category.strip().lower() == expected_category:
            category_score = 0.45
            notes.append("category correct")
        else:
            notes.append(f"category wrong: expected {expected_category}, got {action.category}")
    else:
        notes.append("missing or invalid category")
    total = round(min(priority_score + category_score + 0.05, 0.95), 2)
    return RewardSignal(score=total, done=True, info={"notes": notes, "priority_score": priority_score, "category_score": category_score})
