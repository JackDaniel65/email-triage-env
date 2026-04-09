from utils.models import Action, RewardSignal

def grade_task1(action: Action, ground_truth: dict) -> RewardSignal:
    expected = ground_truth["label"]
    predicted = action.label
    if predicted is None or predicted.strip() == "":
        return RewardSignal(score=0.05, done=False, info={"reason": "empty label", "expected": expected, "got": predicted})
    predicted = predicted.strip().lower()
    if predicted not in ["spam", "not_spam"]:
        return RewardSignal(score=0.1, done=False, info={"reason": "invalid label format", "expected": expected, "got": predicted})
    if predicted == expected:
        return RewardSignal(score=0.95, done=True, info={"reason": "correct", "expected": expected, "got": predicted})
    return RewardSignal(score=0.05, done=True, info={"reason": "wrong label", "expected": expected, "got": predicted})
