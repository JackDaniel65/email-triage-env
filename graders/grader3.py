from utils.models import Action, RewardSignal

def grade_task3(action: Action, ground_truth: dict) -> RewardSignal:
    ideal_keywords = ground_truth.get("ideal_reply_keywords", [])
    reply = action.reply
    if reply is None or reply.strip() == "":
        return RewardSignal(score=0.0, done=True, info={"reason": "empty reply"})
    reply_lower = reply.strip().lower()
    word_count = len(reply_lower.split())
    if word_count < 10:
        return RewardSignal(score=0.1, done=True, info={"reason": "reply too short", "word_count": word_count})
    score = 0.0
    notes = []
    if ideal_keywords:
        matched = [kw for kw in ideal_keywords if kw.lower() in reply_lower]
        keyword_score = round((len(matched) / len(ideal_keywords)) * 0.5, 3)
        score += keyword_score
        notes.append(f"keywords matched: {matched}")
    tone_words = ["thank", "sorry", "please", "happy to help", "assist", "team", "appreciate"]
    tone_hits = [w for w in tone_words if w in reply_lower]
    score += min(len(tone_hits) * 0.1, 0.3)
    notes.append(f"tone signals: {tone_hits}")
    score += 0.2 if 20 <= word_count <= 120 else 0.1
    notes.append(f"word count: {word_count}")
    return RewardSignal(score=round(min(score, 1.0), 3), done=True, info={"notes": notes, "word_count": word_count})
