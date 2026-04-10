from utils.models import Action, RewardSignal


def grade_task3(action: Action, ground_truth: dict) -> RewardSignal:
    ideal_keywords = ground_truth.get("ideal_reply_keywords", [])
    reply = action.reply

    if reply is None or reply.strip() == "":
        return RewardSignal(score=0.05, done=True, info={"reason": "empty reply"})

    reply_lower = reply.strip().lower()
    word_count = len(reply_lower.split())

    if word_count < 10:
        return RewardSignal(score=0.1, done=True, info={"reason": "reply too short", "word_count": word_count})

    score = 0.05
    notes = []

    # keyword matching - up to 0.35
    if ideal_keywords:
        matched = [kw for kw in ideal_keywords if kw.lower() in reply_lower]
        keyword_score = round((len(matched) / len(ideal_keywords)) * 0.35, 3)
        score += keyword_score
        notes.append(f"keywords matched: {matched} ({len(matched)}/{len(ideal_keywords)})")
    else:
        # spam email - short dismissive reply is ok
        if word_count < 20:
            score += 0.15
        notes.append("spam email - no reply needed")

    # tone signals - up to 0.30
    tone_words = ["thank", "sorry", "apologize", "please", "assist", "team", "appreciate", "help", "resolve", "investigate", "review"]
    tone_hits = [w for w in tone_words if w in reply_lower]
    tone_score = min(len(tone_hits) * 0.08, 0.30)
    score += tone_score
    notes.append(f"tone signals: {tone_hits}")

    # professionalism signals - up to 0.15
    prof_words = ["dear", "regards", "sincerely", "best", "contact", "please feel free", "let us know", "we will", "our team"]
    prof_hits = [w for w in prof_words if w in reply_lower]
    prof_score = min(len(prof_hits) * 0.05, 0.15)
    score += prof_score
    notes.append(f"professionalism: {prof_hits}")

    # length bonus - up to 0.15
    if 25 <= word_count <= 100:
        score += 0.15
    elif 15 <= word_count <= 150:
        score += 0.10
    else:
        score += 0.05
    notes.append(f"word count: {word_count}")

    final = round(min(max(score, 0.05), 0.95), 3)
    return RewardSignal(score=final, done=True, info={"notes": notes, "word_count": word_count})
