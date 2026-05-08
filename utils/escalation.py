def should_escalate_to_human(score: float, user_message: str) -> bool:
    if score < 0.6:
        return True
    if any(word in user_message.lower() for word in ["human", "agent", "person", "speak to"]):
        return True
    return False