from middleware.config.db import SessionLocal, ClassifierPrompt, QuestionerPrompt, followUpQuestionerPrompt

def get_classifier_prompt(specialty_name: str, doctor_id: int) -> str:
    session = SessionLocal()
    try:
        prompt = session.query(ClassifierPrompt).filter(
            ClassifierPrompt.specialty_name == specialty_name,
            ClassifierPrompt.doctor_id == doctor_id
        ).first()
        return prompt.prompt_text if prompt else ""
    finally:
        session.close()

def get_questioner_prompt(prompt_key: str) -> str:
    session = SessionLocal()
    try:
        prompt = session.query(QuestionerPrompt).filter(
            QuestionerPrompt.prompt_key == prompt_key
        ).first()
        return prompt.prompt_text if prompt else ""
    finally:
        session.close()

def get_followup_questioner_prompt(prompt_key: str) -> str:
    session = SessionLocal()
    try:
        prompt = session.query(followUpQuestionerPrompt).filter(
            followUpQuestionerPrompt.prompt_key == prompt_key
        ).first()
        return prompt.prompt_text if prompt else None
    finally:
        session.close()