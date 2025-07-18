from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from config.db import get_db, ClassifierPrompt, QuestionerPrompt, followUpClassifierPrompt, followUpQuestionerPrompt, create_tables

app = FastAPI(title="Prompt Management API")

create_tables()

@app.get("/")
def root():
    return {"message": "Prompt Management API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "endpoints": [
        "GET /classifier-prompts/all",
        "GET /classifier-prompts/doctor/{doctor_id}",
        "GET /classifier-prompts/{specialty_name}/{doctor_id}",
        "POST /classifier-prompts",
        "POST /classifier-prompts/upload",
        "PUT /classifier-prompts/{id}",
        "PATCH /classifier-prompts/upload/{id}",
        "GET /questioner-prompts/all",
        "GET /questioner-prompts/classifier/{classifier_id}",
        "POST /questioner-prompts",
        "POST /questioner-prompts/upload",
        "PUT /questioner-prompts/{id}",
        "PUT /questioner-prompts/upload/{id}",
        "GET /followup-classifier-prompts/all",
        "GET /followup-classifier-prompts/doctor/{doctor_id}",
        "GET /followup-classifier-prompts/{specialty_name}/{doctor_id}",
        "POST /followup-classifier-prompts",
        "POST /followup-classifier-prompts/upload",
        "PUT /followup-classifier-prompts/{id}",
        "PATCH /followup-classifier-prompts/upload/{id}",
        "GET /followup-questioner-prompts/all",
        "GET /followup-questioner-prompts/classifier/{classifier_id}",
        "POST /followup-questioner-prompts",
        "POST /followup-questioner-prompts/upload",
        "PUT /followup-questioner-prompts/{id}",
        "PUT /followup-questioner-prompts/upload/{id}"
    ]}

@app.get("/test")
def test_endpoint():
    return {"message": "test endpoint working"}

class ClassifierPromptCreate(BaseModel):
    specialty_name: str
    doctor_id: int
    prompt_text: str
    version: str = "1.0"
    is_active: bool = True
    is_default: bool = False

class QuestionerPromptCreate(BaseModel):
    classifier_prompt_ids: List[int]
    prompt_key: str
    prompt_text: str
    summary_prompt: Optional[str] = None
    version: str = "1.0"
    is_active: bool = True

class FollowUpClassifierPromptCreate(BaseModel):
    specialty_name: str
    doctor_id: int
    prompt_text: str
    version: str = "1.0"
    is_active: bool = True
    is_default: bool = False

class FollowUpQuestionerPromptCreate(BaseModel):
    classifier_prompt_ids: List[int]
    prompt_key: str
    prompt_text: str
    summary_prompt: Optional[str] = None
    version: str = "1.0"
    is_active: bool = True

@app.get("/classifier-prompts/all")
def get_all_classifier_prompts(db: Session = Depends(get_db)):
    return db.query(ClassifierPrompt).all()

@app.get("/classifier-prompts/doctor/{doctor_id}")
def get_classifier_prompts_by_doctor(doctor_id: int, db: Session = Depends(get_db)):
    prompts = db.query(ClassifierPrompt).filter(
        ClassifierPrompt.doctor_id == doctor_id
    ).all()
    
    if not prompts:
        raise HTTPException(status_code=404, detail="No classifier prompts found for this doctor")
    
    return prompts

@app.get("/classifier-prompts/{specialty_name}/{doctor_id}")
def get_classifier_prompt(specialty_name: str, doctor_id: int, db: Session = Depends(get_db)):
    prompt = db.query(ClassifierPrompt).filter(
        ClassifierPrompt.specialty_name == specialty_name,
        ClassifierPrompt.doctor_id == doctor_id
    ).first()
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Classifier prompt not found")
    
    return prompt

@app.post("/classifier-prompts")
def create_classifier_prompt(prompt: ClassifierPromptCreate, db: Session = Depends(get_db)):
    existing = db.query(ClassifierPrompt).filter(
        ClassifierPrompt.specialty_name == prompt.specialty_name,
        ClassifierPrompt.doctor_id == prompt.doctor_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Classifier prompt already exists for this specialty and doctor")
    
    db_prompt = ClassifierPrompt(**prompt.model_dump())
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.post("/classifier-prompts/upload")
async def upload_classifier_prompt(
    file: UploadFile = File(...),
    specialty_name: str = Form(...),
    doctor_id: int = Form(...),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    is_default: bool = Form(False),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    existing = db.query(ClassifierPrompt).filter(
        ClassifierPrompt.specialty_name == specialty_name,
        ClassifierPrompt.doctor_id == doctor_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Classifier prompt already exists for this specialty and doctor")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt = ClassifierPrompt(
        specialty_name=specialty_name,
        doctor_id=doctor_id,
        prompt_text=prompt_text,
        version=version,
        is_active=is_active,
        is_default=is_default
    )
    
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Classifier prompt uploaded successfully",
        "id": db_prompt.id,
        "specialty_name": db_prompt.specialty_name,
        "doctor_id": db_prompt.doctor_id,
        "version": db_prompt.version,
        "text_length": len(prompt_text)
    }

@app.put("/classifier-prompts/{prompt_id}")
def update_classifier_prompt(prompt_id: int, prompt: ClassifierPromptCreate, db: Session = Depends(get_db)):
    db_prompt = db.query(ClassifierPrompt).filter(ClassifierPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Classifier prompt not found")
    
    for key, value in prompt.model_dump().items():
        setattr(db_prompt, key, value)
    
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.patch("/classifier-prompts/upload/{prompt_id}")
async def patch_classifier_prompt_upload(
    prompt_id: int,
    file: UploadFile = File(...),
    specialty_name: str = Form(...),
    doctor_id: int = Form(...),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    is_default: bool = Form(False),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    db_prompt = db.query(ClassifierPrompt).filter(ClassifierPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Classifier prompt not found")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt.specialty_name = specialty_name
    db_prompt.doctor_id = doctor_id
    db_prompt.prompt_text = prompt_text
    db_prompt.version = version
    db_prompt.is_active = is_active
    db_prompt.is_default = is_default
    
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Classifier prompt updated successfully",
        "id": db_prompt.id,
        "specialty_name": db_prompt.specialty_name,
        "doctor_id": db_prompt.doctor_id,
        "version": db_prompt.version,
        "text_length": len(prompt_text)
    }

@app.get("/questioner-prompts/all")
def get_all_questioner_prompts(db: Session = Depends(get_db)):
    return db.query(QuestionerPrompt).all()

@app.get("/questioner-prompts/key/{prompt_key}")
def get_questioner_prompt_by_key(prompt_key: str, db: Session = Depends(get_db)):
    prompt = db.query(QuestionerPrompt).filter(
        QuestionerPrompt.prompt_key == prompt_key
    ).first()
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Questioner prompt not found for this key")
    
    return prompt

@app.get("/questioner-prompts/classifier/{classifier_id}")
def get_questioner_prompts_by_classifier(classifier_id: int, db: Session = Depends(get_db)):
    prompts = db.query(QuestionerPrompt).filter(
        QuestionerPrompt.classifier_prompt_ids.contains([classifier_id])
    ).all()
    
    if not prompts:
        raise HTTPException(status_code=404, detail="No questioner prompts found for this classifier")
    
    return prompts

@app.post("/questioner-prompts")
def create_questioner_prompt(prompt: QuestionerPromptCreate, db: Session = Depends(get_db)):
    for classifier_id in prompt.classifier_prompt_ids:
        classifier_exists = db.query(ClassifierPrompt).filter(
            ClassifierPrompt.id == classifier_id
        ).first()
        
        if not classifier_exists:
            raise HTTPException(status_code=400, detail=f"Classifier prompt ID {classifier_id} does not exist")
    
    existing = db.query(QuestionerPrompt).filter(
        QuestionerPrompt.prompt_key == prompt.prompt_key
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Questioner prompt already exists for this key")
    
    db_prompt = QuestionerPrompt(**prompt.model_dump())
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.post("/questioner-prompts/upload")
async def upload_questioner_prompt(
    file: UploadFile = File(...),
    classifier_prompt_ids: str = Form(...),
    prompt_key: str = Form(...),
    summary_prompt: Optional[str] = Form(None),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    try:
        classifier_ids = [int(x.strip()) for x in classifier_prompt_ids.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="classifier_prompt_ids must be comma-separated integers")
    
    for classifier_id in classifier_ids:
        classifier_exists = db.query(ClassifierPrompt).filter(
            ClassifierPrompt.id == classifier_id
        ).first()
        
        if not classifier_exists:
            raise HTTPException(status_code=400, detail=f"Classifier prompt ID {classifier_id} does not exist")
    
    existing = db.query(QuestionerPrompt).filter(
        QuestionerPrompt.prompt_key == prompt_key
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Questioner prompt already exists for this key")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt = QuestionerPrompt(
        classifier_prompt_ids=classifier_ids,
        prompt_key=prompt_key,
        prompt_text=prompt_text,
        summary_prompt=summary_prompt,
        version=version,
        is_active=is_active
    )
    
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Questioner prompt uploaded successfully",
        "id": db_prompt.id,
        "classifier_prompt_ids": db_prompt.classifier_prompt_ids,
        "prompt_key": db_prompt.prompt_key,
        "text_length": len(prompt_text)
    }

@app.put("/questioner-prompts/{prompt_id}")
def update_questioner_prompt(prompt_id: int, prompt: QuestionerPromptCreate, db: Session = Depends(get_db)):
    db_prompt = db.query(QuestionerPrompt).filter(QuestionerPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Questioner prompt not found")
    
    for key, value in prompt.model_dump().items():
        setattr(db_prompt, key, value)
    
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.put("/questioner-prompts/upload/{prompt_id}")
async def update_questioner_prompt_upload(
    prompt_id: int,
    file: UploadFile = File(...),
    classifier_prompt_ids: str = Form(...),
    prompt_key: str = Form(...),
    summary_prompt: Optional[str] = Form(None),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    db_prompt = db.query(QuestionerPrompt).filter(QuestionerPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Questioner prompt not found")
    
    try:
        classifier_ids = [int(x.strip()) for x in classifier_prompt_ids.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="classifier_prompt_ids must be comma-separated integers")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt.classifier_prompt_ids = classifier_ids
    db_prompt.prompt_key = prompt_key
    db_prompt.prompt_text = prompt_text
    db_prompt.summary_prompt = summary_prompt
    db_prompt.version = version
    db_prompt.is_active = is_active
    
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Questioner prompt updated successfully",
        "id": db_prompt.id,
        "classifier_prompt_ids": db_prompt.classifier_prompt_ids,
        "prompt_key": db_prompt.prompt_key,
        "text_length": len(prompt_text)
    }

@app.get("/followup-classifier-prompts/all")
def get_all_followup_classifier_prompts(db: Session = Depends(get_db)):
    return db.query(followUpClassifierPrompt).all()

@app.get("/followup-classifier-prompts/doctor/{doctor_id}")
def get_followup_classifier_prompts_by_doctor(doctor_id: int, db: Session = Depends(get_db)):
    prompts = db.query(followUpClassifierPrompt).filter(
        followUpClassifierPrompt.doctor_id == doctor_id
    ).all()
    
    if not prompts:
        raise HTTPException(status_code=404, detail="No follow-up classifier prompts found for this doctor")
    
    return prompts

@app.get("/followup-classifier-prompts/{specialty_name}/{doctor_id}")
def get_followup_classifier_prompt(specialty_name: str, doctor_id: int, db: Session = Depends(get_db)):
    prompt = db.query(followUpClassifierPrompt).filter(
        followUpClassifierPrompt.specialty_name == specialty_name,
        followUpClassifierPrompt.doctor_id == doctor_id
    ).first()
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Follow-up classifier prompt not found")
    
    return prompt

@app.post("/followup-classifier-prompts")
def create_followup_classifier_prompt(prompt: FollowUpClassifierPromptCreate, db: Session = Depends(get_db)):
    existing = db.query(followUpClassifierPrompt).filter(
        followUpClassifierPrompt.specialty_name == prompt.specialty_name,
        followUpClassifierPrompt.doctor_id == prompt.doctor_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Follow-up classifier prompt already exists for this specialty and doctor")
    
    db_prompt = followUpClassifierPrompt(**prompt.model_dump())
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.post("/followup-classifier-prompts/upload")
async def upload_followup_classifier_prompt(
    file: UploadFile = File(...),
    specialty_name: str = Form(...),
    doctor_id: int = Form(...),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    is_default: bool = Form(False),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    existing = db.query(followUpClassifierPrompt).filter(
        followUpClassifierPrompt.specialty_name == specialty_name,
        followUpClassifierPrompt.doctor_id == doctor_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Follow-up classifier prompt already exists for this specialty and doctor")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt = followUpClassifierPrompt(
        specialty_name=specialty_name,
        doctor_id=doctor_id,
        prompt_text=prompt_text,
        version=version,
        is_active=is_active,
        is_default=is_default
    )
    
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Follow-up classifier prompt uploaded successfully",
        "id": db_prompt.id,
        "specialty_name": db_prompt.specialty_name,
        "doctor_id": db_prompt.doctor_id,
        "version": db_prompt.version,
        "text_length": len(prompt_text)
    }

@app.put("/followup-classifier-prompts/{prompt_id}")
def update_followup_classifier_prompt(prompt_id: int, prompt: FollowUpClassifierPromptCreate, db: Session = Depends(get_db)):
    db_prompt = db.query(followUpClassifierPrompt).filter(followUpClassifierPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Follow-up classifier prompt not found")
    
    for key, value in prompt.model_dump().items():
        setattr(db_prompt, key, value)
    
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.patch("/followup-classifier-prompts/upload/{prompt_id}")
async def patch_followup_classifier_prompt_upload(
    prompt_id: int,
    file: UploadFile = File(...),
    specialty_name: str = Form(...),
    doctor_id: int = Form(...),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    is_default: bool = Form(False),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    db_prompt = db.query(followUpClassifierPrompt).filter(followUpClassifierPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Follow-up classifier prompt not found")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt.specialty_name = specialty_name
    db_prompt.doctor_id = doctor_id
    db_prompt.prompt_text = prompt_text
    db_prompt.version = version
    db_prompt.is_active = is_active
    db_prompt.is_default = is_default
    
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Follow-up classifier prompt updated successfully",
        "id": db_prompt.id,
        "specialty_name": db_prompt.specialty_name,
        "doctor_id": db_prompt.doctor_id,
        "version": db_prompt.version,
        "text_length": len(prompt_text)
    }

@app.get("/followup-questioner-prompts/all")
def get_all_followup_questioner_prompts(db: Session = Depends(get_db)):
    return db.query(followUpQuestionerPrompt).all()

@app.get("/followup-questioner-prompts/key/{prompt_key}")
def get_followup_questioner_prompt_by_key(prompt_key: str, db: Session = Depends(get_db)):
    prompt = db.query(followUpQuestionerPrompt).filter(
        followUpQuestionerPrompt.prompt_key == prompt_key
    ).first()
    
    if not prompt:
        raise HTTPException(status_code=404, detail="Follow-up questioner prompt not found for this key")
    
    return prompt

@app.get("/followup-questioner-prompts/classifier/{classifier_id}")
def get_followup_questioner_prompts_by_classifier(classifier_id: int, db: Session = Depends(get_db)):
    prompts = db.query(followUpQuestionerPrompt).filter(
        followUpQuestionerPrompt.classifier_prompt_ids.contains([classifier_id])
    ).all()
    
    if not prompts:
        raise HTTPException(status_code=404, detail="No follow-up questioner prompts found for this classifier")
    
    return prompts

@app.post("/followup-questioner-prompts")
def create_followup_questioner_prompt(prompt: FollowUpQuestionerPromptCreate, db: Session = Depends(get_db)):
    for classifier_id in prompt.classifier_prompt_ids:
        classifier_exists = db.query(followUpClassifierPrompt).filter(
            followUpClassifierPrompt.id == classifier_id
        ).first()
        
        if not classifier_exists:
            raise HTTPException(status_code=400, detail=f"Follow-up classifier prompt ID {classifier_id} does not exist")
    
    existing = db.query(followUpQuestionerPrompt).filter(
        followUpQuestionerPrompt.prompt_key == prompt.prompt_key
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Follow-up questioner prompt already exists for this key")
    
    db_prompt = followUpQuestionerPrompt(**prompt.model_dump())
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.post("/followup-questioner-prompts/upload")
async def upload_followup_questioner_prompt(
    file: UploadFile = File(...),
    classifier_prompt_ids: str = Form(...),
    prompt_key: str = Form(...),
    summary_prompt: Optional[str] = Form(None),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    try:
        classifier_ids = [int(x.strip()) for x in classifier_prompt_ids.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="classifier_prompt_ids must be comma-separated integers")
    
    for classifier_id in classifier_ids:
        classifier_exists = db.query(followUpClassifierPrompt).filter(
            followUpClassifierPrompt.id == classifier_id
        ).first()
        
        if not classifier_exists:
            raise HTTPException(status_code=400, detail=f"Follow-up classifier prompt ID {classifier_id} does not exist")
    
    existing = db.query(followUpQuestionerPrompt).filter(
        followUpQuestionerPrompt.prompt_key == prompt_key
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Follow-up questioner prompt already exists for this key")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt = followUpQuestionerPrompt(
        classifier_prompt_ids=classifier_ids,
        prompt_key=prompt_key,
        prompt_text=prompt_text,
        summary_prompt=summary_prompt,
        version=version,
        is_active=is_active
    )
    
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Follow-up questioner prompt uploaded successfully",
        "id": db_prompt.id,
        "classifier_prompt_ids": db_prompt.classifier_prompt_ids,
        "prompt_key": db_prompt.prompt_key,
        "text_length": len(prompt_text)
    }

@app.put("/followup-questioner-prompts/{prompt_id}")
def update_followup_questioner_prompt(prompt_id: int, prompt: FollowUpQuestionerPromptCreate, db: Session = Depends(get_db)):
    db_prompt = db.query(followUpQuestionerPrompt).filter(followUpQuestionerPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Follow-up questioner prompt not found")
    
    for key, value in prompt.model_dump().items():
        setattr(db_prompt, key, value)
    
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@app.put("/followup-questioner-prompts/upload/{prompt_id}")
async def update_followup_questioner_prompt_upload(
    prompt_id: int,
    file: UploadFile = File(...),
    classifier_prompt_ids: str = Form(...),
    prompt_key: str = Form(...),
    summary_prompt: Optional[str] = Form(None),
    version: str = Form("1.0"),
    is_active: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    db_prompt = db.query(followUpQuestionerPrompt).filter(followUpQuestionerPrompt.id == prompt_id).first()
    
    if not db_prompt:
        raise HTTPException(status_code=404, detail="Follow-up questioner prompt not found")
    
    try:
        classifier_ids = [int(x.strip()) for x in classifier_prompt_ids.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="classifier_prompt_ids must be comma-separated integers")
    
    content = await file.read()
    prompt_text = content.decode('utf-8')
    
    db_prompt.classifier_prompt_ids = classifier_ids
    db_prompt.prompt_key = prompt_key
    db_prompt.prompt_text = prompt_text
    db_prompt.summary_prompt = summary_prompt
    db_prompt.version = version
    db_prompt.is_active = is_active
    
    db.commit()
    db.refresh(db_prompt)
    
    return {
        "message": "Follow-up questioner prompt updated successfully",
        "id": db_prompt.id,
        "classifier_prompt_ids": db_prompt.classifier_prompt_ids,
        "prompt_key": db_prompt.prompt_key,
        "text_length": len(prompt_text)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prompt_api:app", host="0.0.0.0", port=8000, reload=True)