import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

load_dotenv()

Base = declarative_base()

class ClassifierPrompt(Base):
    __tablename__ = 'classifier_prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    specialty_name = Column(String(100), nullable=False)
    doctor_id = Column(Integer, nullable=False)
    prompt_text = Column(Text, nullable=False)
    version = Column(String(20), default='1.0')
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class QuestionerPrompt(Base):
    __tablename__ = 'questioner_prompts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    classifier_prompt_ids = Column(ARRAY(Integer), nullable=False)
    prompt_key = Column(String(100), nullable=False)
    prompt_text = Column(Text, nullable=False)
    summary_prompt = Column(Text)
    version = Column(String(20), default='1.0')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_database_url():
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'password')
    db_name = os.getenv('DB_NAME', 'medical_bot_db')
    return f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

engine = create_engine(get_database_url(), echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)