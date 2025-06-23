# config/llm_config.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import lancedb
from dotenv import load_dotenv
import os


BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=BATCH_SIZE)
db = lancedb.connect("./lancedb")