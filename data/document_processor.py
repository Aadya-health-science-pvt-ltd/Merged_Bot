# data/
# Handles data loading and processing, especially for initial data ingestion.

# data/document_processor.py
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from config.constants import BATCH_SIZE, DOCTOR_WEBSITE_URL
from config.llm_config import embeddings
import os

def process_batch(df, chain, sheet_name, file_path):
    """Process a batch of rows with error handling. Extracts specialty from workbook filename."""
    # Extract specialty from the workbook filename (without extension)
    specialty = os.path.splitext(os.path.basename(file_path))[0].lower()
    results = []
    for _, row in df.iterrows():
        try:
            row_dict = row.where(pd.notna(row), None).to_dict()
            description = chain.invoke({"row_data": str(row_dict)})
            results.append({
                "text": description,
                "metadata": {
                    "symptom": str(row_dict.get("Symptoms", "Unknown")).strip(),
                    "is_child": "child" if "child" in sheet_name.lower() else "both",
                    "gender": "female" if row_dict.get("Only Female", 0) else "both",
                    "source": sheet_name,
                    "specialty": specialty
                }
            })
        except Exception as e:
            print(f"Error processing row: {e}")
    return results
def process_sheets(file_path, chain, sheet_filter):
    """Process all sheets with batch parallel processing. Passes workbook path to process_batch."""
    xls = pd.ExcelFile(file_path)
    all_docs = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for sheet_name in xls.sheet_names:
            if not sheet_filter(sheet_name):
                continue

            df = xls.parse(sheet_name)
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                # Pass file_path as workbook_path to process_batch
                futures.append(
                    executor.submit(process_batch, batch, chain, sheet_name, file_path)
                )
        
        for future in futures:
            all_docs.extend(future.result())
    
    return all_docs

def load_and_split_doctor_info(url=DOCTOR_WEBSITE_URL):
    """Loads and splits documents from the doctor's website."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)