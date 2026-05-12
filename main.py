from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from ingest import ingest_pdf
from ask import answer_question

app = FastAPI(title="Document Q&A API")

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Document Q&A API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ingest_pdf(file_path)
    
    return {
        "message": "PDF uploaded and processed successfully",
        "filename": file.filename
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not os.path.exists("faiss_index"):
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )
    
    answer = answer_question(request.question)
    
    return {
        "question": request.question,
        "answer": answer
    }