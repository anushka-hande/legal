from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "FastAPI backend is working"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    result = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents)
    }

    if file.filename.lower().endswith("pdf"):
        pdf_stream = BytesIO(contents)
        reader = PdfReader(pdf_stream)
        result["total_pages"] = len(reader.pages)

        all_text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

        result["total_extracted_characters"] = len(all_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(all_text)

        result["total_chunks"] = len(chunks)
        result["first_chunk_preview"] = chunks[0][:500] if chunks else ""

    return result