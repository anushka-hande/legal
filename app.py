import streamlit as st
import numpy as np
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_documents")
st.title("Legal Document Q&A App")
st.write("Streamlit is working successfully on this machine.")

question = st.text_input("Enter your legal question:")
uploaded_file = st.file_uploader("Upload a legal document", type=["pdf"])
st.write("You typed:", question)

if uploaded_file is not None:
    st.write("Uploaded filename:", uploaded_file.name)

    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    st.write("Total pages in PDF:", total_pages)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text = full_text + page_text + "\n\n"
    
    st.write("Total extracted characters:", len(full_text))
    st.write("Preview of extracted text:")
    st.write(full_text[:500])

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    chunks = splitter.split_text(full_text)

    st.write("Total chunks created:", len(chunks))

    chunk_embeddings = model.encode(chunks)
    st.write("Total embeddings created:", len(chunk_embeddings))
    st.write("Embedding size of first chunk:", len(chunk_embeddings[0]))

    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=chunk_embeddings.tolist()
    )

    st.write("Chunks stored in ChromaDB:", len(chunk_ids))

    if question:
        question_embedding = model.encode([question])[0]

        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=3
        )

        retrieved_chunks = results["documents"][0]
        retrieved_ids = results["ids"][0]
        retrieved_distances = results["distances"][0]

        st.write("Top 3 matching chunks:")

        for i, (chunk_id, distance, chunk_text) in enumerate(
            zip(retrieved_ids, retrieved_distances, retrieved_chunks),
            start=1
        ):
            st.write(f"Rank {i} chunk id:", chunk_id)
            st.write(f"Rank {i} distance:", float(distance))
            st.text_area(f"Chunk rank {i}", chunk_text, height=200)

        top_chunks_text = "\n\n".join(retrieved_chunks)

        st.write("Simple retrieved answer:")
        st.text_area("Answer from retrieved chunks", top_chunks_text, height=250)

    st.write("First chunk preview:")
    st.text_area("Chunk 1", chunks[0], height=200)
    st.write("Full extracted text:")
    st.text_area("Extracted text from all pages", full_text, height=400)
else:
    st.write("No file uploaded yet.")
