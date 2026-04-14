"""
ingest.py - Đọc tài liệu, chunk, embed và lưu vào FAISS index.
Chạy một lần để chuẩn bị dữ liệu trước khi chat.

Usage:
    uv run python src/ingest.py --file data/sample.txt
    uv run python src/ingest.py --file data/my_doc.pdf
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def load_document(file_path: str):
    """Load tài liệu từ file TXT hoặc PDF."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if path.suffix.lower() == ".pdf":
        print(f"📄 Đang load PDF: {file_path}")
        loader = PyPDFLoader(file_path)
    else:
        print(f"📄 Đang load TXT: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")

    return loader.load()


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Chia tài liệu thành các đoạn nhỏ để embed."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Đã tạo {len(chunks)} chunks từ {len(docs)} trang/đoạn")
    return chunks


def embed_and_store(chunks, index_path="faiss_index"):
    """Embed các chunks bằng Gemini và lưu vào FAISS."""
    print("🔢 Đang tạo embeddings với Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    print(f"✅ Đã lưu FAISS index vào '{index_path}/'")
    return db


def ingest(file_path: str, index_path: str = "faiss_index"):
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    embed_and_store(chunks, index_path)
    print("\n🚀 Ingest hoàn tất! Giờ chạy: uv run python src/chat.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest tài liệu vào FAISS")
    parser.add_argument("--file", default="data/sample.txt", help="Đường dẫn tới file")
    parser.add_argument("--index", default="faiss_index", help="Thư mục lưu FAISS index")
    args = parser.parse_args()

    ingest(args.file, args.index)
