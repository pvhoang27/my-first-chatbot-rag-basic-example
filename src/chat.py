"""
chat.py - Chatbot RAG chạy trên terminal.
Nhận câu hỏi, tìm context từ FAISS, gọi Gemini để trả lời.

Usage:
    uv run python src/chat.py
    uv run python src/chat.py --show-sources   # hiển thị nguồn tham khảo
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

PROMPT_TEMPLATE = """Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp bên dưới.
Nếu thông tin không đủ để trả lời, hãy nói "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu."
Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng.

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""


def load_chain(index_path: str = "faiss_index", show_sources: bool = False):
    """Load FAISS index và khởi tạo RAG chain."""
    if not os.path.exists(index_path):
        print(f"❌ Chưa có FAISS index tại '{index_path}/'")
        print("   Hãy chạy trước: uv run python src/ingest.py")
        exit(1)

    print("⏳ Đang load model và index...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.3,
        max_output_tokens=1024,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    print("✅ Sẵn sàng! Gõ câu hỏi của bạn (hoặc 'quit' để thoát)\n")
    return chain


def ask(chain, question: str, show_sources: bool = False) -> str:
    """Gửi câu hỏi và nhận trả lời từ RAG chain."""
    result = chain.invoke({"query": question})
    answer = result["result"]

    if show_sources:
        print("\n📚 Nguồn tham khảo:")
        for i, doc in enumerate(result["source_documents"], 1):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] ...{preview}...")

    return answer


def chat_loop(show_sources: bool = False):
    """Vòng lặp chat trên terminal."""
    chain = load_chain(show_sources=show_sources)

    while True:
        try:
            question = input("🧑 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "thoát"):
            print("👋 Tạm biệt!")
            break

        answer = ask(chain, question, show_sources)
        print(f"\n🤖 Bot: {answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot với Gemini")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Hiển thị đoạn văn bản nguồn được dùng để trả lời",
    )
    args = parser.parse_args()

    chat_loop(show_sources=args.show_sources)
