"""
evaluate.py - Đánh giá chất lượng RAG pipeline.
So sánh câu trả lời của bot với đáp án mẫu.

Usage:
    uv run python src/evaluate.py
"""

import json
import os
import time

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Bộ câu hỏi test (ground truth) ───────────────────────────────────────────
TEST_CASES = [
    {
        "question": "Nhân viên chính thức dưới 5 năm có bao nhiêu ngày phép?",
        "expected_keywords": ["12", "ngày"],
    },
    {
        "question": "Lương được thanh toán vào ngày mấy?",
        "expected_keywords": ["10", "tháng"],
    },
    {
        "question": "Công ty có mấy văn phòng?",
        "expected_keywords": ["2", "Hà Nội", "Hồ Chí Minh"],
    },
    {
        "question": "Budget học tập mỗi năm là bao nhiêu?",
        "expected_keywords": ["5 triệu", "triệu"],
    },
    {
        "question": "Giờ làm việc bắt đầu từ mấy giờ?",
        "expected_keywords": ["8:30", "8"],
    },
]

PROMPT_TEMPLATE = """Trả lời ngắn gọn dựa trên thông tin sau:

{context}

Câu hỏi: {question}
Trả lời:"""


def keyword_score(answer: str, keywords: list) -> float:
    """Tính điểm dựa trên số keywords xuất hiện trong câu trả lời."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def run_evaluation():
    print("=" * 60)
    print("📊 ĐÁNH GIÁ RAG PIPELINE")
    print("=" * 60)

    # Load chain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    results = []
    total_score = 0.0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['question']}")

        start = time.time()
        result = chain.invoke({"query": tc["question"]})
        elapsed = time.time() - start

        answer = result["result"]
        score = keyword_score(answer, tc["expected_keywords"])
        total_score += score

        status = "✅" if score >= 0.5 else "❌"
        print(f"  Trả lời: {answer[:120]}")
        print(f"  Keywords cần có: {tc['expected_keywords']}")
        print(f"  Điểm: {score:.0%}  |  Thời gian: {elapsed:.1f}s  {status}")

        results.append(
            {
                "question": tc["question"],
                "answer": answer,
                "score": score,
                "latency": round(elapsed, 2),
            }
        )

    avg_score = total_score / len(TEST_CASES)
    print("\n" + "=" * 60)
    print(f"📈 TỔNG KẾT: {avg_score:.0%} ({avg_score * len(TEST_CASES):.1f}/{len(TEST_CASES)} câu đúng)")
    print("=" * 60)

    # Lưu kết quả ra file
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"average_score": avg_score, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n💾 Đã lưu kết quả chi tiết vào evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
