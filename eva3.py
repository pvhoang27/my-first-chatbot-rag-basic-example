"""
evaluate_semantic.py - Đánh giá RAG pipeline bằng Semantic Similarity.
Dùng mô hình AI Offline (all-MiniLM-L6-v2) để so sánh vector ngữ nghĩa 
giữa câu trả lời của Bot và Đáp án mẫu.
"""

import json
import time

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# ── Bộ câu hỏi test ────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "Nhân viên chính thức dưới 5 năm có bao nhiêu ngày phép?",
        "expected_answer": "Nhân viên chính thức dưới 5 năm được nghỉ 12 ngày phép/năm.",
    },
    {
        "question": "Lương được thanh toán vào ngày mấy?",
        "expected_answer": "Lương được thanh toán vào ngày 10 hàng tháng.",
    },
    {
        "question": "Công ty có mấy văn phòng?",
        "expected_answer": "Công ty có 2 văn phòng ở Hà Nội và TP. Hồ Chí Minh.",
    },
]

PROMPT_TEMPLATE = """Trả lời ngắn gọn dựa trên thông tin sau:
{context}

Câu hỏi: {question}
Trả lời:"""

def run_evaluation():
    print("=" * 70)
    print("📊 ĐÁNH GIÁ RAG PIPELINE (SEMANTIC SIMILARITY - OFFLINE)")
    print("=" * 70)

    # 1. Load RAG Pipeline (dùng HuggingFaceEmbeddings để embedding, Gemini để sinh câu trả lời)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    bot_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=bot_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    # 2. Load mô hình Giám khảo Offline (Sentence-Transformers)
    print("⏳ Đang tải mô hình Giám khảo Offline (all-MiniLM-L6-v2)...")
    # Lần đầu chạy sẽ tốn chút thời gian tải model (~80MB) về máy, các lần sau sẽ load tức thì
    eval_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Đã tải xong Giám khảo!\n")

    results = []
    total_score = 0.0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc['question']}")

        # Lấy câu trả lời của Bot
        start = time.time()
        bot_result = chain.invoke({"query": tc["question"]})
        elapsed = time.time() - start
        bot_answer = bot_result["result"]

        # ---------------------------------------------------------
        # CHẤM ĐIỂM BẰNG TOÁN HỌC VECTOR (COSINE SIMILARITY)
        # ---------------------------------------------------------
        # Biến 2 câu văn thành 2 ma trận số
        embedding_bot = eval_model.encode(bot_answer, convert_to_tensor=True)
        embedding_expected = eval_model.encode(tc["expected_answer"], convert_to_tensor=True)
        
        # Đo khoảng cách Cosine giữa 2 ma trận (từ 0.0 đến 1.0)
        similarity_score = util.cos_sim(embedding_bot, embedding_expected).item()
        # ---------------------------------------------------------

        total_score += similarity_score

        # Nếu độ tương đồng > 0.75 thì coi như là Trả lời đúng ý
        status = "✅" if similarity_score >= 0.75 else ("⚠️" if similarity_score >= 0.5 else "❌")
        
        print(f"  Bot trả lời: {bot_answer}")
        print(f"  Điểm tương đồng: {similarity_score*100:.1f}%")
        print(f"  Thời gian sinh câu trả lời: {elapsed:.1f}s  {status}\n")

        results.append({
            "question": tc["question"],
            "bot_answer": bot_answer,
            "similarity_score": round(similarity_score, 4),
            "latency": round(elapsed, 2),
        })

    avg_score = total_score / len(TEST_CASES)
    print("=" * 70)
    print(f"📈 TỔNG KẾT: ĐỘ TƯƠNG ĐỒNG TRUNG BÌNH ĐẠT {avg_score*100:.1f}%")
    print("=" * 70)

    # Lưu log
    with open("semantic_eval_results-3.json", "w", encoding="utf-8") as f:
        json.dump({"average_similarity": avg_score, "results": results}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_evaluation()