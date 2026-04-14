"""
evaluate.py - Đánh giá chất lượng RAG pipeline bằng phương pháp LLM-as-a-Judge.
Sử dụng Gemini API để chấm điểm câu trả lời của Bot so với đáp án mẫu.
"""

import json
import time

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Bộ câu hỏi test (ground truth) được nâng cấp ─────────────────────────────
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

# ── Prompt dành cho Giám khảo AI ─────────────────────────────────────────────
EVALUATOR_PROMPT = """Bạn là một Giám khảo AI khắt khe và công tâm.
Nhiệm vụ của bạn là đánh giá câu trả lời của một AI chatbot có khớp với đáp án mẫu (Ground Truth) hay không.

Câu hỏi của người dùng: "{question}"
Đáp án mẫu chuẩn: "{expected_answer}"
Câu trả lời thực tế của Chatbot: "{bot_answer}"

Hãy chấm điểm từ 0 đến 5 dựa trên tiêu chí:
- 5: Chính xác hoàn toàn, đủ ý như đáp án mẫu.
- 3-4: Có ý đúng nhưng thiếu sót một chút hoặc dư thừa thông tin không cần thiết.
- 1-2: Trả lời sai lệch, lạc đề hoặc thiếu thông tin quan trọng.
- 0: Sai hoàn toàn hoặc nói không biết.

CHỈ ĐƯỢC PHÉP trả về kết quả dưới định dạng JSON (không thêm bất kỳ giải thích nào bên ngoài):
{{
    "score": <số nguyên từ 0-5>,
    "reason": "<một câu ngắn gọn giải thích lý do cho điểm>"
}}
"""

def evaluate_with_llm(evaluator_llm, question, expected, bot_answer):
    """Dùng LLM khác để chấm điểm câu trả lời."""
    prompt = EVALUATOR_PROMPT.format(
        question=question, expected_answer=expected, bot_answer=bot_answer
    )
    
    response = evaluator_llm.invoke(prompt)
    raw_text = response.content
    
    # --- ĐOẠN CODE SỬA LỖI ---
    # Ép kiểu và rút trích văn bản an toàn mọi trường hợp
    if isinstance(raw_text, list):
        # Nếu là list, tìm trong phần tử đầu tiên
        if len(raw_text) > 0 and isinstance(raw_text[0], dict):
            raw_text = raw_text[0].get("text", "")
        else:
            raw_text = str(raw_text[0])
    elif isinstance(raw_text, dict):
        raw_text = raw_text.get("text", "")
    
    # Đảm bảo kết quả cuối cùng chắc chắn là chuỗi văn bản
    raw_text = str(raw_text)
    # ---------------------------

    # Xử lý text để bóc tách JSON (phòng trường hợp Gemini trả về markdown ```json)
    clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        result = json.loads(clean_text)
        # Quy đổi thang 5 sang thang 1.0 cho dễ tính trung bình
        score = result.get("score", 0) / 5.0 
        reason = result.get("reason", "Không có lý do")
        return score, reason
    except Exception as e:
        print(f"Lỗi parse JSON từ Giám khảo: {clean_text}")
        return 0.0, "Lỗi phân tích cú pháp"
    """Dùng LLM khác để chấm điểm câu trả lời."""
    prompt = EVALUATOR_PROMPT.format(
        question=question, expected_answer=expected, bot_answer=bot_answer
    )
    
    response = evaluator_llm.invoke(prompt)
    raw_text = response.content
    # Nếu trả về list, lấy phần tử đầu tiên
    if isinstance(raw_text, list):
        raw_text = raw_text[0]
    # Xử lý text để bóc tách JSON (phòng trường hợp Gemini trả về markdown ```json)
    clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(clean_text)
        # Quy đổi thang 5 sang thang 1.0 cho dễ tính trung bình
        score = result.get("score", 0) / 5.0 
        reason = result.get("reason", "Không có lý do")
        return score, reason
    except Exception as e:
        print(f"Lỗi parse JSON từ Giám khảo: {clean_text}")
        return 0.0, "Lỗi phân tích cú pháp"

def run_evaluation():
    print("=" * 60)
    print("📊 ĐÁNH GIÁ RAG PIPELINE (LLM-as-a-Judge)")
    print("=" * 60)

    # Load resources
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # LLM cho Bot (nhiệt độ thấp để chính xác)
    bot_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)
    # LLM cho Giám khảo (có thể dùng model xịn hơn nếu có, ở đây vẫn xài gemini-flash-latest cho miễn phí)
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=bot_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    results = []
    total_score = 0.0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['question']}")

        # 1. Lấy câu trả lời của Bot
        start = time.time()
        bot_result = chain.invoke({"query": tc["question"]})
        elapsed = time.time() - start
        bot_answer = bot_result["result"]

        # 2. Đưa cho Giám khảo chấm điểm
        score, reason = evaluate_with_llm(evaluator_llm, tc["question"], tc["expected_answer"], bot_answer)
        total_score += score

        status = "✅" if score >= 0.8 else ("⚠️" if score >= 0.5 else "❌")
        
        print(f"  Bot trả lời: {bot_answer}")
        print(f"  Giám khảo: Chấm {score*100:.0%} - Lý do: {reason}")
        print(f"  Thời gian: {elapsed:.1f}s  {status}")

        results.append({
            "question": tc["question"],
            "bot_answer": bot_answer,
            "score": score,
            "reason": reason,
            "latency": round(elapsed, 2),
        })

    avg_score = total_score / len(TEST_CASES)
    print("\n" + "=" * 60)
    print(f"📈 TỔNG KẾT: {avg_score:.0%} ĐỘ CHÍNH XÁC")
    print("=" * 60)

    # Lưu log
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"average_score": avg_score, "results": results}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_evaluation()