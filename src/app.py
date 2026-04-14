"""
app.py - Giao diện web cho RAG Chatbot bằng Streamlit.

Usage:
    uv run streamlit run src/app.py
"""

import os

import streamlit as st
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RAG Chatbot")
st.caption("Hỏi về tài liệu nội bộ • Powered by Gemini + FAISS + LangChain")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    show_sources = st.toggle("Hiển thị nguồn tham khảo", value=False)
    temperature = st.slider("Độ sáng tạo (temperature)", 0.0, 1.0, 0.3, 0.1)
    top_k = st.slider("Số đoạn văn tham khảo (k)", 1, 6, 3)

    st.divider()
    st.markdown("**Cách dùng:**")
    st.markdown("1. Chạy `ingest.py` để load tài liệu")
    st.markdown("2. Gõ câu hỏi bên phải")
    st.markdown("3. Bot sẽ trả lời dựa trên tài liệu")

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load chain (cache để không reload mỗi lần) ────────────────────────────────
@st.cache_resource(show_spinner="Đang load model...")
def get_chain(temp: float, k: int):
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=temp,
        max_output_tokens=1024,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


chain = get_chain(temperature, top_k)

# ── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("📚 Nguồn tham khảo"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"[{i}] {src[:200]}...")

# Input
if chain is None:
    st.warning("⚠️ Chưa có FAISS index. Hãy chạy: `uv run python src/ingest.py`")
else:
    if question := st.chat_input("Hỏi về tài liệu..."):
        # Hiển thị câu hỏi
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Gọi RAG và hiển thị trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và trả lời..."):
                result = chain.invoke({"query": question})
                answer = result["result"]
                sources = [doc.page_content for doc in result["source_documents"]]

            st.markdown(answer)
            if show_sources:
                with st.expander("📚 Nguồn tham khảo"):
                    for i, src in enumerate(sources, 1):
                        st.caption(f"[{i}] {src[:200]}...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
