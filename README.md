# 🤖 RAG Chatbot — Gemini + FAISS + LangChain

Chatbot cho phép bạn **hỏi chuyện với tài liệu của mình** (PDF, TXT...) bằng tiếng Việt.  
Sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** với Gemini API (miễn phí), FAISS vector database và LangChain.

---

## 🏗️ Kiến trúc

```
Câu hỏi của user
      │
      ▼
[Embed câu hỏi]  ←── Gemini Embedding API
      │
      ▼
[Vector Search]  ←── FAISS index (local)
      │
      ▼
[Build Prompt]   ←── context + câu hỏi
      │
      ▼
[Gemini LLM]     ←── Trả lời dựa trên context
      │
      ▼
Câu trả lời
```

**Stack kỹ thuật:**
- `LangChain` — orchestrate pipeline RAG
- `Gemini API` — embedding + LLM (miễn phí tại aistudio.google.com)
- `FAISS` — vector database chạy local, không cần server
- `Streamlit` — web UI
- `uv` — quản lý Python dependencies

---

## 🚀 Cài đặt

### Yêu cầu
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — cài bằng: `curl -Lsf https://astral.sh/uv/install.sh | sh`
- Gemini API key miễn phí tại: https://aistudio.google.com/app/apikey

### Bước 1: Clone và cài dependencies

```bash
git clone <repo-url>
cd rag-chatbot

uv sync
```

### Bước 2: Tạo file `.env`

```bash
cp .env.example .env
# Mở .env và điền GOOGLE_API_KEY của bạn
```

File `.env`:
```
GOOGLE_API_KEY=AIzaSy...
```

### Bước 3: Ingest tài liệu

```bash
# Dùng file mẫu
uv run python src/ingest.py

# Hoặc dùng file của bạn
uv run python src/ingest.py --file path/to/your/doc.pdf
uv run python src/ingest.py --file path/to/your/doc.txt
```

### Bước 4: Chat

**Terminal:**
```bash
uv run python src/chat.py

# Hiển thị nguồn tham khảo
uv run python src/chat.py --show-sources
```

**Web UI (Streamlit):**
```bash
uv run streamlit run src/app.py
```
Mở trình duyệt tại: http://localhost:8501

---

## 📊 Đánh giá chất lượng

```bash
uv run python src/evaluate.py
```

Kết quả sẽ in ra terminal và lưu vào `evaluation_results.json`.

---

## 📁 Cấu trúc project

```
rag-chatbot/
├── .env                    ← API key (KHÔNG commit)
├── .env.example            ← Mẫu env cho người khác
├── .gitignore
├── pyproject.toml          ← Dependencies (quản lý bởi uv)
├── README.md
├── src/
│   ├── ingest.py           ← Load, chunk, embed, lưu FAISS
│   ├── chat.py             ← Chat trên terminal
│   ├── app.py              ← Web UI với Streamlit
│   └── evaluate.py         ← Đánh giá chất lượng RAG
├── data/
│   └── sample.txt          ← Dữ liệu mẫu
└── faiss_index/            ← FAISS index (tự tạo sau ingest, KHÔNG commit)
    ├── index.faiss
    └── index.pkl
```

---

## 💡 Mở rộng

- **Thêm nhiều file**: Chạy `ingest.py` nhiều lần với các file khác nhau
- **Hỗ trợ URL**: Dùng `WebBaseLoader` từ LangChain để load webpage
- **Fine-tune prompt**: Chỉnh `PROMPT_TEMPLATE` trong `chat.py` để thay đổi phong cách trả lời
- **Đổi vector DB**: Thay FAISS bằng ChromaDB hoặc Pinecone cho production

---

## 📝 Concepts được áp dụng

| Concept | Implement trong project |
|---|---|
| RAG | `RetrievalQA` chain trong LangChain |
| Embedding | `GoogleGenerativeAIEmbeddings` |
| Vector similarity search | FAISS `similarity_search` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Prompt Engineering | Custom `PromptTemplate` |
| LLM API | `ChatGoogleGenerativeAI` (Gemini) |
| Evaluation | Keyword matching + latency measurement |
