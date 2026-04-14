import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("❌ Chưa cài đặt thư viện google-generativeai. Cài bằng: pip install google-generativeai")
    exit(1)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Không tìm thấy GOOGLE_API_KEY trong .env")
    exit(1)

try:
    genai.configure(api_key=api_key)
    # Thử gọi API đơn giản
    models = list(genai.list_models())
    print("✅ API key hợp lệ! Danh sách model khả dụng:")
    for m in models:
        print(f"- {m.name} (methods: {m.supported_generation_methods})")
except Exception as e:
    print("❌ API key không hợp lệ hoặc có lỗi khi gọi API:")
    print(e)