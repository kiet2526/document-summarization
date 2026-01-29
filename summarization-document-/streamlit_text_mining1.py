import streamlit as st
import pickle
import joblib
from pathlib import Path

# 👉 Chỉ cần thay đường dẫn model_path bằng model của bạn
# Sau đó chạy: streamlit run app.py

MODEL_PATH = "model.pkl"  # Hoặc "model.joblib"

# Cấu hình trang
st.set_page_config(
    page_title="Text Summarization",
    page_icon="📝",
    layout="wide"
)

# CSS tùy chỉnh cho giao diện đẹp hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .summary-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        min-height: 200px;
        font-size: 16px;
        line-height: 1.6;
    }
    .header-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .header-subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model với cache
@st.cache_resource
def load_model(model_path):
    """Load model từ file pickle hoặc joblib"""
    try:
        file_ext = Path(model_path).suffix
        if file_ext == ".pkl":
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif file_ext == ".joblib":
            model = joblib.load(model_path)
        else:
            raise ValueError("File phải có định dạng .pkl hoặc .joblib")
        return model
    except FileNotFoundError:
        st.error(f"❌ Không tìm thấy file model: {model_path}")
        st.info("💡 Vui lòng đặt file model vào đúng đường dẫn hoặc cập nhật MODEL_PATH trong code.")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        return None

def predict(model, text):
    """
    Hàm dự đoán - thay đổi tùy theo model của bạn
    Ví dụ: model có thể là pipeline hoặc có method .predict() hoặc .summarize()
    """
    try:
        # 👉 Thay đổi phần này tùy theo cách model của bạn hoạt động
        # Ví dụ 1: summary = model.predict([text])[0]
        # Ví dụ 2: summary = model.summarize(text)
        # Ví dụ 3: summary = model(text)[0]['summary_text']
        
        summary = model.predict([text])[0]  # Thay đổi theo model của bạn
        return summary
    except Exception as e:
        return f"❌ Lỗi khi tóm tắt: {str(e)}"

# Header
st.markdown('<h1 class="header-title">📝 Text Summarization</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Nhập văn bản và nhận bản tóm tắt ngắn gọn</p>', unsafe_allow_html=True)

# Load model
model = load_model(MODEL_PATH)

# Layout hai cột
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Văn bản gốc")
    st.markdown("*Nhập hoặc dán văn bản cần tóm tắt vào đây*")
    
    text_input = st.text_area(
        label="Input Text",
        placeholder="Ví dụ: Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính...",
        height=300,
        label_visibility="collapsed"
    )
    
    summarize_button = st.button("🚀 Tóm tắt", use_container_width=True, type="primary")

with col2:
    st.markdown("### 📤 Kết quả tóm tắt")
    st.markdown("*Bản tóm tắt sẽ xuất hiện ở đây*")
    
    # Container cho kết quả
    result_container = st.container()

# Xử lý khi nhấn nút
if summarize_button:
    if not text_input.strip():
        st.warning("⚠️ Vui lòng nhập văn bản trước khi tóm tắt!")
    elif model is None:
        st.error("❌ Model chưa được load. Vui lòng kiểm tra đường dẫn model.")
    else:
        with st.spinner("⏳ Đang tóm tắt..."):
            summary = predict(model, text_input)
            
            with result_container:
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Thống kê
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Độ dài gốc", f"{len(text_input.split())} từ")
                with col_stat2:
                    st.metric("Độ dài tóm tắt", f"{len(summary.split())} từ")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>💡 <b>Hướng dẫn:</b> Nhập văn bản bên trái → Nhấn nút Tóm tắt → Xem kết quả bên phải</p>
    </div>
""", unsafe_allow_html=True)