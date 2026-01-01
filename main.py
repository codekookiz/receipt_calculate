import streamlit as st
from datetime import datetime
from typing import List
import os
import re
from huggingface_hub import InferenceClient
import base64
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN"),
    base_url="https://router.huggingface.co"
)

# ---------- OCR Stub (추후 HuggingFace 연결) ----------
def extract_total_from_image(image_bytes: bytes) -> int:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it:nebius",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "다음 영수증 이미지에서 '합계' 또는 'TOTAL'에 해당하는 "
                            "최종 금액만 숫자로 출력해. "
                            "통화 기호, 설명, 문장은 제외하고 숫자만 출력해."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
    )

    content = response.choices[0].message.content

    match = re.search(r"[\d,]+", content)
    if not match:
        return 0

    return int(match.group(0).replace(",", ""))


# ---------- Aggregation ----------
def calculate_monthly_total(images: List[bytes]) -> int:
    totals = []

    for img in images:
        amount = extract_total_from_image(img)
        totals.append(amount)

    return sum(totals)


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="월별 영수증 합계 계산기",
    layout="centered"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 20px;
    }
    h1 {
        font-size: 2.2rem;
    }
    h2 {
        font-size: 1.8rem;
    }
    h3 {
        font-size: 1.5rem;
    }
    button {
        font-size: 1.1rem !important;
        padding: 0.6em 1.2em !important;
    }
    input, label, textarea, select {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 월별 영수증 합계 계산")
st.caption("여러 장의 영수증 이미지를 업로드하면 선택한 월의 총 합계를 계산합니다.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "📤 영수증 이미지 업로드",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="여러 장의 영수증을 한 번에 업로드할 수 있습니다."
    )

with col2:
    month = st.selectbox(
        "📅 대상 월",
        options=[
            f"{datetime.now().year}-{str(m).zfill(2)}"
            for m in range(1, 13)
        ],
        index=datetime.now().month - 1
    )

st.divider()
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    run_button = st.button("▶️ 합계 계산", use_container_width=True)

if run_button:
    if not uploaded_files:
        st.warning("영수증 이미지를 하나 이상 업로드하세요.")
    else:
        image_bytes_list = [file.read() for file in uploaded_files]

        with st.spinner("영수증을 분석 중입니다..."):
            total_amount = calculate_monthly_total(image_bytes_list)

        st.success("계산 완료")

        st.subheader("📊 계산 결과")
        st.markdown(
            f"""
            <div style="padding: 1.2em; border-radius: 12px; background-color: #f6f6f6;">
                <p><strong>대상 월</strong><br>{month}</p>
                <p style="font-size: 1.8rem; margin-top: 0.8em;">
                    <strong>총 합계</strong><br>
                    {total_amount:,} 원
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )