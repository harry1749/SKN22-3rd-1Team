import streamlit as st
import streamlit.components.v1 as components

from src.chain.rag_chain import build_rag_chain_with_sources, prepare_context, stream_answer
from src.config import CLASSIFIER_MODEL, LLM_MODEL

# 페이지 설정
st.set_page_config(
    page_title="의약품 정보 Q&A",
    page_icon="💊",
    layout="wide",
)


# --- CSS 스타일 ---
st.markdown("""
    <style>
    .chat-bubble {
        background-color: white;
        padding: 15px 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        display: inline-block;
        color: black;
        font-family: sans-serif;
        white-space: pre-wrap; 
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        word-break: break-all;
        line-height: 1.6; 
    }
    .user-message-group {
        display: flex;
        align-items: flex-start;
        justify-content: flex-end; 
        gap: 10px;
        width: 100%;
        margin-bottom: 20px;
    }
    .user-icon {
        width: 35px;
        height: 35px;
        background-color: #FF4B4B;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        order: 2; 
    }
    .user-bubble-container { order: 1; }
    </style>
""", unsafe_allow_html=True)


# --- [신규] 답변 포맷팅 함수 (성분명: 효능 형식만 추출) ---
def format_answer_simplified(llm_answer: str) -> str:
    """
    LLM이 생성한 답변에서 "성분명: 효능" 부분만 추출하여
    "- **성분명** : 효능" 형식으로 깔끔하게 표시합니다.
    """
    if not llm_answer:
        return llm_answer
    
    # "병용금지 주의:" 이전까지만 추출
    if "병용금지 주의:" in llm_answer:
        answer = llm_answer.split("병용금지 주의:")[0].strip()
    else:
        answer = llm_answer
    
    # "제안:" 이전까지만 추출
    if "제안:" in answer:
        answer = answer.split("제안:")[0].strip()
    
    # 성분명과 효능 쌍 추출하기
    lines = answer.split("\n")
    result_lines = []
    current_ingredient = None
    current_efficacy = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 성분명: 라인
        if line.startswith("성분명:"):
            # 이전 항목이 있으면 저장
            if current_ingredient and current_efficacy:
                result_lines.append(f"- **{current_ingredient}** : {current_efficacy}")
            # 새 항목 시작
            current_ingredient = line.replace("성분명:", "").strip()
            current_efficacy = None
        
        # 효능: 라인
        elif line.startswith("효능:"):
            current_efficacy = line.replace("효능:", "").strip()
    
    # 마지막 항목 저장
    if current_ingredient and current_efficacy:
        result_lines.append(f"- **{current_ingredient}** : {current_efficacy}")
    
    # 머리글 + 결과
    if result_lines:
        formatted = "💉분석 결과\n\n"
        formatted += "\n".join(result_lines)
        return formatted
    else:
        return llm_answer


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain_with_sources()
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


# 면책동의 다이얼로그
@st.dialog(title="⚠️ 면책사항 동의", width="large")
def disclaimer_dialog():
    """첫 진입 시 표시되는 면책사항 동의 팝업"""
    st.markdown(
        """

# **서비스 이용 약관 및 법적 면책 고지**

### 본 서비스는 식품의약품안전처 공공데이터를 기반으로 정보를 제공하는 데이터 검색 보조 도구입니다.

### 사용자는 본 서비스를 이용함과 동시에 아래의 모든 사항에 **동의**한 것으로 간주됩니다.

---

## **1. 의료 행위의 부인**

### 본 시스템이 제공하는 모든 정보는 일반적인 정보 제공만을 목적으로 하며, 의학적 진단, 치료, 처방 또는 복약 지도를 대신할 수 없습니다.

> **AI가 생성한 답변을 근거로 스스로 질병을 진단하거나 약물을 선택하여 복용하지 마십시오. 이는 오남용으로 인한 심각한 부작용을 초래할 수 있습니다.**

## **2. 정보의 정확성 및 최신성 보장 불가**

### 본 서비스는 생성형 AI(RAG) 기술을 사용합니다.

### AI의 특성상 환각 현상(Hallucination)이 발생할 수 있으며, 공공데이터의 내용과 다른 부정확하거나 왜곡된 정보를 제공할 가능성이 항상 존재합니다.

> 데이터베이스 업데이트 지연으로 인해 최신 의약품 정보나 허가 취소 사항이 반영되지 않았을 수 있습니다.

> 정보의 최종 확인은 반드시 공식적인 식약처 의약품안전나라 또는 전문가를 통해 확인하시기 바랍니다.

## **3. 책임의 제한**

### 서비스 운영 주체는 본 서비스가 제공한 정보의 오류, 누락, 지연으로 인해 발생하는 어떠한 형태의 직접적·간접적·결과적 손해(신체적 부상, 질환의 악화, 경제적 손실 등)에 대해서도 법적 책임을 지지 않습니다.

> **사용자가 본 시스템의 정보를 신뢰하여 행한 모든 결정 및 행동에 대한 책임은 전적으로 사용자 본인에게 있습니다.**

## **4. 전문가 상담 필수**

### 증상이 있거나 의약품 성분에 대해 궁금한 점이 있을 경우, 반드시 전문의 또는 약사와 상담하십시오.

### 응급 상황이 발생한 경우, 본 시스템에 의존하지 말고 즉시 응급 의료 기관(119 등)에 연락하십시오.

## **5. 데이터 출처 및 오용 금지**

### 본 서비스는 식약처의 공공데이터를 인용하나, 식약처가 본 서비스의 운영이나 결과물을 보증하는 것은 아닙니다.

> 사용자는 본 서비스의 결과를 상업적으로 이용하거나, 타인에게 의학적 권고로 전달하여 발생하는 모든 법적 문제에 대해 단독으로 책임을 집니다.

---

## **6. 확인 및 동의**

본인은 위 면책 고지 사항을 충분히 숙지하였으며, 본 서비스가 제공하는 정보는 참고용일 뿐 의료 전문가의 조언을 대체할 수 없음에 동의합니다.

또한, 이를 어기고 발생한 모든 결과에 대해 서비스 제공자에게 책임을 묻지 않을 것을 서약합니다.
        """
    )
    
    # 체크박스 상태 확인
    checked = st.checkbox("**내용을 꼼꼼히 확인 했습니다.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 동의합니다", type="primary", use_container_width=True, disabled=not checked):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    with col2:
        if st.button("❌ 거부합니다", use_container_width=True):
            st.markdown(
                """
            <script>
                window.close();
            </script>
            """,
                unsafe_allow_html=True,
            )
            st.error("서비스를 이용하지 않으셨습니다.")


# 첫 진입 시 면책사항 표시
if not st.session_state.disclaimer_accepted:
    disclaimer_dialog()
    st.stop()


# --- UI 레이아웃 ---
st.title("💊 의약품 정보 Q&A")

# 사이드바 - 정보 표시
with st.sidebar:
    st.markdown("### 📋 시스템 정보")
    st.info(
        f"""
    **분류 모델**: {CLASSIFIER_MODEL}  
    **답변 모델**: {LLM_MODEL}  
    
    본 서비스는 식약처 공공데이터를 기반으로 합니다.
    """
    )
    st.markdown("### 🔍 검색 팁")
    st.markdown(
        """
    - **제품명 검색**: "타이레놀", "게보린" 등
    - **성분명 검색**: "아세트아미노펜", "이부프로펜" 등
    - **증상 검색**: "두통", "감기", "소화불량" 등
    """
    )


# --- 메인 채팅 영역 ---
col_chat = st.columns([1])[0]

# 이전 메시지 표시
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
        <div class="user-message-group">
            <div class="user-bubble-container">
                <div class="chat-bubble">{message["content"]}</div>
            </div>
            <div class="user-icon">👤</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])


# 입력 필드
user_input = st.chat_input("의약품에 대해 궁금한 점을 물어보세요...")

if user_input:
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"""
    <div class="user-message-group">
        <div class="user-bubble-container">
            <div class="chat-bubble">{user_input}</div>
        </div>
        <div class="user-icon">👤</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 답변 생성
    with st.chat_message("assistant"):
        # 컨텍스트 준비
        prepared_context = prepare_context(user_input)
        
        # 스트리밍 답변 - 중간 결과도 실시간으로 표시
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in stream_answer(prepared_context):
            full_answer += chunk
            
            # 스트리밍 중에도 실시간으로 간단한 형식으로 표시
            if "병용금지 주의:" in full_answer:
                display_text = full_answer.split("병용금지 주의:")[0].strip()
            else:
                display_text = full_answer
            
            simplified_answer = format_answer_simplified(display_text)
            answer_placeholder.markdown(simplified_answer)
        
        # 최종 답변 표시
        simplified_answer = format_answer_simplified(full_answer)
        answer_placeholder.markdown(simplified_answer)
        
        # 병용금지 정보 expander에 표시 (DUR이 있을 경우에만)
        if prepared_context.get("dur_context") and prepared_context["dur_context"] != "(병용금지 정보 없음)":
            with st.expander("🚫 병용금지 주의사항 확인"):
                st.markdown(prepared_context["dur_context"])
        
        # 상호 병용금지 정보 (있을 경우)
        if prepared_context.get("mutual_context"):
            with st.expander("⚠️ 약품 간 상호 병용금지 경고"):
                st.markdown(prepared_context["mutual_context"])

    # 어시스턴트 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": simplified_answer})
