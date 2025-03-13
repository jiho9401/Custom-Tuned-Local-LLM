import os
import streamlit as st
import re
from langchain.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import time
from langchain.prompts import PromptTemplate

# 현재 파일의 절대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# =======================
# 1. 모델 선택 드롭다운 추가
# =======================
MODEL_OPTIONS = {
    "라마 3.2 3B": {"model": "llama33B8Q", "base_url": "http://192.168.10.123:11434"},
    "라마 3 8B": {"model": "llama38B4Q", "base_url": "http://192.168.0.123:11434"},
    "deepseek-r1 32b": {"model": "deepseek-r1:32b", "base_url": None},  # 로컬 실행
    "qwq": {"model": "qwq", "base_url": None},  # 로컬 실행
}

st.title("LLM 테스팅")

# =======================
# 2. 세션 상태 초기화 (대화 유지)
# =======================
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "라마 3.2 3B"

# 대화 기억을 위한 메모리 (한 번만 생성)
if "memory" not in st.session_state:
    # 긴 메모리 유지하면서 가중치 조절을 위한 설정
    st.session_state.memory = ConversationBufferWindowMemory(k=5000, return_messages=True, memory_key="history")

# 프롬프트 템플릿 설정 (세션 상태에 저장)
if "template" not in st.session_state:
    st.session_state.template = """당신은 지식이 풍부하고 도움이 되는 AI 어시스턴트입니다.

아래는 우리의 대화 기록입니다:
{history}

가장 최근의 질문은 가장 중요하게 고려하세요. 
이전 대화는 맥락을 이해하는 데 참고만 하고, 현재 질문에 집중해서 답변하세요.

현재 질문: {input}

답변:"""

selected_model = st.selectbox(
    "사용할 AI 모델을 선택하세요:",
    options=list(MODEL_OPTIONS.keys()),
    index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model),  # 기본값 유지
)

# 선택한 모델 정보 가져오기
selected_model_info = MODEL_OPTIONS[selected_model]
model_name = selected_model_info["model"]
base_url = selected_model_info["base_url"]

# =======================
# 3. 모델 변경 시 기존 memory 유지하면서 ConversationChain 업데이트
# =======================
if "conversation_chain" not in st.session_state or st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model  # 선택 모델 저장

    # 새로운 LLM 생성
    if base_url:  # 원격 서버 사용
        llm = Ollama(model=model_name, base_url=base_url)
    else:  # 로컬 실행
        llm = Ollama(model=model_name)

    # 세션 상태의 템플릿 사용
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=st.session_state.template
    )

    # 기존 memory를 유지하면서 새로운 모델과 프롬프트 템플릿 적용
    st.session_state.conversation_chain = ConversationChain(
        llm=llm, 
        memory=st.session_state.memory,
        prompt=prompt,
        verbose=False
    )

# =======================
# 4. 대화 내역 표시
# =======================
for message in st.session_state.memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        clean_response = re.sub(r"<think>.*?</think>", "", message.content, flags=re.DOTALL).strip()  # <think> 태그 제거
        st.chat_message("assistant").write(clean_response)

# =======================
# 5. 채팅 입력창 (하단 고정)
# =======================
if prompt := st.chat_input("메시지를 입력하세요:"):
    st.chat_message("user").write(prompt)  # 사용자 메시지 출력

    if "conversation_chain" in st.session_state:
        ai_msg = st.chat_message("assistant")
        
        # 생각과 응답을 위한 컨테이너 생성
        thinking_container = ai_msg.container()
        response_container = ai_msg.container()
        
        # 응답 표시를 위한 비어있는 텍스트 영역 생성
        with response_container:
            response_area = st.empty()
        
        # 상태 표시
        with st.status("🤔 AI가 생각 중...", expanded=True) as status:
            response = ""
            thinking = ""
            is_thinking = False
            
            # 과거 대화 내용 형식 변환 (프롬프트용)
            past_messages = st.session_state.memory.chat_memory.messages
            history_text = ""
            
            for msg in past_messages:
                if isinstance(msg, HumanMessage):
                    history_text += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    # <think> 태그 제거
                    clean_content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
                    history_text += f"AI: {clean_content}\n"
            
            # 스트리밍 응답
            prompt_with_history = st.session_state.template.format(
                history=history_text,
                input=prompt
            )
            
            for chunk in st.session_state.conversation_chain.llm.stream(prompt_with_history):
                # <think> 태그 처리
                if "<think>" in chunk:
                    is_thinking = True
                    # thinking_container에 생각 영역 생성
                    with thinking_container:
                        st.markdown("#### 🧠 모델이 생각하는 과정:")
                        st.markdown("---")
                        thinking_placeholder = st.empty()
                    continue
                    
                if "</think>" in chunk:
                    is_thinking = False
                    with thinking_container:
                        st.markdown("---")
                    continue
                
                if is_thinking:
                    # 생각 내용 추가
                    thinking += chunk
                    # 생각 과정 표시 (기울임체 사용으로 구분)
                    with thinking_container:
                        thinking_placeholder.markdown(f"*{thinking}*")
                else:
                    # 응답 추가 - 정규화 코드 제거하고 원본 유지
                    response += chunk
                    # 실시간으로 응답 업데이트 (HTML 강제)
                    response_area.markdown(f"<div class='response-text'>{response}</div>", unsafe_allow_html=True)
            
            # 상태 업데이트
            status.update(label="✅ 응답 완료!", state="complete", expanded=False)

        # 최종 응답 저장
        if thinking:
            full_response = f"<think>{thinking}</think>{response}"
        else:
            full_response = response
            
        # 메모리에 대화 저장
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(full_response)
        
    else:
        st.error("⚠️ 모델이 올바르게 초기화되지 않았습니다. 다시 시도해 주세요.")
        
# 페이지 스타일 개선
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 12px;
    }
    .response-text {
        line-height: 1.5;
        letter-spacing: normal !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)
