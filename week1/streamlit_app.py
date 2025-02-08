import streamlit as st
import requests
import json

# Ollama 서버 정보
OLLAMA_SERVER = "http://localhost:11434"
LLM_MODEL = "llama3.2:1b"

# 페이지 설정
st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

# 채팅 기록을 세션 상태에 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 UI 구성 (기존 메시지 표시)
st.title("💬 Ollama AI Chat")

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력창 (아래 배치)
user_input = st.chat_input("메시지를 입력하세요...")

# 사용자가 입력하면 Ollama와 통신
if user_input:
    # 사용자 메시지를 채팅 기록에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)

    # Ollama API 요청 (스트리밍 방식)
    url = f"{OLLAMA_SERVER}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "prompt": user_input, "stream": True}

    # AI 응답을 출력하는 공간 확보
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 스트리밍 응답 받기
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    # JSON 디코딩
                    json_data = json.loads(line.decode("utf-8"))
                    if "response" in json_data:
                        full_response += json_data["response"]
                        message_placeholder.markdown(full_response + "▌")

        # 최종 응답을 채팅 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        message_placeholder.markdown(full_response)  # 커서 제거