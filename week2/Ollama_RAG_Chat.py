import streamlit as st
import requests
import json
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ollama 서버 정보
OLLAMA_SERVER = "http://localhost:11434"
LLM_MODEL = "llama384"
MEMORY_FILE = "chat_memory.json"
VECTOR_DB_PATH = "faiss_index"

# 페이지 설정
st.set_page_config(page_title="Ollama Chat", page_icon="💬", layout="wide")

# 메모리 불러오기 함수
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

# 메모리 저장 함수
def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

# PDF 파일을 읽어와서 문서를 생성하는 함수
def load_pdfs_from_folder(folder_path="data_pdf"):
    documents = []
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file_name)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
    return documents

# PDF 내용을 페이지별로 세분화하여 처리하는 함수
def process_pdfs():
    documents = load_pdfs_from_folder()
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = []
        for doc in documents:
            split_docs.extend(text_splitter.split_text(doc.page_content))
        return split_docs
    return []

# RAG용 VectorDB 생성 함수
def create_vector_db():
    if not os.path.exists(VECTOR_DB_PATH):
        documents = process_pdfs()
        if documents:
            embeddings = HuggingFaceEmbeddings()
            vector_db = FAISS.from_texts(documents, embeddings)
            vector_db.save_local(VECTOR_DB_PATH)
        else:
            st.error("data_pdf 폴더에 PDF 파일이 없거나 파싱된 내용이 없습니다.")

# VectorDB 불러오기 함수
def load_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        embeddings = HuggingFaceEmbeddings()
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        create_vector_db()
        return load_vector_db()

# LangChain 메모리 초기화 (세션 상태에 저장)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=50, human_prefix="User", ai_prefix="Assistant")
    st.session_state.messages = load_memory()
    st.session_state.vector_db = load_vector_db()

# 채팅 UI 구성 (기존 메시지 표시)
st.title("💬 Ollama AI Chat with Enhanced Memory & RAG")

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

    # LangChain 메모리에 저장
    st.session_state.memory.save_context({"input": user_input}, {"output": ""})

    # RAG를 이용한 정보 검색
    search_results = st.session_state.vector_db.similarity_search(user_input, k=5)
    retrieved_context = "\n".join([doc.page_content for doc in search_results])

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)

    # LangChain 메모리에서 전체 대화 히스토리 가져오기
    chat_history = st.session_state.memory.load_memory_variables({})["history"]

    # Ollama API 요청 (스트리밍 방식)
    url = f"{OLLAMA_SERVER}/api/generate"
    headers = {"Content-Type": "application/json"}

    # Ollama에 보낼 프롬프트 (대화 히스토리, 요약, RAG 포함)
    full_prompt = f"""요약된 대화:
{chat_history}

검색된 관련 정보:
{retrieved_context}

User: {user_input}
Assistant:"""

    payload = {
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": True
    }

    # AI 응답을 출력하는 공간 확보
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 스트리밍 응답 받기
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line.decode("utf-8"))
                    if "response" in json_data:
                        full_response += json_data["response"]
                        message_placeholder.markdown(full_response + "▌")

        # 최종 응답을 메모리에 저장
        st.session_state.memory.save_context({"input": user_input}, {"output": full_response})

        # 채팅 기록에도 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 메모리 저장
        save_memory()

        # 커서 제거 후 최종 응답 표시
        message_placeholder.markdown(full_response)

# 메모리 초기화 버튼
if st.sidebar.button("💾 메모리 초기화"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    if os.path.exists(VECTOR_DB_PATH):
        os.system(f"rm -rf {VECTOR_DB_PATH}")
    st.success("메모리와 VectorDB가 초기화되었습니다.")
