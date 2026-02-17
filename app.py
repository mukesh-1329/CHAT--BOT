import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Mistral Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Mistral Chatbot")
st.caption("Powered by Ollama + LangChain")

# -------------------------------
# Initialize Session State
# -------------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

# -------------------------------
# Load LLM
# -------------------------------
llm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    num_ctx=1024,
    num_predict=256,
    num_gpu=0  # Force CPU usage
)
# -------------------------------
# Prompt (Fixed to include history)
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("placeholder", "{history}"),  # Added this line
    ("human", "{input}")
])

chain = prompt | llm

# -------------------------------
# Memory Handler
# -------------------------------
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -------------------------------
# Display Chat History
# -------------------------------
history = get_session_history(st.session_state.session_id)

for msg in history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# -------------------------------
# User Input
# -------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot.invoke(
        {"input": user_input},
        config=RunnableConfig(
            configurable={"session_id": st.session_state.session_id}
        )
    )

    with st.chat_message("assistant"):
        st.markdown(response)  # Fixed typo