import streamlit as st
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()
MODEL_NAME = "gpt-4.1-nano-2025-04-14"

def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "回答には以下の情報も参考にしてください。参考情報: \n{info}",
            ),
            ("placeholder", "{history}"),
            ("human", "{input}")
        ]
    )
    return prompt | ChatOpenAI(model=MODEL_NAME, temperature=0)

# セッション状態を初期化
if "history" not in st.session_state:
    st.session_state.history = []
    # st.session_state.llm = ChatOpenAI(model_name=MODEL_NAME)
    st.session_state.chain = create_chain()



st.title("マルチモーダルRAGチャットボット")

uploaded_file = st.file_uploader("画像を選択してください", type=['jpg', 'png', "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="画像", width=300)

user_input = st.text_input("メッセージを入力してください")


if st.button("送信"):
    response = st.session_state.chain.invoke(
        {
            "input": user_input,
            "history": st.session_state.history,
            "info": "ユーザの年齢は10歳です。",
        }
    )
    st.session_state.history.append(HumanMessage(user_input))
    # response = st.session_state.llm.invoke(st.session_state.history)
    st.session_state.history.append(response)

    for message in reversed(st.session_state.history):
        st.write(f"{message.type}: {message.content}")




