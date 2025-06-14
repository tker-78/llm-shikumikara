import streamlit as st
import os
import contextlib
import io
import dotenv
import matplotlib.pyplot as plt
from matplotlib import font_manager

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter

dotenv.load_dotenv()
# MODEL_NAME = "gpt-4.1-nano-2025-04-14"
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
# ドキュメントを整形
def format_documents(documents):
    return "\n\n".join(document.page_content for document in documents)

def create_chain():
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        persist_directory="data",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "以下の指示に対して、matplotlibを使ったpythonコードだけを出力してください。不要な説明は書かないでください。Streamlitでグラフを描画できる形式にしてください。"),
            ("system", "回答には以下の情報も参考にしてください。参考情報: \n{info}"),
            ("placeholder", "{history}"),
            ("human", "{input}")
        ]
    )
    # return prompt | ChatOpenAI(model=MODEL_NAME, temperature=0)
    return (
        {
            "input": itemgetter("input"),
            "info": itemgetter("input") | retriever | format_documents,
            "history": itemgetter("history")
        }
        | prompt
        | ChatOpenAI(model=MODEL_NAME, temperature=0)
    )

# セッション状態を初期化
if "history" not in st.session_state:
    st.session_state.history = []
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
        }
    )
    code = response.content
    st.write(code)

    if code.startswith("```python"):
        code = code.replace("```python", "", 1)
    if code.endswith("```"):
        code = code.replace("```", "", 1)

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic',
                                               'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

            exec(code)
        except SyntaxError as e:
            st.error(f"構文エラーが発生しました。: \n行数: {e.lineno}\n位置] {e.offset}\nテキスト: {e.text}")
        except Exception as e:
            st.error(f"エラーが発生しました。: {e}")


    # st.session_state.history.append(HumanMessage(user_input))
    # st.session_state.history.append(response)

    # for message in reversed(st.session_state.history):
    #     st.write(f"{message.type}: {message.content}")




