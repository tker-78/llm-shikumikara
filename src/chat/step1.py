import streamlit as st

st.title("マルチモーダルRAGチャットボット")

uploaded_file = st.file_uploader("画像を選択してください", type=['jpg', 'png', "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="画像", width=300)


user_input = st.text_input("メッセージを入力してください")

if st.button("送信"):
    st.write(f"human: {user_input}")