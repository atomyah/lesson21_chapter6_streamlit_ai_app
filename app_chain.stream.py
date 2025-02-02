## ライブラリのインポート: pip install langchain==0.3.0 openai==1.47.0 langchain-community==0.3.0 langchain-openai==0.2.2 httpx==0.27.2
import os
from dotenv import load_dotenv  ## pip install python-dotenvインストール必要
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.schema.output_parser import StrOutputParser ## StrOutputParserチェインを使って回答をストリーミング表示

# from langchain.chains import ConversationChain

load_dotenv()   ## .envファイルの読み込み（ここでOPEN_API_KEYを読み込んでいる．したがって・・・👇）
# openai_api_key = os.getenv("OPENAI_API_KEY")  ## 👉環境変数からAPIキーを取得しているこの１文は不要


st.title("サンプルアプリ: 二人の専門家に相談しよう！")
st.write("##### 専門家1: 医療と先端治療の専門家")
st.write("医療界の先端技術と最新情報を聞くことができます。")
st.write("##### 専門家2: スピリチュアルの専門家")
st.write("スピリチュアルと宇宙の仕組みを聞くことができます。")


selected_item = st.radio(
    "専門家を選択してください。",
    ["医療専門家", "スピリチュアル専門家"]
)

st.divider()    ## 区切り線

## チャットモデルの初期化
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, streaming=True)

# StrOutputParserの初期化
output_parser = StrOutputParser()

## プロンプトのテンプレート
system_template = "あなたは{genre}の専門家です。質問に対して初心者にも分かりやすく簡潔に答えてください。{genre}以外の質問には答えないでください。"
human_template = "ユーザー：{question}"

## ChatPromptTemplateを生成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(  # 履歴用のプレースホルダーを追加. Placeholder（プレースホルダー）とは「何らかの値が入ってくるまで一時的に確保される場所」
        variable_name="chat_history"    ## variable_name="history" を指定してるがこれにより会話履歴が存在する場合その内容がここに挿入される
    ),
    HumanMessagePromptTemplate.from_template(human_template),
])

# チェーンの作成（output_parserはStrOutputParser、文字列表示）
chain = prompt | llm | output_parser



# セッション状態の初期化（ConversationBufferMemoryで保持する履歴データとは別物．UIに会話表示するためのステート）
if "talks" not in st.session_state:
    st.session_state["talks"] = []


# メモリの初期化をセッション状態で管理
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,   ## # メモリの初期化時に返信の形式を指定
        memory_key="chat_history",
        max_token_limit=1000
    )


################################################################################
################ 質問を処理する関数（ストリーミング形式で回答を表示）################
################################################################################
def process_question(genre: str, question: str) -> None:
    """質問を処理し、ストリーミング形式で回答を表示する"""
    # メモリから履歴を取得
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

    # まず、前回までの会話履歴を表示（st.session_state["talks"]の内容を順番に表示）
    for i, message in enumerate(st.session_state["talks"]):
        if i % 2 == 0:
            st.markdown(f"🙂 **あなた**: {message}")
        else:
            st.markdown(f"🤖 **{genre}専門家**: {message}")
    
    # 最新の質問を表示（まだst.session_state["talks"]には追加されていない．その後👇）
    st.markdown(f"🙂 **あなた**: {question}")

    # 回答用のプレースホルダーを作成
    answer_placeholder = st.empty() # UIに空の表示領域を作成(MessagesPlaceholderとは関係ない).
    full_response = ""  # 回答の途中経過を保持する変数

    # ストリーミング形式で回答を生成・表示
    for chunk in chain.stream({
        "genre": genre,
        "question": question,
        "chat_history": chat_history
    }):
        full_response += chunk
        answer_placeholder.markdown(f"🤖 **{genre}専門家**: {full_response}▌")
        # ここでは回答が生成される途中経過を表示しています
        # ▌記号はテキストカーソルを模擬してる．例：「こんに▌」→「こんにち▌」→「こんにちは▌」

    # 最終的な回答を表示（カーソルを消す）
    answer_placeholder.markdown(f"🤖 **{genre}専門家**: {full_response}")  # 最終的な回答を得る（カーソルなし）  

    # 👉ストリーミング完了後に質問と回答を履歴に追加（これらは次回の会話時に「過去の会話履歴」として表示されることになる）
    st.session_state["talks"].append(question)
    st.session_state["talks"].append(full_response)

    # 会話を保存（次回のMessagesPlaceholderで使用するためLangChainのメモリに保存）
    st.session_state.memory.save_context(
        {"input": question},
        {"output": full_response}
    )
################################################################################
############################ 質問を処理する関数、ここまで #########################
################################################################################

    # 自動スクロール用のJavaScript
    js = f"""
    <script>
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    st.components.v1.html(js, height=0)



# メインのインターフェース処理
if selected_item == "医療専門家":
    question = st.chat_input("医療専門家に質問してください。")
    if question:
        process_question("医療", question)

elif selected_item == "スピリチュアル専門家":
    question = st.chat_input("スピリチュアの専門家に質問してください。")
    if question:
        process_question("スピリチュアル", question)






# 会話履歴を表示（デバッグ用）
print("\n=== 会話履歴 ===")
conversation_history = st.session_state.memory.load_memory_variables({})["chat_history"]
print(conversation_history)