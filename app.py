import os
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


load_dotenv() 

### StreamlitCallbackHandlerクラス（カスタムハンドラー）の定義(StreamingStdOutCallbackHandlerを継承)
class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container  ## Streamlitのプレースホルダー（st.empty()で作成した表示領域）。st.delta_generator.DeltaGeneratorはStreamlitの表示要素を扱うクラスの型
        self.text = initial_text    ## self.textは生成されたテキスト全体を蓄積するための変数（初期値は空文字列）
        # self.new_line = True    
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:   #### LLMが新しいトークンを生成するたびに呼び出される関数。
                                                                #### StreamingStdOutCallbackHandler）のメソッドをオーバーライドしている
                                                                #### token: LLMから新しく生成されたテキストの断片. **kwargs: その他の引数(使用していない)
        self.text += token
        self.container.markdown(f"🤖 **専門家**: {self.text}▌") #### Streamlitのプレースホルダーを更新。🤖 絵文字を追加。▌はカーソル

#######################################################################################################################################################
# 【StreamingStdOutCallbackHandlerを継承したStreamlitCallbackHandlerを作成するワケ】
# llm = ChatOpenAI(
#     model_name="gpt-4o-mini",temperature=0.5,streaming=True,callbacks=[StreamingStdOutCallbackHandler()]
# )
# この標準的な書き方だと出力は標準出力（stdout）にのみでStreamlitのユーザーインターフェースには表示されない．トークンの制御や保存もできない．

# クラスを継承してStreamlitのプレースホルダーに表示するようにした結果、このカスタムハンドラーにより、以下のUI表示を実現。
# "こ" → 🤖 **専門家**: こ▌
# "ん" → 🤖 **専門家**: こん▌
# "に" → 🤖 **専門家**: こんに▌
# "ち" → 🤖 **専門家**: こんにち▌
# "は" → 🤖 **専門家**: こんにちは▌
#######################################################################################################################################################




############################### process_question関数　#####################################
def process_question(genre: str, question: str) -> None:
    # メモリから履歴を取得
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
    
    # 会話履歴の表示
    for i, message in enumerate(st.session_state["talks"]):
        if i % 2 == 0:
            st.markdown(f"🙂 **あなた**: {message}")
        else:
            st.markdown(f"🤖 **{genre}専門家**: {message}")
    
    # 最新の質問を表示
    st.markdown(f"🙂 **あなた**: {question}")
    
    # Streamlitのプレースホルダー（表示領域）を作成
    answer_placeholder = st.empty() #### UIに空の表示領域を作成(MessagesPlaceholderとは関係ない).
    
    # カスタムコールバックハンドラーの作成
    callback = StreamlitCallbackHandler(answer_placeholder)     #### answer_placeholder引数は、def __init__()の引数container: st.delta_generator.DeltaGeneratorのこと
    
    # チャットモデルの初期化（コールバック付き）
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.5,
        streaming=True,
        callbacks=[callback]
    )
    
    # プロンプトテンプレート
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "あなたは{genre}の専門家です。質問に対して初心者にも分かりやすく簡潔に答えてください。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    
    # 応答の生成
    result = llm(
        prompt.format_messages(
            genre=genre,
            question=question,
            chat_history=chat_history
        )
    )

    # 履歴の更新
    st.session_state["talks"].append(question)
    st.session_state["talks"].append(callback.text)
    
    # メモリの更新
    st.session_state.memory.save_context(
        {"input": question},
        {"output": callback.text}
    )
############################### process_question関数~ここまで　#####################################



# メインのUI部分
st.title("サンプルアプリ: 二人の専門家に相談しよう！")
st.write("##### 専門家1: 医療と先端治療の専門家")
st.write("医療界の先端技術と最新情報を聞くことができます。")
st.write("##### 専門家2: スピリチュアルの専門家")
st.write("スピリチュアルと宇宙の仕組みを聞くことができます。")

st.divider()    ## 区切り線

# 初期化コード
if "talks" not in st.session_state:
    st.session_state["talks"] = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        max_token_limit=1000
    )

# 専門家の選択
selected_item = st.radio(
    "専門家を選択してください。",
    ["医療専門家", "スピリチュアル専門家"]
)

# 質問入力と処理
if selected_item == "医療専門家":
    question = st.chat_input("医療専門家に質問してください。")
    if question:
        process_question("医療", question)
elif selected_item == "スピリチュアル専門家":
    question = st.chat_input("スピリチュアルの専門家に質問してください。")
    if question:
        process_question("スピリチュアル", question)