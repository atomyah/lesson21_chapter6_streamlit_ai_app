# lesson21_chapter6_kadai

## コマンド集

###【venvの使い方】
#### 仮想環境の作成
python -m venv env (envは任意の名前)

#### 仮想環境の有効化
env¥Scripts¥activate

#### 仮想環境が有効になっているか確認
python --version  # コマンドプロンプトの先頭に (myenv) と表示される

#### インストールされてるパッケージのリスト
pip list

#### requirements.txtからパッケージ一括インストール
pip install -r requirements.txt


#### pipを最新にアップグレード
python -m pip install --upgrade pip

#### 仮想環境の終了
deactivate

### 【Copilot使用方法】
#### 候補が提案された後、「Ctrl + →」候補が単語単位で採用される


### 【Stremalit環境構築】
#### pipを最新にアップグレード
python.exe -m pip install --upgrade pip

#### Streamlitをインストール
pip install streamlit==1.41.1

#### Streamlitの開発サーバーを立ち上げ
streamlit run app.py

### 【Streamlit Cloudへデプロイ前】
#### 仮想環境を立ち上げる
env¥Scripts¥activate

#### パッケージ一覧の記述ファイル作成（デプロイ先のStreamlit Cloud上にrequirement.txtもアップする必要がある）
pip freeze > requirements.txt

#### git push
