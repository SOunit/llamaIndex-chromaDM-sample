import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter

from llama_index.llms.ollama import Ollama

from utils.gitignore_directory_reader import GitIgnoreDirectoryReader
from dotenv import load_dotenv
load_dotenv()

def loadByMyself():
    # OpenAI設定（EmbeddingとLLM）
    Settings.embed_model = OpenAIEmbedding()
    Settings.llm = OpenAI(model="gpt-4o") # this is slow...
    # Settings.llm = Ollama(model="mistral")
    Settings.text_splitter = text_splitter

    # 📁 プロジェクト読み込み（.gitignore＆拡張子フィルタ付き）
    reader = GitIgnoreDirectoryReader(
        # input_dir="sample",  # ← 必要に応じて変更
        input_dir="sample",  # ← 必要に応じて変更
        recursive=True
    )
    documents = reader.load_data()

# timer start
start = time.perf_counter()

# チャンク設定（文単位で分割しつつ、サイズを制御）
text_splitter = SentenceSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# load using repomix
reader = SimpleDirectoryReader(input_dir="sample")
documents = reader.load_data()

# 📚 インデックス作成
index = VectorStoreIndex.from_documents(documents)

# 🔍 クエリエンジン作成
query_engine = index.as_query_engine()

# 💬 テストクエリ
response = query_engine.query("repomix-output--web.txtを読み込みましたね？これはあるプロジェクトのRepomixファイルです。つまり、AIにコンテキストを提供するためのデータファイルです。このプロジェクトは品質が低いのですが、このプロジェクトの特にひどい部分を１０個ほど指摘し、それに対する対応策を示してください。あなたはシニアエンジニアで、このプロジェクトを健全化する義務があります。いいですね。１０個の問題点の指摘と、具体的な改善案を示してください。")

print("\n=== AI RESPONSE ===\n")
print(response)

end = time.perf_counter()
print(f"⏱️ 実行時間: {end - start:.4f} 秒")