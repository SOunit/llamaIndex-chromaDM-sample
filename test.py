from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ① ドキュメントを読み込む（ここではカレントディレクトリ内の「sample」フォルダ）
documents = SimpleDirectoryReader("./sample").load_data()

# ② ドキュメントをチャンク化してベクトル化
index = VectorStoreIndex.from_documents(documents)

# ③ 簡単なクエリエンジンを作る
query_engine = index.as_query_engine()

# ④ 適当に質問してみる
response = query_engine.query("プロジェクトの概要を教えて")

print("\n=== AI RESPONSE ===\n")
print(response)
