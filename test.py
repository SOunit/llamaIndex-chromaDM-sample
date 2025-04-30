from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter

from dotenv import load_dotenv
load_dotenv()

# チャンク設定
text_splitter = SentenceSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)

# Settingsに直接セット
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding()
Settings.text_splitter = text_splitter

# ドキュメント読み込み
documents = SimpleDirectoryReader(
    # input_dir="C:/Users/Sho/source/repos/heymate-dashboard",
    # input_dir="C:/projects/aung-reviewer",
    input_dir="sample",
    recursive=True,
).load_data()

# インデックス作成
index = VectorStoreIndex.from_documents(documents)

# クエリエンジン作成
query_engine = index.as_query_engine()

# テストクエリ
# response = query_engine.query("プロジェクトを読み込んで、問題点や改善点、リファクタできる点を教えて")
response = query_engine.query("この文章についての感想を教えて")

print("\n=== AI RESPONSE ===\n")
print(response)