from fastapi import FastAPI, Request
from pydantic import BaseModel

import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter

from llama_index.llms.ollama import Ollama

import chromadb
# from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

from utils.gitignore_directory_reader import GitIgnoreDirectoryReader
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    query: str

@app.on_event("startup")
def init():
    print("here? - 1")

    # timer start
    start = time.perf_counter()

    # チャンク設定（文単位で分割しつつ、サイズを制御）
    text_splitter = SentenceSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    # load using repomix
    reader = SimpleDirectoryReader(input_dir="stories")
    documents = reader.load_data()

    print("here? - 2")

    # 📚 インデックス作成
    index = VectorStoreIndex.from_documents(documents)

    print("here? - 3")

    # 🔍 クエリエンジン作成
    query_engine = index.as_query_engine()

    print("here? - 4")

    # 💬 テストクエリ
    response = query_engine.query("織田信長の文章を読みましたね？段落ごとの要約を作ってください。また、全体として読み取れるメッセージも要約してくださいい。")

    print("here? - 5")

    print("\n=== AI RESPONSE ===\n")
    print(response)

    end = time.perf_counter()
    print(f"⏱️ 実行時間: {end - start:.4f} 秒")

@app.get("/test")
def test():
     return {"test": "test - get"}

@app.post("/test")
def postTest(req: QueryRequest):
     return {"test": req.query}
    

@app.post("/query")
async def query_codebase(req: QueryRequest):
    # chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("project_chunks"))
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # index = load_index_from_storage(storage_context)

    # query_engine = index.as_query_engine()
    # response = query_engine.query(req.question)
    # return {"answer": str(response)}
    return {"answer": str("test")}

def loadByMyself():
    # OpenAI設定（EmbeddingとLLM）
    Settings.embed_model = OpenAIEmbedding()
    Settings.llm = OpenAI(model="gpt-4o") # this is slow...
    # Settings.llm = Ollama(model="mistral")
    Settings.text_splitter = text_splitter

    # 📁 プロジェクト読み込み（.gitignore＆拡張子フィルタ付き）
    reader = GitIgnoreDirectoryReader(
        # input_dir="sample",  # ← 必要に応じて変更
        input_dir="stories",  # ← 必要に応じて変更
        recursive=True
    )
    documents = reader.load_data()

