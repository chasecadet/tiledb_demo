import os
import logging
import argparse
import kserve
from langchain.vectorstores.tiledb import TileDB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.tiledb import TileDB
from minio import Minio
from utils import download_docs
from utils import process_docs
from utils import get_client
logger = logging.getLogger(__name__)
DEFAULT_NUM_DOCS = 2
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
ACCESS_KEY = os.getenv("ACCESS_KEY", 'minio')
SECRET_KEY= os.getenv("SECRET_KEY",'minio123')

class VectorStore(kserve.Model):
    def __init__(self, name: str, bucket_name: str):
        super().__init__(name)
        self.name = name
        self._prepare_vectorstore(bucket_name)
        self.ready = True
        self.vector_db=None

    def _prepare_vectorstore(self, bucket_name: str):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        documents = process_docs(docs, chunk_size=500, chunk_overlap=0)
        try:
            self.vector_db= TileDB.load("tmp_docs",embeddings)
        except:
            print("index was not present. Creating a new index")
            download_docs("tmp_docs")
            docs = load_docs("tmp_docs")
            documents = process_docs(docs, chunk_size=500, chunk_overlap=0)
            self.db = TileDB.from_documents(documents, embeddings, index_uri=tmp/index, index_type="FLAT"
    def predict(self, request: dict, headers: dict) -> dict:
        data = request["instances"][0]
        query = data["input"]
        num_docs = data.get("num_docs", DEFAULT_NUM_DOCS)
        logger.info(f"Received question: {query}")
        docs = self.vector_db.similarity_search(query, k=num_docs)
        logger.info(f"Retrieved context: {docs}")
        return {"predictions": [doc.page_content for doc in docs]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VectorStore",
                                     description="VectorStore server")
    parser.add_argument("bucket_name", type=str, required=True,
                        help="MinIO bucket location of the persisted VectorStore.")
    args = parser.parse_args()

    model = VectorStore("vectorstore", args.bucket_name)
    kserve.ModelServer(workers=1).start([model])