import argparse
import kserve
import os
import logging
from minio import Minio
import glob
from tqdm import tqdm
from langchain.vectorstores.tiledb import TileDB
from langchain.embeddings import HuggingFaceEmbeddings
from utils import download_docs
from utils import process_docs
from utils import get_client
from utils import load_docs



logger = logging.getLogger(__name__)
DEFAULT_NUM_DOCS = 2
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
ACCESS_KEY = os.getenv("ACCESS_KEY", 'minio')
SECRET_KEY= os.getenv("SECRET_KEY",'minio123')
CLIENT_URL=os.getenv("CLIENT_URL","minio-service.kubeflow.svc.cluster.local:9000")

class VectorStore(kserve.Model):
    def __init__(self, name: str, docs_bucket_name: str):
        super().__init__(name)
        self.name = name
        self._prepare_vectorstore(docs_bucket_name)
        self.ready = True

    def _prepare_vectorstore(self, bucket_name: str):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        try:
            self.vector_db=TileDB.load("tmp_index",embeddings)
        except:
            print("index was not present. Creating a new index")
            client=get_client(ACCESS_KEY,SECRET_KEY,CLIENT_URL)
            download_docs(bucket_name,client)
            docs = load_docs("tmp_docs")
            documents = process_docs(docs, chunk_size=500, chunk_overlap=0)
            self.vector_db = TileDB.from_documents(documents, embeddings, index_uri="tmp_index", index_type="FLAT")
    
    def predict(self, request: dict, headers: dict) -> dict:
        data = request["instances"][0]
        query = data["input"]
        num_docs = data.get("num_docs", DEFAULT_NUM_DOCS)
        logger.info(f"Received question: {query}")
        docs = self.vector_db.similarity_search(query, k=num_docs)
        logger.info(f"Retrieved context: {docs}")
        return {"predictions": [doc.page_content for doc in docs]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VectorStore", description="VectorStore server")
    # Remove required=True for positional arguments
    parser.add_argument("docs_bucket_name", type=str, help="MinIO bucket location of the documents that need embedding.")
    args = parser.parse_args()
    model = VectorStore("vectorstore", args.docs_bucket_name)
    kserve.ModelServer(workers=1).start([model])
