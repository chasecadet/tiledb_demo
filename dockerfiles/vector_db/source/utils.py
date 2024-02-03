from langchain.text_splitter import RecursiveCharacterTextSplitter
from minio import Minio
import os
import glob
from tqdm import tqdm
from langchain.document_loaders import TextLoader

def download_docs(bucket_name,client):
    # Ensure the documentation directory exists
    target_directory = "tmp_docs"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    # List all objects in the bucket
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        # Check if the object is a .txt file
        if obj.object_name.endswith('.txt'):
            # Construct the full path for the file to be downloaded
            destination_path = os.path.join(target_directory, obj.object_name)
            
            # Ensure the subdirectory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Download the object
            client.fget_object(bucket_name, obj.object_name, destination_path)
            print(f"Downloaded {obj.object_name} to {destination_path}")
        else:
            print(f"Skipping non-txt file {obj.object_name}")
def load_doc(fn):
    loader = TextLoader(fn)
    doc = loader.load()
    return doc

def load_docs(source_dir: str) -> list:
    """Load all documents in a the given directory."""
    fns = glob.glob(os.path.join(source_dir, "*.txt"))    
    docs = []
    for i, fn in enumerate(tqdm(fns, desc="Loading documents...")):
        docs.extend(load_doc(fn))
    return docs

def process_docs(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """Load the documents and split them into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts

def get_client(m_access_key,m_secret_key,client_url):
    client = Minio(client_url,
    access_key=m_access_key,
    secret_key=m_secret_key,
    secure=False,   
                  )
    return client