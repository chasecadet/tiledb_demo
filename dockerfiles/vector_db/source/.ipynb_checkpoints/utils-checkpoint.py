import os
import logging
import boto3
import urllib
import posixpath
logger = logging.getLogger(__name__)

import os

# Initialize the Minio client (assuming it's already done elsewhere in your code)
# client = Minio("YOUR_MINIO_ENDPOINT", access_key="YOUR_ACCESS_KEY", secret_key="YOUR_SECRET_KEY", secure=True)

def download_docs(bucket_name):
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

# Example usage
# download_files('your-bucket-name')

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
def return_client(m_access_key,m_secret_key):
    client = Minio("minio-service.kubeflow.svc.cluster.local:9000",
    access_key=m_access_key,
    secret_key=m_secret_key,
    secure=False,   
                  )
    return client