import os
import re
import math
import json
from tqdm import tqdm
import random
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from sklearn.manifold import TSNE
import plotly.graph_objects as go

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
DB = "products_vectorstore"

hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

client = chromadb.PersistentClient(path=DB)
# Check if the collection exists and delete it if it does
collection_name = "products"
existing_collection_names = [collection.name for collection in client.list_collections()]
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

collection = client.create_collection(collection_name)

# Use SentenceTransformer to semantic search
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#vector = model.encode(["Well hi there"])[0]
#print(vector)

def description(item):
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]

#print(description(train[0]))

for i in tqdm(range(0, len(train), 1000)):
    documents = [description(item) for item in train[i: i+1000]]
    vectors = model.encode(documents).astype(float).tolist()
    metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
    ids = [f"doc_{j}" for j in range(i, i+1000)]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )