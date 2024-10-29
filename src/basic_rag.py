import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

MODEL = 'gpt-4o-mini'
db_name = 'vector_db'

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

folders = glob.glob("src/knowledge-base/*")
text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
#print(f"Document types found: {', '.join(doc_types)}")

# Put the chunks into a Vector Store that associates a Vector Embedding with each chunk
embeddings = OpenAIEmbeddings()

# Delete the Chroma data store if it exists
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create the Chroma vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
#print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# # See the dimension of one of the vectors
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])['embeddings'][0]
dimensions = len(sample_embedding)
#print(f"The vectors have {dimensions:,} dimensions")

# Prework
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

# # Visualizing the vector store in 2D plot
# tsne = TSNE(n_components=2, random_state=42)
# reduced_vectors = tsne.fit_transform(vectors)

# # Create the 2D scatter plot
# fig = go.Figure(data=[go.Scatter(
#     x=reduced_vectors[:, 0],
#     y=reduced_vectors[:, 1],
#     mode='markers',
#     marker=dict(size=5, color=colors, opacity=0.8),
#     text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
#     hoverinfo='text'
# )])

# fig.update_layout(
#     title='2D Chroma Vector Store Visualization',
#     scene=dict(xaxis_title='x',yaxis_title='y'),
#     width=800,
#     height=600,
#     margin=dict(r=20, b=10, l=10, t=40)
# )

# #fig.show()

# tsne = TSNE(n_components=3, random_state=42)
# reduced_vectors = tsne.fit_transform(vectors)

# # Create the 3D scatter plot
# fig = go.Figure(data=[go.Scatter3d(
#     x=reduced_vectors[:, 0],
#     y=reduced_vectors[:, 1],
#     z=reduced_vectors[:, 2],
#     mode='markers',
#     marker=dict(size=5, color=colors, opacity=0.8),
#     text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
#     hoverinfo='text'
# )])

# fig.update_layout(
#     title='3D Chroma Vector Store Visualization',
#     scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
#     width=900,
#     height=700,
#     margin=dict(r=20, b=10, l=10, t=40)
# )

# fig.show()

### Implement the RAG
# Create a new chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# Set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Create the retriever which is an abstraction over the VectorStore that RAG will use it
retriever = vectorstore.as_retriever()

# Put everything together
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Can you describe Insurellm in a few sentences"
result = conversation_chain.invoke({"question":query})
print(result["answer"])

# Create a function for this
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat).launch()