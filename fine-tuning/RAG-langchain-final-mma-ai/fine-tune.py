# Populate the Vector Store

import gc
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd

# ✅ Use optimized embedding model with Apple GPU (MPS)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "mps"},  # Run on Apple GPU
    cache_folder="./models"
)

# ✅ Load fighter data
df = pd.read_csv("data-raw/fighter_info.csv")

# ✅ Convert all columns to a single text field
df['text'] = df.apply(lambda row: "\n".join(
    [f"{col}: {row[col]}" for col in df.columns]
), axis=1)

# ✅ Convert into LangChain document format
texts = df['text'].tolist()
documents = [Document(page_content=text) for text in texts]

# ✅ Split documents to optimize vector search
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# ✅ Fix: Use Chroma Persistent Client (Prevents memory overload)
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # NEW WAY
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, client=chroma_client)

# ✅ Process and insert documents in **small batches** (prevents crashes)
batch_size = 2  # Reduce to 25 if crashes still occur
total_documents = len(split_docs)

for i in range(0, total_documents, batch_size):
    batch_docs = split_docs[i : i + batch_size]
    vectorstore.add_documents(batch_docs)  # ✅ Insert smaller batches

    print(f"✅ Inserted {i + batch_size} / {total_documents} documents")

    # ✅ Free unused memory **after each batch** to prevent RAM spikes
    del batch_docs
    gc.collect()

# ✅ Save ChromaDB to disk
vectorstore.persist()
print("\n✅ ChromaDB successfully populated and persisted to disk!")
