import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Load the embedding model (same as before)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "mps"},  # Use Apple GPU
    cache_folder="./models"
)

# ✅ Load the persisted ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, client=chroma_client)

print("✅ ChromaDB successfully loaded!")

query = input("Query: ")

# ✅ Retrieve the **top 5** most similar documents
similar_fights = vectorstore.similarity_search(query, k=5)

# ✅ Display results
print("\n🔍 Top 5 Most Relevant Fights:")
for i, doc in enumerate(similar_fights):
    print(f"\n🔥 Match {i+1}:")
    print(doc.page_content)
