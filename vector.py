import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings.google_generative_ai import GoogleGenerativeAIEmbeddings

CONNECTION_STRING = "postgresql://postgres:1234567890@localhost:5432/my_sample_db"

# Use the LangChain Gemini Embeddings wrapper
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    chunk_size=1,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Create vectorstore from the documents 
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding=embeddings,
    collection_name="documents"
)

query = "How can I use Postgres for vector search?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
