import os
import psycopg2
from google import genai

# Setup client for Gemini embeddings
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="my_sample_db",
    user="postgres",
    password="1234567890",
    host="localhost",
    port=5432
)
cur = conn.cursor()

documents = [
    "Retrieval Augmented Generation (RAG) improves LLMs with context.",
    "LangChain helps orchestrate LLMs and tools together."
]

def get_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=genai.types.EmbedContentConfig(output_dimensionality=3),
    )
    
    return response.embeddings[0].values

for doc in documents:
    emb = get_embedding(doc)
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (doc, emb)
    )

conn.commit()
cur.close()
conn.close()

print("Inserted documents with Gemini embeddings into PostgreSQL.")
