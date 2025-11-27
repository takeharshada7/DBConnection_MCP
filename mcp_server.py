import os
import psycopg2
from mcp.server.fastmcp import FastMCP
from google import genai

mcp = FastMCP("Gemini PostgreSQL MCP Server")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DB_PARAMS = {
    "dbname": "my_sample_db",
    "user": "postgres",
    "password": "1234567890",
    "host": "localhost",
    "port": 5432,
}

def get_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=genai.types.EmbedContentConfig(output_dimensionality=3),
    )
    return response.embeddings[0].values

@mcp.tool()
def insert_document(content: str):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    emb = get_embedding(content)
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)", (content, emb)
    )
    conn.commit()
    cur.close()
    conn.close()
    return "Document inserted successfully."

@mcp.tool()
def query_similar_documents(query: str, k: int = 3):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    query_emb = get_embedding(query)
    cur.execute(
        """
        SELECT content FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_emb, k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return "\n\n".join([row[0] for row in results])


if __name__ == "__main__":
    mcp.run(transport="stdio")
