# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API"))

# Create a dense index with integrated embedding
def create_index():
    index_name = "rag-chatbot"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    return pc.Index(index_name)

def upsert_vectors(index, chunk, embedding):
    records = [
        {
            "id": f"chunk-{i}",
            "values": emb.tolist() if hasattr(emb, 'tolist') else emb,
            "metadata": {"text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunk, embedding))
    ]

    index.upsert(vectors=records, namespace='default')


def query_index(index, query_embedding, top_k=5):
    results = index.query(
        vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
        top_k=top_k,
        namespace='default',
        include_metadata=True
    )
    return results