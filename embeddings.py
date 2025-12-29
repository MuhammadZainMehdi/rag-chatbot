from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, max_tokens=192, overlap=28 ):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    step = max_tokens - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def embedd_text(chunks):
    embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=True)

    return embeddings
