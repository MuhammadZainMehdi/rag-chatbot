import streamlit as st
import PyPDF2
from docx import Document

from embeddings import embedd_text, chunk_text
from pc_index import create_index, upsert_vectors, query_index
from groq import Groq
import os

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# Initialize Pinecone index
if 'index' not in st.session_state:
    st.session_state.index = create_index()

# Initialize Groq client
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def upload_file():
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"]
    )
    text = None

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]

        with st.spinner(f"Reading {file_type.upper()} file..."):
            if file_type == "txt":
                text = uploaded_file.read().decode("utf-8")

            elif file_type == "pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            elif file_type == "docx":
                doc = Document(uploaded_file)
                text = "\n".join(p.text for p in doc.paragraphs)
    
    return text


def generate_response(query, context):
    """Generate response using Groq with retrieved context"""
    
    prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    
    assistant_message = response.choices[0].message.content
    
    return assistant_message


# Sidebar
with st.sidebar:
    st.title("📄 RAG Chatbot")
    st.markdown("---")
    
    # File upload
    st.subheader("Upload Document")
    text = upload_file()
    
    if text:
        # Only process if not already processed
        if 'processed' not in st.session_state or not st.session_state.processed:
            with st.spinner("Chunking text..."):
                chunks = chunk_text(text)
            
            with st.spinner("Creating embeddings..."):
                embedded_text = embedd_text(chunks)
            
            st.success(f"✅ {len(embedded_text)} chunks created")

            with st.spinner("Storing in Pinecone..."):
                upsert_vectors(st.session_state.index, chunks, embedded_text)
            
            st.success("✅ Ready to chat!")
            st.session_state.processed = True
            st.session_state.chunks = chunks
    
    st.markdown("---")
    
    # Controls
    st.subheader("Controls")
    
    if st.button("🔄 Reset Document", use_container_width=True):
        st.session_state.processed = False
        if 'chunks' in st.session_state:
            del st.session_state.chunks
        st.rerun()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    if 'processed' in st.session_state and st.session_state.processed:
        st.subheader("Stats")
        st.metric("Chunks Stored", len(st.session_state.chunks))
        st.metric("Messages", len(st.session_state.messages))


# Main chat interface
st.title("💬 Chat with Your Document")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show context for assistant messages
        if message["role"] == "assistant" and "context" in message:
            with st.expander("📄 View Retrieved Context"):
                st.text_area("Context used:", value=message["context"], height=150, disabled=True)

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if 'processed' not in st.session_state or not st.session_state.processed:
        st.warning("⚠️ Please upload and process a document first!")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Search and generate response
        with st.spinner("Searching and generating answer..."):
            # Create query embedding
            query_embedding = embedd_text([prompt])[0]
            
            # Query the index
            results = query_index(st.session_state.index, query_embedding, top_k=3)
        
        if results.matches:
            # Combine all retrieved chunks
            retrieved_text = "\n\n".join([match.metadata['text'] for match in results.matches])
            
            # Generate AI response
            answer = generate_response(prompt, retrieved_text)
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Show context in expander
                with st.expander("📄 View Retrieved Context"):
                    st.text_area("Context used:", value=retrieved_text, height=150, disabled=True)
                    
                    st.markdown("### Individual Chunks")
                    for i, match in enumerate(results.matches, 1):
                        st.markdown(f"**Chunk {i}** (Similarity: {match.score:.4f})")
                        st.write(match.metadata['text'])
                        st.markdown("---")
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "context": retrieved_text
            })
        else:
            # No results found
            with st.chat_message("assistant"):
                st.warning("No relevant information found in the document.")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "No relevant information found in the document."
            })

# Show instructions if no document uploaded
if 'processed' not in st.session_state or not st.session_state.processed:
    st.info("👈 Please upload a document from the sidebar to get started!")
