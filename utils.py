import os
import uuid
import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI

load_dotenv()

# Initialize once
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
embedding_model = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
llm = ChatMistralAI(api_key=MISTRAL_API_KEY)


def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


def extract_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file")
    return loader.load()  # returns List[Document]


def process_and_embed(file_path):
    docs = extract_text(file_path)
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content.strip() for c in chunks if c.page_content and c.page_content.strip()]
    if not texts:
        return []

    vectors = embedding_model.embed_documents(texts)

    return [
        {
            "id": str(uuid.uuid4()),
            "text": texts[i],
            "embedding": vectors[i]
        }
        for i in range(len(texts))
    ]


def get_supported_formats():
    return ["pdf", "docx"]


def save_to_chromadb(chunks, client, collection_name):
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks]
    )


def answer_query(query, client, collection_name, top_k=5):
    try:
        collection = client.get_collection(collection_name)
    except:
        return "No documents uploaded."

    query_vector = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    # Defensive checks for empty results
    docs_lists = results.get("documents", [])
    if not docs_lists or not docs_lists[0]:
        return "No relevant context found."

    context = "\n".join(docs_lists[0])

    prompt = f"""
Answer ONLY from the context below.
If not found, say you don't know.

Context:
{context}

Question:
{query}
"""
    return llm.invoke(prompt).content
