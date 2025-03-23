# Rewriting the file for download again since the session expired

import openai
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    # STEP 1: Sample documents
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG stands for Retrieval-Augmented Generation, a method to enhance LLM output.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "OpenAI provides powerful language models like GPT-4 for natural language tasks.",
        "Embedding models convert text into vector representations for semantic search.",
        "You can use RAG to combine document search and LLMs to answer complex questions.",
        "SentenceTransformers provides tools for embedding and similarity comparisons.",
    ]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings, index = storeToVectorDB(documents, embedder)

    query = "How does retrieval-augmented generation work?"
    synthesized_context = retrieveFromVectorDB(
        query, documents, doc_embeddings, index, embedder
    )

    # STEP 8: Generate response using OpenAI
    prompt = f"""Use the following context to answer the question:

    Context:
    { synthesized_context }

    Question:
    { query }
    """
    answer = generateAnswer(prompt)
    print("Final Answer:", answer)


def storeToVectorDB(documents, embedder=None):
    # STEP 2: Encode documents
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

    # STEP 3: Build FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    return doc_embeddings, index


def retrieveFromVectorDB(query, documents, doc_embeddings, index, embedder):
    # STEP 4: Encode the user query
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # STEP 5: Retrieve top-k documents
    top_k = 5
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    retrieved_doc_embeddings = [doc_embeddings[i] for i in indices[0]]

    # STEP 6: Rerank using cosine similarity
    cos_scores = [
        util.cos_sim(query_embedding[0], doc_emb)[0]
        for doc_emb in retrieved_doc_embeddings
    ]
    reranked = sorted(zip(retrieved_docs, cos_scores), key=lambda x: x[1], reverse=True)
    top_reranked_docs = [doc for doc, score in reranked[:3]]

    # STEP 7: Synthesize context
    synthesized_context = "\n".join(top_reranked_docs)
    print("Synthesized Context:\n", synthesized_context)

    return synthesized_context


def generateAnswer(prompt):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    main()
