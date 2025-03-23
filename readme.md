# RagAndVector

This project demonstrates the use of **Retrieval-Augmented Generation (RAG)** with **FAISS** for vector similarity search and **OpenAI GPT-4** for generating responses. It combines document embeddings, vector search, and language model capabilities to answer complex questions.

## Features

- **Document Embedding**: Converts text documents into vector representations using `SentenceTransformers`.
- **Vector Search**: Uses FAISS for efficient similarity search and clustering of dense vectors.
- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with GPT-4 to generate context-aware answers.
- **Environment Configuration**: Uses `.env` files to securely manage API keys.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (get it from [OpenAI API Keys](https://platform.openai.com/account/api-keys))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RagAndVector.git
   cd RagAndVector