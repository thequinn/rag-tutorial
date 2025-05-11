# RAG-Tutorial

Fine tuning Local LLMs with RAG using Ollama and Langchain

[Tutorial Source](https://itsfoss.com/local-llm-rag-ollama-langchain/?ref=dailydev)

## Contents

- Part 1: Environment setup
- Part 2: Project workflow
- Part 3: Fine-tuning the LLM for better responses

## Environment Setup

### 1. Setup virtual environment

In terminal, cd into your project root dir:

```bash
python3 -m venv venv1
source venv1/bin/activate
pip install -r requirements.txt
```

### 2. Setup LLM

In terminal, in the same project root folder:

Install Ollama in MacOS:

```bash
curl https://ollama.ai/install.sh | sh
```

After installation, you need to pull a model. For this RAG project, I recommend using one of these models:

```bash
# Option 1: Mistral (good balance of performance and size)
ollama pull mistral

# Option 2: Llama2 (if you want a more powerful model)
ollama pull llama2

# Option 3: Mixtral (if you want the best performance)
ollama pull mixtral
```

Verify the installation:

```bash
ollama list
```

Set the model in your environment:
Create a `.env` file in your project root with:

```
LLM_MODEL=mistral # or llama2 or mixtral, depending on which one you pulled
OLLAMA_HOST=http://localhost:11434
```

## Running the app

### 1. Starting the App

```bash
source venv1/bin/activate
python3 app.py
ollama list
```

Terminal 1: `ollama serve` (running Ollama server)
Terminal 2: `python3 app.py` (Your Flask application)

### 2. Testing the endpoints

For the embed endpoint, you need to send a PDF file:

```bash
curl -X POST http://127.0.0.1:8080/embed \
 -F "file=@/path/test.pdf"
```

For the query endpoint, use curl or Postman to send a POST request:

```bash
curl -X POST http://127.0.0.1:8080/query \
 -H "Content-Type: application/json" \
 -d '{"query": "What is this document about?"}'
```

### Port Information

**Port 8080 (http://127.0.0.1:8080):**

- This is your Flask web application server
- It's where your RAG application runs
- You can access the API endpoints here:
  - POST http://127.0.0.1:8080/embed for embedding documents
  - POST http://127.0.0.1:8080/query for querying

**Port 11434 (http://localhost:11434):**

- This is the Ollama server port
- It's specified in your .env file as OLLAMA_HOST=http://localhost:11434
- This is where the LLM (Language Model) runs
- Your Flask application communicates with Ollama on this port internally

**The workflow is:**

1. Your browser makes requests to your Flask app on port 8080
2. Your Flask app then communicates with Ollama on port 11434 to get LLM responses
3. The results are sent back to your browser

You shouldn't be accessing port 11434 directly in your browser - that's Ollama's internal API. Instead, you should:

- Make sure Ollama is running (it will be on port 11434)
- Access your RAG application through port 8080
- Use the API endpoints on port 8080 to interact with your application

## Project Workflow - High Level

This is a RAG (Retrieval-Augmented Generation) system that uses Ollama and Langchain to create a document Q&A system.

The system follows a typical RAG architecture:

1. Document ingestion → 2. Chunking → 3. Embedding → 4. Storage → 5. Query processing → 6. Response generation

This implementation is particularly interesting because it:

- Uses a local LLM (Ollama) instead of cloud-based solutions
- Implements multi-query retrieval for better context matching
- Has a clean API interface for both document ingestion and querying
- Includes proper error handling and file management

## Project workflow - Low Level

### 1. System Setup:

- The project uses Flask as a web server
- It requires Ollama running locally (default port 11434)
- Uses environment variables for configuration (loaded via dotenv)

### 2. Document Processing Pipeline:

- Document Upload (`/embed` endpoint):
  - Accepts PDF files through a POST request
  - Files are temporarily saved in a `_temp` directory
  - Documents are processed using UnstructuredPDFLoader
  - Text is split into chunks (7500 characters with 100 character overlap)
  - Chunks are embedded and stored in a vector database
  - Temporary files are cleaned up after processing

### 3. Query Processing Pipeline (`/query` endpoint):

- Takes a question as input through a POST request
- Uses a sophisticated retrieval system with:
  - MultiQueryRetriever that generates 5 variations of the user's question
  - Vector database search to find relevant document chunks
  - Ollama LLM for generating responses
- The system uses a specific prompt template that:
  - First generates question variations for better retrieval
  - Then uses the retrieved context to answer the original question

### 4. Key Components:

- `app.py`: Main Flask application with API endpoints
- `embed.py`: Handles document processing and embedding
- `query.py`: Manages the query processing and response generation
- `get_vector_db.py`: Manages the vector database connection

### 5. Dependencies:

- Langchain for the RAG pipeline
- Ollama for the LLM
- Flask for the web server
- Various document processing libraries

## Fine-tuning the LLM for better responses

If Ollama's responses aren't detailed enough, we need to refine how we provide context.

### Tuning strategies:

- **Improve Chunking** – Ensure text chunks are large enough to retain meaning but small enough for effective retrieval.
- **Enhance Retrieval** – Increase `n_results` to fetch more relevant document chunks.
- **Modify the LLM Prompt** – Add structured instructions for better responses.

This ensures that Ollama:

- Uses retrieved text properly
- Avoids hallucinations by sticking to available context
- Provides meaningful, structured answers
