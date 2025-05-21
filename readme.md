# MAKAN AI - RAG Knowledge Base Chatbot

MAKAN AI is a Retrieval-Augmented Generation (RAG) chatbot designed to provide food reccommendations using a curated knowledge base of food review articles. It makes use of embedding generation, and a local vector database on top of a LLM to provide accurate, context-aware responses.

Note: To run project, you need to have Python 3.8+ installed, and also provide your own dataset of food reviews. The project is designed to be modular, allowing for easy integration of new data sources and models.

## 🏗️ Processing Overview

### Data Collection and Processing Pipeline

1. **Data Collection**: Gather unstructured text reviews and store them in the `raw/` and `dataset/` directories.
2. **Data Cleaning and Enrichment**: Raw data is processed and cleaned for consistency, outputting to `cleaned/`. Additionally, key structured information is extracted from text if available and stored along with text chunks. Only quality data with the key structured information available is kept for the next steps.
3. **Embedding Generation**: Cleaned data is converted into vector embeddings using `gen_embeddings.py` and persisted in the subdirectory (`chroma_db/`).
4. **RAG Chatbot**: The `qa.py` script loads the embeddings and answers user queries by retrieving relevant context and generating responses. Additional processing is done on user's query to extract additional metadata filtering for the retrieval stage to provide more relevant recommendations.

## 📦 Project Structure

```
makan-ai/
├── scraper.py                # Scraper for review data
├── gen_embeddings.py         # Script for embedding generation
├── process_data.py           # Data cleaning and processing
├── qa.py                     # Main RAG chatbot script
├── app.py                    # Streamlit Web UI for the chatbot
├── requirements.txt          # Python dependencies
├── Makefile                  # Automation commands
├── chroma_db/                # Local vector database (ChromaDB)
├── cleaned/                  # Cleaned data outputs
├── raw/                      # Raw collected data
├── core/                     # Core modules (e.g., retrieval logic)
└── readme.md                 # Project documentation
```

# 🚀 Getting Started

## Environment Variable Setup

Before running the scripts, set the following environment variables in a `.env` file in the project root. Below are brief descriptions for each variable:

```env
HF_TOKEN                  # HuggingFace Inference API token for model access
START_URL                 # Starting URL for the scraper (e.g., main listing page)
BASE_URL                  # Base URL for the scraper (used to resolve relative links)
ARTICLE_LINK_SELECTOR     # CSS selector for article links on the listing page
NEXT_PAGE_SELECTOR        # CSS selector for the 'next page' button on the listing page
NAME_SELECTOR             # CSS selector for the restaurant or article name on the detail page
LOCATION_SELECTOR         # CSS selector for the location field on the detail page
CUISINE_SELECTOR          # CSS selector for the cuisine type on the detail page
ARTICLE_PARAGRAPH_SELECTOR# CSS selector for article text paragraphs on the detail page
RAW_OUTPUT_PATH           # Output path for raw scraped data (e.g., raw/data.json)
RAW_INPUT_PATH            # Input path for raw data to be cleaned (e.g., raw/data.json)
CLEANED_OUTPUT_PATH       # Output path for cleaned data (e.g., cleaned/cleaned_data.json)
CLEANED_DATA_PATH         # Input path for cleaned data used in embedding generation (e.g., cleaned/cleaned_data.json)
CHROMA_DB_PATH            # Directory path for the persisted Chroma vector database files (e.g., ./chroma_db)
```

## ⚙️ Makefile Commands

### First time setup

- `make venv` : Create and activate a Python virtual environment
- `make install` : Install all dependencies from `requirements.txt`

### Run Project

- `make scrape` : Run all data scrapers to update datasets
- `make process` : Clean and process raw data
- `make embed` : Generate vector embeddings from cleaned data
- `make run` : Start the RAG chatbot for question answering
- `make app` : Start the Streamlit Web UI for interactive chat (see below)

### Misc

- `make clean` : Remove python environment and pycaches

## ✅ How to Use

- Ensure data is loaded and vector store is populated
- To use the web UI, run:

  ```sh
  make app
  ```

  Then open your browser at [http://localhost:8501](http://localhost:8501).

- To ask questions via CLI, run:
  ```sh
  make run
  ```
  and follow the prompts.

---

## 🌐 Web UI

A web-based user interface is now available for interactive chat with the RAG chatbot. This UI allows you to:

- Ask questions and receive recommendations in a chat format
- View retrieved context and sources
- Enjoy a more user-friendly experience compared to the CLI

To start the UI:

1. Ensure all dependencies are installed and data is processed/embedded.
2. Run `make ui` (or the appropriate command for your setup).
3. Open your browser at [http://localhost:8000](http://localhost:8000) (or the configured port).

For more details, see comments in each script and the UI folder README if available.
