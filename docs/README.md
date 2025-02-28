# Semantic Search Engine for Video Segments

## Overview
This project is a **semantic search engine** designed to retrieve **relevant video excerpts** based on **natural language queries**. The system indexes video transcripts and applies **various text retrieval algorithms** to find and rank the most relevant video segments based on user queries.

## Features
- **Speech-to-Text Conversion**: Transcribes videos into text using OpenAI Whisper.
- **Text Chunking by Semantics**: Segments transcriptions into meaningful chunks using embeddings.
- **Multiple Search Algorithms**:
  - **TF-IDF** (Term Frequency - Inverse Document Frequency)
  - **BM25** (Best Matching 25)
  - **n-grams** (linguistic-based similarity)
  - **Sentence Embeddings** (BERT-based models for semantic similarity)
  - **Cross-Encoders** (Deep learning-based refinement)
  - **OpenAI Embeddings** (External API for high-quality semantic search)
- **Ranking and Evaluation**: Compares search effectiveness using MRR (Mean Reciprocal Rank).

## Project Structure
```markdown
📂 SEMANTIC-VIDEO-SEARCH/
│── 📂 data/                # Input data and pre-processed databases
│   ├── 📂 audio/           # Audio files extracted from videos
│   ├── 📂 database/        # Pickle files storing precomputed indices
│   ├── 📂 testing/         # Test datasets for models evaluation
│   ├── 📂 transcriptions/  # Transcribed text data from videos
│── 📂 docs/                # Documentation and reports
│   ├── README.md           # Project overview and usage
|   ├── Semantic_Video_Search.pdf   # Report for technical details
│── 📂 src/                 # Source code for the project
│   ├── chunk_add_dico.py   # Adds chunks to the database dictionary
│   ├── chunk_database.py   # Handles segmentation of transcribed text
│   ├── create_testing_material.py  # Generates data for evaluation
│   ├── index_database.py   # Prepares and indexes data for search
│   ├── main.py             # Core search functionality
│   ├── test_and_compare.py # Evaluates methods
│   ├── transcribe.py       # Converts video/audio to text 
│   ├── update_database.py  # Updates the indexed database
│── 📂 tests/               # Test data
│── 📄 .gitignore           # Ignore unnecessary files
│── 📄 requirements.txt     # List of dependencies
```


## How It Works
### 1. **Transcription**
- Converts video to audio (`transcribe.py`)
- Uses OpenAI Whisper for high-accuracy transcription
- Generates timestamps for synchronization

### 2. **Chunking & Indexing**
- Splits transcriptions into meaningful segments (`chunk_database.py`)
- Creates embeddings with Sentence-BERT or OpenAI API (`index_database.py`)
- Prepares data for lexical searches (TF-IDF, BM25, n-grams)

### 3. **Semantic Search**
- Processes natural language queries (`main.py`)
- Retrieves and ranks video segments based on multiple algorithms
- Uses embeddings and similarity metrics to refine results

### 4. **Evaluation & Comparison**
- Tests different search methods (`test_and_compare.py`)
- Uses a labeled dataset to measure retrieval accuracy
- Calculates Mean Reciprocal Rank (MRR) for performance evaluation

## Setup & Installation
### Prerequisites
- Python 3.8+
- Virtual environment recommended
- Dependencies listed in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/AxelV-hub/semantic-video-search
cd semantic-video-search

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
Running the Project
```

### Running the Project
```bash

# Transcribe and index videos
python src/update_database.py

# Run a search query
python src/main.py "What is the impact of AI on cybersecurity?"

# Compare search algorithms
python src/test_and_compare.py
```

## Results & Performance
The system has been tested on a dataset of 10+ hours of video, with different search models evaluated. OpenAI embeddings provided the best performance in terms of accuracy, while BM25 and TF-IDF were much faster but less precise.

## Future Improvements
- **Vector Database Integration**: Reduce retrieval latency by using specialized databases (e.g., FAISS, Pinecone).
- **Multi-Modal Search**: Incorporate video frame analysis for richer search results.
- **Personalization & Feedback Loop**: Enhance ranking by incorporating user feedback.