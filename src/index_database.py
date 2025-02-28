"""
index_database.py

This module preprocesses the transcription database to generate indices 
for different search and retrieval models, including cosine similarity, BM25, TFIDF, and n-gram.

Main Features:
- Embeds paragraphs using pre-trained sentence transformers and OpenAI embeddings.
- Prepares BM25, TFIDF, and n-gram indices for lexical search.
- Supports incremental updates of specific indexing models.
- Saves the processed data for efficient retrieval.

Usage:
- This script can be executed independently to index the database.
- It can also be imported as a module for integration into a search system.

Example:
```bash
python index_database.py -v -m BM25
```

Dependencies:
- `sentence-transformers` for semantic search embeddings.
- `openai` for alternative embedding-based indexing.
- `nltk`, `numpy`, `pickle` for text processing and storage.
- `sklearn` for similarity computations.
- `bm25` for lexical retrieval.
"""
import argparse

# Basic libraries
import os 
import sys
import json
import pickle
import numpy as np
import time
import random as rd
from tqdm import tqdm
import datetime

# Libraries for speech transcription
import speech_recognition as sr 
from pydub import AudioSegment #for the split of audio
from pydub.silence import split_on_silence #for split by silences
from moviepy.video.io.VideoFileClip import VideoFileClip
import shutil

# Libraries for NLP
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

# Libraries for lexical search
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

# Libraries for handling text
from nltk.metrics.distance import edit_distance
import nltk
from nltk.tokenize import word_tokenize

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries for n-gram
import nltk
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import string
import datetime


# Preprocessing the paragraphs for cosine similarity (embedding the paragraphs into vectors)
def data_preprocessing_cosine_semantic(chunks_dico, trained_model_cosine_semantic, main_verbose = False):
    """
    Converts paragraphs into embeddings for cosine similarity search.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        trained_model_cosine_semantic (SentenceTransformer): Pre-trained model for encoding text.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        list: List of tuples containing (embedding, chunk ID).
    """
    print("Embedding paragraphs...") if main_verbose else None
    # Embedding the paragraphs into vectors
    data_semantic_cosine = []
    for ID, paragraph in tqdm(chunks_dico.items(), desc="Embedding paragraphs"):
        embedding = trained_model_cosine_semantic.encode(paragraph)
        data_semantic_cosine.append((embedding, ID))
    print("Embedding done.") if main_verbose else None
    return data_semantic_cosine


def data_preprocessing_cosine_semantic_vOpenAI(chunks_dico, main_verbose = False):
    """
    Converts paragraphs into embeddings using OpenAI's API.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        list: List of tuples containing (embedding, chunk ID).
    """
    print("Embedding paragraphs with OpenAI...") if main_verbose else None
    client = OpenAI()
    data_semantic_cosine = []
    for ID,paragraph in tqdm(chunks_dico.items(), desc="Embedding paragraphs with OpenAI"):
        try:
            embedding = client.embeddings.create(input=[paragraph.replace("\n"," ")], model="text-embedding-3-large").data[0].embedding
            data_semantic_cosine.append((embedding,ID))
        except:
            print(f"Error with paragraph {ID}.") 
            print(f"Paragraph: {paragraph}")
            embedding = client.embeddings.create(input=["Vide"], model="text-embedding-3-large").data[0].embedding
            data_semantic_cosine.append((embedding,ID))
    print("Embedding with OpenAI done.") if main_verbose else None
    return data_semantic_cosine


# Preprocessing the paragraphs for BM25
def data_preprocessing_BM25(chunks_dico, main_verbose = False):
    """
    Prepares the database for BM25 lexical search.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        tuple: (BM25 index, list of chunk IDs).
    """
    print("Preprocessing paragraphs for BM25...") if main_verbose else None
    
    # We need to use lists for the BM25Okapi function, so we convert the dictionnaries to lists and keep both in the data_BM25 tuple
    paragraphs_list = list(chunks_dico.values())
    IDs_list = list(chunks_dico.keys())

    # We define the tokenizer for BM25
    def BM25_tokenizer(text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            tokenized_doc.append(token)
        return tokenized_doc

    # We use the tokenizer on the paragraphs
    chunks_token = []
    for paragraph in paragraphs_list:
        chunks_token.append(BM25_tokenizer(paragraph))

    # We create the BM25 model
    bm25 = BM25Okapi(chunks_token)
    data_BM25 = (bm25,IDs_list)

    print("Preprocessing for BM25 done.") if main_verbose else None
    return data_BM25

def data_preprocessing_TFIDF(chunks_dico, main_verbose = False):
    """
    Prepares the database for TFIDF-based retrieval.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        tuple: (TFIDF vectorizer, TFIDF matrix, list of chunk IDs).
    """
    vectorizer = TfidfVectorizer()

    paragraphs_list = list(chunks_dico.values())
    IDs_list = list(chunks_dico.keys())

    tfidf_matrix = vectorizer.fit_transform(paragraphs_list)

    return (vectorizer, tfidf_matrix, IDs_list)



def data_preprocessing_ngram(chunks_dico, main_verbose = False):
    """
    Prepares n-gram indexing for text retrieval.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        dict: n-gram index.
    """
    print("Preprocessing paragraphs for n-gram...") if main_verbose else None

    nltk.download('stopwords')
    nltk.download('punkt')

    def preprocess(text):
        stop_words = set(stopwords.words('french'))
        tokens = word_tokenize(text.lower(), language = 'french')
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        return tokens

    index = defaultdict(Counter)
    for ID, paragraph in tqdm(chunks_dico.items(), desc="Preprocessing paragraphs for n-gram"):
        tokens = preprocess(paragraph)
        for n in range(1, 6):
            for gram in ngrams(tokens, n):
                index[gram][ID] += 1
        
    print("Preprocessing for n-gram done.") if main_verbose else None
    return index



def index_database_for_research(chunks_dico, trained_model_cosine_semantic, main_verbose = False):
    """
    Indexes the database for retrieval models: Cosine Similarity, BM25, TFIDF, and n-gram.

    Args:
        chunks_dico (dict): Dictionary containing text chunks.
        trained_model_cosine_semantic (SentenceTransformer): Pre-trained model for encoding text.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
        tuple: Indexed data for different retrieval methods.
    """
    print("Indexing database...") if main_verbose else None

    elapsed_time = {}
    # Transformation de chaque paragraphe en embedding représentant sa sémantique en utilisant les embeddings d'OpenAI
    print(f"Embedding all texts for cosine similarity using OpenAI...") if main_verbose else None
    start_time = time.time()
    data_semantic_cosine_vOpenAI = data_preprocessing_cosine_semantic_vOpenAI(chunks_dico, main_verbose = main_verbose)
    end_time = time.time()
    elapsed_time["Embedding with OpenAI"] = end_time - start_time
    print(f"All texts successfully embedded using OpenAI") if main_verbose else None

    # Transformation de chaque paragraphe en embedding représentant sa sémantique
    print(f"Embedding all texts for cosine similarity...") if main_verbose else None
    start_time = time.time()
    data_semantic_cosine = data_preprocessing_cosine_semantic(chunks_dico, trained_model_cosine_semantic, main_verbose = main_verbose)
    end_time = time.time()
    elapsed_time["Embedding"] = end_time - start_time
    print(f"All texts successfully embedded") if main_verbose else None

    # Indexing of the paragraphs with the BM25 model (is done all at once because the index depend on the whole database)
    print(f"Indexing for BM25 model...") if main_verbose else None
    start_time = time.time()
    data_BM25 = data_preprocessing_BM25(chunks_dico, main_verbose = main_verbose)
    end_time = time.time()
    elapsed_time["BM25"] = end_time - start_time
    print(f"Database successfully indexed") if main_verbose else None

    # Indexing of the paragraphs with the TFIDF model
    print(f"Indexing for TFIDF model...") if main_verbose else None
    start_time = time.time()
    data_TFIDF = data_preprocessing_TFIDF(chunks_dico, main_verbose = main_verbose)
    end_time = time.time()
    elapsed_time["TFIDF"] = end_time - start_time
    print(f"Database successfully indexed") if main_verbose else None

    # Indexing of the paragraphs with the n-gram model
    print(f"Indexing for n-gram model...") if main_verbose else None
    start_time = time.time()
    data_ngram = data_preprocessing_ngram(chunks_dico, main_verbose = main_verbose)
    end_time = time.time()
    elapsed_time["n-gram"] = end_time - start_time
    print(f"Database successfully indexed") if main_verbose else None

    print("Database updated.") if main_verbose else None

    # Save the running time in a text file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"_output_savings/{current_time}_Indexing_running_time.txt", "w") as file:
        file.write("Runnung time for embedding using OpenAI: "+str(elapsed_time["Embedding with OpenAI"])+" seconds \n")
        file.write("Running time for embedding: "+str(elapsed_time["Embedding"])+" seconds \n")
        file.write("Running time for BM25 indexing: "+str(elapsed_time["BM25"])+" seconds \n")
        file.write("Running time for TFIDF indexing: "+str(elapsed_time["TFIDF"])+" seconds \n")
        file.write("Running time for n-gram indexing: "+str(elapsed_time["n-gram"])+" seconds \n")

    return data_semantic_cosine, data_semantic_cosine_vOpenAI, data_BM25, data_TFIDF, data_ngram


if __name__ == "__main__":
    """
    Main execution script for indexing the database.

    This script indexes the transcriptions database to make it searchable using different 
    models, including:
    - Cosine similarity (with a SentenceTransformer model)
    - OpenAI embeddings for semantic search
    - BM25 for lexical search
    - TFIDF for keyword-based search
    - n-gram indexing for phrase-based retrieval

    Command-line Arguments:
        -v, --verbose : Enables verbose mode for detailed logging.
        -m, --model   : Specifies a specific indexing model to update. 
                        If omitted, all models are updated.
                        Options: 'cosine_semantic', 'cosine_semantic_vOpenAI', 'BM25', 'TFIDF', 'ngram'.

    """
    # Here is the script to run when you want to index the databases so they are ready for the research
    parser = argparse.ArgumentParser(description='Update the complete database (add new videos)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('-m', '--model', help="choose if only one indexing method should be updated")
    args = parser.parse_args()
    main_verbose = args.verbose
    indexing_model_to_update = args.model

    # Clean the indexes in the database (data_BM25, data_TFIDF, data_semantic_cosine, data_semantic_cosine_vOpenAI, data_ngram) before updating them
    if indexing_model_to_update == None:
        if os.path.exists("data/database/data_semantic_cosine.pickle"):
            os.remove("data/database/data_semantic_cosine.pickle")
        if os.path.exists("data/database/data_semantic_cosine_vOpenAI.pickle"):
            os.remove("data/database/data_semantic_cosine_vOpenAI.pickle")
        if os.path.exists("data/database/data_BM25.pickle"):
            os.remove("data/database/data_BM25.pickle")
        if os.path.exists("data/database/data_TFIDF.pickle"):
            os.remove("data/database/data_TFIDF.pickle")
        if os.path.exists("data/database/data_ngram.pickle"):
            os.remove("data/database/data_ngram.pickle")
    else:
        if os.path.exists(f"data/database/data_{indexing_model_to_update}.pickle"):
            os.remove(f"data/database/data_{indexing_model_to_update}.pickle")


    # Try to load the embeddings list and the dictionnaries from the database if they exist, otherwise print a message to inform the user that the database is empty
    print("Loading the database...") if main_verbose else None
    if not os.path.exists("Database"):
        os.makedirs("Database")
    try:
        with open('data/database/chunks_dico.pickle', 'rb') as handle:
            chunks_dico = pickle.load(handle)
        print("Database loaded.") if main_verbose else None
    except:
        print("The database is empty. Please add videos to the database before rechunking.") if main_verbose else None
        sys.exit()

    # We only recompute the chunks based on the already in the database transcriptions
    print("Indexing the database...") if main_verbose else None
    if indexing_model_to_update == None:
        trained_model_cosine_semantic = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        data_semantic_cosine, data_semantic_cosine_vOpenAI, data_BM25, data_TFIDF, data_ngram = index_database_for_research(chunks_dico, trained_model_cosine_semantic,  main_verbose = main_verbose)
        
        # Save the videos_dico, chunks_dico, time_stamp_dico, data_semantic_cosine and data_BM25 in the database (piclke files)
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_semantic_cosine.pickle', 'wb') as handle:
            pickle.dump(data_semantic_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/database/data_semantic_cosine_vOpenAI.pickle', 'wb') as handle:
            pickle.dump(data_semantic_cosine_vOpenAI, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/database/data_BM25.pickle', 'wb') as handle:
            pickle.dump(data_BM25, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/database/data_TFIDF.pickle', 'wb') as handle:
            pickle.dump(data_TFIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data/database/data_ngram.pickle', 'wb') as handle:
            pickle.dump(data_ngram, handle, protocol=pickle.HIGHEST_PROTOCOL)



    elif indexing_model_to_update == "cosine_semantic_vOpenAI":
        # Transformation de chaque paragraphe en embedding représentant sa sémantique en utilisant les embeddings d'OpenAI
        print(f"Embedding all texts for cosine similarity using OpenAI...") if main_verbose else None
        data_semantic_cosine_vOpenAI = data_preprocessing_cosine_semantic_vOpenAI(chunks_dico, main_verbose = main_verbose)
        print(f"All texts successfully embedded using OpenAI") if main_verbose else None

        # Saving only the updated database
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_semantic_cosine_vOpenAI.pickle', 'wb') as handle:
            pickle.dump(data_semantic_cosine_vOpenAI, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif indexing_model_to_update == "cosine_semantic":
        # Transformation de chaque paragraphe en embedding représentant sa sémantique
        print(f"Embedding all texts for cosine similarity...") if main_verbose else None
        data_semantic_cosine = data_preprocessing_cosine_semantic(chunks_dico, trained_model_cosine_semantic, main_verbose = main_verbose)
        print(f"All texts successfully embedded") if main_verbose else None

        # Saving only the updated database
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_semantic_cosine.pickle', 'wb') as handle:
            pickle.dump(data_semantic_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif indexing_model_to_update == "BM25":
        # Indexing of the paragraphs with the BM25 model (is done all at once because the index depend on the whole database)
        print(f"Indexing for BM25 model...") if main_verbose else None
        data_BM25 = data_preprocessing_BM25(chunks_dico, main_verbose = main_verbose)
        print(f"Database successfully indexed") if main_verbose else None

        # Saving only the updated database
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_BM25.pickle', 'wb') as handle:
            pickle.dump(data_BM25, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif indexing_model_to_update == "TFIDF":
        # Indexing of the paragraphs with the TFIDF model
        print(f"Indexing for TFIDF model...") if main_verbose else None
        data_TFIDF = data_preprocessing_TFIDF(chunks_dico, main_verbose = main_verbose)
        print(f"Database successfully indexed") if main_verbose else None

        # Saving only the updated database
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_TFIDF.pickle', 'wb') as handle:
            pickle.dump(data_TFIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif indexing_model_to_update == "ngram":
        # Indexing of the paragraphs with the n-gram model
        print(f"Indexing for n-gram model...") if main_verbose else None
        data_ngram = data_preprocessing_ngram(chunks_dico, main_verbose = main_verbose)
        print(f"Database successfully indexed") if main_verbose else None

        # Saving only the updated database
        print("Saving the database...") if main_verbose else None
        with open('data/database/data_ngram.pickle', 'wb') as handle:
            pickle.dump(data_ngram, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print(f"Model {indexing_model_to_update} not recognized. Please choose between 'cosine_semantic', 'cosine_semantic_vOpenAI', 'BM25' and 'TFIDF'")
        sys.exit()


