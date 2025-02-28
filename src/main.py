"""
main.py

This module serves as the entry point for querying the database and retrieving the most relevant paragraphs 
based on different recommendation methods, including semantic similarity, lexical search, and cross-encoder ranking.

Main Features:
- Supports multiple recommendation methods such as cosine similarity, BM25, TFIDF, and n-gram.
- Uses pre-trained models to encode text and compute similarity scores.
- Returns structured recommendations in JSON format.
- Can be executed as a standalone script for interactive searches.

Usage:
- This script can be run independently to perform a query on the database.
- It can also be integrated into a larger system for automated recommendations.

Example:
python main.py "What are the benefits of AI in healthcare?"

Dependencies:
- `sentence-transformers` for semantic similarity.
- `openai` for alternative embedding-based recommendations.
- `bm25`, `tfidf`, `n-gram` for lexical search methods.
- `numpy`, `json`, `pickle` for data processing and storage.
- `argparse` for command-line argument parsing.
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

# Libraries for printing results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Libraries for n-gram
import nltk
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import string



def reco_cosine_semantic(input_prompt, data_semantic_cosine, trained_model_cosine_semantic, n_recommendations_cosine, verbose_cosine = False, **kwargs):
    """
    Retrieves relevant text chunks using cosine similarity on sentence embeddings.

    Args:
        input_prompt (str): The user query or search input.
        data_semantic_cosine (list): List of tuples containing (embedding, chunk ID).
        trained_model_cosine_semantic (SentenceTransformer): Pre-trained model for encoding text.
        n_recommendations_cosine (int): Number of recommendations to return.
        verbose_cosine (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """ 
    print("Recommending paragraphs with cosine similarity on embeddings...") if verbose_cosine else None
    # Embedding the prompt
    prompt_embedding = trained_model_cosine_semantic.encode(input_prompt)

    # Calculating the cosine similarity between the prompt and the paragraphs
    similarities = []
    for paragraph_embedding,ID in data_semantic_cosine:
        similarity = np.dot(paragraph_embedding,prompt_embedding)/(np.linalg.norm(paragraph_embedding)*np.linalg.norm(prompt_embedding))
        similarities.append((similarity,ID))

    # Sorting the paragraphs by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Returning the n most similar paragraphs
    recommendations_IDs = [similarities[i][1] for i in range(n_recommendations_cosine)]
    print("Recommendation with cosine similarity done.") if verbose_cosine else None
    return recommendations_IDs


def reco_cosine_semantic_vOpenAI(input_prompt, data_semantic_cosine_vOpenAI, n_recommendations_cosine_vOpenAI, verbose_cosine_vOpenAI = False, **kwargs):
    """
    Retrieves relevant text chunks using OpenAI's embeddings for cosine similarity.

    Args:
        input_prompt (str): The user query or search input.
        data_semantic_cosine_vOpenAI (list): List of tuples containing (embedding, chunk ID).
        n_recommendations_cosine_vOpenAI (int): Number of recommendations to return.
        verbose_cosine_vOpenAI (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with cosine similarity on embeddings...") if verbose_cosine_vOpenAI else None
    # Embedding the prompt
    client = OpenAI()
    prompt_embedding = client.embeddings.create(input = [input_prompt], model = "text-embedding-3-large").data[0].embedding

    # Calculating the cosine similarity between the prompt and the paragraphs
    similarities = []
    for paragraph_embedding,ID in data_semantic_cosine_vOpenAI:
        similarity = np.dot(paragraph_embedding,prompt_embedding)/(np.linalg.norm(paragraph_embedding)*np.linalg.norm(prompt_embedding))
        similarities.append((similarity,ID))

    # Sorting the paragraphs by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Returning the n most similar paragraphs
    recommendations_IDs = [similarities[i][1] for i in range(n_recommendations_cosine_vOpenAI)]
    print("Recommendation with cosine similarity done.") if verbose_cosine_vOpenAI else None
    return recommendations_IDs



def reco_cross_encoder(input_prompt, data_cross_encoder, trained_model_cross_encoder, n_recommendations_cross_encoder, verbose_cross_encoder = False, **kwargs):
    """
    Retrieves relevant text chunks using a cross-encoder model.

    Args:
        input_prompt (str): The user query or search input.
        data_cross_encoder (list): List of tuples containing (text chunk, chunk ID).
        trained_model_cross_encoder (CrossEncoder): Pre-trained cross-encoder model.
        n_recommendations_cross_encoder (int): Number of recommendations to return.
        verbose_cross_encoder (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with cross-encoder...") if verbose_cross_encoder else None
    # Calculating the similarity between the prompt and the paragraphs with the cross-encoder
    similarities = []
    for paragraph,ID in data_cross_encoder:
        similarity = trained_model_cross_encoder.predict((input_prompt,paragraph))
        similarities.append((similarity,ID))

    # Sorting the paragraphs by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Returning the n most similar paragraphs
    recommendations_IDs = [similarities[i][1] for i in range(n_recommendations_cross_encoder)]
    print("Recommendation with cross-encoder done.") if verbose_cross_encoder else None
    return recommendations_IDs



def reco_cosine_semantic_and_cross_encoder(input_prompt, data_semantic_cosine, trained_model_cosine_semantic, trained_model_cross_encoder, chunks_dico, n_recommendations_finales, verbose_finale=False, **kwargs):
    """
    Retrieves relevant text chunks using cosine similarity, then refines the ranking with a cross-encoder.

    Args:
        input_prompt (str): The user query or search input.
        data_semantic_cosine (list): List of tuples containing (embedding, chunk ID).
        trained_model_cosine_semantic (SentenceTransformer): Pre-trained model for encoding text.
        trained_model_cross_encoder (CrossEncoder): Pre-trained cross-encoder model.
        chunks_dico (dict): Dictionary storing text chunks.
        n_recommendations_finales (int): Number of final recommendations to return.
        verbose_finale (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with cosine similarity and refining with cross-encoder...") if verbose_finale else None

    # Starting with a ranking with cosine similarity, for all the n_reccomendations required
    recommendations_IDs_cosine_similarity = reco_cosine_semantic(input_prompt, data_semantic_cosine, trained_model_cosine_semantic, 100)

    # Then we rerank the top50 to get a more precised ranking with the cross-encoder
    data_cross_encoder = [(chunks_dico[ID],ID) for ID in recommendations_IDs_cosine_similarity[:50]]
    recommendations_IDs_cross_encoder = reco_cross_encoder(input_prompt, data_cross_encoder, trained_model_cross_encoder, 50)

    # We concatenate the reranked top50 and the remaining recommendations from the cosine similarity
    if n_recommendations_finales > 50:
        recommendations_IDs_cross_encoder.extend(recommendations_IDs_cosine_similarity[50:])

    print("Recommendation with cosine similarity and cross-encoder done.") if verbose_finale else None
    return recommendations_IDs_cross_encoder[:n_recommendations_finales]




# Implementation of BM25 search (lexical search), Work in progress, maybe will also implement other lexical search methods 
def reco_BM25(input_prompt, data_BM25, input_preprocessing_BM25, n_recommendations_BM25, verbose_BM25 = False, **kwargs):
    """
    Retrieves relevant text chunks using BM25 lexical search.

    Args:
        input_prompt (str): The user query or search input.
        data_BM25 (tuple): Tuple containing BM25 model and corresponding chunk IDs.
        input_preprocessing_BM25 (function): Preprocessing function for BM25.
        n_recommendations_BM25 (int): Number of recommendations to return.
        verbose_BM25 (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with BM25...") if verbose_BM25 else None

    # We extract the bm25 model and the IDs list from the data_BM25 tuple, and compute the scores for each chunk
    bm25,IDs_list = data_BM25
    BM25_scores = bm25.get_scores(input_preprocessing_BM25(input_prompt))
    
    # We extract the n_recommendations best scores' indexes and return the corresponding IDs, sorted from the best to the worst
    best_scores_indexes = np.argsort(BM25_scores)[-n_recommendations_BM25:][::-1]
    recommendations_IDs = [IDs_list[i] for i in best_scores_indexes]

    print("Recommendation with BM25 done.") if verbose_BM25 else None

    return recommendations_IDs

def reco_TFIDF(input_prompt, data_TFIDF, n_recommendations_TFIDF, verbose_TFIDF = False, **kwargs):
    """
    Retrieves relevant text chunks using TFIDF-based similarity.

    Args:
        input_prompt (str): The user query or search input.
        data_TFIDF (tuple): Tuple containing TFIDF vectorizer, matrix, and chunk IDs.
        n_recommendations_TFIDF (int): Number of recommendations to return.
        verbose_TFIDF (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with TFIDF...") if verbose_TFIDF else None

    # We extract the tfidf model and the IDs list from the data_TFIDF tuple, and compute the scores for each chunk
    vectorizer, tfidf, IDs_list = data_TFIDF
    query_tfidf = vectorizer.transform([input_prompt])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf).flatten()
    
    # We extract the n_recommendations best scores' indexes and return the corresponding IDs, sorted from the best to the worst
    best_scores_indexes = np.argsort(cosine_similarities)[-n_recommendations_TFIDF:][::-1]
    recommendations_IDs = [IDs_list[i] for i in best_scores_indexes]

    print("Recommendation with TFIDF done.") if verbose_TFIDF else None

    return recommendations_IDs

def reco_ngram(input_prompt, data_ngram, n_recommendations_ngram, chunks_dico, verbose_ngram = False, **kwargs):
    """
    Retrieves relevant text chunks using n-gram matching.

    Args:
        input_prompt (str): The user query or search input.
        data_ngram (dict): n-gram index mapping n-grams to chunk IDs.
        n_recommendations_ngram (int): Number of recommendations to return.
        chunks_dico (dict): Dictionary storing text chunks.
        verbose_ngram (bool, optional): If True, prints process details. Defaults to False.

    Returns:
        list: List of recommended chunk IDs.
    """
    print("Recommending paragraphs with ngram...") if verbose_ngram else None

    input_tokens = input_preprocessing_ngram(input_prompt)
    scores = {}
    for n in range(1, 6):
        input_ngrams = ngrams(input_tokens, n)
        for ngram in input_ngrams:
            for ID, count in data_ngram[ngram].items():
                scores[ID] = count**n + scores.get(ID, 0)

    all_scores = {}
    for ID in chunks_dico.keys():
        all_scores[ID] = scores.get(ID, 0)
    recommendations_IDs = [ID for ID, _ in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations_ngram]]

    print("Recommendation with ngram done.") if verbose_ngram else None
    return recommendations_IDs


def input_preprocessing_ngram(text):
    """
    Tokenizes input text for n-gram-based search.

    Args:
        text (str): Input text query.

    Returns:
        list: List of preprocessed tokens.
    """
    stop_words = set(stopwords.words('french'))
    tokens = word_tokenize(text.lower(), language = 'french')
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

# Input preprocessing for BM25, which is the tokenizer
def input_preprocessing_BM25(text):
    """
    Tokenizes input text for BM25-based search.

    Args:
        text (str): Input text query.

    Returns:
        list: List of tokenized words.
    """
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        tokenized_doc.append(token)
    return tokenized_doc


# Function to put the recommendations in a json format and send it to the application
def send_reco_to_application(recommendations_IDs, videos_dico, time_stamps_dico, chunks_dico):
    """
    Converts recommended chunk IDs into structured JSON format.

    Args:
        recommendations_IDs (list): List of recommended chunk IDs.
        videos_dico (dict): Dictionary mapping chunk IDs to video names.
        time_stamps_dico (dict): Dictionary mapping chunk IDs to timestamps.
        chunks_dico (dict): Dictionary mapping chunk IDs to transcriptions.

    Returns:
        str: JSON formatted recommendation output.
    """
    result = {}

    # Retrieve the transcription, the timestamp, and the video path from the ID
    for i,ID in enumerate(recommendations_IDs):
        result["video_path_"+str(i+1)] = videos_dico[ID]
        result["time_stamp_"+str(i+1)] = time_stamps_dico[ID]
        result["script_"+str(i+1)] = chunks_dico[ID]

    # Sending the recommendations to the application
    result_json = json.dumps(result,ensure_ascii=False,indent=4)
    return result_json






if __name__ == '__main__':
    """
    Main execution script for retrieving relevant transcriptions.

    This script:
    - Loads the preprocessed transcription database.
    - Accepts user input queries.
    - Uses cosine similarity-based methods for text chunk retrieval.
    - Outputs results in JSON format.

    Command-line Arguments:
        - input_prompt (str): User's search query.
        - -v, --verbose: Enables verbose mode for detailed logging.

    Example Usage:
        python main.py "How does AI impact finance?"

    Dependencies:
        - Requires preprocessed embeddings and chunk dictionaries.
        - Requires `data/database/videos_dico.pickle`, `data/database/chunks_dico.pickle`, and `data/database/time_stamp_dico.pickle`.

    Outputs:
        - JSON formatted recommendations are printed to standard output.
    """
    # Retrieve the input prompt, update_database_command, and verbose option from the command line
    parser = argparse.ArgumentParser(description='Make a research on the database and return the most relevant paragraphs.')
    parser.add_argument('input_prompt', type=str, help='The query of the user')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    input_prompt = args.input_prompt
    main_verbose = args.verbose

    print(f"Input prompt is: \"{input_prompt}\"") if main_verbose else None
    
    # Some global variables that will be used later
    trained_model_cosine_semantic = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    trained_model_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Try to load the embeddings list and the dictionnaries from the database if they exist, otherwise create them and force updating the database
    if not os.path.exists("Database"):
        os.makedirs("Database")
    try:
        with open('data/database/videos_dico.pickle', 'rb') as handle:
            videos_dico = pickle.load(handle)
        with open('data/database/chunks_dico.pickle', 'rb') as handle:
            chunks_dico = pickle.load(handle)
        with open('data/database/time_stamp_dico.pickle', 'rb') as handle:
            time_stamp_dico = pickle.load(handle)
        # with open('data/database/data_semantic_cosine.pickle', 'rb') as handle:
        #     data_semantic_cosine = pickle.load(handle)
        with open('data/database/data_semantic_cosine_vOpenAI.pickle', 'rb') as handle:
            data_semantic_cosine_vOpenAI = pickle.load(handle)
        # with open('data/database/data_BM25.pickle', 'rb') as handle:
        #     data_BM25 = pickle.load(handle)
        # with open('data/database/data_TFIDF.pickle', 'rb') as handle:
        #     data_TFIDF = pickle.load(handle)
        # with open('data/database/data_ngram.pickle', 'rb') as handle:
        #     data_ngram = pickle.load(handle)
        print("Database loaded.") if main_verbose else None
    except:
        print("The database in not complete, please update it.") if main_verbose else None
        sys.exit()


    # We retrieve the recommendations from the cosine similarity and cross-encoder
    recommendations_IDs = reco_cosine_semantic_vOpenAI(input_prompt, data_semantic_cosine_vOpenAI, 3, verbose_cosine_vOpenAI = main_verbose)
    result_json = send_reco_to_application(recommendations_IDs, videos_dico, time_stamp_dico, chunks_dico)
    print(result_json)


