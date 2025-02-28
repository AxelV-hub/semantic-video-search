"""
test_and_compare.py

This module evaluates and compares different ranking models for retrieving the most relevant paragraphs 
based on query inputs. It computes performance metrics such as Mean Reciprocal Rank (MRR) and running time.

Main Features:
- Evaluates multiple ranking models including TFIDF, BM25, n-gram, and semantic embeddings.
- Computes rankings and Mean Reciprocal Rank (MRR) for each model.
- Logs and visualizes performance metrics through boxplots and bar charts.
- Generates structured reports with evaluation statistics.

Usage:
- This script can be executed independently to compare different models.
- It can also be integrated into a larger system for continuous model evaluation.

Example:
python test_and_compare.py -v

Dependencies:
- `sentence-transformers` for semantic embeddings.
- `bm25`, `tfidf`, `n-gram` for lexical search models.
- `numpy`, `pandas`, `matplotlib`, `seaborn` for data processing and visualization.
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

from main import reco_cosine_semantic, reco_cosine_semantic_vOpenAI, reco_cross_encoder, reco_cosine_semantic_and_cross_encoder, reco_BM25, input_preprocessing_BM25, reco_TFIDF, input_preprocessing_ngram, reco_ngram
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

from codecarbon import EmissionsTracker



# def score_ranking(data_semantic_cosine, trained_model_cosine_semantic, trained_model_cross_encoder, chunks_dico, videos_dico, time_stamp_dico, main_verbose=False):
def score_ranking(reco_model, reco_model_parameters, chunks_dico, videos_dico, time_stamp_dico, verbose_score_ranking=False, n_recommendations_for_comparison=100):
    """
    Computes rankings for correct answers based on retrieved results from a given recommendation model.

    Args:
        reco_model (function): The recommendation function used for retrieving relevant paragraphs.
        reco_model_parameters (dict): Parameters required for the selected recommendation function.
        chunks_dico (dict): Dictionary storing text chunks.
        videos_dico (dict): Dictionary mapping chunk IDs to video names.
        time_stamp_dico (dict): Dictionary storing timestamps for each chunk.
        verbose_score_ranking (bool, optional): If True, prints process details. Defaults to False.
        n_recommendations_for_comparison (int, optional): Number of top recommendations to evaluate. Defaults to 100.

    Returns:
        np.array: Rankings for the correct answers in the retrieved results.
    """
    # We create a useful function to transform the timestamps fromat into a float to compare them
    def convert_timestamp(timestamps):
        """
        Converts timestamp format (mm:ss:cs) into a float representation in seconds.

        Args:
            timestamps (list): List of timestamp strings.

        Returns:
            list: List of timestamps converted to seconds.
        """
        total_seconds = []
        for timestamp in timestamps:
            time = timestamp.split(":")
            minutes = float(time[0])
            seconds = float(time[1])
            centiseconds = float(time[2])
            total_seconds.append(minutes * 60 + seconds + centiseconds / 100)
        return total_seconds

    # Open pickle file of testing_material.pkl in the data/testing folder
    with open('data/testing/testing_material.pkl', 'rb') as handle:
        correct_video_name, correct_timestamps, testing_queries = pickle.load(handle)

    # For each query, we retrieve the recommendations 
    rankings = np.full(len(testing_queries), n_recommendations_for_comparison, dtype=int)
    for i,query in enumerate(testing_queries):
        recommendations_IDs = reco_model(query, **reco_model_parameters)

        # We go through the list of IDs, take the video name and the timestamps, and compare them to the correct ones
        for j,ID in enumerate(recommendations_IDs):
            video_name = videos_dico[ID]
            timestamps = time_stamp_dico[ID]
            if video_name == correct_video_name[i] :
                # We verify if the timestamps overlap with the correct ones 
                correct_timestamps_float = convert_timestamp(correct_timestamps[i])
                timestamps_float = convert_timestamp(timestamps)
                if (timestamps_float[0] <= correct_timestamps_float[1] and timestamps_float[1] >= correct_timestamps_float[0]) or (timestamps_float[0] <= correct_timestamps_float[0] and timestamps_float[1] >= correct_timestamps_float[1]):
                    rankings[i] = (j + 1)
                    print(f"Ranking --> {rankings[i]} for test input \"{query}\"") if verbose_score_ranking else None
                    break
    print(f"The rankings are: {rankings}") if verbose_score_ranking else None
    
    return rankings


def compare_models(data_semantic_cosine, data_semantic_cosine_vOpenAI, data_BM25, input_preprocessing_BM25, trained_model_cosine_semantic, trained_model_cross_encoder, data_TFIDF, data_ngram, input_preprocessing_ngram, chunks_dico, videos_dico, time_stamp_dico, main_verbose=False, n_recommendations_for_comparison=100):
    """
    Compares the performance of different ranking models and logs evaluation metrics.

    Args:
        data_semantic_cosine (list): Semantic embeddings dataset.
        data_semantic_cosine_vOpenAI (list): OpenAI embeddings dataset.
        data_BM25 (tuple): BM25 index and corresponding IDs.
        input_preprocessing_BM25 (function): Preprocessing function for BM25.
        trained_model_cosine_semantic (SentenceTransformer): Pre-trained model for semantic embeddings.
        trained_model_cross_encoder (CrossEncoder): Pre-trained cross-encoder for re-ranking.
        data_TFIDF (tuple): TFIDF vectorizer and transformed matrix.
        data_ngram (dict): n-gram index.
        input_preprocessing_ngram (function): Preprocessing function for n-gram.
        chunks_dico (dict): Dictionary storing text chunks.
        videos_dico (dict): Dictionary mapping chunk IDs to video names.
        time_stamp_dico (dict): Dictionary storing timestamps for each chunk.
        main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.
        n_recommendations_for_comparison (int, optional): Number of top recommendations to evaluate. Defaults to 100.

    Outputs:
        - Saves evaluation results in structured reports.
        - Generates and saves visualizations of ranking distributions.
    """
    # Compute the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    reco_cosine_semantic_parameters = {
        "data_semantic_cosine": data_semantic_cosine,
        "trained_model_cosine_semantic": trained_model_cosine_semantic,
        "n_recommendations_cosine":n_recommendations_for_comparison,
        "verbose_cosine":False
    }
    reco_cosine_semantic_parameters_vOpenAI = {
        "data_semantic_cosine_vOpenAI": data_semantic_cosine_vOpenAI,
        "n_recommendations_cosine_vOpenAI":n_recommendations_for_comparison,
        "verbose_cosine_vOpenAI":False
    }
    reco_cosine_semantic_and_cross_encoder_parameters = {
        "data_semantic_cosine": data_semantic_cosine,
        "trained_model_cosine_semantic": trained_model_cosine_semantic,
        "trained_model_cross_encoder": trained_model_cross_encoder,
        "chunks_dico": chunks_dico,
        "n_recommendations_finales": n_recommendations_for_comparison,
        "verbose_finale":False
    }
    reco_BM25_parameters = {
        "data_BM25": data_BM25,
        "input_preprocessing_BM25": input_preprocessing_BM25,
        "n_recommendations_BM25":n_recommendations_for_comparison,
        "verbose_BM25":False
    }
    reco_TFIDF_parameters = {
        "data_TFIDF": data_TFIDF,
        "n_recommendations_TFIDF":n_recommendations_for_comparison,
        "verbose_TFIDF":False
    }
    reco_ngram_parameters = {
        "data_ngram": data_ngram,
        "input_preprocessing_ngram": input_preprocessing_ngram,
        "n_recommendations_ngram":n_recommendations_for_comparison,
        "chunks_dico": chunks_dico,
        "verbose_ngram":False
    }


    models = {
        'TFIDF': (reco_TFIDF, reco_TFIDF_parameters),
        'BM25': (reco_BM25, reco_BM25_parameters),
        'n-gram': (reco_ngram, reco_ngram_parameters),
        'Embedding': (reco_cosine_semantic, reco_cosine_semantic_parameters),
        'Cross-encoder': (reco_cosine_semantic_and_cross_encoder, reco_cosine_semantic_and_cross_encoder_parameters),
        'OpenAI': (reco_cosine_semantic_vOpenAI, reco_cosine_semantic_parameters_vOpenAI)
    }

    # take the number of testing queries from the testing_material.pkl file
    with open('data/testing/testing_material.pkl', 'rb') as handle:
        correct_video_name, correct_timestamps, testing_queries = pickle.load(handle)
    nb_testing_queries = len(testing_queries)

    # Initialize the dictionaries that will store the results
    MRR = {}
    avg_running_time = {}
    avg_emissions = {}
    df = pd.DataFrame()

    # Put the queries in the dataframe 
    df['Queries'] = testing_queries

    # For each model, compute the ranking of the correct answer for each recommendation method
    for model_name, (model, parameters) in models.items():
        print(f"Using {model_name} model...") if main_verbose else None
        # tracker = EmissionsTracker()
        start_time = time.time()
        # tracker.start()
        rankings = score_ranking(model, parameters, chunks_dico, videos_dico, time_stamp_dico, verbose_score_ranking=main_verbose, n_recommendations_for_comparison=n_recommendations_for_comparison)
        end_time = time.time()
        # emissions = tracker.stop()
        print(f"{model_name} model took {end_time - start_time} seconds") if main_verbose else None

        # Add rankings to dataframe
        df[model_name] = rankings

        # Compute MRR
        MRR[model_name] = np.mean(1 / rankings)

        # Compute average running time
        avg_running_time[model_name] = (end_time - start_time) / nb_testing_queries

        # Compute average emissions
        # avg_emissions[model_name] = emissions / nb_testing_queries

    # Save the rankings in a csv file
    df.to_csv(f'_output_savings/{current_datetime}_Rankings.csv', encoding='utf-8')

    # Print the average running time and emissions for each model
    for model_name in models.keys():
        print(f"Average running time for {model_name} model: {avg_running_time} seconds")
    # for model_name in models.keys():
        # print(f"Average emissions for {model_name} model: {avg_emissions} kgCO2")

    # Save the running time and emissions in a text file with the date and time
    with open(f"_output_savings/{current_datetime}_Running_time.txt", "w", encoding = "utf-8") as file:
        file.write("-" * 30 + "\n")
        file.write(f"Date and time: {current_datetime}\n")
        file.write("Average running time for each model:\n")
        for model_name in models.keys():
            file.write(f"{model_name}: {avg_running_time[model_name]} seconds\n")

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.yscale('log')
    plt.title('Ranking of the correct answer for each recommendation method', fontsize=16)
    plt.ylabel('Ranking of the correct answer (log)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'_output_savings/{current_datetime}_Ranking_plot.png')
    # plt.show()

    # Plot MRR bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(MRR.keys(), MRR.values())
    plt.title('Mean Reciprocal Rank for each recommendation method', fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'_output_savings/{current_datetime}_MRR_plot.png')
    # plt.show()



def info_database(chunks_dico, videos_dico, time_stamp_dico):
    """
    Computes and displays information about the database, including video and chunk statistics.

    This function provides insights such as:
    - Number of videos and chunks in the database.
    - Total and average duration of videos.
    - Minimum, maximum, and average chunk duration.
    - Distribution of chunk durations (saved as a histogram).
    - Saves this information in a text report.

    Args:
        chunks_dico (dict): Dictionary storing text chunks.
        videos_dico (dict): Dictionary mapping chunk IDs to video names.
        time_stamp_dico (dict): Dictionary storing timestamps for each chunk.

    Returns:
        int: The total number of chunks in the database.
    """
    # For each video in the database, print the number of chunks and the total duration
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Information about the database:")
    for video_name in set(videos_dico.values()):
        n_chunks = len([ID for ID in chunks_dico.keys() if videos_dico[ID] == video_name])
        timestamps = [time_stamp_dico[ID] for ID in chunks_dico.keys() if videos_dico[ID] == video_name]
        durations_seconds = []
        for timestamp in timestamps:
            start, finish = timestamp
            start_time_seconds = int(start.split(":")[0]) * 60 + int(start.split(":")[1]) + int(start.split(":")[2]) / 100
            finish_time_seconds = int(finish.split(":")[0]) * 60 + int(finish.split(":")[1]) + int(finish.split(":")[2]) / 100
            durations_seconds.append(finish_time_seconds - start_time_seconds)
        total_duration_seconds = sum(durations_seconds)
        total_duration_minutes = int(total_duration_seconds / 60)
        total_duration_seconds = int(total_duration_seconds % 60)
        min_duration = int(min(durations_seconds))
        max_duration = int(max(durations_seconds))
        mean_duration = int(np.mean(durations_seconds))
        print("-" * 30)
        print(f"Video '{video_name}':")
        print(f"Number of chunks: {n_chunks}")
        print(f"Total duration: {total_duration_minutes} minutes and {total_duration_seconds} seconds")
        print(f"Minimum duration of a chunk: {int(min_duration)} seconds")
        print(f"Maximum duration of a chunk: {int(max_duration)} seconds")
        print(f"Mean duration of a chunk: {int(mean_duration)} seconds")

    # Number of videos	
    n_videos = len(set(videos_dico.values()))
    print("-" * 30)
    print(f"Number of videos: {n_videos}")

    # Total duration of videos (in hours), based on the timestamps
    videos = set(videos_dico.values())
    total_duration = 0
    for video_name in videos:
        duration = 0
        for _,timestamp in [timestamp for ID,timestamp in time_stamp_dico.items() if videos_dico[ID] == video_name]:
            # Retrieve the finish time of the chunk
            finish_time = int(timestamp.split(":")[0]) / 60 + int(timestamp.split(":")[1]) /3600 + int(timestamp.split(":")[2]) / 360000
            if finish_time > duration:
                duration = finish_time
        total_duration += duration
    total_duration_hours = int(total_duration)
    total_duration_minutes = int((total_duration - total_duration_hours) * 60)
    print(f"Total duration of videos: {total_duration_hours} hours and {total_duration_minutes} minutes")

    # Number of chunks
    n_chunks = len(chunks_dico)
    print(f"Number of chunks: {int(n_chunks)}")

    # Average number of chunks per video
    avg_chunks_per_video = n_chunks / n_videos
    print(f"Average number of chunks per video: {round(avg_chunks_per_video, 2)}")

    # Average duration of a chunk (in seconds), and plot a histogram of the durations
    durations = []
    for timestamps in time_stamp_dico.values():
        start, finish = timestamps
        start_time_seconds = int(start.split(":")[0]) * 60 + int(start.split(":")[1]) + int(start.split(":")[2]) / 100
        finish_time_seconds = int(finish.split(":")[0]) * 60 + int(finish.split(":")[1]) + int(finish.split(":")[2]) / 100
        durations.append(finish_time_seconds - start_time_seconds)
    avg_duration = np.mean(durations)
    avg_duration_minutes = int(avg_duration / 60)
    avg_duration_seconds = int(avg_duration % 60)
    durations_minutes = [duration / 60 for duration in durations]
    print(f"Average duration of a chunk: {avg_duration_minutes} minutes and {avg_duration_seconds} seconds")

    # plot and save the histogram of the durations
    plt.figure(figsize=(10, 6))
    plt.hist(durations_minutes, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of the durations of the chunks', fontsize=16)
    plt.xlabel('Duration of (minutes)', fontsize=14)
    plt.ylabel('Number of chunks', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'_output_savings/{current_datetime}_Histogram_durations.png')
    # plt.show()

    # Save all the information in a text file with the date and time
    with open(f"_output_savings/{current_datetime}_Database_info.txt", "w", encoding = "utf-8") as file:
        file.write("-" * 30 + "\n")
        file.write(f"Date and time: {current_datetime}\n")
        file.write("Information about the database at this date:\n")
        file.write(f"Number of videos in the database: {n_videos}\n")
        file.write(f"Total duration of videos: {total_duration_hours} hours and {total_duration_minutes} minutes\n")
        file.write(f"Number of chunks: {int(n_chunks)}\n")
        file.write(f"Average number of chunks per video: {round(avg_chunks_per_video, 2)}\n")
        file.write(f"Average duration of each chunk: {avg_duration_minutes} minutes and {avg_duration_seconds} seconds\n")
    file.close()

    return n_chunks


if __name__ == '__main__':
    # Retrieve the input prompt, update_database_command, and verbose option from the command line
    parser = argparse.ArgumentParser(description='Compare the different ranking models available.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    main_verbose = args.verbose

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
        with open('data/database/data_semantic_cosine.pickle', 'rb') as handle:
            data_semantic_cosine = pickle.load(handle)
        with open('data/database/data_semantic_cosine_vOpenAI.pickle', 'rb') as handle:
            data_semantic_cosine_vOpenAI = pickle.load(handle)
        with open('data/database/data_BM25.pickle', 'rb') as handle:
            data_BM25 = pickle.load(handle)
        with open('data/database/data_TFIDF.pickle', 'rb') as handle:
            data_TFIDF = pickle.load(handle)
        with open('data/database/data_ngram.pickle', 'rb') as handle:
            data_ngram = pickle.load(handle)
        print("Database loaded.") if main_verbose else None
    except:
        print("The database in not complete, please update it.") if main_verbose else None
        sys.exit()


    n_chunks = info_database(chunks_dico, videos_dico, time_stamp_dico)

    compare_models(data_semantic_cosine, data_semantic_cosine_vOpenAI, data_BM25, input_preprocessing_BM25, trained_model_cosine_semantic, trained_model_cross_encoder, data_TFIDF, data_ngram, input_preprocessing_ngram, chunks_dico, videos_dico, time_stamp_dico, main_verbose=main_verbose, n_recommendations_for_comparison=n_chunks)

