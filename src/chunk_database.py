"""
chunk_database.py

This module is responsible for re-chunking the transcription database 
by processing existing transcriptions without re-performing speech-to-text conversion.

Main Features:
- Identifies unprocessed transcriptions and applies chunking.
- Supports optional forced re-chunking of all transcriptions.
- Saves structured chunked data with timestamps into the database.

Usage:
- This script can be executed independently to update the database.
- It can also be imported as a module for integration into a larger system.

Example:
```bash
python chunk_database.py -v -e
```

Dependencies:

- `sentence-transformers` for text embeddings.
- `openai` for alternative embeddings.
- `nltk`, `numpy`, `pickle` for text processing and storage.
- `matplotlib`, `seaborn` for visualization (optional).
- `sklearn` for similarity computations.
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
from sklearn.metrics.pairwise import cosine_similarity

# Libraries for lexical search
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string

# Libraries for handling text
from nltk.metrics.distance import edit_distance
import nltk
from nltk.tokenize import word_tokenize

from transcribe import convert_video_to_audio, transcribe_audio
from index_database import data_preprocessing_cosine_semantic, data_preprocessing_BM25

from chunk_add_dico import chunk_text_add_dico

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA




# Function that rechunk the database without computing the transcriptions
def chunk_database_function(chunks_dico, videos_dico, time_stamps_dico, enforce = False, main_verbose = False):
	""" Rechunks the transcription database without re-performing speech recognition.

	Args:
		chunks_dico (dict): Dictionary storing chunked text.
		videos_dico (dict): Dictionary mapping video IDs to transcriptions.
		time_stamps_dico (dict): Dictionary storing timestamps for each chunk.
		enforce (bool, optional): If True, forces re-chunking of all videos. Defaults to False.
		main_verbose (bool, optional): If True, prints detailed processing logs. Defaults to False.

	Returns:
		tuple: Updated dictionaries (chunks_dico, videos_dico, time_stamps_dico).
	"""
	print("Chunking database...") if main_verbose else None

	transcription_folder = "data/transcriptions"

	# Check in the chunks_dico which video has already been chunked, and which has not
	# If enforce is True, we rechunk all the videos
	to_chunk = []
	if enforce:
		videos_dico = {}
		chunks_dico = {}
		time_stamps_dico = {}
		for filename in os.listdir(transcibe_folder_name):
			if filename.endswith(".txt"):
				if filename.replace(".txt", "_chunks.pickle") in os.listdir(transcibe_folder_name) and filename.replace(".txt", "_timestamps.pickle") in os.listdir(transcibe_folder_name):
					to_chunk.append(filename)
	else:
		video_list = list(set(videos_dico.values()))
		for filename in os.listdir(transcibe_folder_name):
			if filename.endswith(".txt"):
				video_name = os.path.basename(filename).split('.')[0]
				if video_name+"_chunks.pickle" in os.listdir(transcibe_folder_name) and video_name+"_timestamps.pickle" in os.listdir(transcibe_folder_name):
					if video_name not in video_list:
						to_chunk.append(filename)
		if to_chunk == []:
			print("No new video to chunk.") if main_verbose else None
			return chunks_dico, videos_dico, time_stamps_dico


	for filename in to_chunk:
		print(f"Rechunking transcription file '{filename}'...") if main_verbose else None
		script_path = os.path.join(transcription_folder, filename)
		chunks_dico, videos_dico, time_stamps_dico = chunk_text_add_dico(script_path, chunks_dico, videos_dico, time_stamps_dico, main_verbose = main_verbose)

	print("Database Chunked.") if main_verbose else None
			
	return chunks_dico, videos_dico, time_stamps_dico



if __name__ == "__main__":
	""" Runs the chunking process on the transcription database.

	Command-line Arguments:
		-v, --verbose: Enables verbose mode for detailed logging.
		-e, --enforce: Forces re-chunking of all videos.
	"""
	
	# Here is the script to run when you want to only rechunk all the transcriptions in the database
	parser = argparse.ArgumentParser(description='Update the complete database (add new videos)')
	parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
	parser.add_argument('-e', '--enforce', action='store_true', help='Enforce rechunking of all the videos')
	args = parser.parse_args()
	main_verbose = args.verbose
	enforce = args.enforce

	# Some global variables that will be used later
	video_folder_name = "_video_files"
	transcibe_folder_name = "_transcriptions_files"

	# Try to load the embeddings list and the dictionnaries from the database if they exist, otherwise print a message to inform the user that the database is empty
	print("Loading the database...") if main_verbose else None
	if not os.path.exists("data/database"):
		os.makedirs("data/database")
	try:
		with open('data/database/videos_dico.pickle', 'rb') as handle:
			videos_dico = pickle.load(handle)
		with open('data/database/chunks_dico.pickle', 'rb') as handle:
			chunks_dico = pickle.load(handle)
		with open('data/database/time_stamp_dico.pickle', 'rb') as handle:
			time_stamp_dico = pickle.load(handle)
		print("Database loaded.") if main_verbose else None
	except:
		videos_dico = {}
		chunks_dico = {}
		time_stamp_dico = {}

	# We only recompute the chunks based on the already in the database transcriptions
	print("Rechunking the database...") if main_verbose else None
	chunks_dico, videos_dico, time_stamp_dico = chunk_database_function(chunks_dico, videos_dico, time_stamp_dico, enforce = enforce, main_verbose=main_verbose)

	# Save the videos_dico, chunks_dico, time_stamp_dico, data_semantic_cosine and data_BM25 in the database (piclke files)
	print("Saving the database...") if main_verbose else None
	with open('data/database/videos_dico.pickle', 'wb') as handle:
		pickle.dump(videos_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/chunks_dico.pickle', 'wb') as handle:
		pickle.dump(chunks_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/time_stamp_dico.pickle', 'wb') as handle:
		pickle.dump(time_stamp_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
