"""
update_database.py

This module processes new video files by converting them to audio, transcribing them,
and segmenting the transcriptions into structured chunks with timestamps. It updates
the database accordingly.

Main Features:
- Identifies and processes new video files not yet included in the database.
- Converts video to audio and applies speech-to-text transcription.
- Segments transcriptions into meaningful chunks with timestamp alignment.
- Saves structured data into the database.

Usage:
- This script can be executed independently to update the database with new videos.
- It can also be integrated into a larger pipeline.

Example:
```bash
python update_database.py -v -e
```

Dependencies:
- `moviepy` for video-to-audio conversion.
- `speech_recognition` for Google Speech-to-Text.
- `pydub` for audio processing.
- `sentence-transformers` for text embedding and similarity calculations.
- `nltk`, `numpy`, `pickle` for text processing and storage.
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
from chunk_database import chunk_database_function


from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import datetime


def update_database(video_folder_name, videos_dico, chunks_dico, time_stamp_dico, main_verbose = False):
	"""
	Updates the transcription database by processing new video files.

	Args:
		video_folder_name (str): Path to the folder containing video files.
		videos_dico (dict): Dictionary mapping video IDs to transcriptions.
		chunks_dico (dict): Dictionary storing chunked text data.
		time_stamp_dico (dict): Dictionary storing timestamps for each chunk.
		main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

	Returns:
		tuple: Updated dictionaries (videos_dico, chunks_dico, time_stamp_dico).
	"""
	print("Updating database...") if main_verbose else None

	# Search for new videos in the _video_files folder that are not in the database yet
	new_videos_to_be_processed = []
	for filename in os.listdir(video_folder_name):
		if filename.endswith(".mp4"):
			video_name = os.path.basename(filename).split('.')[0]
			if video_name not in videos_dico.values():
				new_videos_to_be_processed.append(video_name)
	if new_videos_to_be_processed:
		print("The following videos are new in the database and will be processed:") if main_verbose else None
		for video_name in new_videos_to_be_processed:
			time.sleep(0.2)
			print(f"  -->  \"{video_name}\"")
	else:
		print("No new videos need to be processed.") if main_verbose else None

	# Create the Transcription folder and the Temp audio folder if they don't exist
	if not os.path.exists("data/transcriptions"):
		os.makedirs("data/transcriptions")
	if not os.path.exists("_audio_files"):
		os.makedirs("_audio_files")

	# Create the output savings folder if it doesn't exist
	if not os.path.exists("_output_savings"):
		os.makedirs("_output_savings")

	# Get the current date and time
	current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	# Create the output file name
	output_file_name = f"{current_datetime}_Running_time_to_create_database.txt"
	output_file_path = os.path.join("_output_savings", output_file_name)

	# Open the output file for writing
	with open(output_file_path, "w", encoding = "utf-8") as output_file:
		# Initialize variables to store total running time for each step
		total_conversion_time = 0
		total_transcription_time = 0
		total_chunking_time = 0

		# Iterate in each new video to be processed
		for i,video_name in enumerate(new_videos_to_be_processed):
			print("-----------------------------------") if main_verbose else None
			print(f"({i+1}/{len(new_videos_to_be_processed)} - Start processing video '{video_name}'...") if main_verbose else None

			# Construct the file path of the video, the audio and the transcription
			video_path = os.path.join(video_folder_name, f"{video_name}.mp4")
			audio_path = os.path.join("_audio_files", f"{video_name}.wav")
			script_path = os.path.join("data/transcriptions", f"{video_name}.txt")

			# Convert video to audio
			print(f"Converting video file '{video_name}' to audio...") if main_verbose else None
			start_time = time.time()
			convert_video_to_audio(video_path, audio_path, main_verbose = main_verbose)
			end_time = time.time()
			running_time = end_time - start_time
			total_conversion_time += running_time
			output_file.write(f"Conversion of video '{video_name}' to audio: {running_time} seconds\n")
			print(f"Video file '{video_name}' converted to audio successfully") if main_verbose else None

			# Get transcription
			print(f"Transcribing audio file '{video_name}'...") if main_verbose else None
			start_time = time.time()
			transcribe_audio(audio_path,script_path, main_verbose = main_verbose)
			end_time = time.time()
			running_time = end_time - start_time
			total_transcription_time += running_time
			output_file.write(f"Transcription of audio file '{video_name}': {running_time} seconds\n")
			print(f"Audio file '{video_name}' transcribed successfully") if main_verbose else None

			# Chunk transcriptions based on semantic similarity between paragraphs
			print(f"Chunking transcription of '{video_name}'...") if main_verbose else None
			start_time = time.time()
			chunks_dico, videos_dico, time_stamp_dico = chunk_text_add_dico(script_path, chunks_dico, videos_dico, time_stamp_dico, main_verbose=main_verbose) 
			end_time = time.time()
			running_time = end_time - start_time
			total_chunking_time += running_time
			output_file.write(f"Chunking transcription of '{video_name}': {running_time} seconds\n")
			print(f"Transcription of '{video_name}' chunked successfully") if main_verbose else None

			# Saving the chunks_dico, videos_dico and time_stamp_dico in the database
			with open('data/database/chunks_dico.pickle', 'wb') as handle:
				pickle.dump(chunks_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
			with open('data/database/videos_dico.pickle', 'wb') as handle:
				pickle.dump(videos_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
			with open('data/database/time_stamp_dico.pickle', 'wb') as handle:
				pickle.dump(time_stamp_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# Write total running time for each step to the output file
		output_file.write(f"Total conversion time: {total_conversion_time} seconds\n")
		output_file.write(f"Total transcription time: {total_transcription_time} seconds\n")
		output_file.write(f"Total chunking time: {total_chunking_time} seconds\n")

	print("Database updated.") if main_verbose else None
	return videos_dico, chunks_dico, time_stamp_dico



if __name__ == "__main__":
	"""
	Runs the database update process.

	Command-line Arguments:
		-v, --verbose: Enables verbose mode for detailed logging.
		-e, --enforce: Forces reprocessing of all videos, even those already in the database.
	"""
	# Here is the script to run when you want to update the database (add new videos)
	parser = argparse.ArgumentParser(description='Update the complete database (add new videos)')
	parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
	parser.add_argument('-e', '--enforce', action='store_true', help='Force the reprocessing of all videos, even the one already in the database')
	args = parser.parse_args()
	main_verbose = args.verbose
	force = args.enforce

	# Some global variables that will be used later
	video_folder_name = "_video_files"
	transcibe_folder_name = "data/transcriptions"

	# Try to load the embeddings list and the dictionnaries from the database if they exist, otherwise create them and force updating the database
	if force:   # Si l'option force est activée, on repart d'une base de données vide
		chunks_dico = {}
		videos_dico = {}
		time_stamp_dico = {}
		data_semantic_cosine = []
		# clean the database folder and the transcription folder (delete every file inside them)
		if os.path.exists("data/database"):
			shutil.rmtree("data/database")
		if os.path.exists("data/transcriptions"):
			shutil.rmtree("data/transcriptions")
	else:
		if not os.path.exists("Database"):
			os.makedirs("Database")
		try:
			with open('data/database/videos_dico.pickle', 'rb') as handle:
				videos_dico = pickle.load(handle)
			with open('data/database/chunks_dico.pickle', 'rb') as handle:
				chunks_dico = pickle.load(handle)
			with open('data/database/time_stamp_dico.pickle', 'rb') as handle:
				time_stamp_dico = pickle.load(handle)
			print("Database loaded.") if main_verbose else None
		except:
			chunks_dico = {}
			videos_dico = {}
			time_stamp_dico = {}
			data_semantic_cosine = []

	# Update the videos_dico, chunks_dico, time_stamp_dico and data_semantic_cosine
	videos_dico, chunks_dico, time_stamp_dico = update_database(video_folder_name, videos_dico, chunks_dico, time_stamp_dico, main_verbose=main_verbose)

	# Save the videos_dico, chunks_dico, time_stamp_dico, data_semantic_cosine and data_BM25 in the database (piclke files)
	with open('data/database/videos_dico.pickle', 'wb') as handle:
		pickle.dump(videos_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/chunks_dico.pickle', 'wb') as handle:
		pickle.dump(chunks_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/time_stamp_dico.pickle', 'wb') as handle:
		pickle.dump(time_stamp_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)

