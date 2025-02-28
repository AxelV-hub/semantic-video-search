"""
chunk_add_dico.py

This module updates the transcription database by chunking transcriptions 
into meaningful segments and generating timestamps for each chunk.

Main Features:
- Splits transcriptions into chunks based on semantic similarity.
- Generates unique chunk identifiers for indexing.
- Assigns timestamps to each chunk using sentence-based alignment.
- Saves the processed data into a structured database.

Usage:
- This script can be run independently or integrated into a larger pipeline.

Example:
```bash
python update_database.py
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

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from openai import OpenAI
from tqdm import tqdm



def chunk_text_add_dico(text_file_path, chunks_dico, videos_dico, time_stamps_dico, main_verbose = False, visualisation = False):
	""" 
	Processes a text file to create chunks based on semantic similarity.

	Args:
		text_file_path (str): Path to the text file to be chunked.
		chunks_dico (dict): Dictionary to store chunked text.
		videos_dico (dict): Dictionary to store video references for chunks.
		time_stamps_dico (dict): Dictionary to store timestamps for each chunk.
		main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.
		visualisation (bool, optional): If True, generates visualizations of embeddings. Defaults to False.

	Returns:
		tuple: Updated dictionaries (chunks_dico, videos_dico, time_stamps_dico).
	"""

	print("Chunking text...") if main_verbose else None

	def chunk_text_main(sentences_list, main_verbose = main_verbose, visualisation = visualisation):
		"""
		Segments a list of sentences into meaningful chunks based on semantic similarity.

		Args:
			sentences_list (list): List of sentences to be chunked.
			main_verbose (bool, optional): If True, prints process details. Defaults to False.
			visualisation (bool, optional): If True, generates visualizations. Defaults to False.

		Returns:
			list: List of chunked text segments.
		"""

		def embed_text(sentences_list):
			"""
			Generates embeddings for sentences using SentenceTransformer.

			Args:
				sentences_list (list): List of sentences for embedding.

			Returns:
				list: List of sentence embeddings.
			"""		
			model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
			embeddings_list = []
			for i in range(len(sentences_list)):
				if i == 0:
					embeddings_list.append(model.encode(sentences_list[i]+"."+sentences_list[i+1]))
				elif i == len(sentences_list)-1:
					embeddings_list.append(model.encode(sentences_list[i-1]+"."+sentences_list[i]))
				else:
					embeddings_list.append(model.encode(sentences_list[i-1]+"."+sentences_list[i]+"."+sentences_list[i+1]))
			return embeddings_list

		def embed_text_openAI(sentences_list, client):
			"""
			Generates embeddings for sentences using OpenAI API.

			Args:
				sentences_list (list): List of sentences for embedding.
				client (OpenAI): OpenAI API client instance.

			Returns:
				list: List of sentence embeddings.
			"""		
			
			text_list = []
			for i in range(len(sentences_list)):
				if i == 0:
					text_list.append(sentences_list[i]+"."+sentences_list[i+1])
				elif i == len(sentences_list)-1:
					text_list.append(sentences_list[i-1]+"."+sentences_list[i])
				else:
					text_list.append(sentences_list[i-1]+"."+sentences_list[i]+"."+sentences_list[i+1])
			embeddings_list = []
			for text in tqdm(text_list):
				embeddings_list.append(client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding)
			return embeddings_list



		if visualisation:
			# plot the histogram of the length of the sentences in number of words
			sentences_length = [len(sentence.split()) for sentence in sentences_list]
			plt.hist(sentences_length, bins=max(sentences_length)-min(sentences_length))
			plt.xlabel('Number of words')
			plt.ylabel('Frequency')
			plt.title('Histogram of sentences length')
			plt.savefig(f"_output_savings/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_histogram_sentences_number_of_words.png")
			# plt.show()

		print("Embedding the text sentences...") if main_verbose else None
		client = OpenAI()
		text_embeddings = embed_text_openAI(sentences_list, client)
		# text_embeddings = embed_text(sentences_list)

		if visualisation:
			print("Ploting some visualisation about the text...") if main_verbose else None
			# project the embeddings on a 2D space using t-SNE

			def project_embeddings(embeddings_list):
				embeddings_list = np.array(embeddings_list)
				tsne = TSNE(n_components=2, metric='cosine', random_state=42)
				projections = tsne.fit_transform(embeddings_list)
				return projections

			# project the embeddings on a 2D space using PCA to compare with t-SNE
			def project_embeddings_pca(embeddings_list):
				embeddings_list = np.array(embeddings_list)
				pca = PCA(n_components=2)
				projections = pca.fit_transform(embeddings_list)
				return projections

			# plot the projection of the text embeddings with a link between consecutive sentences
			def plot_projections(projections, method='t-SNE'):
				plt.figure(figsize=(8, 6))
				plt.scatter(projections[:, 0], projections[:, 1], c=[i for i in range(projections.shape[0])], cmap='brg', s=100)
				plt.colorbar()
				plt.title('Projection of text embeddings with '+method)
				for i in range(projections.shape[0]-1):
					plt.plot([projections[i, 0], projections[i+1, 0]], [projections[i, 1], projections[i+1, 1]], 'k-', alpha=0.3)
				plt.savefig(f"_output_savings/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_projection_text_{method}.png")
				# plt.show()


			text_projections_pca = project_embeddings_pca(text_embeddings[:100])
			plot_projections(text_projections_pca, method='PCA')

			text_projections = project_embeddings(text_embeddings[:100])
			plot_projections(text_projections, method='t-SNE')

		print("Computing the similarity between the sentences...") if main_verbose else None
		# Compute cosine similarity on each consecutive pair of sentences in the text 
		def compute_similarity(embeddings_list):
			"""
			Computes cosine similarity between consecutive sentence embeddings.

			Args:
				embeddings_list (list): List of sentence embeddings.

			Returns:
				list: List of similarity scores.
			"""
			similarity_list = []
			for i in range(len(embeddings_list)-1):
				similarity = cosine_similarity([embeddings_list[i]],[embeddings_list[i+1]])
				similarity = similarity[0][0]
				similarity_list.append(similarity)
			return similarity_list

		text_similarity = compute_similarity(text_embeddings)

		print("Creation of the chunks based on the similarities...") if main_verbose else None
		# We go through the similarity scores and if (1-similarity) is above a threshold, we chunk the text. The threshold starts at 1.5 and decreases by 0.05 each step, and is reset to 1.5 after each chunking

		def chunk_text(text_similarity, sentences_list):
			"""
			Creates text chunks based on similarity scores.

			Args:
				text_similarity (list): List of similarity scores between sentences.
				sentences_list (list): List of sentences to be chunked.

			Returns:
				tuple: List of text chunks and corresponding thresholds used for segmentation.
			"""
			min_nb_words = 25
			max_nb_words = 100
			chunks = []
			i = 0
			thresholds_used = [0.5]
			# find the index of sentences where we are above the min_nb_words in total
			while i < len(text_similarity):
				# find the minimum index where we are above the min_nb_words
				j = i
				nb_words = 0
				while nb_words < min_nb_words and j < len(text_similarity):
					nb_words += len(sentences_list[j].split())
					j += 1
				# find the maximum index before we are above the max_nb_words
				k = i
				nb_words = 0
				while nb_words < max_nb_words and k < len(text_similarity):
					nb_words += len(sentences_list[k].split())
					k += 1
				# find the transition in the text_similarity between j and k where the similarity is the lowest
				min_similarity = 1
				min_index = j
				for l in range(j, k):
					if text_similarity[l] < min_similarity:
						min_similarity = text_similarity[l]
						min_index = l
				chunks.append(sentences_list[i:min_index])
				i = min_index
				thresholds_used += [1 - min_similarity] * len(chunks[-1])
			return chunks, thresholds_used


		chunks, thresholds_used = chunk_text(text_similarity, sentences_list)
		thresholds_used[0] = thresholds_used[1]

		if visualisation:
			print("Ploting some visualisation about the similarity...") if main_verbose else None
			# plot similarity scores 

			def plot_similarity(similarity_list, thresholds_used):
				similarity_list = [1 - similarity for similarity in similarity_list]
				plt.figure(figsize=(10, 6))
				plt.plot(similarity_list[:200])
				plt.plot(thresholds_used[:200], 'r-')
				plt.xlabel('Sentence pairs', fontsize=14)
				plt.ylabel('Difference score', fontsize=14)
				plt.title('Difference score and threshold used to cut', fontsize=16)
				plt.xticks(fontsize=14)
				plt.yticks(fontsize=14)
				plt.savefig(f"_output_savings/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_similarity_scores.png")
				# plt.show()


			plot_similarity(text_similarity[:500], thresholds_used[:500])

			# plot similarity scores graph
			def plot_similarity_histogram(similarity_list):
				plt.figure(figsize=(10, 6))
				plt.hist(similarity_list, bins=20)
				plt.xlabel('Cosine similarity')
				plt.ylabel('Frequency')
				plt.title('Histogram of similarity scores')
				plt.savefig(f"_output_savings/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_histogram_similarity_scores.png")
				# plt.show()

			plot_similarity_histogram(text_similarity)


		return chunks

	# Run the chunking function on the file
	with open(text_file_path.replace(".txt","_chunks.pickle"), 'rb') as handle:
		chunks_list = pickle.load(handle)
	clusters = chunk_text_main(chunks_list)

	if visualisation:    # Si l'objectif est juste de visualiser, on ne run pas le reste de la fonction
		return None
	
	# Load the timestamps of the sentences
	with open(text_file_path.replace(".txt","_timestamps.pickle"), 'rb') as handle:
		timestamps_list = pickle.load(handle)

	# Creation of chunks unique IDs, then added to dictionnary of chunks (ID defined using the 4 first letters of the paragraph, then the size of the paragraph in 4 digits, then the 4 last letters of the paragraph and finally 6 random digits)
	total_segments_yet = 0  # useful to calculate the timestamp of the chunks according to the number of sentences in each 


	for i,text_list in enumerate(clusters):
		cluster_txt = "".join(text_list)
		cluster_len = len(cluster_txt)
		# Starting with the ID generation, saving the chunk text and the video name
		ID = cluster_txt[0:4]+str(cluster_len).zfill(4)+cluster_txt[-4:]+str(rd.random())[-6:]
		chunks_dico[ID] = cluster_txt
		videos_dico[ID] = os.path.basename(text_file_path).split('.')[0]
		# Calculating the timestamp of the chunk, if k in the number of sentences in the chunk txt, then take start = timestamps_list[total_segments_yet][0] and end = timestamps_list[total_segments_yet+k][1] and increment total_segments_yet by k
		nb_segments_in_cluster = len(text_list)
		start = timestamps_list[total_segments_yet][0]
		end = timestamps_list[total_segments_yet+nb_segments_in_cluster-1][1]
		total_segments_yet += nb_segments_in_cluster
		# Calculate the timestamp of the chunk
		start_minutes = int(start // 60)
		start_seconds = int(start % 60)
		start_centiseconds = int((start % 1) * 100)
		end_minutes = int(end // 60)
		end_seconds = int(end % 60)
		end_centiseconds = int((end % 1) * 100)

		# Format the timestamp
		start_timestamp = f"{start_minutes}:{start_seconds}:{start_centiseconds}"
		end_timestamp = f"{end_minutes}:{end_seconds}:{end_centiseconds}"

		time_stamps_dico[ID] = (start_timestamp, end_timestamp)

	# Saving the chunks, videos and timestamps dico in the database
	if not os.path.exists("data/database"):
		os.makedirs("data/database")
	with open('data/database/chunks_dico.pickle', 'wb') as handle:
		pickle.dump(chunks_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/videos_dico.pickle', 'wb') as handle:
		pickle.dump(videos_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('data/database/time_stamp_dico.pickle', 'wb') as handle:
		pickle.dump(time_stamps_dico, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("Chunking done.") if main_verbose else None
	return chunks_dico, videos_dico, time_stamps_dico



if __name__ == "__main__":
	# Add an example of text chunking, with the visualisation of the embeddings and the similarity scores
	script_path = "_transcriptions_files/Guide Complet De La Regression Lineaire En Python - Machine Learning_chunks.pickle"
	chunk_text_add_dico(script_path, {}, {}, {}, main_verbose = False, visualisation = True)