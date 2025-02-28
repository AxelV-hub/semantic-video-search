"""
transcribe.py

This module provides functionalities for transcribing audio extracted from video files.
It supports two transcription methods:
- **OpenAI Whisper API** for high-quality transcription with timestamps.
- **Google Speech-to-Text API** for an alternative method.

Main features:
- Extracts audio from video files.
- Splits audio into chunks based on silence detection.
- Transcribes the extracted audio into text.
- Generates timestamps for improved text alignment.
- Supports evaluation and testing using predefined datasets.

Usage:
- The module can be used as a standalone script to test transcription accuracy.
- It can be imported as a module for integration into larger projects.

Example:
```bash
python transcribe.py -v
```

Dependencies:
- `speech_recognition` for Google Speech-to-Text.
- `pydub` for audio manipulation.
- `moviepy` for video processing.
- `sentence_transformers` for text processing.
- `openai` for Whisper API transcription.
- `nltk`, `numpy`, `pandas`, `seaborn`, and `matplotlib` for evaluation.
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

# Libraries for speech transcription
import speech_recognition as sr 
from pydub import AudioSegment #for the split of audio
from pydub.silence import split_on_silence #for split by silences
from moviepy.video.io.VideoFileClip import VideoFileClip

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
from nltk.tokenize import word_tokenize, sent_tokenize
from deepmultilingualpunctuation import PunctuationModel

# Libraries for printing results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime


def convert_video_to_audio(video_path, audio_path, main_verbose=False):
    """
    Extracts the audio from a video file and saves it as a separate audio file.
    
    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.
        main_verbose (bool): If True, print process details.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, verbose=False)
    audio_clip.close()
    video_clip.close()


def transcribe_audio(path, script_path, verbose=False, main_verbose=False, version = "whisper"):
    """
    Transcribes an audio file into text using either OpenAI Whisper or Google Speech-to-Text.
    
    Args:
        path (str): Path to the audio file to transcribe.
        script_path (str): Path to save the transcribed text.
        verbose (bool): If True, prints additional processing details.
        main_verbose (bool): If True, provides more detailed logging.
        version (str): Defines the transcription model to use ("whisper" or "Google").
    """    
    if version == "whisper":

        from openai import OpenAI
        client = OpenAI()

        # Open the audio file using pydub
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(16000)

        # Split the audio file into chunks on silence
        print("Splitting audio file on silences...") if main_verbose else None

        chunks = split_on_silence(
            sound,
            min_silence_len = 1500,  
            silence_thresh = sound.dBFS - 14,  
            keep_silence = True,
        )

        print("Audio file splitted on silences successfully.") if main_verbose else None
        
        # Create a directory to store the audio files
        folder_name = "data/audio/_temp_audio_chunks_files"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        total_chunks = len(chunks)
        print(f"Total number of chunks: {total_chunks}") if main_verbose else None
        whole_text = ""
        timestamps_list = []
        position_in_full_video = 0
        chunks_list = []

        # Transcribe each audio chunk
        with tqdm(total=total_chunks,desc = "Transcribtion...") as pbar:
            for i, audio_chunk in enumerate(chunks):

                chunk_filename = os.path.join(folder_name, f"chunk{i}.mp3")
                audio_chunk.export(chunk_filename, format="mp3")
                audio_file = open(chunk_filename, "rb")
                audio_length = audio_chunk.duration_seconds

                # Transcribe the audio chunk using OpenAI Whisper
                transcribe = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    prompt="On parlera de : IA, Cloud, Cloud provider, Cloud souverain, Cloud de confiance, Machine Learning, ML, Cybersécurité, Big Data, Data, IoT, etc."
                )

                # Extract the transcribed text and timestamps
                for seg in transcribe.segments:
                    text = seg["text"]
                    start = seg["start"]
                    end = seg["end"]
                    whole_text += text 
                    chunks_list.append(text)
                    timestamps_list.append((position_in_full_video + start, position_in_full_video + end))
                position_in_full_video += audio_length
                
                pbar.update(1)

        # Save the transcribed text, timestamps, and chunks in pickle files
        with open(script_path.replace(".txt","_chunks.pickle"), 'wb') as handle:
            pickle.dump(chunks_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(script_path, 'w', encoding="utf-8") as file:
            file.write(whole_text)
        with open(script_path.replace(".txt","_timestamps.pickle"), 'wb') as handle:
            pickle.dump(timestamps_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # The Google version is now only used for comparison with the whisper version
    elif version == "Google" :
        print("Splitting audio file on silences...") if main_verbose else None

        # Create a speech recognition object
        recognizer = sr.Recognizer()

        # Define the optimal number of chunks for 5 minutes of audio
        min_chunks_in_5_minutes = 5
        max_chunks_in_5_minutes = 10

        # Open the audio file using pydub
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(16000)

        # Starting with some tests on a 5 minutes sample to find the best parameters for a good sentence flow
        duration = sound.duration_seconds
        start = duration/2 - 150 
        end = duration/2 + 150 
        if duration > 300:
            sound_test = sound[start*1000:end*1000]
        else:
            sound_test = sound

        # Initial parameters
        silence_length = 1000
        silence_threshold = sound.dBFS - 14    # threshold for silence
        correct_chunk_lenght = False
        iterations = 0
        correction_amplitude = 200

        # We test different silence lengths and silence threshold to find the optimal one
        while not(correct_chunk_lenght):
            print(f"Testing silence length : {silence_length} ms") if verbose else None
            chunks_test = split_on_silence(
                sound_test,
                min_silence_len=silence_length,  
                silence_thresh=silence_threshold,  
            )
            print(f"Obtained : {len(chunks_test)} chunks for 5 minutes") if verbose else None

            # If we have between 5 and 10 chunks for 5 minutes, we consider that the silence length is optimal
            if len(chunks_test) >= min_chunks_in_5_minutes and len(chunks_test) <= max_chunks_in_5_minutes:
                correct_chunk_lenght = True
                print(f"Optimal silence length found : {silence_length} ms") if verbose else None
            # Otherwise, we adjust the silence length and the silence threshold
            else : 
                # Every 3 iterations, we reduce the correction amplitude by 2
                if iterations % 5 == 0:
                    correction_amplitude = correction_amplitude//2
                # If we already tried 20 times, we lower the threshold for silence and start again
                if iterations > 20:
                    silence_threshold = sound.dBFS - 10
                    silence_length = 500
                    correction_amplitude = 400
                # We stop after 40 iterations
                if iterations == 40:
                    break	
                # If we have less than 40 chunks for 5 minutes, that is not enough sentences, we lower the silence length, and if it is above 60, we increase it
                if len(chunks_test) < min_chunks_in_5_minutes:
                    silence_length -= correction_amplitude
                else:
                    silence_length += correction_amplitude
                
            iterations += 1

        # Apply the best parameters to the whole audio file
        chunks = split_on_silence(
                sound,
                min_silence_len=silence_length,  
                silence_thresh=silence_threshold,  
                keep_silence=1500,
            )

        print("Audio file splitted on silences successfully.") if main_verbose else None
        folder_name = "_temp_audio_chunks_files"

        # Create a directory to store the audio files
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        whole_text = ""
        total_chunks = len(chunks)
        print(f"Total number of chunks: {total_chunks} with silence length: {silence_length} ms") if main_verbose else None
        timestamps_list = []

        # Transcribe each audio chunk
        with tqdm(total=total_chunks,desc = "Transcribtion...") as pbar:
            for i, audio_chunk in enumerate(chunks):

                chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
                audio_chunk.export(chunk_filename, format="wav")
                with sr.AudioFile(chunk_filename) as source:
            
                    audio_chunk = AudioSegment.from_wav(chunk_filename)
                    audio_length = audio_chunk.duration_seconds - 1.5 if version == "v2" else audio_chunk.duration_seconds
                    if i == 0:
                        start = 0
                        end = audio_length
                    else:
                        start = timestamps_list[-1][1]
                        end = start + audio_length

                    # Use Google's API for the transcription of the chunk
                    try : 
                        audio_listened = recognizer.record(source)
                        text = recognizer.recognize_google(audio_listened, language="fr-FR")
                        text = f"{text.capitalize()}. "
                        timestamps_list.append((start,end))
                    except :
                        text = ""
                        if i != 0:
                            timestamps_list[-1] =  (timestamps_list[-1][0],end)
                        else:
                            timestamps_list.append((0,0))
                    
                whole_text += text
                pbar.update(1)
        
        # Case of an empty text, we remove the corresponding timestamp
        if timestamps_list[0] == (0,0):
            timestamps_list.pop(0)

        # Verify if the number of points in the paragraph is identical to the number of timestamps
        if len(whole_text.split(".")) != len(timestamps_list):
            print("Error: the number of points and timestamps are different")
        
        # Function to redo the punctuation and capitalization using a transformer model
        def redo_punctuation(text):
            punctuations = """!;:,.?"""
            for punctuation in punctuations:
                text = text.replace(punctuation, "")

            model = PunctuationModel()
            text = model.restore_punctuation(text)
            text = text.lower()
            text = sent_tokenize(text)
            text = [sentence.capitalize() for sentence in text]
            text = " ".join(text)
            return text

        # Re-building the correct punctuation
        print("Re-creating punctuation...") if main_verbose else None
        new_text = redo_punctuation(whole_text)
        print("Punctuation restored successfully.") if main_verbose else None

        # Function to get the position of the points in the text
        def place_of_points(text):
            # Delete all punctuation in the text, except the points
            text = text.replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace(",", "")
            text = text.replace(".", " .") # we separate the points from the words

            # Add a point at the end of the text to ensure that the last sentence is taken into account
            text = text + "."

            # Split the text into words
            words = text.split(" ")

            # Take the position of the points in the text and put them in a list. We don't count the points as words
            points = []
            for i in range(len(words)):
                if words[i] == ".":
                    points.append(i - len(points)) # we subtract the number of points we have already seen to get the position in number of words only. 
            
            return points

        # Function that takes a new list of points positions and returns the corresponding timestamps, using linear approximation based on a given list of points and timestamps. The format of the timestamps is (start, end)
        def get_timestamps(points, timestamps, new_points):
            if len(points) != len(timestamps):
                print("Error: the number of points and timestamps is different")
            timestamps_points = [timestamp[1] for timestamp in timestamps[:len(points)]]

            new_timestamps = []
            for j,point in enumerate(new_points):
                # Find the two points between which the new point is
                i = 0
                while points[i] < point:
                    i += 1
                # Linear approximation
                approx = timestamps_points[i - 1] + (timestamps_points[i] - timestamps_points[i - 1]) * (point - points[i - 1]) / (points[i] - points[i - 1])
                if j == 0:
                    new_timestamps.append((0, approx))
                else:
                    new_timestamps.append((new_timestamps[-1][1], approx))
            return new_timestamps

        # Function that takes the current text, the current timestamps and a new text, and returns the timestamps for the new text, based on the position of the points
        def get_new_timestamps(text, timestamps, new_text):
            points = place_of_points(text)
            new_points = place_of_points(new_text)
            return get_timestamps(points, timestamps, new_points)

        # Get the new timestamps
        print("Re-creating timestamps...") if main_verbose else None
        timestamps_list = get_new_timestamps(whole_text, timestamps_list, new_text)
        both_text = "The transcribed text before punctuation restoration:\n" + whole_text + "\n\nThe transcribed text after punctuation restoration:\n" + new_text

        # Save in a pickle text file the full text
        with open(script_path, 'w', encoding="utf-8") as file:
            file.write(new_text)
        with open(script_path+"_before_punctuation.txt", 'w', encoding="utf-8") as file:
            file.write(whole_text)


    else : 
        print("Error: version not recognized")


def test_transciption(main_verbose=False):
    """
    Runs a test suite on the transcription process using predefined test files.
    
    Args:
        main_verbose (bool): If True, provides detailed logs.
    """

    # Create a temporary folder to store the transcriptions of the test
    if not os.path.isdir("_testing_transcriptions"):
        os.mkdir("_testing_transcriptions")
    
    # We test several versions of the transcription function
    versions = ["whisper", "Google"]
    distances = {}

    # Load the audio files and their corresponding scripts from the _testing_files folder
    audio_files = [f"_testing_files/audio_{i}.wav" for i in range(6)]
    original_script = [f"_testing_files/script_{i}.txt" for i in range(6)]
    tested_script = {version: [f"_testing_transcriptions/script_{i}_{version}.txt" for i in range(6)] for version in versions}

    for version in versions:

        # Apply the transcription function to each audio file
        for i in range(1):
            transcribe_audio(audio_files[i], tested_script[version][i], verbose=True, main_verbose = main_verbose, version=version)
        
        # Compare the transcriptions to the original scripts with Levenshtein distance
        distances[version] = []
        for i in range(1):
            with open(original_script[i], 'r') as file:
                text1 = file.read()
            with open(tested_script[version][i], 'r') as file:
                text2 = file.read()
            tokens1 = word_tokenize(text1.lower())
            tokens2 = word_tokenize(text2.lower())
            distance = edit_distance(tokens1, tokens2)/len(tokens1)
            distances[version].append(distance)
        
    # Print the results, with the mean of the 6, and in brackets the precision of each distance
    for version in versions:
        print(f"Mean distance for version {version}: {np.mean(distances[version])}")
        print(f"All distances for version {version} : {distances[version]}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save in a text file the distances and the mean distance for each version in the _output_savings folder
    with open(f"_output_savings/{current_time}_distances.txt", "w", encoding = "utf-8") as file:
        for version in versions:
            file.write(f"Mean distance for version {version}: {np.mean(distances[version])}\n")
            file.write(f"All distances for version {version} : {distances[version]}\n")

    # Plot the distances for each version in a boxplot and save it in the _output_savings folder
    df = pd.DataFrame({
        'Google': distances['Google'],
        'whisper': distances['whisper'],
    })
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df)
    plt.title("Levenshtein distance between the original scripts and the transcriptions", fontsize=14)
    plt.ylabel("Levenshtein distance", fontsize = 14)
    plt.xlabel("Version", fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"_output_savings/{current_time}_Test_transcription_score.png")
    # plt.show()
        
    # Take an extract of the transcriptions, and the correct answer, and save them in a text file in the _output_savings folder
    with open(f"_output_savings/{current_time}_test_transcriptions.txt", "w", encoding = "utf-8") as file:
        i = 0
        file.write("-"*30+"\n")
        file.write(f"Correct answer {i}:\n")
        with open(original_script[i], 'r', encoding = "utf-8") as file2:
            text = file2.read()
            file.write(text)
        file.write("\n\n---------------------\n")
        file.write(f"Transcription {i}:\n")
        file.write(f"Version whisper:\n")
        with open(tested_script["whisper"][i], 'r', encoding = "utf-8") as file3:
            text = file3.read()
            file.write(text)
        file.write("\n\n---------------------\n")
        file.write(f"Version Google:\nText after the punctuation restoration:\n")
        with open(tested_script["Google"][i], 'r', encoding = "utf-8") as file4:
            text = file4.read()
            file.write(text)
        file.write(f"\n\nText before the punctuation restoration:\n")
        with open(tested_script["Google"][i]+"_before_punctuation.txt", 'r', encoding = "utf-8") as file5:
            text = file5.read()
            file.write(text)
        file.write("\n")
    
    # Delete the temporary folder _testing_transcriptions
    shutil.rmtree("_testing_transcriptions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the quality of the transcription function')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    main_verbose = args.verbose
    test_transciption(main_verbose = main_verbose)