"""
create_testing_material.py

This module generates test question-answer pairs for evaluating a question-answering 
algorithm based on a database of transcriptions.

Main Features:
- Randomly selects text chunks from the database.
- Uses OpenAI's GPT-3.5 to generate relevant questions.
- Ensures questions are answerable solely from the selected text chunk.
- Provides user validation before saving generated questions.
- Saves generated test questions and associated metadata for further evaluation.

Usage:
- This script can be executed independently to generate test questions.
- It can also be imported as a module for integration into a testing pipeline.

Example:
python create_testing_material.py

Dependencies:
- `openai` for generating questions.
- `pickle` for storing and retrieving testing data.
- `random` for selecting chunks randomly.
"""

import random as rd
from openai import OpenAI
import pickle

def generate_testing_questions(chunks_dico, videos_dico, time_stamp_dico, n_questions=1, main_verbose=False):
	"""
	Generates test questions based on text chunks from the database.

	Args:
		chunks_dico (dict): Dictionary containing text chunks.
		videos_dico (dict): Dictionary mapping chunk IDs to video names.
		time_stamp_dico (dict): Dictionary storing timestamps for each chunk.
		n_questions (int, optional): Number of questions to generate. Defaults to 1.
		main_verbose (bool, optional): If True, prints detailed logs. Defaults to False.

	Returns:
		tuple: Updated testing dataset containing (correct_video_name, correct_timestamps, testing_queries).
	"""

	# open the testing material
	try : 
		with open('data/testing/testing_material.pkl', 'rb') as handle:
			correct_video_name, correct_timestamps, testing_queries = pickle.load(handle)
	except:
		testing_queries = []
		correct_video_name = []
		correct_timestamps = []

	client = OpenAI()

	validated_questions = 0

	while validated_questions < n_questions:
		# Choose a random chunk
		good_chunk = False
		while not good_chunk:
			chunk_ID = rd.choice(list(chunks_dico.keys()))
			chunk_size = len(chunks_dico[chunk_ID].split())
			if main_verbose:
				print(f"Chunk size: {chunk_size}")
			if chunk_size > 100:
				good_chunk = True

		chunk_text = chunks_dico[chunk_ID]
		video_name = videos_dico[chunk_ID]
		timestamps = time_stamp_dico[chunk_ID]

		# Prepare the message for the model
		messages = [
			{"role": "system", "content": """Vous êtes un assistant intelligent conçu pour générer des questions qui, bien que basées sur un paragraphe spécifique, pourraient être posées par quelqu'un sans connaissance préalable de ce texte. La réponse à la question doit se trouver uniquement dans ce paragraphe. Pour vous donner le contexte, je veux générer des pairs questions-réponses de test pour évaluer un algorithme de réponse aux questions basé sur une base de données. Pour ce faire, je dois générer des questions que quelqu'un pourrait poser sur l'outil, et l'accoler avec le paragraphe de ma base de données qui y répond. La question doit donc être spécifique au paragraphe, en ce sens que la réponse ne doit pas se trouver dans un autre paragraphe, mais la question doit être suffisamment générale pour que quelqu'un qui ne sait pas que le paragraphe existe, puisse la poser de nulle part dans un outil de réponse aux questions. Il ne faut donc pas ajouter à la question des termes comme 'comme mentionné dans le paragraph' ou 'd'après le texte ci-dessus'."""},
			{"role": "user", "content": chunk_text}
		]

		# Generate a question related to the chunk with OpenAI API
		response = client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=messages
		)

		question = response.choices[0].message.content

		print("-"*30) if main_verbose else None
		print(f"Question: {question}") if main_verbose else None
		print(f"Text with the answer: {chunk_text}") if main_verbose else None

		# Ask for user validation
		print("Do you validate this question? [y/n/modified question]")
		user_input = input().strip().lower()
		if user_input == "y":
			testing_queries.append(question)
			correct_video_name.append(video_name)
			correct_timestamps.append(timestamps)
			validated_questions += 1
			# save the testing material
			testing_material = (correct_video_name, correct_timestamps, testing_queries)
			with open('data/testing/testing_material.pkl', 'wb') as handle:
				pickle.dump(testing_material, handle, protocol=pickle.HIGHEST_PROTOCOL)
		elif user_input == "n":
			print("The question will not be saved.") if main_verbose else None
		else:
			testing_queries.append(user_input)
			correct_video_name.append(video_name)
			correct_timestamps.append(timestamps)
			validated_questions += 1
			# save the testing material
			testing_material = (correct_video_name, correct_timestamps, testing_queries)
			with open('data/testing/testing_material.pkl', 'wb') as handle:
				pickle.dump(testing_material, handle, protocol=pickle.HIGHEST_PROTOCOL)


	return correct_video_name, correct_timestamps, testing_queries


if __name__ == "__main__":
	"""
	Main execution script for generating testing questions.

	This script:
	- Loads the database containing video transcriptions.
	- Loads existing test questions if available.
	- Prompts the user to define the number of new test questions.
	- Generates the specified number of test questions using OpenAI.
	- Saves the updated test dataset.

	Dependencies:
		- Requires `Database/videos_dico.pickle`, `Database/chunks_dico.pickle`, and `Database/time_stamp_dico.pickle`.
		- Requires OpenAI API access for question generation.

	Example Usage:
		python create_testing_material.py
	"""

	# open the database
	with open('Database/videos_dico.pickle', 'rb') as handle:
		videos_dico = pickle.load(handle)
	with open('Database/chunks_dico.pickle', 'rb') as handle:
		chunks_dico = pickle.load(handle)
	with open('Database/time_stamp_dico.pickle', 'rb') as handle:
		time_stamp_dico = pickle.load(handle)

	# open testing material
	with open('data/testing/testing_material.pkl', 'rb') as handle:
		correct_video_name, correct_timestamps, testing_queries = pickle.load(handle)

	print(f"Number of questions in the testing material: {len(testing_queries)}")

	# Ask how many new question the user want to add, and generate them
	print("How many new questions do you want to generate?")
	n_questions = int(input())
	correct_video_name, correct_timestamps, testing_queries = generate_testing_questions(chunks_dico, videos_dico, time_stamp_dico, n_questions=n_questions, main_verbose=True)


