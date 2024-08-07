# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:32:09 2024

@author: narze
"""
import spacy 
import random
from collections import Counter
nlp = spacy.load("en_core_web_sm")

from flask import Flask, render_template, request
from PyPDF2 import PdfReader


def generate_mcqs(text, num_questions=5):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentences from the text
    sentences = [sent.text for sent in doc.sents]
    
    # Randomly select sentences to form questions
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))

    # Initialize list to store generated MCQs
    mcqs = []

    # Generate MCQs for each selected sentence
    for sentence in selected_sentences:
        # Process the sentence with spaCy
        sent_doc = nlp(sentence)

        # Extract entities (nouns) from the sentence
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        # Ensure there are enough nouns to generate MCQs
        if len(nouns) < 2:
            continue

        # Count the occurrence of each noun
        noun_counts = Counter(nouns)

        # Select the most common noun as the subject of the question
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]

            # Generate the question stem
            question_stem = sentence.replace(subject, "_______")

            # Generate answer choices
            answer_choices = [subject]

            # Add some random words from the text as distractors
            for _ in range(3):
                distractor = random.choice(list(set(nouns) - set([subject])))
                answer_choices.append(distractor)

            # Shuffle the answer choices
            random.shuffle(answer_choices)

            # Append the generated MCQ to the list
            correct_answer = chr(64 + answer_choices.index(subject) + 1)  # Convert index to letter
            mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

import joblib
model=joblib.load('GenQue.pkl')

def printMCQ(text):
    num_questions=len(text)//300
    
    results = model(text, num_questions)
    output=''
    for i, mcq in enumerate(results,start=1):
        question_stem, answer_choices, correct_answer = mcq
    
        output=output + f"Q{i}: {question_stem}"+'\n\n'
    
        for j, choice  in enumerate(answer_choices, start=1):
            output=output+ f"{chr(64+j)}: {choice}"+'\n'
        output=output+f"Correct Answer: {correct_answer}"+'\n'
        output=output+'\n'
    return output

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    if request.method == 'POST':
        pdf_file = request.files['file']
        if pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            return render_template('result.html', text=printMCQ(text))
    return 'No file provided or invalid file format.'

if __name__ == '__main__':
    app.run(debug=True)
