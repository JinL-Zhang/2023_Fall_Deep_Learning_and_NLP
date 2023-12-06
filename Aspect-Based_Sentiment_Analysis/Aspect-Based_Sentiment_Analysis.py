# -*- coding: utf-8 -*-
"""
Aspect-Based Sentiment Analysis
Use a custom-trained Named Entity Recognition (NER) model stored in 'model-best' folder to identify the
dishes in the reviews (i.e. the aspects) and then use a pre-trained ABSA model
to analyze the sentiment expressed about those dishes.
"""

import spacy
import pandas as pd
import re
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
import warnings

# file path
extract_to_path = '.'
data_file = 'reviews.csv'
ner_path = "model-best"

# load the review data
data = pd.read_csv(data_file, sep='\t')
reviews = data['Review']

# load the pre-trained NER model
ner = spacy.load(ner_path)


# Define a helper function to help tagging the review
def tag_dishes_in_review(review, unique_dishes):
    '''
    Define a function to replace all the dish name with [B-ASP]dish_name[E-ASP]
    correctly, as there is potential overlap in entity names,
    such as 'tacos' and 'tacos al pastor'
    '''
    # Sort unique dishes by length in descending order
    sorted_dishes = sorted(unique_dishes, key=len, reverse=True)

    # Function to replace dish with tagged version
    def replace_dish(match):
        dish = match.group(0)
        return f"[B-ASP]{dish}[E-ASP]"

    # Replace each dish in the review with tagged version
    for dish in sorted_dishes:
        # Escape special regex characters in dish names
        escaped_dish = re.escape(dish)
        # Create a regex pattern to match the dish as a whole word, not tagged
        pattern = r'(?<!\[B-ASP\])\b' + escaped_dish + r'\b(?!\[E-ASP\])'
        # Replace in the review
        review = re.sub(pattern, replace_dish, review)
    return review


# Load the ABSA model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Define the function that accepts raw reviews as a list of string
# and outputs a dataframe
def absa(reviews):
    '''
    This function should take an input as a list of sentences
    Return a dataframe with the review id
    (i.e. the index of the review in the list),
    dish, corresponding sentiment score, and confidence.

    The loaded NER model is utilized in this function to
    locate the dishes in a given review.
    Then, add tags to the review that indicates the start and end of the dish.
    Example: "Best [B-ASP]tandoori chicken[E-ASP] I've ever had."

    NOTE: The reviews must not be tagged before
    '''
    warnings.filterwarnings('ignore')

    # Intialize a DataFrame with the results
    df = pd.DataFrame(columns=['review_id', 'dish', 'sentiment', 'confidence'])
    review_id = 0
    # Loop through the reviews and make predictions
    for review in reviews:
        # preprocess the review by adding [B-ASP] and [E-ASP] between a dish
        # Apply the NER model
        doc = ner(review)
        # Get the unique dish in each review
        dish_name = set(ent.text for ent in doc.ents if ent.label_ == "DISH")
        # Tag the identified dishes in the review
        tagged_review = tag_dishes_in_review(review, dish_name)
        for dish in dish_name:
            result = classifier(tagged_review, text_pair=dish)
            # Add a row using append
            new_row = {'review_id': review_id, 'dish': dish,
                       'sentiment': result[0]['label'],
                       'confidence': result[0]['score']}
            df = df.append(new_row, ignore_index=True)
        review_id += 1

    return df
