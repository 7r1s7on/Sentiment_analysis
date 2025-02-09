"""
Script loads and prepares the data, runs the training, and saves the model.
"""

import os
import sys
import logging
import warnings
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score 
from sklearn.linear_model import LogisticRegression     
from sklearn.model_selection import train_test_split    

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

#downloads nltk packages 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#configures warnings
warnings.filterwarnings("ignore") 

#configures logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the script...")

#defining ROOT directory, and appending it to the sys.path
# so that python know which file should be included within the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(ROOT_DIR))

#defining data and model directories
from utils import get_root_dir
DATA_PATH = get_root_dir('data')
MODEL_PATH = get_root_dir('outputs/models')

#loading dataset
logging.info(f"Loading 'Polar Movie Reviews' dataset...")
df = pd.read_csv(os.path.join(DATA_PATH,'raw/train.csv'))

#preprocessing data
logging.info(f"1. Data preprocessing\n-----------------------------------------------\n")

logging.info(f"a. Lower casing...")
#lower casing
df['review'] = df['review'].str.lower() 

logging.info(f"b. Removing unnecessary characters (HTML tags, URLs, numbers, etc)..." )
#removing urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df['review'] = df['review'].apply(lambda text: remove_urls(text))

#removing html tags (they appear so often in text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
df['review'] = df['review'].apply(lambda text: remove_html(text))    

#removing non-alpha-numeric chars
def remove_non_alpha_numeric(text):
    return re.sub('[^a-zA-Z]', ' ', text)
df['review'] = df['review'].apply(lambda text: remove_non_alpha_numeric(text))  

logging.info(f"c. Tokenizing...")
#tokenization
def tokenize(text):
    return word_tokenize(text)
df['review_tokenized'] = df['review'].apply(tokenize)

logging.info(f"d. Lemmatizing... (takes some time)")
#lemmatization
lemmatizer = WordNetLemmatizer()
wordnet_map = {'N':wordnet.NOUN, 'V':wordnet.VERB, 'J':wordnet.ADJ, 'R':wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return ' '.join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df['lemma_review'] = df['review'].apply(lambda text: lemmatize_words(text))
df['lemma_review_tokenized'] = df['lemma_review'].apply(tokenize)

#clearing small length words
def clean_small_length(token):
    return [i for i in token if len(i)>2]
df['cleaned_review'] = df['lemma_review_tokenized'].apply(clean_small_length)

logging.info(f"Converting back to string...")
#converting back to string
def convert_to_string(token):
    return ' '.join(token)
df['cleaned_review'] = df['cleaned_review'].apply(convert_to_string)

logging.info(f"Filtering and removing stopwords, frequent and small words...")
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    additional_words_to_keep = ['not', 'never', 'nor']
    return ' '.join([word for word in str(text).split() if (word not in STOPWORDS) or (word in additional_words_to_keep)])
df['cleaned_review'] = df['cleaned_review'].apply(lambda text: remove_stopwords(text))

words_del = 'movie film one make would'.split()
def remove_freq_words(text):
    return ' '.join([i for i in text.split() if i not in words_del])
df['cleaned_review'] = df['cleaned_review'].apply(lambda text: remove_freq_words(text))

#saving cleaned data to "data/processed" dir
logging.info(f'Saving data to "data/processed" path...')

def save_clean_data():
    '''saves cleaned data to 'data/processed' directory'''
    clean_data_path = os.path.join(DATA_PATH, 'processed')
    if not os.path.exists(clean_data_path):
        os.makedirs(clean_data_path)
        clean_train_data_path = os.path.join(clean_data_path, 'clean_train.csv')
        df.to_csv(clean_train_data_path, index=False)
        logging.info(f'Cleaned train data saved successfully!\n-----------------------------------------------\n')

def train_model():
    '''perfoms word embedding, trains the model and evaluates it'''
    logging.info(f'2. Training model\n-----------------------------------------------\n')
    logging.info(f'Encoding labels...')
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

    logging.info(f'Splitting data into train and test parts...')
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'],
        df['sentiment'],
        test_size=0.1,
        shuffle=True,
        random_state=42
        )

    logging.info(f'Word embedding...')
    unigram_vec = TfidfVectorizer(max_features=7000)
    X_train_unigram = unigram_vec.fit_transform(X_train).toarray()
    X_test_unigram = unigram_vec.transform(X_test).toarray()

    logging.info(f'Model training...')
    model = LogisticRegression(penalty='l2', random_state=42)
    model.fit(X_train_unigram, y_train)
    predictions = model.predict(X_test_unigram)
    log_reg_acc = accuracy_score(y_test, predictions)
    logging.info(f"Model Accuracy: {log_reg_acc * 100:.2f}%")
    logging.info(f"Precision: {precision_score(y_test,predictions) * 100:.2f}%")
    logging.info(f"Recall: {recall_score(y_test,predictions) * 100:.2f}%")
    logging.info(f"F1 score: {f1_score(y_test,predictions) * 100:.2f}%")
    return model

def save_trained_model(model):
    '''saves the model in 'output/models' directory'''
    logging.info("Saving the model...")
    model_name = os.environ.get('MODEL_NAME', 'final_model.joblib')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    model_path = os.path.join(MODEL_PATH, model_name)
    joblib.dump(model, model_path)
    logging.info(f'"{model_name}" saved successfully!\n-----------------------------------------------\n')

def main():
    save_clean_data()
    model = train_model()
    save_trained_model(model)

if __name__ == "__main__":
    main()
    