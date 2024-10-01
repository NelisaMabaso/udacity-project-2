# import libraries
import sys
import zipfile
from sqlalchemy import create_engine
import pickle
import tempfile
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score


def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        tuple: Tuple containing:
            - X (pd.Series): Messages for classification.
            - y (pd.DataFrame): DataFrame containing the target variables (categories).
            - category_names (Index): Names of the categories.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:, 3:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize and clean text data.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        list: List of cleaned and lemmatized tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to identify whether a sentence starts with a verb.
    """
    
    def starting_verb(self, text):
        """
        Check if the first word of the text is a verb.

        Args:
            text (str): Input text to check.

        Returns:
            bool: True if the first word is a verb, False otherwise.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """
        Fit the transformer. This method does nothing but is required for compatibility.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            self: Returns an instance of the transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by applying the starting verb check.

        Args:
            X (pd.Series): Input series of text data.

        Returns:
            pd.DataFrame: DataFrame with boolean values indicating the presence of starting verbs.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model(clf=RandomForestClassifier()):
    """
    Build a machine learning pipeline with text and starting verb features.

    Args:
        clf (sklearn estimator): The classifier to use. Defaults to RandomForestClassifier.

    Returns:
        Pipeline: A scikit-learn pipeline that includes feature extraction and classification.
    """
    model = Pipeline([
        ('features', FeatureUnion([
            ('text_transform', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance and print classification metrics.

    Args:
        model: The trained model to evaluate.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): True labels for the test set.
        category_names (Index): Names of the categories for evaluation.
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the trained model as a compressed pickle file.

    Args:
        model: The model to save.
        model_filepath (str): Path where the model will be saved, should end with .pkl.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a temporary pickle file
        temp_pickle_path = os.path.join(tmpdirname, 'classifier.pkl')

        # Save the model to the temporary pickle file
        with open(temp_pickle_path, 'wb') as f:
            pickle.dump(model, f)

        # Create the zip file
        zip_filepath = model_filepath.replace('.pkl', '.zip')
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_pickle_path, arcname='classifier.pkl')

    print(f'Model saved and compressed as: {zip_filepath}')


def main():
    """
    Main function to load data, train the model, evaluate it, and save the trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the zip file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.zip')


if __name__ == '__main__':
    main()
