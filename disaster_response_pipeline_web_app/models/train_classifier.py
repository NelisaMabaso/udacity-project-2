# import libraries
import sys
from sqlalchemy import create_engine
import pickle
import os

# Importing necessary libraries from nltk for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer

# Other necessary libraries for data manipulation and machine learning
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Importing machine learning and evaluation libraries
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
    Load data from SQLite database and split into feature (X) and target (y) variables.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        X (pd.Series): Messages data (features).
        y (pd.DataFrame): Categories data (target variables).
        category_names (list): List of category names for classification.
    """
    # Create database engine connection
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load data from SQL table into a pandas DataFrame
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    
    # Define feature and target variables
    X = df['message']
    y = df.iloc[:, 3:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Process and tokenize text data.

    Args:
        text (str): Input message text.

    Returns:
        tokens (list): List of processed and tokenized words.
    """
    # Define regex for URL detection and replace them with placeholders
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Initialize stopwords and lemmatizer
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # Clean the text by removing non-alphanumeric characters and converting to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # Tokenize the cleaned text and remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer class to check if a sentence starts with a verb.
    """

    def starting_verb(self, text):
        """
        Identify if the first word in the sentence is a verb.

        Args:
            text (str): Input text data.

        Returns:
            bool: True if the first word is a verb, False otherwise.
        """
        # Tokenize sentences from the text
        sentence_list = nltk.sent_tokenize(text)
        
        # For each sentence, check the POS tag of the first word
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """
        This function doesn't modify the input data, it just returns the transformer.
        """
        return self

    def transform(self, X):
        """
        Apply the starting verb check to a series of text data.

        Args:
            X (pd.Series): Input text data.

        Returns:
            pd.DataFrame: DataFrame containing boolean values indicating the presence of a starting verb.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model(clf=RandomForestClassifier()):
    """
    Build a machine learning pipeline with a multi-output classifier.

    Args:
        clf (estimator): Classifier model to be used. Default is RandomForestClassifier.

    Returns:
        model (Pipeline): Machine learning pipeline model.
    """
    # Create a pipeline that includes both text processing and classification
    model = Pipeline([
        ('features', FeatureUnion([  # Combining multiple feature processing steps
            ('text_transform', Pipeline([  # Text vectorization steps
                ('vect', CountVectorizer(tokenizer=tokenize)),  # Tokenize and vectorize
                ('tfidf', TfidfTransformer())  # Apply TF-IDF transformation
            ])),
            ('starting_verb', StartingVerbExtractor())  # Add custom transformer
        ])),
        ('clf', MultiOutputClassifier(clf))  # Multi-output classifier with chosen estimator
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the machine learning model using test data.

    Args:
        model (Pipeline): Trained model.
        X_test (pd.Series): Test features (messages).
        Y_test (pd.DataFrame): True labels for the test set.
        category_names (list): List of category names.

    Prints:
        Classification report for each category.
    """
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Generate and print the classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
        model: Trained model to be saved.
        model_filepath (str): The filepath where the pickle file will be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model to the specified filepath
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_filepath}")


def main():
    """
    Main function to load data, build the model, train it, evaluate, and save the model.
    """
    # Check for correct number of command-line arguments
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        # Load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Build and train model
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # Save the trained model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        # Error message if command-line arguments are not provided correctly
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
