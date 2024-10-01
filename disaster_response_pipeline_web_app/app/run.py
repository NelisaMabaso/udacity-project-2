import json
import plotly
import pandas as pd
import zipfile
import os
import tempfile

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin  # Add this import

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize input text.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of cleaned and lemmatized tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to identify whether a sentence starts with a verb.
    """
    
    def starting_verb(self, text):
        """
        Check if the first word of the text is a verb.

        Args:
            text (str): The input text to check.

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

# Load data from SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

zip_filepath = '../models/classifier.zip'
model = None

try:
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        # Check if the classifier.pkl file exists in the zip
        if 'classifier.pkl' not in zip_ref.namelist():
            raise FileNotFoundError("classifier.pkl not found in the zip file")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_ref.extract('classifier.pkl', tmpdirname)
            model_path = os.path.join(tmpdirname, 'classifier.pkl')
            model = joblib.load(model_path)

except FileNotFoundError as e:
    print(f"Error: {e}")

except zipfile.BadZipFile:
    print("Error: The file is not a zip file or is corrupted")
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")

if model is None:
    print("Failed to load the model. The application may not function correctly.")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the index page with visualizations of the disaster response data.

    Returns:
        str: Rendered HTML for the index page with visualizations.
    """
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Creating a custom visual for category distribution
    category_counts = df.iloc[:, 4:].sum() 
    category_names = list(category_counts.index)
    
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name='Message Genres',
                    marker=dict(color='blue')
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    name='Message Categories',
                    marker=dict(color='orange')
                )
            ],
            'layout': {
                'title': 'Distribution of Messages by Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                } 
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handle user query and display model results.

    Returns:
        str: Rendered HTML for the results page with user query and classification results.
    """
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Start the Flask application.

    This function runs the Flask app with debugging enabled.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
