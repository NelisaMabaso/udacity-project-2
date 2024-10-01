# Udacity Project 2: Disaster Presponce Pipeline
This project was done in partial completion to the Udacity Data Science Nano-Degree Program.
This project aimed to develop a machine learning pipeline using NLP techniques to categorize messages according the category for aid/assistance for Figure8.
Supervised machine learning techniques were used to predict based off of labelled twitter messages, the classification result.
A flask app was updated to demonstrate the use of this project.

## Instructions
*Please note the pickled model could not be uploaded due to file size constraints.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

Acknoledgements: 
Udacity for the coursework and the base templates of code
Figure8 for providing the messages and categories datasets
