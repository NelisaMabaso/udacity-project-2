# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets from two CSV files.

    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        df (pd.DataFrame): Merged dataframe of messages and categories.
    """
    # Load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting data types, and removing duplicates.

    Args:
        df (pd.DataFrame): Merged dataframe containing messages and categories.

    Returns:
        df (pd.DataFrame): Cleaned dataframe ready for analysis or storage.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Extract column names for the categories from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # Convert category values to 0 and 1 by taking the last character of each string
    for column in categories.columns:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # Drop the original 'categories' column from the dataframe
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenate the new `categories` dataframe with the original dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates from the dataframe
    df = df.drop_duplicates()

    # Drop the 'original' column (optional) and fill missing values with 0
    df = df.drop(columns=['original'])
    df = df.fillna(0)

    # Fix any potential issues with the 'related' column where values may be 2
    df['related'] = df['related'].apply(lambda a: 1 if a == 2 else a)
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe into an SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe containing messages and categories.
        database_filename (str): Filepath for the SQLite database to save the data.
    """
    # Create a SQLite engine and save the dataframe to the specified database
    engine = create_engine('sqlite:///' + database_filename)
    
    # Save the dataframe to the 'DisasterResponse' table
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') 


def main():
    """
    Main function to load, clean, and save data from CSV files to a SQLite database.

    The function expects three command-line arguments:
    - Path to the messages CSV file.
    - Path to the categories CSV file.
    - Path to the SQLite database where cleaned data will be saved.
    """
    if len(sys.argv) == 4:
        # Extract file paths from command-line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save cleaned data to SQLite database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Print usage instructions if the required arguments are not provided
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
