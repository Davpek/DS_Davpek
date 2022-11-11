# Project: Disaster Response Pipeline
# Autor: David Klapetek
# Date: 11.11.2022
# Description:
# Extract, Transform, and Load process for the message and category data.
# Read the dataset, clean the data, and then store it in a SQLite database

# imports
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(file):
    """ Loads data from csv file """
    data = pd.read_csv(file)
    return data

def transform_data(messages,categories):
    """Merge data for later model"""
    df = messages.merge(categories, on='id')

    """Exctract, transfrom and clean the category columns"""
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)

    categories['related'] = categories['related'].replace(2, 1)

    # drop the original categories column from `df`
    df=df.drop(columns=['categories'],axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def data_to_sql(df,db):
    """Save the clean dataset into an sqlite database"""
    engine = create_engine(f'sqlite:///{db}')
    df.to_sql('Message_Clean', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_file, categories_file, database = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_file, categories_file))
        messages = load_data(messages_file)
        categories = load_data(categories_file)

        print('Cleaning data...')
        df = transform_data(messages,categories)

        print('Saving data...\n    DATABASE: {}'.format(database))
        data_to_sql(df, database)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')

# run
if __name__ == '__main__':
    main()