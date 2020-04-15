import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    categories = df['categories'].str.split(';',expand=True)

    rename_dict=dict(zip(categories,categories.iloc[0].values))
    categories.rename(columns=rename_dict, inplace=True)
    for column in categories.columns:
    # set each value to be the last character of the string
        categories[column] = np.array(categories[column].str.split('-').tolist())[:,1]

    df.drop(columns='categories',inplace=True)

    df = pd.concat([df,categories],axis=1)

    return df

def clean_data(df):
    df.drop_duplicates()
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()