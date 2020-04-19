import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
import collections
nltk.download(['punkt','wordnet'])
nltk.download('stopwords')
stopword_list = set(stopwords.words('english'))

def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: File path to csv file with messages
        categories_filepath: File path to csv file with categories
    output:
        df: Pandas dataframe with preprocessed and merged datasets
    '''
    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id', how='outer')

    categories = df['categories'].str.split(';',expand=True)

    rename_dict=dict(zip(categories,categories.iloc[0].values))
    categories.rename(columns=rename_dict, inplace=True)
    for column in categories.columns:
    # set each value to be the last character of the string
        categories[column] = np.array(categories[column].str.split('-').tolist())[:,1]

    df.drop(columns='categories', axis=1, inplace=True)

    df = pd.concat([df,categories.astype(int)],axis=1)

    return df

def clean_data(df):
    '''
    input:
        df: Pandas dataframe with merged data from messages and categories
    output:
        df: Pandas dataframe removed of dublicates
    '''
    df['related-2']=df['related-1']*0
    df['related-2'][df['related-1']==2]=1
    df['related-1'][df['related-1']==2]=0
    df.drop(columns=['child_alone-0'],inplace=True)
    df.drop_duplicates()

    return df
    


def save_data(df, database_filename):
    '''
    input:
        df: Cleaned pandas dataframe
    output:
        database_filename: Save file path of cleaned data set
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)


def tokenize(text):
    '''
    input:
        text: Text of message to be tokenized
    output:
        clean_tokens: Preprocessed text data
    '''
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    text=nltk.word_tokenize(text)
    lemmatizer= nltk.stem.WordNetLemmatizer()
    clean_tokens=[lemmatizer.lemmatize(i).strip() for i in text if i not in stopword_list]
    return clean_tokens

def get_most_common_words_of_category(df):
    '''
    input:
        df: Pandas dataframe with messages and categories
    output:
        most_common_df: Pandas dataframe containing most common word per category and corresponding count
    '''
    label_names = list(df.columns[4:])
    most_common_df=pd.DataFrame([])
    for col in label_names:
        match_df=df[df[col]==1]
        messages=match_df['message'].values
        cleaned_messages=[]
        for mes in messages:
            cleaned_messages = cleaned_messages + tokenize(mes)
        word_counter = collections.Counter(cleaned_messages)
        most_common_word = max(word_counter, key=word_counter.get)
        most_common_df.at[col,'word']=most_common_word
        most_common_df.at[col,'count']=word_counter[most_common_word]

    return(most_common_df)

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

        print('Saving csv with most common words')
        most_common_dict=get_most_common_words_of_category(df)
        most_common_dict.to_csv('MostCommonWords.csv')


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