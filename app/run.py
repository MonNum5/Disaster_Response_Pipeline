import json
import plotly
import pandas as pd
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import numpy as np

import nltk
from nltk.corpus import stopwords
import collections
nltk.download(['punkt','wordnet'])
nltk.download('stopwords')
stopword_list = set(stopwords.words('english'))


app = Flask(__name__)

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



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Added Plotly graphs

    # Distribution for each label
    label_names = list(df.columns[4:])
    number_per_category = [df[cat].sum() for cat in label_names]

    most_common_dict=pd.read_csv(open('../data/MostCommonWords.csv','rb'))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
                    x=label_names,
                    y=number_per_category
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
       
                'yaxis': {
                    'title': "Number of Datasets"
                },
                'xaxis': {
                    'title': "Name of Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=most_common_dict.index,
                    y=most_common_dict['count'],
                    text=most_common_dict['word'],
                    textposition='auto',
                )
            ],

            'layout': {
                'title': 'Most Common Words per Category',
       
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Name of Category"
                }
            }
        },
     
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()