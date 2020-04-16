import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
nltk.download(['punkt','wordnet'])
nltk.download('stopwords')
stopword_list = set(stopwords.words('english'))
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    con = engine.connect()
    df=pd.read_sql('messages',con)
    X=df['message'].values
    Y=df[df.columns[4:]].values.astype(int)
    return X, Y, df.columns[4:]


def tokenize(text):
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    text=nltk.word_tokenize(text)
    lemmatizer= nltk.stem.WordNetLemmatizer()
    clean_tokens=[lemmatizer.lemmatize(i).strip() for i in text if i not in stopword_list]
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=0)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(i, col)
        print(classification_report(y_pred[:,i],Y_test[:,i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()