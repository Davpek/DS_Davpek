# Project: Disaster Response Pipeline
# Autor: David Klapetek
# Date: 11.11.2022
# Description:
# Split the data into a training set and a test set. Then creates a machine learning pipeline that uses NLTK
# as well as scikit-learn's Pipeline and GridSearchCV to output a final model

import sys
import sys
import nltk
import pickle
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(db):
    engine = create_engine(f'sqlite:///{db}')
    df = pd.read_sql_table('Message_Clean', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.iloc[:, 4:].columns)
    return X,Y,category_names

def tokenize(text):
    """Tokenize text data"""
    # tokenize text
    tokens = word_tokenize(text)

    # remove stop words
    stopwords_ = stopwords.words("english")
    tokens = [word for word in tokens if tokens not in stopwords_]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """Create model pipeline"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__criterion': ["gini", "entropy"]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test)
    for index, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ Export the final model as a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database))
        X, Y, category_names = load_data(database)
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()