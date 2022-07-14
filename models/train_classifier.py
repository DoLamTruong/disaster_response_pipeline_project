import sys
import pandas as pd
import numpy as np
import sqlalchemy
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

import nltk
nltk.download(['punkt', 'wordnet','omw-1.4'])




def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Args:
    database_filepath: path of sqlite database
    
    Returns:
    X: Features
    Y: Target
    """
    
    engine = sqlalchemy.create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql_table("disaster_data", engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    
    Args:
    text: Text to be tokenized
    
    Returns:
    tokens: tokens of input text
    """
    # tokenize text
    tokens = word_tokenize(text)
    # clean token
    clean_tokens = [WordNetLemmatizer().lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Make a pipeline for classify and tune model using GridSearchCV.
    
    Returns:
    pileline_Model: pipeline model
    """    
    pipeline = Pipeline([
    ('vecter_text', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {'clf__estimator__n_estimators' : [50, 100]}
    
    pileline_Model = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return pileline_Model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model and return report. 
    
    Args:
    model: ski-kit learn trained model  
    X_test: feature in test dataset
    Y_test: labels for test dataset
    category_names: category name to predict
    
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Save model as a pickle file.

    Args:
    model: ski-kit learn trained model
    model_filepath: path to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath) 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        category_names = Y.columns.values
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