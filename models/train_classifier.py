import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
import numpy as np
import pandas as pd
import pickle
import sklearn
from sqlalchemy import create_engine

from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def load_data(database_filepath):

    '''
    FUNCTION: 
        Load database file containing the cleaned data

    INPUTS:
        database_filepath - database file containing the cleaned data

    OUTPUTS
        X - The messages from the database
        y - the categories the messages fall in marked 0 or 1
        y.columns - the names of the categories

    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    y = df.iloc[:,4:]
    return X, y, y.columns


def tokenize(text):

    '''
    FUNCTION: 
        clean the input text, simplify it, and tokenize it

    INPUTS:
        text - a string to clean and tokenize

    OUTPUTS
        clean_tokens - a list of tokenized words from text 
    '''
    
    #remove punctuation and convert to lowercase
    message = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    #tokenize
    tokens = word_tokenize(message)
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def gscv_scores(y_test, y_pred):

    '''
    Function: calculate the f-1 score average of all classes to use in 
              GridSearch evaluation
    Inputs:
        -y_test: outputs of test data set from train_test_split()
        -y_pred: pipeline predicted output based on predicted x from 
                 train_test_split() 
    Outputs:
        -average f-1 score of all classes to determine if a parameter improved 
         predictibility
    '''
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    report = pd.DataFrame()

    for col in y_test.columns:
        class_rep = classification_report(y_test.loc[:,col], y_pred.loc[:,col], 
                                          output_dict=True, zero_division=0)
        scoring_df = pd.DataFrame.from_dict(class_rep)

        #print(scoring_df)

        #drop un-needed columns and rows
        scoring_df = scoring_df[['macro avg']]
        scoring_df.drop(index='support', inplace=True)
        scoring_df = scoring_df.transpose()
        report = report.append(scoring_df, ignore_index=True)

    report.index = y_test.columns
    return report['f1-score'].mean()


def build_model():

    '''
    FUNCTION: create and return a pipeline
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    #comment different parameters to decrease run time
    parameters = {
        #'vect__ngram_range': ((1,1), (1,2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__estimator__n_estimators': [10, 25, 50],
        #'clf__estimator__min_samples_split': [2, 4, 6],
        'clf__estimator__bootstrap': (True, False)
    }

    scorer = make_scorer(gscv_scores)

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3, verbose=4)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    '''
    FUNCTION: 
        Evaluate the effectiveness of the pipeline in modeling the data

    INPUTS:
        model - pipeline fit to the data
        X_test - test set of messages from the train_test_split
        Y_test - test set of categories from the train_test_split
        category_names - names of the categories the messages can fit in

    OUTPUTS
        report - the precision, recall, and f-1 scores for each category
        scores - the average precision, recal, and f-1 scores for the data set
    '''

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)

    report = pd.DataFrame()

    for col in Y_test.columns:
        
        class_rep = classification_report(Y_test.loc[:,col], 
                                          y_pred.loc[:,col], 
                                          output_dict=True, 
                                          zero_division=0)
        scoring_df = pd.DataFrame.from_dict(class_rep)

        scoring_df = scoring_df[['macro avg']]
        scoring_df.drop(index='support', inplace=True)
        scoring_df = scoring_df.transpose()
        report = report.append(scoring_df, ignore_index=True)

    report.index = category_names
    scores = {'mean precision score':report['precision'].mean(),
              'mean recall score':report['recall'].mean(),
              'mean f-1 score':report['f1-score'].mean()}
    
    print(report)
    print(scores)


def save_model(model, model_filepath):

    '''
    FUNCTION: 
        save the model as a pickle file

    INPUTS:
        model - the pipeline trained to the data
        model_filepath - where to save the file

    OUTPUTS
        None
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


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