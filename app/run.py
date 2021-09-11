import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as grph
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    '''
    FUNCTION: 
        clean the input text, simplify it, and tokenize it

    INPUTS:
        text - a string to clean and tokenize

    OUTPUTS
        clean_tokens - a list of tokenized words from text 
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath='DisasterResponse/data/DisasterResponse.db'
engine = create_engine('sqlite:///{}'.format(database_filepath))
df = pd.read_sql_table(database_filepath, engine)

# load model
model = joblib.load("DisasterResponse/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    '''
    FUNCTION: Create the app visuals
    '''
    
    #Plot 1 - Count of Messages by Genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = list(df.iloc[:,4:].columns)
    lengths_list = df['message'].str.split().str.len()
    
    #Plot 2 - Average Length of Messages by Genre
    len_count, len_div = np.histogram(lengths_list, range=(0, lengths_list.quantile(0.95)))
    
    #Plot 3 - Count of Messages by Category for Direct Messages
    dir_count = df.loc[df.genre == 'direct', categories].shape[0]
    dir_cat = df.loc[df.genre == 'direct', categories].sum().sort_values(ascending = False)
    dir_cat_count = dir_cat/dir_count
    dir_cat_names = list(dir_cat_count.index)
    
    #Plot 4 - Count of Messages by Category for News Messages
    news_count = df.loc[df.genre == 'news', categories].shape[0]
    news_cat = df.loc[df.genre == 'news', categories].sum().sort_values(ascending = False)
    news_cat_count = news_cat/news_count
    news_cat_names = list(news_cat_count.index)
    
    #Plot 5 - Count of Messages by Category for Social Messages
    soc_count = df.loc[df.genre == 'social', categories].shape[0]
    soc_cat = df.loc[df.genre == 'social', categories].sum().sort_values(ascending = False)
    soc_cat_count = soc_cat/soc_count
    soc_cat_names = list(soc_cat_count.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Plot 1 - Count of Messages by Genre
        {
            'data': [
                grph.Bar(
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
        #Plot 2 - Average Length of Messages by Genre
        {
            'data': [
                grph.Bar(
                    x=len_div,
                    y=len_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
        #Plot 3 - Count of Messages by Category for Direct Messages
        {
            'data': [
                grph.Bar(
                    x=dir_cat_names,
                    y=dir_cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Direct Messages by Category',
                'yaxis': {
                    'title': "% of Total Direct Messages"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        #Plot 4 - Count of Messages by Category for News Messages
        {
            'data': [
                grph.Bar(
                    x=news_cat_names,
                    y=news_cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of News Messages by Category',
                'yaxis': {
                    'title': "% of Total News Messages"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        #Plot 5 - Count of Messages by Category for Social Messages
        {
            'data': [
                grph.Bar(
                    x=soc_cat_names,
                    y=soc_cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Social Messages by Category',
                'yaxis': {
                    'title': "% of Total Social Messages"
                },
                'xaxis': {
                    'title': "Message Category"
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