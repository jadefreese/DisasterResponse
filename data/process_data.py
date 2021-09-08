import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    FUNCTION: 
        Load csv files containing the messages and categories

    INPUTS:
        messages_filepath - csv file containing messages
        categories_filepath - csv file containing categories

    OUTPUTS
        df - data frame of the merged data sets of messages and categories

    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    
    return df


def clean_data(df):

    '''
    FUNCTION: 
        clean the data in the give dataframe to remove duplicates

    INPUTS:
        df - data frame imported from the load_data function

    OUTPUTS
        df - data frame of the cleaned input data frame

    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # use the first row of the categories dataframe to extract a list of new 
    # column names for categories
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x[:-2] , row))
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    

    # replace categories column in df with new category columns
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df.related.replace(2,1,inplace=True)
    df.drop_duplicates('message', inplace=True)
    df.reset_index()

    return df

def save_data(df, database_filename):

    '''
    FUNCTION: 
        save the data fram as an SQL Database

    INPUTS:
        df - data frame of cleaned data
        database_filename - the file name/location of where to save the database

    OUTPUTS
        SQL Database
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')  


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