## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
### Required Libraries:
In order to run this project, install Python 3 or higher with the following Python libraries: sys, nltk, re, numpy, pandas, pickle, sklearn, sqlalchemy, json, plotly, flask, and joblib
### To Run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database `python DisasterResponse/data/process_data.py DisasterResponse/data/disaster_messages.csv DisasterResponse/data/disaster_categories.csv DisasterResponse/data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves `python DisasterResponse/models/train_classifier.py DisasterResponse/data/DisasterResponse.db DisasterResponse/models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python DisasterResponse/app/run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

This project aims to build an ETL pipeline that processes messages and category data from a CSV file and load them in to a SQLite database then use a machine learning pipeline to create and save a multi-output supervised learning model. This model will analyze disaster data from Figure Eight to assist in classifying disaster messages as they come in and send the nessages to the appropriate disaster relif agency.

## File Descriptions<a name="files"></a>
1. Web App Screenshots Folder: screen shots taken of the final web app
2. app Folder:

    - template Folder - html templates provided by Udacity for the web app
    - run.py - python file to create visuals of the processed data and run the web app
3. data Folder:

    - DisasterResponse.db - the output file from the process_data.py program
    - disaster_categories.csv - data provided by Udacity to categorize the disaster response messages
    - disaster_messages.csv - data provided by Udacity of the recorded disaster response messages
    - process_data.py - python script to intake the data, clean it, and output a single database of response messages and their respective categories
4. models Folder:

    - classifier.pkl - the ouput file from the train_classifier.py program
    - train_classifier.py - python script to read the database and use it to train a machine learning pipeline to categorize messages

## Results<a name="results"></a>

![File1](https://github.com/jadefreese/DisasterResponse/blob/main/Web%20App%20Screenshots/Classifying%20Message.JPG)
![File2](https://github.com/jadefreese/DisasterResponse/blob/main/Web%20App%20Screenshots/Home%20Page1.JPG)
![File3](https://github.com/jadefreese/DisasterResponse/blob/main/Web%20App%20Screenshots/Home%20Page2.JPG)
![File4](https://github.com/jadefreese/DisasterResponse/blob/main/Web%20App%20Screenshots/Home%20Page3.JPG)
![File5](https://github.com/jadefreese/DisasterResponse/blob/main/Web%20App%20Screenshots/Home%20Page4.JPG)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.


