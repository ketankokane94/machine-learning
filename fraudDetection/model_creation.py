import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter

output_file_path            = 'fraudDetection'
trained_model_pickle_name   = 'model_v1.pk'

def save_model(clf):
    #save the trained model
    import dill as pickle
    #filename = 'model_v1.pk'
    with open(os.path.join(output_file_path, trained_model_pickle_name), 'wb') as file:
        pickle.dump(clf, file)
        


def load_model():
    with open(os.path.join(output_file_path, trained_model_pickle_name) ,'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def drop_columns(df):
    #check if the columsn to be dropped exists or no 
    for _ in ['nameDest','nameOrig','isFlaggedFraud']:
        if _ in df.columns:           
            df.drop(_,inplace=True,axis=1)
        
      
def encode_cat_data(df):
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
   
def predict():
    data = pd.read_json()
    drop_columns(data)
    encode_cat_data(data)
    classfication_model = load_model()
    classfication_model.predict(data)

def evaluate_model(clf, X_test,y_test):
    y_pred = clf.predict(X_test)
    return np.mean(y_pred==y_test)
    
def train():
    df = pd.read_csv('data2.csv')
    drop_columns(df)
    encode_cat_data(df)
    X = df.drop('isFraud',axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    training_set_length = len(X_train)
    testing_set_length= len(X_test)
    training_label_distribution = dict(Counter(y_train))
    testing_label_distribution = dict(Counter(y_test))
    model_report = {}
    model_report.update({'training_set_length':training_set_length})
    model_report.update({'testing_set_length':testing_set_length})
    model_report.update({'training_label_distribution':training_label_distribution})
    model_report.update({'testing_label_distribution':testing_label_distribution})
    clf = SGDClassifier(max_iter=3)   
    clf.fit(X_train,y_train)
    save_model(clf)
    accuracy = 0
    accuracy = evaluate_model(clf,X_test,y_test)
    model_report.update({'acuracy':accuracy})
    print(json.dumps(model_report))
    
    
    
    
train()