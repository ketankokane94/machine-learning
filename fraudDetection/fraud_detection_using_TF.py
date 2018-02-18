#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:10:59 2018

@author: ketankokane
"""

#fraud detection model on Synthetic transaction using TF's estimator API 

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    df = pd.read_csv('data2.csv')
    preprocess_data(df)



def preprocess_data(df):
    #drop irrelevant columns 
    df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)
    #normalize numerical columns
    columns_to_norm =['amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest']
    df[columns_to_norm]= df[columns_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
    create_feature_columns()


def create_feature_columns():
    amount = tf.feature_column.numeric_column('amount')
    oldbalanceOrg = tf.feature_column.numeric_column('oldbalanceOrg')
    newbalanceOrig = tf.feature_column.numeric_column('newbalanceOrig')
    oldbalanceDest = tf.feature_column.numeric_column('oldbalanceDest')
    newbalanceDest = tf.feature_column.numeric_column('newbalanceDest')
    #step = tf.feature_column.categorical_column_with_hash_bucket('step',hash_bucket_size=800)
    type = tf.feature_column.categorical_column_with_hash_bucket('type',hash_bucket_size=10)
    
    
def split_data_for_train_test(df):
    X= df.drop('isFraud',axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def train_model():
    pass

def save_model():
    pass

def load_model():
    pass

def evaluate_model():
    pass


read_data()

