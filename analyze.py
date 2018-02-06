# comma seperated values library to parse titanic data file
import csv

# RandomForestClassifier used to predict survival using sample data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load numpy
import numpy as np

# Load pandas
import pandas as pd

# Set randomizer seed
np.random.seed(0)

def analyzer(train_x, train_y):
    clf = RandomForestClassifier()
    #print train_x
    clf.fit(train_x, train_y)
    return clf

def encode_gender(df):
    for idx, row in df.iterrows():
        if row['Sex'] == "male":
            df.set_value(idx, 'Sex', 0)
        elif row['Sex'] == "female":
            df.set_value(idx, 'Sex', 1)
    return df

def clean_data (df):
    del df['Cabin']
    del df['Ticket']
    del df['Name']

    df = encode_gender(df)
    print df.describe()
    return df

if __name__ == "__main__":
    test_df = pd.read_csv('test.csv')
    train_df = pd.read_csv('train.csv')

    # clean data and encode strings to numerical values

    # comment for now test_df = clean_data(test_df)
    train_df = clean_data(train_df)

    #print train_df

    test_x = test_df

    train_x = train_df.drop('Survived', axis=1)
    train_y = train_df['Survived']

    # trained_rforest = analyzer(train_x, train_y)
    #print trained_rforest
