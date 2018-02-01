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

def analyzer(data):
    clf = RandomForestClassifier()



def handle_missing_vals (df):
    del df['Age']
    del df['Cabin']
    return df

if __name__ == "__main__":
    test_df = pd.read_csv('test.csv')
    train_df = pd.read_csv('train.csv')

    print train_df['Survived']
    # remove columns with missing data
    train_df = handle_missing_vals(train_df)

    train_x = train_df.drop('Survived', axis=1)
    train_y = train_df['Survived']


    analyzer(train_df)



