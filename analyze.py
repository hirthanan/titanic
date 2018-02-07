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

def encode_embarked(df):
    for idx, row in df.iterrows():
        if row['Embarked'] == 'Q':
            df.set_value(idx, 'Embarked', 0)
        elif row['Embarked'] == 'S':
            df.set_value(idx, 'Embarked', 1)
        else:
            df.set_value(idx, 'Embarked', 2)
    return df

def handle_missing_data(df):
    age_median = df['Age'].median()
    fare_median = df['Fare'].median()
    df['Age'].fillna(age_median, inplace=True)
    df['Fare'].fillna(fare_median, inplace=True)
    return df

def clean_data (df):
    del df['Cabin']
    del df['Ticket']
    del df['Name']

    df = encode_gender(df)
    df = encode_embarked(df)
    df = handle_missing_data(df)
    return df

if __name__ == "__main__":
    test_df = pd.read_csv('test.csv')
    train_df = pd.read_csv('train.csv')

    # clean data and encode strings to numerical values

    test_df = clean_data(test_df)
    train_df = clean_data(train_df)

    test_x = test_df

    train_x = train_df.drop('Survived', axis=1)
    train_y = train_df['Survived']

    rforest_model = analyzer(train_x, train_y)
    predictions = rforest_model.predict(test_x)
    prediction = pd.DataFrame(range(892, 1310), columns=['PassengerId'])
    prediction['Survived'] = predictions
    prediction.to_csv('prediction.csv', index=False)
