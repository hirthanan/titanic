import csv

def analyzer(data):
    for row in data:
        print row["PassengerId"]

def csv_reader(fileObj):
    reader = csv.DictReader(fileObj, delimiter=',')
    return reader

if __name__ == "__main__":
    ''' format train.csv data for python to interpret '''
    with open("train.csv") as f:
        data = csv_reader(f)
        analyzer(data)
