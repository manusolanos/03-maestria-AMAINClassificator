import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def random_forest(data_file):
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        #print(df.head())

        # count the rows for the dependent variable
        #sizes = df.iloc[:,0].value_counts(sort=1)
        #print(sizes)


        #To access all columns except the first column, use usecols=cols[:-1]

        # Handle missing values removing from dataset
        # df = df.dropna()

        # ... define dependent variable
        Y = df['data'].values
        
        # ... define independent variables => all data except for the first one with title 'data'
        X = df.drop(labels=['data'], axis=1)


        # ... split data into train and test 
        # | 0.4 means usage of 40% randomly selected as test data 
        # | with random_state it keeps the same selection (no changes with different runs)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20) 

        model = RandomForestClassifier(n_estimators=10, random_state=30)
        model.fit(X_train, Y_train)

        Y_prediction_test = model.predict(X_test)
        # ... check prediction
        print("Accuracy = ", metrics.accuracy_score(Y_test, Y_prediction_test))        
    else:
        print(f"The file {data_file} doesn't exist!")


if __name__ == '__main__':
    random_forest("/home/manuel/Maestria/maestria-trainer/data/mix_data/data_frame1_1_data_frame2_1_ext.csv")