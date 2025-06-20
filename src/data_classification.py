import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

def random_forest(working_dir, data_file):
    data_file = working_dir + "/" + data_file
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20) 

        
        # | n_estimators is the number of trees in the forest
        # | random_state is used for reproducibility of results 
        # | You can change n_estimators to a higher value for better accuracy
        # | but it will take more time to train the model        
        # | You can also use model.predict(X_test) to predict the test data
        # | and model.score(X_test, Y_test) to get the accuracy of the model
        # | You can also use model.feature_importances_ to get the importance of each feature
        # | You can also use model.get_params() to get the parameters of the model
        # | You can also use model.set_params() to set the parameters of the model
        # | You can also use model.n_features_in_ to get the number of features in the model
        # | You can also use model.n_classes_ to get the number of classes
        # | You can also use model.classes_ to get the classes of the model
        # | You can also use model.n_outputs_ to get the number of outputs
        # | You can also use model.estimators_ to get the estimators of the model
        # | You can also use model.estimators_features_ to get the features of the estimators
        # | You can also use model.estimators_samples_ to get the samples of the estimators
        # | You can also use model.estimators_weights_ to get the weights of the estimators
        # | You can also use model.estimators_n_outputs_ to get the number of outputs of the estimators
        # | You can also use model.estimators_n_features_in_ to get the number of features in the estimators        

        # Initialize the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=10, random_state=30)

        # Train the model
        model.fit(X_train, Y_train)
        
        Y_prediction_test = model.predict(X_test)           
        # ... check prediction
        accuracy =  metrics.accuracy_score(Y_test, Y_prediction_test)     
        report = metrics.classification_report(Y_test, Y_prediction_test)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)

        # Save the Trained Model        
        model_filename = working_dir + "/random_forest_model.joblib"
        joblib.dump(model, model_filename)

        print(f"Model saved successfully as {model_filename}")
    else:
        print(f"The file {data_file} doesn't exist!")

def run_prediction(working_dir, data_file, model):    
    if model:        
        data_file = working_dir + "/" + data_file
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            #print(df.head())

            # Prepare the Data:
            # Separate your data into features (X) and the target variable (y). Features are the independent 
            # variables used for prediction, and the target is the variable you aim to predict. You may also 
            # need to handle missing values or encode categorical variables if present.
            y = df['data']
            
            # ... define independent variables => all data except for the first one with title 'data'
            X = df.drop(labels=['data'], axis=1)

            predictions = model.predict(X)
            #print("Predictions: ", predictions)

            # ... check prediction
            accuracy =  metrics.accuracy_score(y, predictions)
            report = metrics.classification_report(y, predictions)

            print(f"Accuracy: {accuracy}")
            print("Classification Report:\n", report)
        else:
            print(f"The file {data_file} doesn't exist!")
    else:
        print(f"The model doesn't exist!")


def load_model(working_dir):
    model_file = working_dir + "/random_forest_model.joblib"
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print("Model loaded successfully.")
        return model
    else:
        print(f"The model file {model_file} doesn't exist!")
        return None


if __name__ == '__main__':
    # Creates the Random Forest model using the data from the specified directory and file
    #random_forest("/home/manuel/Maestria/maestria-trainer/data/mix_data_reduced","all_data_mixed_ext.csv")

    # Loads the Random Forest model from the specified directory and file and runs predictions
    #'''
    model = load_model("/home/manuel/Maestria/maestria-trainer/data/mix_data_reduced")
    if model:
        run_prediction("/home/manuel/Maestria/maestria-trainer/data/mix_data_reduced", "all_data_mixed_ext.csv", model)
    else:
        print("Model could not be loaded, predictions cannot be made.")
    #'''