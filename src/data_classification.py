import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

def reload_model():
    # 1. Recover the model
    filename = 'random_forest_model.joblib'
    loaded_rf_model = joblib.load(filename)

    print(f"Random Forest model loaded from {filename}")

    # 2. Use the loaded model for predictions
    # (Assuming you have new data X_new to predict on)
    # For demonstration, we'll use X_test from the original training data.
    predictions = loaded_rf_model.predict(X_test)
    print(f"Predictions using loaded model: {predictions[:5]}") # Print first 5 predictions
    print(f"Original true labels: {y_test[:5]}")

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


        # Initialize the Random Forest Classifier
        # | n_estimators is the number of trees in the forest
        # | random_state is used for reproducibility of results 
        # | You can change n_estimators to a higher value for better accuracy
        # | but it will take more time to train the model
        # | You can also use joblib to save the model after training
        # | joblib.dump(model, 'random_forest_model.pkl') to save the model
        # | model = joblib.load('random_forest_model.pkl') to load the model
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

        rf_model = RandomForestClassifier(n_estimators=10, random_state=30)

        # Train the model
        rf_model.fit(X_train, Y_train)
        '''
`       # Evaluate the model        
        # | You can use model.score(X_test, Y_test) to get the accuracy of the model
        # | You can also use model.predict(X_test) to predict the test data
        # | You can also use metrics.accuracy_score(Y_test, Y_prediction_test)
        # | to get the accuracy of the model
        # | You can also use metrics.confusion_matrix(Y_test, Y_prediction_test)
        # | to get the confusion matrix of the model
        # | You can also use metrics.classification_report(Y_test, Y_prediction_test)
        # | to get the classification report of the model
        # | You can also use metrics.roc_auc_score(Y_test, Y_prediction_test)
        # | to get the ROC AUC score of the model
        # | You can also use metrics.f1_score(Y_test, Y_prediction_test)
        # | to get the F1 score of the model
        # | You can also use metrics.precision_score(Y_test, Y_prediction_test)
        # | to get the precision score of the model
        # | You can also use metrics.recall_score(Y_test, Y_prediction_test)
        # | to get the recall score of the model
        # | You can also use metrics.log_loss(Y_test, Y_prediction_test)
        # | to get the log loss of the model
        # | You can also use metrics.balanced_accuracy_score(Y_test, Y_prediction_test
        # | to get the balanced accuracy score of the model
        # | You can also use metrics.roc_curve(Y_test, Y_prediction_test) 
        # | to get the ROC curve of the model
        # | You can also use metrics.auc(fpr, tpr) to get the
        # | area under the ROC curve of the model
        # | You can also use metrics.precision_recall_curve(Y_test, Y_prediction_test
        # | to get the precision-recall curve of the model
        # | You can also use metrics.average_precision_score(Y_test, Y_prediction_test)
        # | to get the average precision score of the model
        # | You can also use metrics.brier_score_loss(Y_test, Y_prediction_test)
        # | to get the Brier score loss of the model
        # | You can also use metrics.hamming_loss(Y_test, Y_prediction_test)
        # | to get the Hamming loss of the model
        # | You can also use metrics.jaccard_score(Y_test, Y_prediction_test)
        # | to get the Jaccard score of the model
        # | You can also use metrics.matthews_corrcoef(Y_test, Y_prediction_test
        # | to get the Matthews correlation coefficient of the model
        # | You can also use metrics.roc_auc_score(Y_test, Y_prediction_test)
        # | to get the ROC AUC score of the model
        # | You can also use metrics.fbeta_score(Y_test, Y_prediction_test, beta
        # | to get the F-beta score of the model
        # | You can also use metrics.cohen_kappa_score(Y_test, Y_prediction_test
        # | to get the Cohen's kappa score of the model
        # | You can also use metrics.mean_squared_error(Y_test, Y_prediction_test)
        # | to get the mean squared error of the model
        # | You can also use metrics.mean_absolute_error(Y_test, Y_prediction_test)
        # | to get the mean absolute error of the model
        # | You can also use metrics.explained_variance_score(Y_test, Y_prediction
        # | to get the explained variance score of the model
        # | You can also use metrics.r2_score(Y_test, Y_prediction_test)
        # | to get the R-squared score of the model
        # | You can also use metrics.mean_squared_log_error(Y_test, Y_prediction_test)
        # | to get the mean squared logarithmic error of the model
        # | You can also use metrics.davies_bouldin_score(Y_test, Y_prediction_test)
        # | to get the Davies-Bouldin score of the model
        # | You can also use metrics.silhouette_score(Y_test, Y_prediction_test)
        # | to get the silhouette score of the model
        # | You can also use metrics.calinski_harabasz_score(Y_test, Y
        # | prediction_test) to get the Calinski-Harabasz score of the model    
        '''
        Y_prediction_test = rf_model.predict(X_test)   
        # ... check prediction
        print("Accuracy = ", metrics.accuracy_score(Y_test, Y_prediction_test))        

        # --- 3. Save the Trained Model ---
        # Define the filename for your saved model
        model_filename = 'random_forest_model.joblib'

        # Save the model using joblib
        joblib.dump(rf_model, model_filename)

        print(f"Model saved successfully as {model_filename}")
    else:
        print(f"The file {data_file} doesn't exist!")


if __name__ == '__main__':
    random_forest("/home/manuel/Maestria/maestria-trainer/data/mix_data/data_frame1_1_data_frame2_1_ext.csv")