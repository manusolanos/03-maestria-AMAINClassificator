import argparse
import csv
import os
import pandas as pd
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score



def feature_extraction_all(feature_csv):
    features = []

    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        for line in islice(data, 0, None):
            if line[0] != 'file1':
                feature = [float(i) for i in line[2:]]
                features.append(feature)
    print('length:')
    print(len(features))
    return features


def obtain_dataset(clone_featureCSV, cloneSet=True):    
    Vectors = []
    Labels = []

    if os.path.exists(clone_featureCSV):
        features = feature_extraction_all(clone_featureCSV)

        if cloneSet:
            pass
        else:
            pass

        Vectors.extend(features)
        Labels.extend([(1 if cloneSet else 0) for i in range(len(features))])
    else:
        print(f"The file:({clone_featureCSV}) does not exist!")
    
    print('len of Vectors:')
    print(len(Vectors))
    print('len of Labels:')
    print(len(Labels))
    return Vectors, Labels


def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]


from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=64, random_state=0)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_randomforest.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


from sklearn.neighbors import KNeighborsClassifier
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_knn_1.pkl')
        y_pred = clf.predict(test_X)


        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_knn_3.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


from sklearn import tree
def decision_tree(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)

        # save model
        #joblib.dump(clf, 'clf_decision_tree.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]

############
############
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0
  
    # End value of upper half
    end_upper = row_number
  
    # Start value of lower half
    start_lower = row_number
  
    # End value of lower half
    end_lower = df.shape[0]
  
    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]
  
    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]
  
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]
  
    # Combine the two lists
    index_ = upper_half + lower_half
  
    # Update the index of the dataframe
    df.index = index_
  
    # Insert a row at the end
    df.loc[row_number] = row_value
   
    # Sort the index labels
    df = df.sort_index()
  
    # return the dataframe
    return df

def mix_dataframes(dataFrame1, dataFrame2, limit=None):
    index = 1
    
    if dataFrame1.shape[0] <= dataFrame2.shape[0]:
        for i, row in dataFrame1.iterrows():        
            dataFrame2 = Insert_row(index, dataFrame2, row)        
            index += 2

            if limit is not None and limit <= index:
                break

        # ... cut dataset
        return dataFrame2.truncate(after=index)
    else:
        for i, row in dataFrame2.iterrows():        
            dataFrame1 = Insert_row(index, dataFrame1, row)        
            index += 2

            if limit is not None and limit <= index:
                break
        # ... cut dataset
        return dataFrame1.truncate(after=index)

def example():
    d1 = {'col1':[1, 3, 5, 7, 9]}
    df1 = pd.DataFrame(data=d1)
    d2 = {'col2' : [2, 4, 6, 8, 10]}
    df2 = pd.DataFrame(data=d2)
    #print(df1, "\n", df2)

    
    df1.insert(0, "x", value=101010101010)
    #print(df1)
    df1.columns = range(df1.shape[1]) # removes column title
    #print(df1)

    df2.insert(0, "x", value=999999999999)
    #print(df2)
    df2.columns = range(df2.shape[1]) # removes column title
    #print(df2)

    print("Proceso simplificado:\n")
    # Iterating over rows
    
    '''
    index = 1
    #for row in df2.itertuples():        
    for i, row in df2.iterrows():
        #df1 = Insert_row(index, df1, [88888888888, 1])        
        df1 = Insert_row(index, df1, row)        
        index += 2'
    print(df1)
    '''
    print(mix_dataframes(df2, df1))

def mix_data(dir_path, data_frame1, data_frame2):
    cvs_file1 = dir_path + '/data_frame1.csv'
    cvs_file2 = dir_path + '/data_frame2.csv'

    # ... step 1: create csv files
    data_frame1.to_csv(cvs_file1, index=False)
    # release memory
    '''
    try:
        del data_frame1
    except:
        pass
    '''

    data_frame2.to_csv(cvs_file2, index=False)
    # release memory
    '''
    try:
        del data_frame2
    except:
        pass
    '''
    
    # ... combine files
    '''
    with open(cvs_file1, 'r', newline='') as csv_1, \
         open(cvs_file2, 'r', newline='') as csv_2, \
         open(dir_path + '/data_frame_mixed.csv', 'w', newline='') as csv_mixed:
        
        writer = csv.writer(csv_mixed)
        count = 0

        while True:            
            line_csv_1 = csv_1.readline()
            line_csv_2 = csv_2.readline()

            count += 1
            if not line_csv_1 or not line_csv_2 or count == 10:
                break
            
            writer.writerow(line_csv_1)
            writer.writerow(line_csv_2)
    '''

def prepare_data(dir_path, clone_csv, nonclone_csv):
    Vectors = []
    Labels = []

    if os.path.exists(dir_path):
        clone_csv_file = dir_path + "/" + clone_csv
        nonclone_csv_file = dir_path + "/" + nonclone_csv

        
        if os.path.exists(clone_csv_file) and os.path.exists(nonclone_csv_file):
            clone_df = pd.read_csv(clone_csv_file)
            #print(clone_df.head())
            nonclone_df = pd.read_csv(nonclone_csv_file)
            #print(nonclone_df.head())

            # ... removes the pair identifier and adds the label (1 = clone, 0 = nonclone)
            clone_df.drop(clone_df.columns[[0,1]], axis=1, inplace=True)            
            #print(clone_df.head())
            clone_df.insert(0, 1, 1) # Note the usage of the value of 1 in the second parameter that indicates column name, just to fulfill the first value in the first row
            #print(clone_df.head())
            nonclone_df.drop(nonclone_df.columns[[0,1]], axis=1, inplace=True)
            #print(clone_df.head())
            nonclone_df.insert(0, 0, 0) # Note the usage of the value of 0 in the second parameter that indicates column name, just to fulfill the first value in the first row
            #print(clone_df.head())

            '''
            # ... consume mucha memoria
            df = mix_dataframes(clone_df, nonclone_df, limit=10)
            print(df.head())
            '''
            mix_data(dir_path, clone_df, nonclone_df)

        else:
            print(f"One of the files :(clone={clone_csv_file}, nonclone={nonclone_csv_file}) does not exist!")

        #Vectors.extend(features)
        #Labels.extend([(1 if cloneSet else 0) for i in range(len(features))])
    else:
        print(f"The path:({dir_path}) does not exist!")
    
    print('len of Vectors:')
    print(len(Vectors))
    print('len of Labels:')
    print(len(Labels))
    return Vectors, Labels

if __name__ == '__main__':
    dir_path = '/Users/manuelsolano/Documents/Maestria/maestria-trainer/data'
    prepare_data(dir_path, "BCB_clone_ast.csv", "BCB_nonclone_ast.csv")
    #example()
