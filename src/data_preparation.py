import csv
import os
import pandas as pd
#import numpy as np

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

def mix_data_old(dir_path, data_frame1, data_frame2):
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

def chop_data(working_dir, csv_file, limit=5000):
    results_dir = working_dir + '/chop_data/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    working_file = working_dir + "/" + csv_file
    if os.path.exists(working_file):
        # creates dir that will contains file's chunks
        file_name = csv_file.split('.')[0]
        chunks_dir = results_dir + "/" + file_name + "_chunks"
        os.mkdir(chunks_dir)

        with open(working_file, 'r', newline='') as source_csv:
            reader = csv.reader(source_csv)

            chunk_counter = 0;
            row_counter = 0;
            chunk = []
            for row in reader:
                chunk.append(row)
                row_counter += 1

                if (row_counter >= limit):
                    with open(chunks_dir + '/' + file_name + "_" + str(chunk_counter) + ".csv", 'w', newline='') as destination_csv:                
                        writer = csv.writer(destination_csv)
                        writer.writerows(chunk)
                    
                    chunk_counter += 1
                    row_counter = 0
                    chunk = []

            # checks for a residual chunck, it means a chunk with size less than the limit
            if len(chunk) > 0: #if not chunk: # this has to works, because empty list is false
                with open(chunks_dir + '/' + file_name + "_" + str(chunk_counter) + ".csv", 'w', newline='') as destination_csv:                
                        writer = csv.writer(destination_csv)
                        writer.writerows(chunk)
    else:
        print(f"The file: ({working_file}) does not exist!")

    #Vectors.extend(features)
    #Labels.extend([(1 if cloneSet else 0) for i in range(len(features))])

def mix_data(working_dir, file_1, file_2):
    if os.path.exists(working_dir) and os.path.exists(file_1) and os.path.exists(file_2):
        results_dir = working_dir + '/mix_data/'
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        # ...
        results = pd.read_csv(file_1)         
        number_files_1 = len(results) + 1 # + 1 because data has no col title and the first row is considered as title

        results = pd.read_csv(file_2)
        number_files_2 = len(results) + 1 # + 1 because data has no col title and the first row is considered as title

        # usage the min between dir_1 and dir_2 as limit in the combination
        limit = number_files_1 if number_files_1 <= number_files_2 else number_files_2
        new_file_name = os.path.basename(file_1).split('.')[0] + "_" + os.path.basename(file_2).split('.')[0] + ".csv"
        print(f"Writing the new file {new_file_name} with the limit of {limit} and mix of {limit * 2} rows")
        
        with open(file_1, 'r', newline='') as csv_1, \
             open(file_2, 'r', newline='') as csv_2, \
             open(results_dir + '/' + new_file_name, 'w', newline='') as csv_mix:
        
            writer = csv.writer(csv_mix)            
            for i in range(limit):
                try:
                    line = csv_1.readline().split(',')
                    line[-1] = line[-1].strip()
                    writer.writerow(line)

                    line = csv_2.readline().split(',')
                    line[-1] = line[-1].strip()
                    writer.writerow(line)
                except:
                    print("An exception writing rows has been ocurred!")
        
    else:
        print("Some directory does not exist!")

def add_columns_title(data_file):
    if os.path.exists(data_file):        
        data_file_with_titles = os.path.dirname(data_file) + "/" + os.path.basename(data_file).split('.')[0] + "_ext.csv"

        with open(data_file, 'r', newline='') as csv_source, \
             open(data_file_with_titles, 'w', newline='') as csv_destination:
            
            writer = csv.writer(csv_destination)

            #df = pd.read_csv(data_file)
            #cols_counter = len(df.T[0])
            cols_counter = 229
            #print(f"The column counter is {cols_counter}")        

            # Create a range object
            cols_range = range(1, cols_counter)
            # Convert the range to a list      
            aux = list(cols_range)        
            cols = ['data']        
            cols.extend(aux)
            #print(cols)
            writer.writerow(cols)


            reader = csv.reader(csv_source)
            for row in reader:
                writer.writerow(row)
    else:
        print(f"The file {data_file} does not exist!")


if __name__ == '__main__':
    working_dir = '/Users/manuelsolano/Documents/Maestria/maestria-trainer/data'


    # Step 1: CHOP in small pieces
    '''
    chop_data(working_dir, "data_frame1.csv", limit=5000)
    chop_data(working_dir, "data_frame2.csv", limit=5000)
    #'''

    # Step 2: Mix Files
    '''
    for i in range(54):
        file1 = f"{working_dir}/chop_data/data_frame1_chunks/data_frame1_{(i)}.csv"
        file2 = f"{working_dir}/chop_data/data_frame2_chunks/data_frame2_{(i)}.csv"
        mix_data(working_dir,file1, file2)        
    #'''

    # Step 3: Complement format for pandas works (read files with titles)
    #add_columns_title(f"{working_dir}/mix_data/data_frame1_1_data_frame2_1.csv")
    '''
    for i in range(1,53):
        add_columns_title(f"{working_dir}/mix_data/data_frame1_{str(i)}_data_frame2_{str(i)}.csv")
    '''
