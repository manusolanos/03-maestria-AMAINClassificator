import csv
import os
import pandas as pd
#import numpy as np
import shutil

def prepare_data(dir_path, clone_csv, non_clone_csv):
    if os.path.exists(dir_path):
        clone_csv_file = dir_path + "/" + clone_csv
        non_clone_csv_file = dir_path + "/" + non_clone_csv

        if os.path.exists(clone_csv_file):
            # Open the CSV file explicitly
            with open(clone_csv_file, "r") as f:
                # Read the CSV file into a DataFrame using the file object
                clone_df = pd.read_csv(f)
                #print(clone_df.head())
                
                # Step A.1 : removes the pair identifiers positionated in the first two columns
                clone_df.drop(clone_df.columns[[0,1]], axis=1, inplace=True)            
                
                # Step A.2 : adds the label (1 = clone)
                # Note the usage of the value of 1 in the second parameter that indicates column name, just to fulfill the first value in the first row
                clone_df.insert(0, 1, 1) 
                
                # Step A.3 : save the clone dataframe with the label
                cvs_file_clone_pair_labeled = dir_path + f'/{clone_csv.split('.')[0]}_labeled.csv'
                clone_df.to_csv(cvs_file_clone_pair_labeled, index=False)
        else:
            print(f"The file :(clone={clone_csv_file}) does not exist!")  

        if os.path.exists(non_clone_csv_file):
            # Open the CSV file explicitly
            with open(non_clone_csv_file, "r") as f:
                # Read the CSV file into a DataFrame using the file object
                non_clone_df = pd.read_csv(f)
                #print(nonclone_df.head())
        
                # Step B.1 : removes the pair identifiers positionated in the first two columns
                non_clone_df.drop(non_clone_df.columns[[0,1]], axis=1, inplace=True)
                
                # Step B.2 : adds the label (0 = non_clone)
                # Note the usage of the value of 0 in the second parameter that indicates column name, just to fulfill the first value in the first row
                non_clone_df.insert(0, 0, 0) 
                
                # Step B.3 : save the non_clone dataframe with the label
                cvs_file_non_clone_pair_labeled = dir_path + f'/{non_clone_csv.split('.')[0]}_labeled.csv'
                non_clone_df.to_csv(cvs_file_non_clone_pair_labeled, index=False)
        else:
            print(f"The file :(non_clone={non_clone_csv_file}) does not exist!")  
    else:
        print(f"The path:({dir_path}) does not exist!")

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
        print(f"The file {new_file_name} has been created in {results_dir}")
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

def remove_dir_and_contents(path_to_dir):
  """
  Removes a directory and all its contents.

  Args:
    path_to_dir: The path to the directory to remove.
  """
  try:
    shutil.rmtree(path_to_dir)
  except OSError as e:
    print(f"Error deleting {path_to_dir}: {e}")


def append_files_in_dir(working_dir, dir_name, output_file):
    if os.path.exists(working_dir + "/" + dir_name):
        # remove the previous results file
        results_file = working_dir + "/" + dir_name + f"/{output_file}.csv"
        if os.path.exists(results_file):
            print(f"Removing the previous results file: {results_file}")
            os.remove(results_file)

        # append all files in the directory
        files = os.listdir(working_dir + "/" + dir_name)
        if len(files) > 0:            
            with open(results_file, 'w', newline='') as csv_results:
                writer = csv.writer(csv_results)
                for file in files:
                    with open(working_dir + "/" + dir_name + "/" + file, 'r', newline='') as csv_file:
                        reader = csv.reader(csv_file)
                        for row in reader:
                            writer.writerow(row)
            print(f"All files have been appended in {results_file}")
        else:
            print(f"The directory {dir_name} is empty!")
    else:
        print(f"The directory {dir_name} does not exist!")


if __name__ == '__main__':
    working_dir = '/home/manuel/Maestria/maestria-trainer/data'
    clone_file_name = 'BCB_clone_gast'
    non_clone_file_name = 'BCB_nonclone_gast'


    # Step 1: Prepare data (remove columns and insert labels with clone and non-clone)
    '''
    prepare_data(working_dir, f"{clone_file_name}.csv", f"{non_clone_file_name}.csv")
    #'''

    # Step 2: CHOP in small pieces
    '''
    chop_data(working_dir, f"{clone_file_name}_labeled.csv", limit=5000)
    chop_data(working_dir, f"{non_clone_file_name}_labeled.csv", limit=5000)
    #'''

    # Step 3: Mix Files
    '''
    for i in range(53):
        file_clones = f"{working_dir}/chop_data/{clone_file_name}_labeled_chunks/{clone_file_name}_labeled_{(i+1)}.csv"
        file_non_clones = f"{working_dir}/chop_data/{non_clone_file_name}_labeled_chunks/{non_clone_file_name}_labeled_{(i+1)}.csv"
        mix_data(working_dir,file_clones, file_non_clones)        
    #'''

    # Step 4: Complement format for pandas works (read files with titles)
    '''
    # Step 4.1: Merge all files
    append_files_in_dir(working_dir, "mix_data_reduced_gast", output_file="all_data_mixed")
    # Step 4.2: Add columns titles just to the first file
    add_columns_title(f"{working_dir}/mix_data_reduced_gast/all_data_mixed.csv")    
    #'''
