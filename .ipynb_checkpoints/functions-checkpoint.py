# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#impute-the-missing-data-and-score
# https://scikit-learn.org/stable/modules/impute.html

from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

import os
import pandas as pd
import numpy as np
from io import BytesIO
from io import TextIOWrapper
import zipfile
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')
sns.set()
pd.options.display.max_columns = None

def get_dataset(dictionary):
    
    """ https://scikit-learn.org/0.16/datasets/index.html """
    """ https://scikit-learn.org/stable/datasets/index.html """

    for values in dictionary.values():
        #key = pd.DataFrame.from_dict(dictionary.values)
        if np.isscalar(values):
            pass
        else:
            #print(pd.DataFrame.from_dict(values))
            feature_names = dictionary["feature_names"]
            data = pd.DataFrame(dictionary["data"], columns=feature_names)
            target = pd.DataFrame(dictionary["target"], columns=["TARGET"])
            output = pd.concat([data,target],axis=1)
        
        return output


#for dataset in [load_boston(), load_iris(), load_diabetes()]:
#print(get_dataset(dataset)[:5])

def get_current_working_directory():

        """
        :return:
        """
        
        current_path = os.getcwd()

        return current_path


def change_current_working_directory(directory):
    """
    :param directory:
    :return:
    """
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("\n" + "Directory Does Not Exists. Working Directory Have Not Been Changed." + "\n")

    current_path = str(os.getcwd())
    
    return current_path

def get_list_of_files_from_directory(directory):
    """
    :param directory:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        list_of_files.append(item)

    return list_of_files

def get_list_of_zip_files(directory):
    """
    :param directory:
    :return:
    """
    os.chdir(directory)
    zip_files = []

    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename.endswith(".zip"):
                zip_files.append(filename)

    return zip_files

def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def unzip_files(directory, output_directory, zip_file_name):
    """
    :param input_directory:
    :param output_directory:
    :return:
    """

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    print("Unpacked " + str(zip_file_name) + " to: " + str(output_directory) + "\n")
    
def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def count_unique_values(dataframe, column):
    count_unique = dataframe[str(column)].value_counts()
    count_null = pd.Series(dataframe[str(column)].isnull().sum(),index=["nan"])
    count_unique = count_unique.append(count_null, ignore_index=False,)
    
    return count_unique

def choose_imputer_and_visualise(dataframe, columns, target, imputer=None):
    """ 
    :SimpleImputer:
    :IterativeImputer:
    :KNNImputer:
    """
    if imputer == None:
        output = dataframe.fillna(0)
        
    elif imputer == SimpleImputer:
        SI = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        SI.fit(dataframe[columns])
        output = SI.transform(dataframe[columns])
        
    elif imputer ==  IterativeImputer:
        II = IterativeImputer(max_iter=10, random_state=0)
        II.fit(dataframe[columns])
        output = II.transform(dataframe[columns])
        
    elif imputer == KNNImputer:
        KNNI = KNNImputer(missing_values=np.nan, weights="distance", add_indicator=False)
        output = pd.DataFrame(KNNI.fit_transform(dataframe[columns]))
    else:
        output = "error"
        
    #print(output.isnull().sum())
    
    for column in range(len(columns)):
        sns.distplot(output[column], fit=norm);
        fig = plt.figure()
        res = stats.probplot(output[column], plot=plt)
        fig = plt.figure()
    
    for column in range(len(columns)):
        target_column = pd.DataFrame(dataframe.iloc[:,-1])
        test_output = pd.merge(target_column, output, left_index=True, right_index=True)
        sns.jointplot(x=column, y=target, data=test_output, kind='reg', marker="+", color="b")
        
    return output