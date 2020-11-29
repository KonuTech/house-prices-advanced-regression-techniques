from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits

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