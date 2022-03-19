######################################################
# 8,330: Machine Learning (MiQEF)
# Assignment 4: AdaBoost Algorithm
# Niklas Leander Kampe | 16-611-618
######################################################

# Utility Libraries
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

def data():
    df = pd.read_csv('Data/diabetes.csv')
    df.rename(columns = {'Outcome': 'y'}, inplace = True)
    y = df['y'].to_numpy()
    X = df.loc[:, df.columns != 'y'].to_numpy()
    print("\n############ Raw Data ############\n")
    print(df)
    return y, X, df



if __name__ == '__main__':
    y, X, df = data()