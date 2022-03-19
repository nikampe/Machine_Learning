######################################################
# 8,330: Machine Learning (MiQEF)
# Assignment 3: kNN Algorithm
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

def accuracy(data):
    n = len(data)
    correct = 0
    for i in range(n):
        if data.iloc[i,0] == data.iloc[i,1]:
            correct += 1
        else:
            continue
    acc = round((correct/n) * 100, 2)
    return acc
        
def data():
    df = pd.read_csv('Data/diabetes.csv')
    df = df[['Outcome', 'Glucose']]
    df.rename(columns = {'Outcome': 'y', 'Glucose': 'x'}, inplace = True)
    y = df['y']
    X = df['x']
    print("\n############ Raw Data ############\n")
    print(df)
    return y, X, df

class kNN_Algorithm():
    def __init__(self, y, X, df, n_splits):
        self.y = y
        self.X = X
        self.df = df
        self.n_splits = n_splits
    # Function to Split Train/Test Data for Cross-Validation
    def split(self):
        data = self.df
        kf_n = KFold(n_splits = self.n_splits, shuffle = False)
        split = list(kf_n.split(data))
        data_train = []
        data_test = []
        test_indices = {n: "" for n in range(1,len(split)+1)}
        for i in range(0, len(split)):
            train_index = data.iloc[split[i][0]]
            test_index = data.iloc[split[i][1]]
            data_train.append(train_index)
            data_test.append(test_index)
            test_indices[i+1] = str(test_index.index[0]) + "-" + str(test_index.index[-1])
        print("\n############ Test Indices ############\n")
        for i in range(0, len(split)):
            print(f'Split {i+1}: {test_indices[i+1]}')
        return data_train, data_test
    # Utility Function to Sort Values in a Data Frame Ascending
    def sort(self):
        data_train, data_test = self.split()
        for i in range(0, len(data_train)):
            data_train[i].sort_values(by = 'x', axis = 0, inplace = True)
            data_test[i].sort_values(by = 'x', axis = 0, inplace = True)
        return data_train, data_test
    # kNN Classification Algorithm based on Cross-Validation
    def classifier(self):
        data_train, data_test = self.sort()
        k_benchmark = math.floor(np.sqrt(len(data_train[0])))
        if k_benchmark % 2 == 0:
            k_benchmark = k_benchmark - 1
        k_arr = np.arange(3, k_benchmark * 3, 4)
        if k_benchmark not in k_arr:
            k_arr = np.append(k_arr, k_benchmark)
        accuracies_dict = {k: 0 for k in k_arr}
        y_preds = []
        for k in k_arr:
            accuracies = []
            for i in range(0, len(data_train)):
                y_train = data_train[i]['y'].to_numpy()
                X_train = data_train[i]['x'].to_numpy()
                X_test = data_test[i]['x'].to_numpy()
                y_pred = []
                for obs in X_test:
                    closest = min(X_train, key = lambda x: abs(x - obs))
                    closest_index = list(X_train).index(closest)
                    lower_bound = closest_index - math.floor(k/2)
                    upper_bound = closest_index + math.floor(k/2)
                    k_closest = y_train[lower_bound:upper_bound]
                    if np.mean(k_closest) >= 0.5:
                        y_pred.append(1.0)
                    else:
                        y_pred.append(0.0)
                y_preds.append(y_pred)
                data_pred = pd.DataFrame(columns = ['y_pred', 'y', 'x'])
                data_pred['y_pred'] = y_pred
                data_pred['y'] = data_test[i]['y'].to_numpy()
                data_pred['x'] = X_test
                acc = accuracy(data_pred)
                accuracies.append(acc)
            accuracies_dict[k] = round(np.mean(accuracies), 2)
        print("\n############ Classification Accuracies ############\n")
        for k in k_arr:
            print(f"k={k}: {accuracies_dict[k]}%")
        print("\n############ Best Classifier ############\n")
        k_optimal = max(accuracies_dict, key = accuracies_dict.get)
        print(f"Optimal k: {k_optimal}")
        print(f"Optimal Accuracy: {accuracies_dict[k_optimal]}%")
        print("-----")
        print(f"Benchmark k: {k_benchmark}")
        print(f"Behcmark Accuracy: {accuracies_dict[k_benchmark]}%")
        
if __name__ == '__main__':
    n_splits = 5
    y, X, df = data()
    kNN_Algorithm(y, X, df, n_splits).classifier()
    
    
    
    
    
    