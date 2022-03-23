######################################################
# 8,330: Machine Learning (MiQEF)
# Assignment 4: AdaBoost Algorithm
# Niklas Leander Kampe | 16-611-618
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data():
    df = pd.read_csv('Data/titanic.csv')
    print("\n############ Raw Data ############\n")
    print(df)
    return df

def data_preprocessing():
    df = data()
    for i in df.index:
        # Recode Target Variable
        df.loc[i, 'Survived'] = -1 if df.loc[i, 'Survived'] == 0 else 1
        # Recode Gender Covariate
        df.loc[i, 'Sex'] = 0 if df.loc[i, 'Sex'] == 'male' else 1
    # Drop Name Covariate
    df.drop(columns = 'Name', inplace = True)
    # Extract Target Vector and Covariate matrix
    y = df['Survived'].to_numpy()
    X = df.loc[:, df.columns != 'Survived'].to_numpy()
    return y, X

class StumpClassifier():
    def __init__(self):
        self.polarity = 1
        self.feature = None
        self.threshold = None
        self.alpha = None
    # Stump Classifier Prediction
    def predict(self, X):
        n_samples = X.shape[0]
        feature = X[:, self.feature]
        # Initialize Predictions Vector of Ones
        predictions = np.ones(n_samples)
        # Update Predictions Vector
        if self.polarity == 1:
            predictions[feature < self.threshold] = -1
        else:
            predictions[feature > self.threshold] = -1
        return predictions

class AdaBoost():
    def accuracy(self, y, y_pred):
        n = len(y)
        correct = 0
        # Compute Ratio of Correct Classification Predictions
        for i in range(n):
            if y[i] == y_pred[i]:
                correct += 1
            else:
                continue
        acc = round((correct/n) * 100, 2)
        return acc
    def fit(self, M):
        y, X = data_preprocessing()
        n_samples, n_features = X.shape
        # Initialize on first classifier: w_i = 1/n
        w = np.full(n_samples, (1 / n_samples))
        # Initialize Vector of Classifiers and Errors
        clfs = []
        errors = []
        # Iterate through classifiers
        for _ in range(M):
            # Initialize the Stump Classifier
            clf = StumpClassifier()
            # Initialize the minimum error for optimization
            min_error = float('inf')
            # Find Best Threshold and Feature + Updating the Stump Classifier
            for i in range(n_features):
                feature = X[:, i]
                # Initialize Thresholds Vector
                thresholds = np.unique(feature)
                for threshold in thresholds:
                    # Predict with Polarity 1
                    p = 1
                    # Initialize and Update Predictions
                    y_pred = np.ones(n_samples)
                    y_pred[feature < threshold] = -1
                    # Compute Error: err_j = sum [w_i * L(Y_i, C_j(X_i))] / sum[w_i]
                    error = (sum(w * (np.not_equal(y, y_pred)).astype(int)))/sum(w)
                    # Invert Weak Classifiers that Perform Worse than 0.5 (random)
                    if error > 1:
                        error = 1 - error
                        p = -1
                    # Update Best Configuration of Classifier
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature = i
                        min_error = error
            # Compute Alpha: alpha_j = log((1-err_j)/err_j)
            clf.alpha = np.log((1.0 - min_error) / (min_error))
            # Calculate Predictions
            y_pred = clf.predict(X)
            # Updating Weights from Second Classfier on: w_i = exp(alpha_j * L(Y_i,C_j(X_i)))
            w = w * np.exp(clf.alpha * (np.not_equal(y, y_pred)).astype(int))
            # Save Classifier
            clfs.append(clf)
            errors.append(min_error)
            return clfs, errors
    def predict(self):
        y, X = data_preprocessing()
        M_arr = [5, 10, 20, 50, 100]
        accs = pd.DataFrame(index = range(len(M_arr)), columns = ['M', 'Accuracy', 'Error'])
        for M, i in zip(M_arr, range(len(M_arr))):
            # Fit Stump Classifiers
            clfs, errors = self.fit(M)
            # Estimate Final Predictions: sign(sum[alpha_j * C_j(x)])
            clf_preds = [clf.alpha * clf.predict(X) for clf in clfs]
            y_pred = np.sign(np.sum(clf_preds, axis = 0))
            acc = self.accuracy(y, y_pred)
            # Append Accuracy Results in Dependence of M
            accs.loc[i, 'M'] = M
            accs.loc[i, 'Accuracy'] = acc
            accs.loc[i, 'Error'] = errors[0]
        print("\n############ Prediction Accuracy ############\n")
        print(accs)
        # Error Plot in Dependence of M
        plt.plot(accs['M'], accs['Error'], color = 'red')
        plt.title("Error in Dependence of M", size = 14)
        plt.xlabel('M', size = 12)
        plt.ylabel('Error', size = 12)
        plt.show()
        return y_pred
    
if __name__ == '__main__':
    AdaBoost().predict()