######################################################
# 8,330: Machine Learning (MiQEF)
# Assignment 2: Bitcoin Classifier
# Niklas Leander Kampe | 16-611-618
######################################################

# Utility Libraries
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
pd.set_option('display.max_columns', None)

# Data API
import yfinance as yf

# Utility Functions
def hist_plot(data, bins):
    for i,j in zip(data.columns, range(0, len(bins))):
        plt.hist(data[i].to_numpy(), bins = bins[j], color = 'grey')
        plt.title(f'Distribution of {i}', size = 14)
        plt.axvline(data[i].mean(), color = 'red', linestyle = 'dashed', linewidth = 2)
        min_ylim, max_ylim = plt.ylim()
        min_xlim, max_xlim = plt.xlim()
        plt.text(max_xlim * 0.4, max_ylim * 0.9, f'Mean: {round(data[i].mean(), 4)}', color = 'red')
        plt.text(max_xlim * 0.4, max_ylim * 0.825, f'Var: {round(data[i].var(), 4)}', color = 'k')
        plt.text(max_xlim * 0.4, max_ylim * 0.75, f'Skew: {round(data[i].skew(), 4)}', color = 'k')
        plt.text(max_xlim * 0.4, max_ylim * 0.675, f'Kurt: {round(data[i].kurt(), 4)}', color = 'k')
        plt.xticks(rotation = 'vertical')
        plt.xlabel(f'{i}', size = 12)
        plt.ylabel('Count', size = 12)
        plt.show()

def scatter_plot(data, color_categorical):
    colors = {0: 'blue', 1: 'red'}
    for combo in combinations(data.columns, 2):
        plt.scatter(data[combo[0]], data[combo[1]], color = color_categorical.map(colors))
        plt.title(f'{combo[0]} vs. {combo[1]}', size = 14)
        plt.xticks(rotation = 'vertical')
        plt.xlabel(f'{combo[0]}', size = 12)
        plt.ylabel(f'{combo[1]}', size = 12)
        plt.show()

def line_plot(data_train, data_test):
    plt.plot(data_train, color = 'red')
    plt.plot(data_test, color = 'green')
    plt.title(f'Train vs. Test Data', size = 14)
    plt.xticks(rotation = 'vertical')
    plt.xlabel(f'Time', size = 12)
    plt.ylabel(f'Price', size = 12)
    plt.show()
    
def maximum(a):
    if a >= 0:
        return a
    else:
        return 0

# Data Object
class Data():
    def __init__(self):
        self.start_date_asset = '2019-10-30'
        self.start_date_feature = '2020-01-01'
        self.end_date = '2021-12-31'
        self.interval = '1d'
        self.ticker_asset = 'BTC-USD'
        self.ticker_feature = ['ES=F', 'GC=F', 'CL=F']
        self.api = 'https://api.alternative.me/fng/?limit=0'
        
    def data_asset(self):
        data_y = yf.Ticker(self.ticker_asset)
        data_y = data_y.history(interval = self.interval, start = self.start_date_asset, end = self.end_date)
        data_y = data_y[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_y = data_y['Close']
        return data_y
    
    def data_features(self):
        data_y = self.data_asset()
        data_X = pd.DataFrame()
        # X1-X3: S&P500, Gold & Crude Oil - Daily Prices/Returns
        x_financial_markets = pd.DataFrame(columns = ['S&P 500', 'Gold', 'Oil'])
        for i,j in zip(self.ticker_feature[0:3], range(0,3)):
            x = yf.Ticker(i)
            x = x.history(interval = self.interval, start = self.start_date_feature, end = self.end_date)
            x_financial_markets.iloc[:,j] = x['Close'].to_numpy()
        x_financial_markets.index = x.index
        data_X = x_financial_markets
        # X4: MACD - Moving Average Convergence/Divergence (Technical Indicator)
        ema_12 = data_y.ewm(span = 12, adjust = False, min_periods = 12).mean()
        ema_26 = data_y.ewm(span = 26, adjust = False, min_periods = 26).mean()
        macd = ema_12 - ema_26
        data_X['MACD'] = data_X.index.map(macd)
        # X5: Corporate Bond - Daily Yields/Yield Changes
        bond_yields = pd.read_csv("Data/BAMLH0A0HYM2.csv", index_col = "DATE")
        bond_yields.index = pd.to_datetime(bond_yields.index)
        bond_yields['BAMLH0A0HYM2'] = pd.to_numeric(bond_yields['BAMLH0A0HYM2'], errors = 'coerce')
        for i in data_X.index:
            data_X.loc[i, 'Bond Yields'] = bond_yields.loc[i, 'BAMLH0A0HYM2']
        # X6: Fear and Greed Index (Sentiment Indicator)
        response = requests.get(self.api)
        response_json = response.json()['data']
        response_df = pd.DataFrame()
        for i in range(0, len(response_json)):
            response_df.loc[response_json[i]['timestamp'], 'Sentiment'] = response_json[i]['value']
        response_df.index = pd.to_datetime(response_df.index, unit='s')
        for i in data_X.index:
            data_X.loc[i, 'Sentiment'] = int(response_df.loc[i, 'Sentiment'])
        return data_X
    
    def data_preprocessing(self):
        y = self.data_asset()
        X = self.data_features()
        data = pd.DataFrame()
        # Convert Prices to Returns
        y = y.pct_change()
        for col in X.columns:
            X[col] = X[col].pct_change()
        for i,j in zip(X.index, range(0, len(X.index))):
            if y.loc[i] < 0:
                data.loc[i, 'BTC'] = 0
            else:
                data.loc[i, 'BTC'] = 1
            for k,l in zip(X.columns, range(0, len(X.columns))):
                data.loc[i, k] = X.loc[i, k]
        # Drop Implausible Values
        data = data.loc[np.abs(data['S&P 500']) <= 1]
        data = data.loc[np.abs(data['Gold']) <= 1]
        data = data.loc[np.abs(data['Oil']) <= 1]
        data = data.loc[np.abs(data['MACD']) <= 1]
        data = data.loc[np.abs(data['Sentiment']) <= 2]
        data[['S&P 500', 'Gold', 'Oil', 'MACD', 'Sentiment', 'Bond Yields']] = data[['S&P 500', 'Gold', 'Oil', 'MACD', 'Sentiment', 'Bond Yields']].shift(1)
        # Replace Infinite Values with N/A
        data.replace([np.inf, -np.inf], np.nan, inplace = True)
        # Drop N/A values
        data.dropna(inplace = True)
        # Rename Columns according to Date Shift
        data.rename(columns = {'S&P 500': 'S&P 500 (-1d)', 'Gold': 'Gold (-1d)', 'Oil': 'Oil (-1d)', 'MACD': 'MACD (-1d)', 'Sentiment': 'Sentiment (-1d)', 'Bond Yields': 'Bond Yields (-1d)'}, inplace = True)
        # Train Test Split
        len_train = math.floor(len(data) * 0.8) #0.8
        y_train = data.iloc[:len_train,:]['BTC']
        X_train = data.iloc[:len_train,:][['S&P 500 (-1d)', 'Gold (-1d)', 'Oil (-1d)', 'MACD (-1d)', 'Sentiment (-1d)', 'Bond Yields (-1d)']]
        y_test = data.iloc[len_train:,:]['BTC']
        X_test = data.iloc[len_train:,:][['S&P 500 (-1d)', 'Gold (-1d)', 'Oil (-1d)', 'MACD (-1d)', 'Sentiment (-1d)', 'Bond Yields (-1d)']]
        # Visuzalization
        data_train = pd.concat([y_train, X_train], axis = 1).reindex(y_train.index)
        data_test = pd.concat([y_test, X_test], axis = 1).reindex(y_test.index)
        hist_plot(X_train, bins = [50,50,50,50,50])
        scatter_plot(X_train, y_train)
        # Summary
        print("\n", "################## Training Data Set ##################", "\n")
        print(round(data_train, 4))
        print("\n", "################## Testing Data Set ##################", "\n")
        print(round(data_test, 4))
        return y_train, X_train, y_test, X_test

# Model Object
class Classifier():
    def __init__(self, y_train, X_train, y_test, X_test, y_raw):
        self.y_train = y_train
        self.X_train = X_train
        self.y_test = y_test
        self.X_test = X_test
        self.y_raw = y_raw
        self.initial_cash_balance = 1000000
        self.initial_btc_balance = 0
    def model(self):
        model = GaussianNB()
        return model
    def model_fit(self):
        model_lda = LinearDiscriminantAnalysis()
        model_lda.fit(self.X_train, self.y_train)
        model_qda = QuadraticDiscriminantAnalysis()
        model_qda.fit(self.X_train, self.y_train)
        return model_lda, model_qda
    def model_predict(self):
        model_lda, model_qda = self.model_fit()
        models = [model_lda, model_qda]
        model_names = ['LDA', 'QDA']
        accuracies = []
        print("\n", "################## In-Sample (IS) Performance ##################", "\n")
        for model, model_name in zip(models, model_names):
            y_pred_is = model.predict(self.X_train)
            accuracy_is = round(((y_train == y_pred_is).sum() / y_train.shape[0])*100, 2)
            print(f"Classification Success Rate ({model_name}): ", f'{accuracy_is}%')
        print("\n", "################## Out-of-Sample (OOS) Performance ##################", "\n")
        for model, model_name, i in zip(models, model_names, range(0,2)):
            y_pred_oos = model.predict(self.X_test)
            accuracy_oos = round(((y_test == y_pred_oos).sum() / y_test.shape[0])*100, 2)
            accuracies.append(accuracy_oos)
            print(f"Classification Success Rate ({model_name}): ", f'{accuracy_oos}%')
            CM = confusion_matrix(y_test, y_pred_oos)
            disp = ConfusionMatrixDisplay(confusion_matrix = CM, display_labels = model.classes_)
            disp.plot()
        best_model = models[accuracies.index(max(accuracies))]
        return best_model
    def model_backtest(self):
        # Short Selling Restricted; Partial Buys/Sells Restricted; Final BTC Balance must be Zero
        cash_balance = self.initial_cash_balance
        cash_balances = []
        btc_balance = self.initial_btc_balance
        btc_balances = []
        model = self.model_predict()
        predictions = model.predict(self.X_test)
        prices = self.y_raw[self.y_raw.index.isin(self.y_test.index)]
        buy_transactions = 0
        sell_transactions = 0
        for price, i in zip(prices.to_numpy(), range(0, len(prices))):
            date = prices.index[i]
            if (date == prices.index[-1]) & (btc_balance > 0):
                cash_purchase = btc_balance * price
                cash_balance += cash_purchase
                btc_balance = 0
                sell_transactions += 1
            elif (predictions[i] == 1) & (cash_balance > 0):
                btc_purchase = cash_balance / price
                btc_balance += btc_purchase
                cash_balance = 0
                buy_transactions += 1
            elif (predictions[i] == 0) & (btc_balance > 0):
                cash_purchase = btc_balance * price
                cash_balance += cash_purchase
                btc_balance = 0
                sell_transactions += 1
            else:
                continue
            cash_balances.append(cash_balance)
            btc_balances.append(btc_balance)
        print("\n", "################## Model Backtesting ##################", "\n")
        print(f"Initial Cash Balance: ${self.initial_cash_balance}")
        print(f"Initial BTC Balance: {self.initial_btc_balance}")
        print(f"Buy Transactions: {buy_transactions}")
        print(f"Sell Transactions: {sell_transactions}")
        print(f"Final Cash Balance: ${round(cash_balance, 2)}")
        print(f"Final BTC Balance: {btc_balance}")
        print(f"Gross Return (in $): ${round(cash_balance - self.initial_cash_balance, 2)}")
        print(f"Net Return (in %): {round(((cash_balance / self.initial_cash_balance) - 1) * 100, 2)}%")
        print(f"Net Return (in %) of (Not) Holding BTC over Testing Period: {round(maximum(((prices[-1] / prices[0]) - 1) * 100), 2)}%")
        print(f"Excess Return (in %) against (Not) Holding BTC over Testing Period: {round(((cash_balance / self.initial_cash_balance) - 1) * 100, 2) - round(maximum(((prices[-1] / prices[0]) - 1) * 100), 2)}%")
                
if __name__ == "__main__":
    y_raw = Data().data_asset()
    y_train, X_train, y_test, X_test = Data().data_preprocessing()
    Classifier(y_train, X_train, y_test, X_test, y_raw).model_backtest()
    
    
    
    
    