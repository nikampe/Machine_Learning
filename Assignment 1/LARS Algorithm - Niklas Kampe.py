######################################################
# 8,330: Machine Learning (MiQEF)
# Assignment 1: LARS Algorithm
# Niklas Leander Kampe | 16-611-618
######################################################

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

def data():
    X, y = make_regression(n_samples = 50, n_features = 5, noise = 0.1)
    # Normalization
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    y = (y - np.mean(y)) / np.std(y)
    # Overview
    print("\n############ Raw Data ############\n")
    data = pd.DataFrame(columns = ['y', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5'])
    data['y'] = y
    data[['x_1', 'x_2', 'x_3', 'x_4', 'x_5']] = X
    print(data.head(n=10))
    return X, y

class LARS_Algorithm():
    
    def __init__(self, X, y):
        # Instance of the Underlying Data Set
        self.X = X
        self.y = y
    
    # Cholesky Factorization
    def cholinsert(self, R, x, X):
        # Support/Utility function
        diag_k = np.dot(x.T,x)
        if R.shape == (0,0):
            R = np.array([[np.sqrt(diag_k)]])
        else:
            col_k = np.dot(x.T,X)
            R_k = np.linalg.solve(R,col_k)
            R_kk = np.sqrt(diag_k - np.dot(R_k.T,R_k))
            R = np.r_[np.c_[R,R_k],np.c_[np.zeros((1,R.shape[0])),R_kk]]
        
        return R
    
    def model(self):
        # Get Data
        X, y = self.X, self.y
        
        # Find Number of Observations (n) and Features/Predictors (p)
        n,p = X.shape
        
        # Initial State of mu_hat (Regressed rediction of y)
        mu_hat = np.zeros(n)
        
        # Initial State of Active and Inactive Set
        active_set = []
        inactive_set = list(range(p))
    
        # Initial State of Coefficients (beta) and Correlations (corr)
        beta = np.zeros((p+1,p))
        corr = np.zeros((p+1,p))
        
        # initial cholesky decomposition of the gram matrix
        # since the active set is empty this is the empty matrix
        R = np.zeros((0,0))
    
        # Looping through all Features/Predictors
        for k in range(p):
            print(f"\n############ Run {k+1} ############\n")
            
            # Vector of Current Correlations: c^ = X' * (y - mu^_A) --> (Hastie, 2003, p.7)
            c_hat = np.dot(X.T, y - mu_hat)
            # Add Current Correlation to Correlation Array
            corr[k,:] = c_hat
            
            # Find Feature/Predictor with Highest Absolute Correlation: C^= max_j{|c^_j|} --> (Hastie, 2003, p.7)
            j_max = inactive_set[np.argmax(abs(c_hat[inactive_set]))]
            print(f"Predictor with Highest Correlation in Inactive Set: x_{j_max+1}")
            C_hat = c_hat[j_max]
            print(f"Highest Correlation in Inactive Set: {round(C_hat,2)}")
            
            # Add Feature/Predictor with Highest Absolute Correlation to Active Set: A = {j: |c^_j| = C^} --> (Hastie, 2003, p.7)
            R = self.cholinsert(R, X[:,j_max], X[:,active_set])
            active_set.append(j_max)
            print(f"Updated Active Set: x_{active_set+np.ones(len(active_set))}")
            inactive_set.remove(j_max)
            print(f"Updated Inactive Set: x_{inactive_set+np.ones(len(inactive_set))}")
            
            # Find the Sign of the Current Correlations: s_j = sign{c^_j} for j∈A --> (Hastie, 2003, p.7)
            s = np.sign(c_hat[active_set])
            print(f"Signs of Correlations in Active Set (1=Positive, 0=Negative): {s}")
            s = s.reshape((1, len(active_set)))
            
            # Vector of Active Features with Strictly Positive Correlation: X_A = (...s_j*x_j...) --> (Hastie, 2003, p.7)
            X_A = np.copy(X[:, active_set] * s)

            # Find Equiangular Vector
            ## G_A = X'_A * X_A with X_A = (...s_j * x_j...) for j∈A --> (Hastie, 2003, p.7)
            G_A = X_A.T @ X_A
            G_A_inv = np.linalg.inv(G_A)
            ## A_A = (1'_A * G^(-1)_A * 1_A)^(1/2) --> (Hastie, 2003, p.7)
            one = np.ones((len(active_set), 1))
            A_A = (1. / np.sqrt(one.T @ G_A_inv @ one)).flatten()[0]
            
            # Equiangluar Vector: u_A = X_A * w_A with w_a = A_A * G_A^(-1) * 1_A --> (Hastie, 2003, p.7)
            w_A = A_A * G_A_inv @ one
            u_A = X_A @ w_A
            
            """
            # Find the Sign of the Current Correlations: s_j = sign{c^_j} for j∈A --> (Hastie, 2003, p.7)
            s = np.sign(c_hat[active_set])
            print(f"Signs of Correlations in Active Set (1=Positive, 0=Negative): {s}")
            s = s.reshape(len(s),1)
            
            # Find Equiangular Vector
            ## G_A = X'_A * X_A with X_A = (...s_j * x_j...) for j∈A --> (Hastie, 2003, p.7)
            G_A = np.linalg.solve(R,np.linalg.solve(R.T, s))
            ## A_A = (1'_A * G^(-1)_A * 1_A)^(1/2) --> (Hastie, 2003, p.7)
            A_A = 1/np.sqrt(sum(G_A * s))
            # Equiangluar Vector: u_A = X_A * w_A with w_a = A_A * G_A^(-1) * 1_A --> (Hastie, 2003, p.7)
            w_A = A_A * G_A
            u_A = np.dot(X[:,active_set], w_A).reshape(-1)
            """

            if k == p:
                # Full Least Angle Solutions
                gamma_hat = C_hat / A_A
            else:
                # Inner Product Vector: a = X' * u_A --> (Hastie, 2003, p.7)
                a = np.dot(X.T,u_A)
                a = a.reshape((len(a),))
                
                # Optimal Gamma: gamma^ = min(+)_(j∈A) {(C-c_j)/(A_A-a_j), (C+c_j)/(A_A+a_j)} --> (Hastie, 2003, p.7)
                tmp = np.r_[(C_hat - c_hat[inactive_set])/(A_A - a[inactive_set]), (C_hat + c_hat[inactive_set])/(A_A + a[inactive_set])]
                gamma_hat = min(np.r_[tmp[tmp > 0], np.array([C_hat/A_A]).reshape(-1)])
            
            # mu_hat_A+ = mu_hat_A+gamma_tilde_A*u_A with A+ = A-{j} ---> (Hastie, 2003, p.7,11)
            mu_hat = mu_hat + gamma_hat * u_A.flatten()

            # Add Coefficient of New Feature/Predictor to Coefficient Vector
            if beta.shape[0] < k:
                beta = np.c_[beta, np.zeros((beta.shape[0],))]
            beta[k+1,active_set] = beta[k,active_set] + gamma_hat * w_A.T.reshape(-1)
        
        return beta
    
if __name__ == "__main__":
    X, y = data()
    p = X.shape[1]
    beta = LARS_Algorithm(X,y).model()
    print(f"\n############ Final Results ############\n")
    print('Coefficients Evolving over Time:\n', beta)
    print('\nFinal Coefficients:\n', f'x_1: {round(beta[p][0],4)}, x_2: {round(beta[p][1],4)}, x_3: {round(beta[p][2],4)}, x_4: {round(beta[p][3],4)}, x_5: {round(beta[p][4],4)}')
    
    
    
    
    