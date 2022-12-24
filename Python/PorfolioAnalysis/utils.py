import numpy as np
from scipy import optimize
import pandas_datareader.data as web

class portfolio:
    """
    Portfolio class defines portfolios
    """
    def __init__(self, R, W, C):
        """
        Initializer
        R: array of stock's expected returns
        W: stock weights in the portfolio
        C: stock's covariance matrix    
        """
        self.R = R
        self.C = C
        self.W = W
        self.mean = np.dot(self.R, self.W) # Portfolio expected return
        self.var = np.dot(np.dot(self.W, self.C), self.W) # Portfolio expected variance
        
    def sharpeRatio(self, rf = 0.015):
        """
        Portfolio's Sharpe ratio
        """
        return (self.mean - rf) / np.sqrt(self.var)
    
    def getMinVarWeights(self):
        """
        Weights of the minimum variance portfolio found by solving 
        the matrix equation A z = b where (details in portfolio_class.ipynb)        
         - A is a matrix of the form: A11 = 2C, A12 = 1, A21 = 1.T, A22 = 0
         - z is the column vector (w, lambda)
         - b is the column vector (0, 1)
        Inputs:
         - R: individual asset returns
         - C: covariance of the assets
        Returns:
         - wmin: portfolio weights
         - rmin: portfolio return
         - vmin: portfolio variance        
        """

        Sigma = self.C.as_matrix() # covariance matrix        
        n_assets = Sigma.shape[0]
        
        A11 = 2*Sigma
        A12 = np.ones([n_assets])
        A21 = np.ones([n_assets])
        A22 = 0
        
        row1 = np.c_[A11, A12]
        row2 = np.hstack([A21, A22])
        row2.shape = (1, row2.shape[0])
        
        A = np.r_[row1, row2]
        b = np.r_[np.zeros([n_assets]), 1]

        # minimal weights
        wmin = np.linalg.solve(A, b)[0:n_assets]
        
        # return
        rmin = np.dot(self.R, wmin)

        # vmin
        vmin = np.dot(np.dot(wmin, Sigma), wmin)
        
        return wmin, rmin, vmin
        
    def getEfficientWeights(self, r0 = 0.15):
        """
        Weights of the efficient portfolio for a given expected return r0
        Details in portfolio_class.ipynb:
         - A is a matrix of the form:
             A11 = 2C,      A12 = mu, A13 = 1_(nx1)
             A21 = mu.T,    A22 = 0,  A23 = 0
             A31 = 1_(1xn), A22 = 0,  A23 = 0
         - z is the column vector (w, lambda_1, lambda_2)
         - b is the column vector (0_(nx1), r0, 1)
        Inputs:
         - R: individual asset returns
         - C: covariance of the assets
        Returns:
         - weff: portfolio weights
         - reff: portfolio return (=r0)
         - veff: portfolio variance        
        """
        
        Sigma = self.C.as_matrix() # covariance matrix        
        n = Sigma.shape[0]
        
        A11 = 2*Sigma
        A12 = self.R
        A13 = np.ones([n])
        
        row1 = np.c_[A11, A12, A13]
        
        A21 = self.R
        A22 = 0
        A23 = 0
        
        row2 = np.hstack([A21, A22, A23])
        row2.shape = (1, row2.shape[0])

        A31 = np.ones([n])
        A32 = 0
        A33 = 0

        row3 = np.r_[A31, A32, A33]
        row3.shape = (1, row3.shape[0])

        A = np.r_[row1, row2, row3]
        b = np.r_[np.zeros([n]), r0, 1]

        # efficient weights
        weff = np.linalg.solve(A, b)[0:n]
        
        # return
        reff = np.dot(self.R, weff)
        #print 'check: ', r0, '=?', reff

        # vmin
        veff = np.dot(np.dot(weff, Sigma), weff)
        
        return weff, reff, veff
                
    def tangencyWeights(self, rf = 0.015):
        """
        
        CHECK: IT DOESN'T WORK!!!!
        
        Weights of the tangency portfolio with 
        respect to Sharpe ratio maximization.
   
        Input:
         - rf:     risk-free rate
         - self.R: assets returns
         - self.C: asset Covariances
        
        Outputs:
         - tangency portfolio weights
        """

        n = len(self.R)

        """        
        # Using scipy.optimize:        
        fun = lambda w: np.sqrt(np.dot(np.dot(w, self.C), w))/(np.dot(self.R, w)-rf)            
        w = np.ones([n])/n                 # start with equal weights
        b_ = [(0.,1.) for i in range(n)]    # weights between 0%..100%. 
                                            # No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda w: sum(w)-1.})   # Sum of weights = 100%
        optimized = optimize.minimize(fun, w, method='SLSQP', constraints=c_, bounds=b_)
        
        if not optimized.success: 
            raise BaseException(optimized.message)

        # tangency weights:
        wt =  optimized.x        
        """

        # Using matrix algebra (description in portfolio_class.ipynb):
        Sigma_inv = np.linalg.inv(self.C)
        one_vec = np.ones([n])
        mu = self.R
                
        mu_minus_rf = mu - rf*one_vec
        top = np.dot(Sigma_inv, mu_minus_rf)
        bottom = np.dot(one_vec, top_vec)
        
        # tangency weights:
        wt = top/bottom
         
        # tangency return:
        rt = np.dot(self.R, wt)
        
        # tangency variance:
        vt = np.dot(np.dot(wt, self.C), wt)
            
        return wt, rt, vt


    
    
    
    
