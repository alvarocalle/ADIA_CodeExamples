# Links
#https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1
#http://francescopochetti.com/scrapying-around-web/
#https://github.com/FraPochetti/StocksProject
#https://doc.scrapy.org/en/latest/intro/tutorial.html
#https://github.com/Aryal007/predict_stock_prices/blob/master/predict_stock.py
#https://www.python.org/dev/peps/pep-0008/#class-names
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/
#http://web.stanford.edu/~zlotnick/TextAsData/Web_Scraping_with_Beautiful_Soup.html
#https://textblob.readthedocs.io/en/dev/
#http://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.sentiment
#http://www.gestaltu.com/2015/11/tactical-alpha-in-theory-and-practice-part-ii-principal-component-analysis.html/
#http://www.quantandfinancial.com/2013/07/mean-variance-portfolio-optimization.html
#https://github.com/omartinsky/QuantAndFinancial/blob/master/black_litterman/black_litterman.ipynb
#http://www.quantandfinancial.com/2013/08/portfolio-optimization-ii-black.html

## Import necesary modules
import sys
import random
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import utils   # Contains my utilities

from scipy import stats
import statsmodels.api as sm
from sklearn.decomposition import KernelPCA, PCA # PCA library

def main():    
       
    # Font used in plots:    
    font = {'family' : 'serif',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : 12,
            }

    # Get list of stocks in the S&P 500 index:
    SP500_tickers = pd.read_csv('SP500_tickers.csv', sep=';', index_col=None, 
                                skiprows=1, header=0,
                                skip_blank_lines=True, skipinitialspace=True)                          
    
    keys = SP500_tickers['Ticker']
    values = SP500_tickers['Security']
    dic = dict(zip(keys, values))
   
    # Set dates:
    start_date = datetime.datetime(2000, 1, 1)
    end_date   = datetime.date.today() 

    ## Download daily prices from Yahoo!
    symbols = random.sample(keys, 10)   
    stockPrices = web.DataReader(symbols, 'yahoo', start_date, end_date).Close

    SP500 = pd.DataFrame(web.DataReader('^GSPC', 'yahoo', start_date, end_date).Close)
    SP500.columns = ['Close']
   
    # calculate SP500 returns:
    SP500['Return'] = SP500.pct_change()[1:]    
    SP500['LogReturn'] = np.log(SP500['Close']/SP500['Close'].shift(1))[1:]

    # remove first row
    SP500 = SP500[1:]
    
    # -----------------------------
    # Portfolios
    # -----------------------------
    
    # Set dates:
    start_date = datetime.datetime(2015, 1, 1)
    end_date   = datetime.date.today() 
    
    # Download daily prices from Yahoo!
    symbols = keys
    stockPrices = web.DataReader(symbols, 'yahoo', start_date, end_date).Close
    print 'number of retrieved stocks:' , stockPrices.shape[1]

    # Drop stocks that did not trade in the time period
    stockPrices = stockPrices.dropna(axis='columns', how='any')
    print 'number of actual traded stocks:' , stockPrices.shape[1]

    # Calculate stock returns:
    simpleReturns = stockPrices.pct_change()[1:]
    logReturns = np.log(stockPrices/stockPrices.shift(1))[1:]
    
    # Largest cap companies in the SP500:
    SP500_largecaps = pd.read_csv('SP500_company_weights.csv', sep=';',
                                  index_col=None,
                                  skiprows=1, header=0,
                                  skip_blank_lines=True, skipinitialspace=True)                          

    # store info in a dictionary:
    large_keys = SP500_largecaps['Symbol']
    large_values = SP500_largecaps['Weight']
    large_dic = dict(zip(large_keys, large_values))

    print 'Number of stocks:', SP500_largecaps.shape[0]
    print SP500_largecaps.head()
        
    # Portfolio mean return-variance of the 50 largest cap companies    
    # Select the 50 largest cap companies
    prices = stockPrices[large_keys[:50]]

    # daily returns
    simpleReturns = prices.pct_change()[1:]
    logReturns = np.log(prices/prices.shift(1))[1:]

    returns = logReturns

    # anualyzed expected returns
    expReturns = (1 + returns.mean(axis=0))** 251 - 1

    # anualyzed expected variances
    expVariances = returns.var(axis=0)* np.sqrt(251)

    # annualized covariances
    covars = returns.cov() * 251

    # create dataframe with returns, vars and weights
    largeCaps = pd.DataFrame(expReturns, columns=['ExpReturns'])
    largeCaps['ExpVariances'] = expVariances
    largeCaps['Weights'] = np.array(SP500_largecaps.Weight[:50]/sum(SP500_largecaps.Weight[:50]))    

    # mean-variance return of the portfolio:
    print largeCaps.head(3)

    W = largeCaps['Weights']
    R = largeCaps['ExpReturns']
    V = largeCaps['ExpVariances']
    C = covars

    largeCapsPF = utils.portfolio(R, W, C)

    rp = largeCapsPF.mean
    vp = largeCapsPF.var
    print 'mean: ', np.round(rp*100), '%'
    print 'variance: ', np.round(vp*100), '%'

    # Minimum variance portfolio for the large caps
    wmin, rmin, vmin = largeCapsPF.getMinVarWeights()

    print 'minimum variance pf (mean): ', np.round(rmin*100), '%'
    print 'minimum variance pf (variance): ', np.round(vmin*100), '%'

    # Tangency portfolio for the large caps
    wt, rt, vt = largeCapsPF.tangencyWeights(0.015)

    print 'Tangency pf (mean): ', np.round(rt*100), '%'
    print 'Tangency pf (variance): ', np.round(vt*100), '%'

    # Efficient portfolio for a given expected return
    weff, reff, veff = largeCapsPF.getEfficientWeights(0.05)

    print 'Efficient pf (mean): ', reff*100, '%'
    print 'Efficient pf (variance): ', veff*100, '%'
    
    # Efficient portfolio corresponding to the large cap portfolio return
    weff, reff, veff = largeCapsPF.getEfficientWeights(largeCapsPF.mean)
    print 'Large cap variance: ', largeCapsPF.var*100, '%'
    print 'Efficient pf variance: ', veff*100, '%'

    # Maximum return stock
    wx, rx, vx = largeCapsPF.getEfficientWeights(max(largeCaps.ExpReturns))

    # Minumum return stock
    wy, ry, vy = largeCapsPF.getEfficientWeights(min(largeCaps.ExpReturns))
        
    sxy  = np.dot(np.dot(wx, C), wy)

    interval = np.linspace(0,1,100)
    frontier = []
    
    for a in interval:
        rz = a *rx + (1 - a) *ry
        vz = a**2 * vx + (1 - a)**2 *vy + 2*a*(1 - a) * sxy
        frontier.append([rz, vz])

    frontier = pd.DataFrame(frontier,columns=['rz','vz'])

    # Efficient frontier:
    plt.figure()
    plt.plot(frontier.vz, frontier.rz, linestyle='-', color='r',
             linewidth=1.5, label='Frontier')
    plt.plot(vt, rt, 'yo', label='Tangency Portfolio')
    plt.plot(veff, reff, 'bo', label='Efficient Portfolio')
    plt.plot(vmin, rmin, 'go', label='minVar Portfolio')
    plt.title('Large Cap Efficient frontier', fontdict=font)
    plt.xlabel('Variance')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.legend(loc='right', bbox_to_anchor=(.3, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=8)
    plt.grid(False)
    plt.show()

    ## --- Apply PCA to prices:
    # http://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
    # Hilpisch: Python for Finance
    
    # normalizer
    normalizer = lambda x: (x - x.mean()) / x.std()
    
    data = stockPrices
    
    # PCA with Kernel method
    pcaKer = KernelPCA().fit(data.apply(normalizer))
    numPCAKer = len(pcaKer.lambdas_)
    pcaKer.lambdas_.round()

    get_weight = lambda x: x / x.sum()
    pcaKerVarExplain = get_weight(pcaKer.lambdas_)[:10]
    get_weight(pcaKer.lambdas_)[:10].sum()
    
    # PCA with SVD method
    pcaSVD = PCA().fit(data.apply(normalizer))
    pcaSVDVarExplain = pcaSVD.explained_variance_ratio_
    
    # plot variance explained by components
    plt.title('Principal Component Analysis', fontdict=font)
    plt.xlabel('Number of components', fontdict=font)
    plt.ylabel('Variance explained', fontdict=font)
    plt.plot(pcaSVDVarExplain, linestyle='-',  color='r', linewidth=1.5, label= 'SVD')
    plt.plot(pcaKerVarExplain, 'bo', label= 'Kernel')
    plt.legend(loc='right', bbox_to_anchor=(.9, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=8)
    plt.xlim([0,20])
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    # create table of principal portfolios
    I = np.identity(data.shape[1])  # identity matrix
    coef = pcaSVD.transform(I)
    np.linalg.norm(coef,axis=0) # check eigen-vector norms
    colnames = ['PC-'+str(i) for i in range(1, pcaSVD.n_components_+1)]
    pd.DataFrame(coef, columns=colnames, index=data.columns)

    df = pd.DataFrame(SP500.ix[data.index].Close)
    df.columns = ['SP500']
    df['PC1'] = pcaSVD.transform(-data)[:,1]
    df['PC1-10'] = pcaSVD.transform(-data)[:,10]
    
    # plot PC new index
    plt.figure()   
    df.apply(normalizer).plot(linewidth=1.5)
    plt.title('SP500 vs PC index', fontdict=font)
    plt.ylabel('Normalized Prices (USD)', fontdict=font)
    plt.legend(loc='left', bbox_to_anchor=(.2, 0.95), ncol=1, fancybox=True, shadow=True, fontsize=8)
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    
    ## --- Stock selection with PCA:
    # The selection is done applying PCA to correlation matrix of returns
    # Paper: Yang & Rea (2015) ISSN 1179-3228
    
    ## Download daily prices for all the securities in the SP500
    start_date = datetime.datetime(2014, 1, 1)
    end_date   = datetime.date.today() 
    prices = web.DataReader(keys, 'yahoo', start_date, end_date).Close
    print 'number of retrieved stocks:' , prices.shape[1]
    
    # drop stocks that did not trade in the whole study time period
    prices = prices.dropna(axis='columns', how='any')
    print 'number of actual traded stocks:' , prices.shape[1]
   
    # calculate returns:
    simpleReturns = prices.pct_change()[1:]
    logReturns = np.log(prices/prices.shift(1))[1:]

    # correlation matrix of returns:
    assetCorr = logReturns.corr()
    
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(assetCorr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.axis('off')
    plt.show()

    # PCA to correlation matrix
    data = assetCorr.apply(normalizer)
    #data = assetCorr
    
    eps = 0.0001 # selection factor
    
    # perform PCA to select securities
    while data.shape[0] > 50:
        
        #pcaKer = KernelPCA().fit(data)
        pcaSVD = PCA().fit(data)

        var = pcaSVD.explained_variance_
        diff = [abs(t - s) for s, t in zip(var, var[1:])]
        
        # select low eigenvalue PCs based on selection factor
        invalid = sum(np.array(diff) < eps)
        I = np.identity(data.shape[1])  # identity matrix
        low_coef = pcaSVD.transform(I)[:, -invalid:] # set of small eigenvalued PCs
        sum(np.linalg.norm(low_coef,axis=0).round() != 1.0) # check eigen-vector norms
                
        #colnames = ['PC-'+str(i) for i in range(npc-invalid, npc+1)]
        low_pfs = pd.DataFrame(low_coef, index=data.columns)

        # select stocks with the highest weights in low eigenvalue PCs
        low_assets = abs(low_pfs).idxmax().values
    
        # remove those assets from data
        data = data.drop(low_assets).drop(low_assets, axis=1)
    
    print 'Number of stocks remained: ', data.shape[0]

    # variance explained
    plt.xlabel('Number of components', fontdict=font)
    plt.ylabel('Variance explained', fontdict=font)
    plt.plot(pcaSVD.explained_variance_, 'bo', label= 'SVD')
    plt.legend(loc='right', bbox_to_anchor=(.9, 0.85), ncol=1, fancybox=True, shadow=True, fontsize=8)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    # select num of component: 5
    npc = 5
    
    coef = pcaSVD.transform(I)[:,:npc]
    np.linalg.norm(coef,axis=0) # check eigen-vector norms
    #colnames = ['PC-'+str(i) for i in range(1, valid + 1)]
    colnames = ['PC-'+str(i) for i in range(1, npc + 1)]
    ppalPortfolios = pd.DataFrame(coef, columns=colnames, index=data.columns)

    # select stocks with highest weights in the first five components
    security = abs(ppalPortfolios).max(axis=1).sort_values(ascending=False)[:20].index.values
    
    ### Porfolio overview
    
    
    # Create api from class TwitterKeys in utils
    api = utils.TwitterAPI().createAPI()

    # Calculate the sentiment of each security from Twitter
    num_tweets = 10

    for key,value in dic.iteritems():
        query = key + ' ' + value
        print key, ':', utils.twitterSentiment(api, query, num_tweets)
    

if __name__ == '__main__':
    main()

          
