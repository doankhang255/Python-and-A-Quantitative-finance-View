import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix 
import numpy as np

# housing = pd.read_csv(r"C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\housing.csv",index_col = 0)
# print(housing.head())
# print("\n")
# print(housing.tail())
# print(housing.cov())
# print(housing.corr())

# # ---------------------------sm = scatter_matrix(housing)
# housing.plot(kind = 'scatter', x='LSTAT', y='MEDV')
# plt.show()

# ~~~~~~~~~~~~~~~~~~~ guess line 
# b0 = 1
# b1 = 2

# housing["guess_response"] = b0 + b1*housing["RM"]
# housing['observed_error'] = housing["MEDV"] - housing["guess_response"]
# # indices = [7, 20, 100]
# # print(housing["observed_error"].loc[indices])
# print("sum of squared error", ((housing["observed_error"])**2).sum())

# #~~~~~~~~~~~~~~~~~ best fit line
import statsmodels.formula.api as smf
# model = smf.ols(formula = "MEDV~RM", data = housing).fit()
# b0_ols = model.params[0]
# b1_ols = model.params[1]
# housing["best_response"] = b0_ols + b1_ols*housing["RM"]
# # print(b0,b1)
# housing["error"] = housing["MEDV"] - housing["best_response"]
# print((housing['error']**2).sum())
# plt.scatter(housing["RM"], housing["MEDV"], color = "green", label = "real")
# plt.scatter(housing["RM"], housing["guess_response"], color = "blue", label = "guess")
# plt.scatter(housing["RM"], housing["best_response"], color = "pink", label = "best_model")
# plt.plot(housing["RM"], housing["guess_response"], color = "red")
# plt.plot(housing["RM"], housing["best_response"], color = "yellow")
# plt.ylabel("MEDV/1000$")
# plt.xlabel("RM/number")
# plt.xlim(np.min(housing["RM"])-2, np.max(housing["RM"])+2)
# plt.ylim(np.min(housing["MEDV"]) - 5, np.max(housing["MEDV"]) + 5 )
# plt.legend()
# plt.show()
# print(model.summary())
# plt.plot(housing.index, housing['error'], color = "purple")
# plt.axhline(y=0, color = 'red')
# plt.show()
#~~~~~~~~~~~~~~~~ Durbin Watson test
# Durbin-Watson test giúp kiểm tra xem mô hình có vi phạm giả định “residual độc lập” hay không.
# d > 2.5 → nghi ngờ autocorrelation âm.
# d < 1.5 → nghi ngờ autocorrelation dương.
# d ~ 2 Không có tự tương quan (residual độc lập).

# diagnotis for Normality
# import scipy.stats as stats 
# z = (housing["error"] - housing['error'].mean())/(housing['error'].std(ddof = 1))
# print(z)
# stats.probplot(z, dist = 'norm', plot = plt)
# plt.show() 

# ~~~~~~~~~~ diagnotis equal variance 
# housing.plot(kind = 'scatter', x = 'RM', y = 'error', color = 'green')
# plt.axhline(y = 0, color = 'red')
# plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~multiple linear regression 

indice_panel = pd.DataFrame()
aord = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\ALLOrdinary.csv')
nikkei = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\Nikkei225.csv')
hsi = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\HSI.csv')
daxi = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\DAXI.csv')
cac40 = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\CAC40.csv')
sp500 = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\SP500.csv')
dji = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\DJI.csv')
nasdaq = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\nasdaq_composite.csv')
spy = pd.read_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\SPY.csv')
# print(spy.head())
indice_panel['spy'] = spy['Open'].shift(-1) - spy['Open']
indice_panel['spy_lag1'] = indice_panel['spy'].shift(1)
indice_panel['sp500'] = sp500['Open'] - sp500['Open'].shift(1)
indice_panel['nasdaq'] = nasdaq['Open'] - nasdaq['Open'].shift(1)
indice_panel['dji'] = dji['Open'] - dji['Open'].shift(1)
indice_panel["cac40"] = cac40["Open"] - cac40['Open'].shift(1)
indice_panel['daxi'] = daxi['Open'] - daxi['Open'].shift(1)
indice_panel["aord"] = aord['Close'] - aord["Open"]
indice_panel['hsi'] = hsi["Close"] - hsi['Open']
indice_panel['nikkei'] = nikkei["Close"] - nikkei['Open']
indice_panel["Price"] = spy['Open']
# print(indice_panel.head())
indice_panel = indice_panel.ffill()
indice_panel = indice_panel.dropna()
# print(indice_panel.head())
# print(indice_panel.isnull().sum())
indice_panel.to_csv(r'C:\Users\doank\OneDrive\Documents\Python and Statistics for Financial Analysis\indicepanel.csv')
# print(indice_panel.shape) # number of row and column
train = indice_panel.iloc[-2000:-1000, :]
test = indice_panel.iloc[-1000:, :]
train = indice_panel.iloc[-2000:-1000, :].copy()
test = indice_panel.iloc[-1000:, :].copy()
# print(train.shape, test.shape)
# sm = scatter_matrix(train)
# plt.show()
# train.iloc[:, :-1].corr()['spy']
lm  = smf.ols(formula = 'spy~spy_lag1 + sp500 + nasdaq + dji + cac40 + aord + daxi + nikkei + hsi', data = train).fit()
# lm.summary()
train['PredictedY'] = lm.predict(train)
test['PredictedY'] = lm.predict(test)
# print(train['PredictedY'])
# plt.scatter(train['spy'], train['PredictedY'])
# plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~RMSE - ROOT MEAN SQUARED ERROR, adjusted R^2
def adjustedMetric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjusted_r2 = 1-(1-r2)*(data.shape[0] - 1) / (data.shape[0] - model_k - 1)
    RMSE = (SSE/(data.shape[0] - model_k - 1)) **0.5
    return adjusted_r2, RMSE

def assesstable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index = ['R2', 'RSME'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

result = assesstable(test, train, lm, 9, 'spy')
print(result)

# signal-based strategy
train['order'] = [1 if sig > 0 else -1 for sig in train['PredictedY']]
train['profit'] = train['spy']* train['order']
train['Wealth'] = train['profit'].cumsum()
# print("total profit in train", train['profit'].sum())
plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(train['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

test['order'] = [1 if sig > 0 else -1 for sig in test['PredictedY']]
test['profit'] = test['spy']* test['order']
test['Wealth'] = test['profit'].cumsum()
# # print("total profit in test", test['profit'].sum())
# plt.figure(figsize=(10, 10))
# plt.title('Performance of Strategy in Train')
# plt.plot(test['Wealth'].values, color='green', label='Signal based strategy')
# plt.plot(test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
# plt.legend()
# plt.show()

# Evaluation model - practical standard
train['Wealth'] = train['Wealth'] + train.loc[train.index[0], 'Price']
test['Wealth'] = test['Wealth'] + test.loc[test.index[0], 'Price']
# Sharpe Ratio on Train data
train['Return'] = np.log(train['Wealth']) - np.log(train['Wealth'].shift(1))
dailyr = train['Return'].dropna()
print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))
# Sharpe Ratio in Test data
test['Return'] = np.log(test['Wealth']) - np.log(test['Wealth'].shift(1))
dailyr = test['Return'].dropna()
print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))

# Maximum Drawdown in Train data
train['Peak'] = train['Wealth'].cummax()
train['Drawdown'] = (train['Peak'] - train['Wealth'])/train['Peak']
print('Maximum Drawdown in Train is ', train['Drawdown'].max())

# Maximum Drawdown in Test data
test['Peak'] = test['Wealth'].cummax()
test['Drawdown'] = (test['Peak'] - test['Wealth'])/test['Peak']
print('Maximum Drawdown in Test is ', test['Drawdown'].max()) 