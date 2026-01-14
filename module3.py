import pandas as pd
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.stats import norm , stats

ms = pd.read_csv(r'C:\Users\doank\OneDrive\Personal Vault\Python and Statistics for Financial Analysis\microsoft.csv', index_col = 0)
ms.index = pd.to_datetime(ms.index)

data = pd.DataFrame()
data["population"] = [47, 48, 85, 20, 19, 13, 72, 16, 50, 60] 
# a_sample_without_replacement = data["population"].sample(5, replace = False)
# a_sample_with_replacement = data["population"].sample(5, replace = True)
# print(a_sample_without_replacement,"\n",a_sample_with_replacement)

# print(data["population"].mean())
# print(data["population"].var(ddof = 0))
# print(data["population"].std(ddof = 0))
# print(data["population"].shape[0])
# print("~~~~~~~~~~~~~")
# a_sample = data["population"].sample(10, replace = False)
# print(a_sample.mean())
# print(a_sample.var(ddof = 1))
# print(a_sample.std(ddof = 1))
# print(a_sample.shape[0])

# Fstsample = pd.DataFrame(np.random.normal(10,5, size = 30))
# print(Fstsample[0].mean())
# print(Fstsample[0].std(ddof = 1))

#~~~~~~~~~~~~ Empirical distribution of sample mean and variance 
# meanlist = []
# varlist = []
# for t in range (1000):
#     sample = pd.DataFrame(np.random.normal(10, 5 , size = 30))
#     meanlist.append(sample[0].mean())
#     varlist.append(sample[0].var(ddof = 1))

# collection = pd.DataFrame()
# collection["meanlist"] = meanlist
# collection["varlist"] = varlist 
# collection['meanlist'].hist (bins = 500, density = True, color = "blue" )
# # collection['varlist'].hist(bins = 500, density = True, color = "red")
# pop = pd.DataFrame(np.random.normal(10, 5 , size = 1000))
# pop[0].hist (bins = 500, color = "pink", density = True)
# plt.show()

# sample_mean_list = []
# sample_mean_list1 = []
# apop = pd.DataFrame([1,0,1,0,1])
# for i in range (100000):
#     sample = apop[0].sample(10, replace = True )
#     sample_mean_list. append (sample.mean())
#     sample1 = apop[0].sample(1000, replace = True )
#     sample_mean_list1. append (sample1.mean())

# acollec = pd.DataFrame()
# acollec['mean_list'] = sample_mean_list
# acollec["mean_list1"] = sample_mean_list1 
# acollec['mean_list'].hist (bins = 500, density = True)
# acollec['mean_list1'].hist (bins = 500, density = True)
# plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~ Confidence interval
ms["LogReturn"] = np.log(ms["Close"].shift(-1)) - np.log(ms["Close"])
# z_left = norm.ppf (0.05) # cal 90% confidence interval
# z_right = norm.ppf (0.95) 
# sample_mean = ms["LogReturn"].mean()
# sample_std = (ms["LogReturn"].std(ddof=1)/(ms.shape[0])) **0.5
# interval_left = sample_mean + z_left * sample_mean
# interval_right = sample_mean + z_right * sample_mean
# print(sample_mean)
# print(f"{interval_left} , {interval_right} ")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Hypothesis Testing 
# ```````````````two tailed test 
# xbar = ms["LogReturn"].mean()
# s = ms["LogReturn"].std(ddof = 1)
# n = ms["LogReturn"].shape[0]
# z_hat = (xbar - 0) / (s/(n**0.5))
# print (z_hat) 

# alpha = 0.025
# zleft = norm.ppf(alpha/2, 0, 1) #(p,mean,std) của normal distribution
# zright = -zleft
# print(f"({zleft} , {zright})")
# print("Significant interval:", alpha)
# print("Reject or not:", z_hat > zright or z_hat <zleft)

# `````````````one tail test 
# alpha1 = 0.05
# zright1 = norm.ppf (1 - alpha1, 0 , 1)
# print(zright1)
# print("Significant interval:", alpha1)
# print("Reject or not:", z_hat > zright1)
# tương tự với zleft1

#```````````` p value of two tails test  
# alpha2 = 0.05 
# p_value = 1 - (norm.cdf(abs(z_hat), 0 , 1))
# print("Significant level", alpha2)
# print("Reject:", p_value < alpha2)
# print(p_value)

ms["LogReturn"].plot(color = "orange")
plt.axhline(0, color = "green")
plt.show()