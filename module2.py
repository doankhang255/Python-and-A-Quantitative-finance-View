import pandas as pd
import pandas
import matplotlib.pyplot as plt
import numpy as np

# die = pd.DataFrame({"value" : [1,2,3,4,5,6]})
# sum_of_dice = die.sample(2, replace = True).sum().loc["value"]
# # print("Sum:", sum_of_dice)
# trial = 2000
# result = [int(die.sample(2,replace = True).sum().loc["value"]) for i in range (trial)]
# result = [int(x) for x in result]
# print (result[:100])

# freq = pd.DataFrame({"Sum" : result})["Sum"].value_counts()
# sort_freq = freq.sort_index()
# print(sort_freq)
# sort_freq.plot (kind = "bar", color = "red")
# relative_freq = sort_freq/trial
# relative_freq.plot(kind = "bar", color = "blue")
# plt.show()

# X_distr = pd.DataFrame({"index" : [2,3,4,5,6,7,8,9,10,11,12]}) #xi
# X_distr["prob"] = [1,2,3,4,5,6,5,4,3,2,1]
# X_distr["prob"] = [x/36 for x in X_distr["prob"]] #pi

# # mean and variance 
# X_distr["Sum1"] = [X_distr.loc[x, "index"] * X_distr.loc[x, "prob"] for x in X_distr.index]
# Mean = X_distr["Sum1"].sum()
# X_distr["Sum2"] = [((X_distr.loc[x, "index"] - Mean)**2)*X_distr.loc[x,"prob"] for x in X_distr.index]
# Var = X_distr["Sum2"].sum()
# print(f"({Mean} , {Var})")

#.......... Log daily return 
from scipy.stats import norm , stats
ms = pd.read_csv(r'C:\Users\doank\OneDrive\Personal Vault\Python and Statistics for Financial Analysis\microsoft.csv', index_col = 0)
ms.index = pd.to_datetime(ms.index)

ms["Log_return"] = np.log((ms["Close"].shift(-1))/ms["Close"])
mu = ms["Log_return"].mean()
sigma = ms["Log_return"].std(ddof = 1)

density = pd.DataFrame()
density["x"] = np.arange(ms["Log_return"].min()-0.01 , ms["Log_return"].max()+0.01 , 0.001 )
density["pdf"] = norm.pdf(density["x"] , mu , sigma)
# ms["Log_return"].hist(bins=50, figsize=(10,5))
density["cdf"] = norm.cdf(density["x"] , mu , sigma)
# plt.plot(density["x"] , density["pdf"] , color = "red")
# plt.plot(density["x"] , density["cdf"] , color = "red")
# prob_return = norm.cdf(-0.01, mu, sigma)
# print(prob_return) #probabolity to drop 1% in a day
plt.plot(density["x"] , density["pdf"] , color = "red")
plt.ylim(0,40)
plt.fill_between(density["x"], density["pdf"] , where = density["x"] < -0.01, color = "green", alpha = 0.5) # paint area
plt.show()

#............ in 200 days
# mu200 = 220*mu
# sigma200 = (220**0.5) * sigma
# prob_return200 = norm.cdf(-0.2, mu200, sigma200) #in 200 days
# print(prob_return200)
  

  
# #VaR (value at risk)
# VaR = norm.ppf (0.05, mu, sigma) # 5% quantile
# print(VaR)




