from numpy.core.fromnumeric import cumsum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.utils import iqr


psdataset = pd.read_csv("C:/Users/svahi/Desktop/Indeed/PS_Homework_data_set.csv")
psdataset.columns #['advertiser_id', 'assigned', 'date_assignment_starts','date_assignment_ends', 'first_revenue_date', 'date_created', 'age','assign_days', 'revenue']
psdataset.iloc[:, 0]
psdataset = psdataset.iloc[: , 1:] #remove the first column
psdataset.shape
psdataset.info()
psdataset.describe()
psdataset.corr()
psdataset.rank()
psdataset[pd.isnull(psdataset)]
""" plt.scatter(psdataset['assigned'],psdataset['revenue'])
#plt.show()
psdataset.advertiser_id.isna().sum()
psdataset.assigned.isna().sum()
 """
# columns that have null values
nullCols= [colname for colname,colval in psdataset.iteritems() if colval.isna().sum()> 0]

print(nullCols) #['first_revenue_date', 'revenue']

# fill null values in revenue with 0
psdataset['revenue'].fillna(value = 0,inplace=True)
corr = psdataset.corr()

# visualise the data with seaborn
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.set_style(style = 'white')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(10, 250, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, square=True,linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
#-------------------------------------------------
'''from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn import preprocessing
X = psdataset.drop(['revenue','date_assignment_starts','date_assignment_ends', 'first_revenue_date', 'date_created'], axis=1)
y = psdataset['revenue']
X_normalized = preprocessing.normalize(X, norm='l2')
y = y
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=2)
selector = selector.fit(X, y)
print("Features selected", selector.support_)
print("Feature ranking", selector.ranking_)'''

X = psdataset.drop(['revenue','date_assignment_starts','date_assignment_ends', 'first_revenue_date', 'date_created'], axis=1)
X.columns
y = psdataset['revenue']
correlation_matrix = psdataset.corr().round(2)
abs(correlation_matrix.revenue).nlargest(2) 
#plt.figure(figsize=(14, 12))  
#sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()
# it seems that the feature which mainly affects the price is LSTAT (percentage of the lower status of the population),
#  with a Pearson coefficient of -0.74.

#Let’s visualize this relationship graphically:
#plt.scatter(psdataset['age'], psdataset['revenue'], color='blue')
#plt.xlabel('kooft')  
#plt.ylabel('revenue')
#plt.show()

#Question 1:
# How many leads are represented in this dataset? Describe both the assigned 
# and unassigned populations. What is the average revenue of each group?

psdataset.shape[0] # 77891
notAssigned = psdataset[psdataset['assigned'].values == 0]
notAssigned.describe(include='number')
# population describtion: mean, sd, median, distribution
notAssigned.describe(include='number')['revenue']['mean'] #1039001.1140105851


print(notAssigned.shape) #(40812, 9)
assigned = psdataset[psdataset['assigned'].values == 1]
assigned.describe(include='number')['revenue']['mean'] #3238846.4468836808
print(assigned.shape) #(37079, 9)

psdataset.groupby('assigned').count()
psdataset.groupby('assigned').mean()
#           advertiser_id        age  assign_days       revenue
#assigned                                                      
#0          1.068360e+07   11.907919   124.014971  1.039001e+06
#1          4.887142e+06  638.015966   116.594487  3.238846e+06

psdataset.groupby('assigned').median()
#            advertiser_id  age     assign_days  revenue
#assigned                                            
#0            10804573.5    0.0        137.0      0.0
#1             4115693.0  544.0        138.0      0.0

# getting the outliers
outliers=[]
def detect_outlier(data):
    threshold=3
    mean_1 = np.mean(data)
    std_1 =np.std(data)
    for y in data:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

detect_outlier(psdataset['revenue'])
#--------------------------------------
# detect outlier using IQR tells how spread the middle values are. 
# It can be used to tell when a value is too far from the middle.

def iqr_outlier(data):
    sorted(data)
    #Finding first quartile and third quartile
    q1, q3= np.percentile(data,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr) 
    return [lower_bound, upper_bound]

iqr(psdataset['revenue']) #0.0
iqr_outlier(psdataset['revenue']) #[0.0, 0.0]
#---------------------------
# avg revenue of each group:
avgRevenue = psdataset.groupby('assigned').mean()['revenue']
bars = ('Not Assigned', 'Assigned')
""" plt.figure(figsize = (12,7))
plt.bar(bars, round(avgRevenue,3), width= 0.9, align='center',color='blue', edgecolor = 'red')
i = 1.0
j = 2000
for i in range(len(bars)):
    plt.annotate(round(avgRevenue,3)[i], (-0.1 + i, round(avgRevenue,3)[i] + j))
plt.legend(labels = ['Mean Revenue'])

plt.title("Bar plot representing the average revenue by lead assignemnt status")
plt.xlabel('Lead Assignment Status')
plt.ylabel('Average Revenue ($)')
plt.show()
plt.savefig('QA1_3.png')
 """#----------------------------------------
#-----------------------------------------------------
# What are the most important metrics to consider when answering the problem statement? Why?
# For this dataset, we can see revenue is the dependent variable and looking into the relationship between other variables and revenue 
# depicts that column assigned and revenue will tell us about impact of assiging in revenue. In order to do that
# we can take a look at mean, median and other statistics.
# The first thing we can consider is looking into the mean and median of the revenue. Since we have many None values for
# revenue (I assumed that none values for revenue can be replaced by 0), the median can not be reliable since it is zero 
psdataset.groupby('assigned').median()['revenue']
#assigned
#0    0.0
#1    0.0

# therefore, mean is more realiable here:
psdataset.groupby('assigned').mean()['revenue']
#assigned
#0    1.039001e+06
#1    3.238846e+06

#-------------------------
#Analyze any existing relationship between account age and revenue.
'''plt.scatter(psdataset['age'],psdataset['revenue'])
plt.show()
plt.savefig("QA3_age_revenue")'''
# looking into the scatter plot of age and revenue, it seems that revenue is decreasing as the age increases, but 
# if we consider summation of the revenue for a different ages it will give us the better undestanding

#print(percentage.reset_index().to_markdown())

sortdata = psdataset.sort_values('age')
sortdata.groupby(['age','revenue']).sum().groupby('age').cumsum()
#plt.plot(sortdata['age'],sortdata['revenue'].cumsum())
#plt.show()
sortdata['cum_sum'] = sortdata['revenue'].cumsum()
sortdata['cum_perc'] = 100*sortdata['cum_sum']/sortdata['revenue'].sum()
'''plt.plot(sortdata['age'],sortdata['cum_sum'].cumsum())
plt.show()
plt.plot(sortdata['age'],sortdata['cum_perc'].cumsum())'''
#--------------------------------------------------------
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
reg1 = smf.ols('revenue ~  assigned', data=psdataset).fit()
print (reg1.summary())
#----------------------------------------------------
y = psdataset.revenue
X = psdataset.assigned
est = sm.OLS(y,X)
est = est.fit()
est.summary()
est.params
X_prime = np.linspace(X.min(), X.max(), 100)[:, np.newaxis] 
# add constant as we did before 
# Now we calculate the predicted values 
y_hat = est.predict(X_prime) 
plt.scatter(X, y, alpha=0.3) 
# Plot the raw data 
plt.xlabel("Gross National Product") 
plt.ylabel("Total Employment") 
plt.plot(X_prime, y_hat, 'r', alpha=0.9) 
# Add the regression line, colored in red P

# import formula api as alias smf import statsmodels.formula.api as smf 
# formula: response ~ predictors 
est = smf.ols(formula='revenue ~ assigned', data=psdataset).fit() 
est.summary()
#-------------------------------------------
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(psdataset[['age']], psdataset['revenue'], random_state=0)
from sklearn.linear_model import LinearRegression  
lm = LinearRegression()  
lm.fit(X_train,y_train)  
y_pred = lm.predict(X_test)

plt.scatter(X_test, y_test,  color='black')  
plt.plot(X_test, y_pred, color='blue', linewidth=3)  
plt.xticks(())  
plt.yticks(())  
plt.xlabel('assigned')  
plt.ylabel('revenue')   
plt.show()
from sklearn.metrics import mean_squared_error, r2_score  
print("MSE: {:.2f}".format(mean_squared_error(y_test, y_pred)))  
print("R2: {:.2f}".format(r2_score(y_test, y_pred)))
#_________________________________________________________

# Training the Polynomial Regression model on the whole dataset
X = psdataset['age'].values#, 'assigned','assign_days']]
y = psdataset['revenue'].values
from sklearn.preprocessing import PolynomialFeatures
X= X.reshape(-1, 1)
y= y.reshape(-1, 1)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#_________________________________________________________
#For the sake of completeness, I’m attaching here the Multiple Linear Regression version of our task, 
# where I’m going to consider all the 13 features of our problem:
X_train, X_test, y_train, y_test= train_test_split(psdataset[['assigned', 'age','assign_days']], psdataset['revenue'],random_state=0)  
from sklearn.linear_model import LinearRegression  
lm = LinearRegression()  
lm.fit(X_train,y_train)  
y_pred = lm.predict(X_test)  
#let's evaluate our results
from sklearn.metrics import mean_squared_error, r2_score  
print("MSE: {:.2f}".format(mean_squared_error(y_test, y_pred)))  
print("R2: {:.2f}".format(r2_score(y_test, y_pred)))







#----------------------------------------------------
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(psdataset[['assigned']], psdataset[['revenue']])
#Visualizing the Linear Regression results
import matplotlib.pyplot as plt
plt.scatter(psdataset[['assigned']], psdataset[['revenue']], color = 'red')
plt.plot(psdataset[['assigned']], model.predict(psdataset[['assigned']]), color = 'blue')
plt.title('revene vs assigned')
plt.xlabel('assigned')
plt.ylabel('revenue')
plt.show()
#Adjusted R-squared
model.score(psdataset[['assigned']], psdataset[['revenue']])













import math
m=100
ctr = math.ceil(len(sortdata['age'])/m)-1
ageArray = []
ageArray = [0 for i in range(math.ceil(len(sortdata['age'])/m))] 

for i in range(0, len(sortdata['age']), m):
    for j in range(m):
        if sortdata.index.max()-i-j<0:
            break
        ageArray[ctr]+=sortdata['revenue'][sortdata.index.max()-i-j]
    ctr-=1
print(ageArray)
plt.plot(range(math.ceil(len(sortdata['age'])/m)),ageArray)
plt.show()


    
    

#from pandas.stats.api import ols