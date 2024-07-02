#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression
# 
# Build a linear regression model to predict `Sales` using an appropriate predictor variable.

# ## Step 1: Reading and Understanding the Data
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import the numpy and pandas package

import numpy as np
import pandas as pd


# In[3]:


# Read the given CSV file, and view some sample records

advertising = pd.read_csv("advertising.csv")
advertising.head()


# inspect the various aspects of our dataframe

# In[4]:


advertising.shape


# In[5]:


advertising.info()


# In[6]:


advertising.describe()


# ## Step 2: Visualising the Data
# 
# visualise data using seaborn. We'll first make a pairplot of all the variables present to visualise which variables are most correlated to `Sales`.

# In[7]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[8]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales',size=4, aspect=1, kind='scatter')
plt.show()


# In[9]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# As is visible from the pairplot and the heatmap, the variable `TV` seems to be most correlated with `Sales`. So let's go ahead and perform simple linear regression using `TV` as our feature variable.

# ---
# ## Step 3: Performing Simple Linear Regression
# 
# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 
# ---

# ### Generic Steps in model building using `statsmodels`
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[10]:


X = advertising['TV']
y = advertising['Sales']


# #### Train-Test Split

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[12]:


X_train.head()


# In[13]:


y_train.head()


# #### Building a Linear Model
# import the `statsmodel.api` library using which we'll perform the linear regression.

# In[14]:


import statsmodels.api as sm


# By default, the `statsmodels` library fits a line on the dataset which passes through the origin. But in order to have an intercept, we need to manually use the `add_constant` attribute of `statsmodels`. And once added the constant to `X_train` dataset, we can go ahead and fit a regression line using the `OLS` (Ordinary Least Squares) attribute of `statsmodels` as shown below

# In[15]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[16]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[17]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# ---
# The fit is significant. Let's visualize how well the model fit the data.
# 
# From the parameters that we get, our linear regression equation becomes:
# 
# $ Sales = 6.948 + 0.054 \times TV $

# In[18]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# ## Step 4: Residual analysis 

# #### Distribution of the error terms
# We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[19]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[20]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# The residuals are following the normally distributed with a mean 0. All good!

# #### Looking for patterns in the residuals

# In[21]:


plt.scatter(X_train,res)
plt.show()


# We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.
# 
# Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

#  the regression line is a pretty good fit to the data

# ## Step 5: Predictions on the Test Set
# 

# In[22]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[23]:


y_pred.head()


# In[24]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ##### Looking at the RMSE

# In[25]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# ###### Checking the R-squared on the test set

# In[26]:


r_squared = r2_score(y_test, y_pred)
r_squared


# ##### Visualizing the fit on the test set

# In[27]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


#  

#  

#  

#  

# ### Linear Regression using `linear_model` in `sklearn`
# 
# Apart from `statsmodels`, there is another package namely `sklearn` that can be used to perform linear regression. We will use the `linear_model` library from `sklearn` to build the model. Since, we hae already performed a train-test split, we don't need to do it again.
# 
# There's one small step that we need to add, though. When there's only a single feature, we need to add an additional column in order for the linear regression fit to be performed successfully.

# In[28]:


from sklearn.model_selection import train_test_split
X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[29]:


X_train_lm.shape


# In[30]:


X_train_lm = X_train_lm.reshape(-1,1)
X_test_lm = X_test_lm.reshape(-1,1)


# In[31]:


print(X_train_lm.shape)
print(y_train_lm.shape)
print(X_test_lm.shape)
print(y_test_lm.shape)


# In[32]:


from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# Fit the model using lr.fit()
lm.fit(X_train_lm, y_train_lm)


# In[33]:


print(lm.intercept_)
print(lm.coef_)


# The equationwe get is the same as what we got before!
# 
# $ Sales = 6.948 + 0.054* TV $

# Sklearn linear model is useful as it is compatible with a lot of sklearn utilites (cross validation, grid search etc.)

#  
