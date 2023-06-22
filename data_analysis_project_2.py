the link to the full project with the visualisation :https://colab.research.google.com/drive/1JbECA-nFsYo__FBdwRbPyI0Kbo2anxO8?usp=sharing

##I- preamble :

This project's topic is linear regression : Understanding linear regression is important because it provides a scientific calculation for identifying and predicting future outcomes. The ability to find predictions and evaluate them can help provide benefits to many businesses and individuals, like optimized operations and detailed research materials.

##II- Linear regression with a data set :

In this second part, we will try to draw the regression line that represents a certai dataset and use it in fact to check the accuracy of our vector X :

We start off by transforming the data giving into numpy tables so we can manipulate this data a lot easier.

[17]
import numpy as np #we import the numpy library
x_train=np.array([1,2,3,4,5,6])  
print(x_train) #we transform the data into numpy tables and print them out
y_train=np.array([2,3,2,3,4,5])
print(y_train)
[1 2 3 4 5 6]
[2 3 2 3 4 5]

Now ,we will be scatter ploting this data to visualize it .

[18]
import matplotlib.pyplot as plt #we import the matplotlib library needed here

plt.scatter(x_train, y_train, color="purple") #we scatter plot the data
plt.xlabel('Xi :')
plt.ylabel('Yi :')

plt.show()



Now , we will try to find beta_0 as well as beta_1 using decomposition QR :

[19]
import numpy as np #we import the numpy library
import numpy.linalg as alg  #we import the  inversion function from the numpy library
A = np.array([[1, 1], [1, 2], [1,3],[1,4],[1,5],[1,6]]) #we lay down our matrix A that we have constracted from the data in x_train 
Q, R = np.linalg.qr(A) #We decomposite our matrix A into QR 
print('\nQ:\n', Q)#we print out both our matrixes Q and R to make sure they are logical and correct 
print('\nR:\n', R)

Qt=np.transpose(Q) #we define Qt as the transpose of Q and we calculate it
X= (np.linalg.inv(R)@Qt@y_train) #then we define our vector X which is the inverse of R multiplied by Q transpose as well as the output data(y_train)

print(X) #now we print our vector
print('beta_0 est :', X[0])
print('beta_1 est :', X[1])

Q:
 [[-0.40824829 -0.5976143 ]
 [-0.40824829 -0.35856858]
 [-0.40824829 -0.11952286]
 [-0.40824829  0.11952286]
 [-0.40824829  0.35856858]
 [-0.40824829  0.5976143 ]]

R:
 [[-2.44948974 -8.5732141 ]
 [ 0.          4.18330013]]
[1.26666667 0.54285714]
beta_0 est : 1.2666666666666657
beta_1 est : 0.5428571428571431

In this part of the project ,we are asked to prove that certain statistics formulas(stated in the project) can in fact calculate our beta_0 and beta_1 too.In order to do that, we will calculate these formulas by transforming them into python code and we will then check if they actually can help calculate the vector or not.

[26]
from statistics import mean #we import the mean function from the statistics library so we can use it
beta_1= sum(((x_train)-mean(x_train))*((y_train)-mean(y_train)))/sum((x_train -mean(x_train))**2) #we calculate the formula given for beta_1
beta_1 #we print the result out
0.5263157894736842
[27]
beta_0= mean(y_train)- beta_1*mean(x_train) #we calculate the formula given
beta_0 #we print out the result

1.4210526315789473
After calculating them, we realize that their values are very close and can be considered equal to the vector's ones that we have found earlier using the QR decomposition.Therefore, we have proved that these formulas can easily calculate our X vector containing beta_0 and beta_1.

Right now, we will be creating a python function (using the formulas above) that takes the data given as input and returns the vector X containing beta_0 and beta_1 as an output.

[22]
def simple_linear_Regression(data_x,data_y): #we name our function and give the input which is the data

  
  BETA_1=sum(((data_x)-mean(data_x))*((data_y)-mean(data_y)))/sum((data_x-mean(data_x))**2)#we use the formula that calculates beta_1 that we have proved in the previous question
  BETA_0=mean(data_y)- BETA_1*mean(data_x)#we use the formula that calculates beta_0 that we have proved in the previous question
  X=np.zeros(2) #we define a random (2,1)vector and fill it with zeros as a start
  X[0]=BETA_0#now we fill the first element of the vector with beta_0 
 
  X[1]=BETA_1#now we fill the second element of the vector with beta_1
  
  return(X)# we return the vector X
Now we will use this function to make sure that the results that we recieved previously were correct and to check if the function works.

[23]
a=simple_linear_Regression(x_train,y_train)#we give the data given in the project as input
print(a) #we print out the vector returned by our function

[1.42105263 0.52631579]

At this point, we will try to scatter plot the regression line for our data .

[28]
plt.plot(x_train, y_train, "bo", label="donnees") # the data
plt.plot( 
    [x_train[0], x_train[-1]],        # x values
    [beta_1 * x_train[0] + beta_0, 
     beta_1 * x_train[-1] + beta_0],  #  y values
    "r-",                           # red color with a continuous line(the regression line)
    label="regression")             # legend
plt.xlabel("x") # name of the axe x
plt.ylabel("y") # name of the axe y
plt.xlim(0, 11) # ladder for the axe x
plt.legend() # the legend
plt.title("Regression Lineaire") # title of the graphic presentation

Text(0.5, 1.0, 'Regression Lineaire')

And after observing the line regression that we got, we can tell that the values of beta_0 and beta_1 that we have gotten previously are accurate and seem to be raisonable since the line is the closest that it can be to all the points representing the data in the graph.

##III- Construction of a model using statsmodels and sklearn :

Int this part , We will try to eventually make a regression model out of our dataset using statsmodels .

First of all, We will write a python code that again returns the values of our vector X while using the code in the link given as an inspiration.

[29]
import numpy as np 

import statsmodels.api as sm #we import the numpy library and statsmodels needed for the execution of this code

                          
X = sm.add_constant(x_train, prepend=False) #we add a colonne to the X matrix

#  We Fit and summarize OLS model :
mod = sm.OLS(y_train, X)

res = mod.fit()

print("the vector needed is : ", res.params)


the vector needed is :  [0.54285714 1.26666667]

[30]
dir(res) #this is what we used to figure out that in order to get the vector, we need to use params
['HC0_se',
 'HC1_se',
 'HC2_se',
 'HC3_se',
 '_HCCM',
 '__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_abat_diagonal',
 '_cache',
 '_data_attr',
 '_data_in_cache',
 '_get_robustcov_results',
 '_is_nested',
 '_use_t',
 '_wexog_singular_values',
 'aic',
 'bic',
 'bse',
 'centered_tss',
 'compare_f_test',
 'compare_lm_test',
 'compare_lr_test',
 'condition_number',
 'conf_int',
 'conf_int_el',
 'cov_HC0',
 'cov_HC1',
 'cov_HC2',
 'cov_HC3',
 'cov_kwds',
 'cov_params',
 'cov_type',
 'df_model',
 'df_resid',
 'eigenvals',
 'el_test',
 'ess',
 'f_pvalue',
 'f_test',
 'fittedvalues',
 'fvalue',
 'get_influence',
 'get_prediction',
 'get_robustcov_results',
 'initialize',
 'k_constant',
 'llf',
 'load',
 'model',
 'mse_model',
 'mse_resid',
 'mse_total',
 'nobs',
 'normalized_cov_params',
 'outlier_test',
 'params',
 'predict',
 'pvalues',
 'remove_data',
 'resid',
 'resid_pearson',
 'rsquared',
 'rsquared_adj',
 'save',
 'scale',
 'ssr',
 'summary',
 'summary2',
 't_test',
 't_test_pairwise',
 'tvalues',
 'uncentered_tss',
 'use_t',
 'wald_test',
 'wald_test_terms',
 'wresid']
Now, we will visualize the statistics of the regression :

[31]
import warnings   
warnings.filterwarnings('ignore')
print(res.summary()) #we print out the summary to get all the statistics for this regression 
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.755
Model:                            OLS   Adj. R-squared:                  0.693
Method:                 Least Squares   F-statistic:                     12.31
Date:                Fri, 02 Dec 2022   Prob (F-statistic):             0.0247
Time:                        17:54:58   Log-Likelihood:                -4.6879
No. Observations:                   6   AIC:                             13.38
Df Residuals:                       4   BIC:                             12.96
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.5429      0.155      3.508      0.025       0.113       0.972
const          1.2667      0.603      2.102      0.103      -0.407       2.940
==============================================================================
Omnibus:                          nan   Durbin-Watson:                   1.919
Prob(Omnibus):                    nan   Jarque-Bera (JB):                0.501
Skew:                          -0.468   Prob(JB):                        0.778
Kurtosis:                       1.939   Cond. No.                         9.36
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

At this point,We are asked to define R-squared as well as Adj.R-squared :

-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable. In other words, r-squared shows how well the data fit the regression model (the goodness of fit).

-The adjusted R-squared is a modified version of R-squared that accounts for predictors that are not significant in a regression model. In other words, the adjusted R-squared shows whether adding additional predictors improve a regression model or not.

[32]
print(res.rsquared) #we print out the values of the two indicators
print(res.rsquared_adj)

0.7547038327526133
0.6933797909407666

R-squared is defined as the percentage of the response variable variation that is explained by the predictors in the model collectively.Therefore, based on its value , we can see that 25% of the variability in the outcome data cannot be explained by the model . So, an R-squared of 0.75 means that the predictors explain about 75% of the variation in our response variable. Based on the value of adj.R-squared , we can conclude that adding data to the model will help better it . So based on these conclusions, we can say that this regression is good since most of the data can be in fact explained and visualized by the model, but its quality can definitely become a whole lot better if we add more data to the model.

Now, we will be using sklearn to do a regression.

We will take inspiration from the link given (just like what we did for statsmodels) in order to write a python code that return as an output the two variables beta_0 and beta_1.

[33]
import numpy as np
from sklearn.linear_model import LinearRegression #We import all the libraries and functions needed for this code

x_train=x_train.reshape(-1,1) # we reshape the data so we can use the linear regression function
y_train=y_train.reshape(-1,1) # we reshape the data so we can use the linear regression function

reg = LinearRegression().fit(x_train, y_train) #we apply and fit the linear regression to the data

print(reg.coef_) #we print out beta_0
print(reg.intercept_)#we print out beta_1


[[0.54285714]]
[1.26666667]

IV- Regression with a real dataset via scikit-learn :
In this part, we will try to use the library scikit-learn in order to make a regression for a reel dataset :

We will start off by uploading the file that we will be using in this part.

[34]
from google.colab import files 
mtcars= files.upload() #we will import the data
Saving mtcars.csv to mtcars.csv


[35]
import pandas as pd #we import the pandas library
carset= pd.read_csv('mtcars.csv') #we visualize the data imported from the file that we have named carset
carset

Now, we will split the dataset(carset) into two (train-carset and test-carset).

[36]
from sklearn.model_selection import train_test_split 
train_carset, test_carset= train_test_split(carset, test_size=0.2, random_state=42)
train_carset #we split the data while giving 20% of it to the test_carset dataset and we visualize train_carset



[37]
test_carset #here we visualize test_carset after the split and we can see that only 20% of the carset is in this dataset

Now we will print the dimensions of each dataset after the big split

[38]
print(carset.shape) #we print the dimension of carset
(32, 12)

[39]
print(train_carset.shape) #we print the dimension of train_carset
(25, 12)

[40]
print(test_carset.shape)#we print the dimension of test_carset
(7, 12)

After that , we will construct y_train :

[41]
y_train= train_carset['mpg'] #we construct y_train now and define it as the mpg colonne in train_carset
print(y_train) #we print it so we can visualize it
25    27.3
12    17.3
0     21.0
4     18.7
16    14.7
5     18.1
13    15.2
11    16.4
23    13.3
1     21.0
2     22.8
26    26.0
3     21.4
21    15.5
27    30.4
22    15.2
18    30.4
31    21.4
20    21.5
7     24.4
10    17.8
14    10.4
28    15.8
19    33.9
6     14.3
Name: mpg, dtype: float64

After checking the carset, it seems like hp and wt could both be good predictors since the values of data in those two are the closest to those in mpg.

We will now scatter plot each of them so we can decide which one is the better predictor and fit for mpg :

[42]
x1_train=train_carset['hp']
#we will scatter plot hp
plt.scatter(x1_train,y_train)

<matplotlib.collections.PathCollection at 0x7fe4088eba90>

[43]
x2_train=train_carset['wt']
#we will scatter plot wt
plt.scatter(x2_train,y_train)
<matplotlib.collections.PathCollection at 0x7fe4088c63d0>

After the visualization of both predictors, we choose wt as the best one since its visualization is the closer to a regression line shape. Now we will sklearn to fit the data with a simple linear regression(we will use the predictor that we judged as the best one(wt)). After that , we will effect predictors to the data_for_test.

[44]
from sklearn.linear_model import LinearRegression
X_train =  np.array(train_carset.wt)
X_train = X_train.reshape(X_train.shape[0], 1)
y_test = np.array(test_carset.mpg) #we define the predictors for the test_carset
X_test =  np.array(test_carset.wt)
X_test = X_test.reshape(X_test.shape[0], 1)
reg = LinearRegression().fit(X_train, y_train)#we fit the data and do a simple linear regression
[45]
y_predicted=reg.predict(X_test)
y_predicted #we predict the values for the test dataset
array([22.15398263,  7.98974016, 16.41677063, 25.19603923, 20.1259449 ,
       18.5782319 , 17.88442951])
[46]
reg.score(X_test, y_test)#we print out the score of the regression of the test_carset
0.6879761857596277
We will be representing now the data_test that we have :

[47]
plt.plot(X_test, y_test, "bo", label="donnees") # the data
           
plt.xlabel("x") # name of the axe x
plt.ylabel("y") # name of the axe y
plt.xlim(0, 11) # ladder for the axe x
plt.legend() # the legend

<matplotlib.legend.Legend at 0x7fe408827b80>

We will now represent the predictions that we have done .

[48]
plt.plot(X_test, y_predicted, "bo", label="predictions") # the data
           # legende
plt.xlabel("x") # name of the axe x
plt.ylabel("y") # name of the axe y
plt.xlim(0, 11) # ladder for the axe x
plt.legend() # the legend
<matplotlib.legend.Legend at 0x7fe408802e50>

Finally, we will attempt to calculate the mean squared errors of the test and training as well as the coefficients of the regression model :

[49]

from sklearn.metrics import mean_squared_error #we import what's needed to calculate the mean squared errors

print(mean_squared_error(reg.predict(X_test), y_test))
print(mean_squared_error(y_train, reg.predict(X_train))) #we print out the mean squared errors based on the dataset that we have as well as the predictions that we got from the model
print('Coefficients: \n', reg.coef_[0], reg.intercept_) #we print out the coefficients of the regression model
12.475985659918818
7.7736977663875155
Coefficients: 
 -5.336941400557079 36.93731031351841

Now , we will be scatter plotting the data as well as the predicted values on top of each other; so we can visualize how accurate these predictions were.

[50]
plt.plot(X_test, y_test, "bo", label="donnees",color= 'royalblue') # the data
           
plt.plot(X_test, y_predicted, "bo", label="predictions",color='tomato')#the predictions
plt.xlabel("x") # name of the axe x
plt.ylabel("y") # name of the axe y
plt.xlim(0, 11) # ladder for the axe x
plt.legend() # the legend
<matplotlib.legend.Legend at 0x7fe4087b3f40>

Bonus :
In this bonus question, we are asked to represent mpg in function of both of her predictors: hp and wt and do a regression to all of that.

In order to accomplish this, we have to define all the data and then find the new x vector for this 3D regression by the the QR decomposition, then we will find our c0,c1,c2 variables which are in fact the elements of the (3,1) vector x and we will use them to predict the outcome of the model as well as visualize the data and the data_predicted by the model :

[70]
train_carset1, test_carset1= train_test_split(carset, test_size=0.2, shuffle=False) #we start off by splitting the original data into train_carset1 and test_carset1
train_carset1
x1_train_1=train_carset1['hp']
x2_train_1=train_carset1['wt'] #we define x1_train_1, x2_train_2,y_train_1 respectively as the hp,wt and mpg data in train_carset1(the original outcome and the two best predictors for it)
y_train_1=train_carset1['mpg']
x2_train_1
x_0=np.ones(25)
Z=np.stack((x_0,x1_train_1,x2_train_1)) 
Z=np.transpose(Z)#we define the matrix Z for this regression and transpose it
Z 
Q1,R1= np.linalg.qr(Z)
np.shape(Z)
Q1t=np.transpose(Q1)     #we decomposite our Z matrix into a QR factorisation then we transpose Q and inverse R and use it all to get the x vector needed for this regression
R1inv=np.linalg.inv(R1)
x=R1inv@Q1t@y_train_1
print(x)
c0=x[0]
c1=x[1]                #we get the three elements of the (3,1) x vector and name them so we can manipulate them
c2=x[2]
c=len(x1_train_1)
y_train_carset_predicted_s=np.zeros(c)  #we define the  data predicted by the model
for i in range(c):
  y_train_carset_predicted_s[i]=c0+(x1_train_1[i])*c1+(x2_train_1[i])*c2 #here, we use the elements of the vector x to get the predicted data using a for loop in the range of the dimension of the data that we have

from mpl_toolkits.mplot3d import axes3d #we import what's needed to do a 3D with three axes visualization
fig = plt.figure()
ax = fig.gca(projection='3d')
l=np.zeros(25)
ax.scatter3D(x1_train_1,x2_train_1,y_train_carset_predicted_s)#we scatter plot the predicted data in red
ax.scatter3D(x1_train_1,x2_train_1,y_train_1)#we scatter plot the original data 
[36.50301827 -0.04620683 -3.07564075]

<mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x7fe40738efd0>

