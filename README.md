# Retail-Forecasting
This project uses historical sales data for 45 retail stores and predicts the department-wide weekly sales of each store for the following year. The data comes from [Kaggle](https://www.kaggle.com/manjeetsingh/retaildataset) and contains information about the store type, store size, temperature, price of fuel, store department, consumer price index each week, whether a holiday occurred that week, and the sales that week.

## Table of Contents

1. Introduction

2. Overview of the data

3. Exploratory Data Analysis

4. Modeling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a. Nearest Neighbors

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b. Linear Models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; d. Decision Tree Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; e. Random Forest Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; f. Gradient Boosted Trees

5. Conclusion

## 1. Introduction

One of the key challenges for retail companies is to predict how well a store or department will perform in the future based on past performance. Given past information about a particular store department--its geographical location, weather, weekly consumer price index, previous sales etc--can we predict what its sales will be for a given week next year? This project attempts to predict the 2012 weekly sales for each department of a set of 45 retail stores, based on data from 2011-2011 (see section 2 for the full list of variables). It makes use of several data-science libraries in Python--Pandas for data cleaning and analysis, Matplotlib for data visualization, and scikit-learn for machine learning. We will be concerned not only with coming up with the best possible model, but also comparing and contrasting the performance of different algorithms. 

The code is written using Python inside of Jupyter notebooks (ipynb files). There are 7 notebooks in total: one for data preparation, one for data exploration, and one for each of the 5 types of models that will be trained, tested, and evaluated. This readme contains only minimal code and visualizations needed to express the main insights: the full code can be found in the notebooks, which can be downloaded should one wish to experiment on their own with the data. Additionally, I have created an easy to use application, written in Flask, where the user can input information, such as the store, department, and week, and immediately receive a prediction for weekly sales. 

## 2. Overview of the Data

The data is contained in three csv files: stores, sales, and features. After merging this data and eliminating rows with negative values
we are left with a single dataframe containing 418660 rows and 16 columns. Each row represents a week of sales
for a particular store department. Each column is a variable that pertains to some aspect
of that week of sales. Our task is to use the first 15 variables (called "features") to predict the 
variable "Weekly_Sales" (called the "target"). A description of each variable is as follows:

Store - The store number, ranging from 1 to 45

Date - The week for which sales were calculcated

Temperature - Average temperature (Farenheit) in the region in which the store is located

Fuel_Price - Cost of fuel (dollars per gallon) in the region in which the store is located

MarkDown1-5 - Promotional markdowns (discounts) that are only tracked from Nov. 2011 onward

CPI - Consumer Price Index, which measures the overall costs of goods and services bought by
a typical customer. As CPI rises, the the average customer has to spend more to maintain the same
standard of living. It is calculated as follows:

(price of a basket of goods in current year / price of the basket in the base year) x 100

Unemployment - Unemployment Rate

IsHoliday - True if the week contains a holiday

Dept - The department number

Type - The type of store (A, B, or C). No further information on type is provided, but it appears to be correlated to the size of the store (see section 3).

Size - The size of the store

Weekly_Sales - The sales for a given department within a given store that week. This is the target variable that we are trying to predict.

## 3. Exploratory Data Analysis

To begin, we can examine some general summary statistics for each variable:

~~~
df.describe().T
~~~

![Plot1](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/describe.png)





