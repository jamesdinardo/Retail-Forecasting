# Retail-Forecasting
This project uses historical sales data for 45 retail stores and predicts the department-wide sales of each store for the following year. The data comes from [Kaggle](https://www.kaggle.com/manjeetsingh/retaildataset) and contains information about the store type, store size, temperature, price of fuel, store department, consumer price index each week, whether a holiday occurred that week, and the sales that week.

## Table of Contents

1. Introduction

2. Overview of the data

3. Exploratory Data Analysis

4. Modeling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a. Nearest Neighbors

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b. Linear Regression and Regularized Regression

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c. Support Vector Machines

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; d. Decision Tree Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; e. Random Forest Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; f. Gradient Boosted Trees

5. Conclusion

## 1. Introduction

One of the key challenges for retail companies is to predict how well a store or item will perform in the future based on past performance. Given five years of information about a particular store--it's geographical location, weather, weekly consumer price index, weekly sales--can we predict what it's sales are likely to be next year? This project attempts to predict the following year's sales for 45 retail stores based on these and other variables (see section 2 for the full list). It makes use of several data-science libraies in Python--Pandas for data cleaning and analysis, Matplotlib for data visualization, and scikit-learn for machine learning. We will be concerned not only with coming up with the best possible model, but also comparing and contrasting the performance of different algorithms. 

The code is written using Python inside of Jupyter notebooks. There are 7 notebooks in total: one for data exploration, and one for each of the 7 models that will be trained, tested, and evaluated. The notebooks can be downloaded should one wish to experiment on their own with the data. Moreover, I have created an easy to use eapplication, written in Flask, where the user can enter a particular store and get predictions for that store for the following year. 

## 2. Overview of the Data
