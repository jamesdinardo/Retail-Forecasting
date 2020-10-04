# Retail-Forecasting
This project uses historical sales data for 45 retail stores and predicts the department-wide weekly sales of each store for the following year. The data comes from [Kaggle](https://www.kaggle.com/manjeetsingh/retaildataset) and contains information about the store type, store size, temperature, price of fuel, store department, consumer price index each week, whether a holiday occurred that week, and the sales that week.

## Table of Contents

1. Introduction

2. Overview of the data

3. Exploratory Data Analysis

4. Modeling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a. K-Nearest Neighbors

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

Our dataset has 45 unique stores, 81 unique departments, 3 unique types, and 3323 unique store department combinations. The following table displays general summary statistics for the continuous variables:

~~~
df.describe().T
~~~

![Plot1](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/describe.png)

We see that the average store deparment does $16027.10 in sales per week, with a standard deviation of $22726.51. We can plot the average weekly sales as a function of date, using a line plot:

~~~
average_sales_per_week_per_department = df.groupby('Date')['Weekly_Sales'].mean()

fig, ax = plt.subplots(figsize=(15, 5))
_ = ax.set_ylabel('Weekly Sales')
_ = ax.set_title('Average Weekly Sales Per Store Department')
_ = average_sales_per_week_per_department.plot()
~~~

![Plot2](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/sales_per_week.png)

Sales appear steady for most of the year up until the holidays, where there is a noticable increase in sales for both 2010 and 2011. The following plot shows the same data, only with each year separated vertically. Note that the final year of data, 2012, only has sales data up until 2012-12-10, which is why the line flattens out toward the end of the year:

~~~
df_indexed = df.set_index('Date')

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))
_ = ax[0].plot(df_indexed['2010'].groupby('Date')['Weekly_Sales'].mean())
_ = ax[1].plot(df_indexed['2011'].groupby('Date')['Weekly_Sales'].mean())
_ = ax[2].plot(df_indexed['2012'].groupby('Date')['Weekly_Sales'].mean())

_ = ax[0].set_yticks([10000, 15000, 20000, 25000])
_ = ax[1].set_yticks([10000, 15000, 20000, 25000])
_ = ax[2].set_yticks([10000, 15000, 20000, 25000])

_ = ax[0].set_title("Average Weekly Sales Per Store Department, Per Year")
_ = ax[1].set_ylabel("Sales")
_ = ax[2].set_xlabel("Date")
~~~

![Plot3](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/subplots_per_year.png)

Stores and departments vary considerably. Moreover, they are not equally represented in the dataset.

The average store does $1046624.03 in sales per week, but there is a large difference between the store with the highest, lowest, and median sales:

~~~
min_max_median_store = df[df['Store'].isin([5, 20, 45])].groupby(['Date', 'Store'])['Weekly_Sales'].sum().dropna().reset_index()

fig, ax = plt.subplots(figsize=(15, 5))
_ = sns.lineplot(x='Date', y='Weekly_Sales', hue='Store', data=min_max_median_store)
_ = plt.xticks(rotation='90')
_ = plt.legend(['Minimum: Store 5', 'Maximum: Store 20', 'Median: Store 45'])
~~~

![Plot4](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/min_median_max_stores.png)

The trends observed earlier appear to be consistent among stores with different sales volumes, as each line follows the same general pattern.

There is also a disparity between sales volume for different departments:

~~~
df.groupby('Dept').agg({'Weekly_Sales':'mean'}).sort_values(by='Weekly_Sales', ascending=False)
~~~

![Plot5](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/average_weekly_sales_by_dept.png)

In addition, some of the departments only appear for a few weeks, suggesting that those departments might have been phased out or merged with other departments. Since we are predicting store department sales, each of these variables are likely to be important when building models. Let's look at the linear correleation between each of the continuous variables in the dataset. An easy way to view linear correlation is to construct a heatmap:

~~~
fix, ax = plt.subplots(figsize=(7, 5))
_ = sns.heatmap(df.corr(), square=True, cmap='coolwarm', ax=ax)
~~~

![Plot6](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/heatmap.png)

Values closer to 0 indicate weak or no correlation, positive values indicate positive correlation, and negative values indicate negative correlation. The only variable with much correlation to the target (weekly sales) is Size, which make sense since larger stores tend to sell more. But it is improtant to note that the heatmap only shows linear one-to-one correlation, so it is possible that variables are correlated to the target in tandem with eachother or in non-linear ways.

Another useful type of plot is the countplot, which counts the number of instances of each unique categorical variable. The following code creates three categorical variables by binning the continuous variable, Size, into three categories: small, medium, and large. Then it maps the Type variable to color and counts how many instances there are for each:

~~~
df['Size_Category'] = pd.cut(df['Size'], bins=[0, 100000, 200000, np.inf], labels=['Small', 'Medium', 'Large'])

_ = sns.countplot(x='Size_Category', hue='Type', data=df)
_ = plt.xlabel("Size Category")
_ = plt.ylabel("Count")
~~~

![Plot7](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/size_category_count.png)

What this plot shows is that Size and Type are correlated. All large stores (>200000) are of Type A, and all stores of Type C are small (<100000). 

## 4. Modeling

Before training any machine learning models, we have to "preprocess" our data, which means getting it into a format that the models can understand and perform well on. The scikit-learn API cannot directly work with columns of type "object." So we create dummy variables in Pandas, using the following code:

~~~
df_dummies = pd.get_dummies(df_
~~~

### a. K-Nearest Neighbors

The K-Nearest Neighbors or KNN algorithm works by mapping out the feature values for the training data, and then comparing each new data point that we want to predict to those values. Imagine that are dataset had only 1 feature, Size, and 1 target, Weekly Sales. The sizes for all training datapoints are stored, and we compare the size of a new datapoint to predict. The algorithm finds the K-Nearest Neighbors--that is, the K datapoints that have the most similar size to the new datapoint--takes the mean target value of those datapoints, and uses that value for the prediction. K is a hyperparamter that we set manually, so if we set K=5, the algorithm will find the closest 5 datapoints based on store size, and predict that our new datapoint's target value (weekly sales) is that of the mean of those 5 points.



### b. Linear Models

Linear models attempt to fit a straight line through the datapoints, and use this line to make predictions on new datapoints. 


### c. Decision Tree Regressor

A decision tree asks a series of true or false questions about the data in order to sort them into nodes. For example, we might first ask "Is the deptartment 92?" and move data into the left hand node if not, and right hand node if yes. This is, in fact, the first question (root node) asked by the algorithm of our dataset:

![Plot9](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/tree.png)



### d. Random Forest Regressor

The last two classes of algorithms we will try are called "ensemble methods," since they combine several machien learning models into a meta-model. Random Forests work by creating multiple decision trees and then averaging out their predictions.

### e. Gradient Boosted Trees

Boosting is the process by which we build models sequentially, with each model making adjustments to improve the results of previous models. In our case, we build decision trees one at a time, with each tree (called a base learner) learning from the mistakes of the previous tree.

## 5. Conclusion

~~~
from pandas.plotting import autocorrelation_plot

time_series = df_indexed.groupby(df_indexed.index)['Weekly_Sales'].mean()

fig, ax = plt.subplots(figsize=(15, 5))
autocorrelation_plot(time_series, ax=ax)
_ = plt.xlim(0, 60)
_ = plt.ylim(-.4, .4)
_ = plt.xticks(range(0, 60, 1))
_ = plt.title('Autocorrelation for Various Lag Values of Weekly Sales')
_ = plt.annotate('Lag = 1 year', xy=(52, .35))
~~~

![Plot10](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/autocorrelation.png)
