# Retail-Forecasting
This project uses historical sales data for 45 retail stores and predicts the weekly sales for each store department for the following year. The data comes from [Kaggle](https://www.kaggle.com/manjeetsingh/retaildataset) and contains information about the store type, store size, temperature, price of fuel, store department, consumer price index each week, whether a holiday occurred that week, and the sales that week.

## Table of Contents

1. Introduction

2. Overview of the data

3. Exploratory Data Analysis

4. Modeling

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a. K-Nearest Neighbors

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b. Linear Models

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; d. Decision Tree Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; e. Random Forest Regressor

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; f. Boosted Trees

5. Conclusion

## 1. Introduction

One of the key challenges for retail companies is to predict how well a store or department will perform in the future based on past performance. Given past information about a particular store department--its size, previous sales, the temperature and unemployment rate that week etc--can we predict what its sales will be for a given week next year? This project attempts to predict the 2012 weekly sales for each department of a set of 45 retail stores, based on data from 2010-2011 (see section 2 for the full list of variables). It makes use of several data-science libraries in Python--pandas for data cleaning and analysis, matplotlib for data visualization, and scikit-learn for machine learning. The goal us to implement several different models (including linear, tree-based, and ensemble methods) to come up with a best model that can predict sales within one standard deviation of the mean.

The code is written in Python inside of Jupyter notebooks (ipynb files). There are 7 notebooks in total: one for data preparation, one for data exploration, and one for each of the 5 types of models that will be trained, tested, and evaluated. This readme contains only minimal code and visualizations needed to express the main insights: the full code can be found in the notebooks, which can be downloaded should one wish to experiment on their own with the data. Additionally, I have created an easy to use application, written in Flask, where the user can input information, such as the store, department, and week, and immediately receive a prediction for weekly sales. 

## 2. Overview of the Data

The data is contained in three csv files: stores, sales, and features. After merging this data and eliminating rows with negative values
we are left with a single dataframe containing 418660 rows and 16 columns. Each row represents a week of sales
for a particular store department. Each column is a variable that pertains to some aspect
of that week of sales. Our task is to use the first 15 variables (called "features") to predict the 
variable "Weekly_Sales" (called the "target"). A description of each variable is as follows:

Store - The store number, ranging from 1 to 45

Date - The week for which sales were calculcated

Temperature - Average temperature (Fahrenheit) in the region in which the store is located

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

The average store does $1046624.03 in sales per week, but there is a large difference between the stores with the highest, lowest, and median sales:

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

Values closer to 0 indicate weak or no correlation, positive values indicate positive correlation, and negative values indicate negative correlation. The only variable with much correlation to the target (Weekly_Sales) is Size, which make sense since larger stores tend to sell more. But it is improtant to note that heatmaps only show linear one-to-one correlation, so it is possible that variables are correlated to the target in tandem with eachother or in non-linear ways.

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

Before training any machine learning models, we have to "preprocess" our data, which means getting it into a format that the models can understand and perform well on. Each modeling notebook begins with the following preprocessing steps:

- Create three categorical features from Date--Week, Month, and Year--and then drop Date

- The scikit-learn API cannot directly work with columns of type "object." So we create dummy variables in Pandas, using the following code:

~~~
df_dummies = pd.get_dummies(df, drop_first=True)
~~~

- split the data into train and test sets, where X_train and X_test contain the features and y_train and y_test contain the target:

~~~
X_train = df_dummies.loc[(df['Year']==2010) | (df['Year']==2011), :].drop('Weekly_Sales', axis=1).values
X_test = df_dummies.loc[df['Year']==2012, :].drop('Weekly_Sales', axis=1).values
y_train = df_dummies.loc[(df['Year']==2010) | (df['Year']==2011), 'Weekly_Sales'].values.reshape(-1, 1)
y_test = df_dummies.loc[df['Year']==2012, 'Weekly_Sales'].values.reshape(-1, 1)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
~~~

Shape of X_train: (313995, 201)\
Shape of X_test: (104665, 201)

Now, roughly 75% of the data will be used to train the model, and 25% will be used to test (evaluate) the model. Note that creating dummy variables greatly increases the number of features to 201. We can use feature selection techniques to reduce this number, or keep it as is depending on how the model is performing.

The goal is to optimize the performance of each model, and then select the model with the best performance. We will use to metrics from the scikit-learn library to measure performance. R2 (pronounced "R squared") evaluates the fit of the model, with 1.0 being a perfect fit. The Root Mean Squared Error (RMSE) squares the difference between each prediction and the real value, sums them all up, and then takes the square root. Since the standard deviation of weekly sales is around $22000, a good model should have an RMSE well below that.

### a. K-Nearest Neighbors

The K-Nearest Neighbors or KNN algorithm works by mapping out the feature values for the training data, and then comparing each new datapoint that we want to predict to those values. Imagine that our dataset had only 1 feature, Size, and we wanted to predict the Weekly Sales. The sizes for all training datapoints are stored, and we compare the size of a new datapoint for which we want to predict sales. The algorithm finds the K-Nearest Neighbors--that is, the K datapoints that have the most similar size to the new datapoint--takes the mean target value of those datapoints, and uses that value for the prediction. K is a hyperparamter that we set manually, so if we set K=5, the algorithm will find the closest 5 datapoints based on store size, and predict that our new datapoint's target value (weekly sales) is that of the mean of those 5 points.

The best KNN model is one where we use feature selection to retain only the top 50 features, which improves performance and speed.

~~~
from sklearn.feature_selection import SelectPercentile

selection = SelectPercentile(percentile=25)
selection.fit(X_train, y_train)
X_train_selected = selection.transform(X_train)

knn = KNeighborsRegressor(n_jobs=-1)
knn.fit(X_train_selected, y_train)

X_test_selected = selection.transform(X_test)

y_pred = knn.predict(X_test_selected)

print('R2: {}'.format(metrics.r2_score(y_test, y_pred)))
print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
~~~

R2: 0.48\
RMSE: 15892.47

KNN does a decent job but is especially slow to train. Next we will consider various linear models.

### b. Linear Models

Linear models attempt to fit a straight line through the datapoints, and use this line to make predictions on new datapoints. We will try three linear models: Linear Regression, Lasso, and Ridge. 

Linear Regression models create a best fit line by finding the line that minimizes a cost function. Specifically, the best fit line is the one that minimizes the squared difference (residuals) between each datapoint and the line. Linear Regression is also known as Ordinary Least Squares (OLS), since it tries to minimize the sum of the squared residuals. Whichever line does this the best is used as a model for predicting new datapoints. Again, imagine that we had only a single feature, Size, and wanted to predict Weekly Sales. In this case, the prediction (y) would be the function of the input w * x + b, where w is the coefficient and b is the y-intercept, both of which are learned in training the model.

Lasso and Ridge are examples of "regularization," which means penalizing models that have very large coefficients to keep them from overfitting. Lasso (L1 regularization) adds a penalty to the cost function equal to the sum of the absolute value of the coefficients, while Ridge (L2 regularizaiton) adds a penalty to the cost function equal to the sum of squares of the coefficients. Both have the effect of reducing the coefficients that appear in the final equation for the model. 

The best performing linear model is a Ridge regression trained on a dataset that has been reduced to 124 features by keeping track of only the week of the month (from 0 to 4) and dropping some departments:

~~~
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print('R2 using L2 regularization, week of the month, best dept: {:.2f}'.format(metrics.r2_score(y_test, y_pred)))
print('RMSE using L2 regularization, week of the month, best dept: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
~~~

R2 using L2 regularization, week of the month, best dept: 0.63\
RMSE using L2 regularization, week of the month, best dept: 13447.08

The results are an improvement from KNN.

### c. Decision Tree Regressor

A decision tree asks a series of true or false questions about the data in order to sort them into nodes. For example, we might first ask "Is the department 92?" and move data into the left hand node if not, and right hand node if yes. This is, in fact, the first question (root node) that is asked of our dataset:

~~~
dt_pruned = DecisionTreeRegressor(random_state=0, max_depth=4)
dt_pruned.fit(X_train, y_train)

features = list(df_dummies.drop('Weekly_Sales', axis=1).columns)

fig, ax = plt.subplots(figsize=(16,10))
tree.plot_tree(dt_pruned, feature_names=features, fontsize=8, filled=True)
plt.show()
~~~

![Plot8](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/tree.png)

The first thing our model does is separate datapoints based on whether or not they are from department 92. Then it asks a number of questions relating to store type, size, and other department information. The plotted tree was limited to a max depth of 4, which means that only 4 splits happen for any datapoint as it makes its way down the tree. We can inspect the R2 results of different values for max depth, keeping in mind that asking too many questions (too large a depth) will result in overfitting the model:

~~~
md_values = np.array([4, 10, 15, 20, 30, 40, None])

for i in md_values:
    dt = DecisionTreeRegressor(random_state=0, max_depth=i)
    dt.fit(X_train, y_train)
    print('Max Depth of {}: {:.2f}'.format(i, dt.score(X_test, y_test)))
~~~

Max Depth of 4: 0.42\
Max Depth of 10: 0.70\
Max Depth of 15: 0.79\
Max Depth of 20: 0.81\
Max Depth of 30: 0.85\
Max Depth of 40: 0.87\
Max Depth of None: 0.87

We can use a modest depth of 10 and then try to reduce the number of features. Decision Tree Regressors have a feature_importances_ method that tells you how important each feature was in building the model:

~~~
print('Total features: {}'.format(len(dt.feature_importances_)))
feature_importances = pd.DataFrame({'Feature': features, 'Feature Importance':dt.feature_importances_}).sort_values(by='Feature Importance', ascending=False)
display(feature_importances.iloc[:50, :])
~~~

![Plot9](https://github.com/jamesdinardo/Retail-Forecasting/blob/master/img/feature_importances.png)

This again suggests that departments are an important predictor of sales, along with store size, and certain holiday weeks (47 and 51). Dropping all but the top 50 features (plus the 3 years) makes our model more robust against overfitting, so we will do that and inspect the results:

~~~
y_pred = dt.predict(X_test)

print('R2 with max depth of 10, 53 features: {:.2f}'.format(dt.score(X_test, y_test)))
print('RMSE with max depth of 10, 53 features: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
~~~

R2 with max depth of 10, 53 features: 0.70\
RMSE with max depth of 10, 53 features: 12156.50

Since our model performs almost as well with a quarter of the features, this is an improvement.

### d. Random Forest Regressor

The last two classes of algorithms we will try are called "ensemble methods," since they combine several machine learning models into a single meta-model. Random Forests work by creating multiple decision trees and then averaging their predictions. Specifically, each tree is allowed to use only a subset of rows, so that each tree will make slightly different predictions.

The best model is one in which we calculate feature importances, reduce our features to only the best 50 (plus the 3 years), and train a model with a modest max depth and number of trees"

~~~
features_to_drop = feature_importances.iloc[50:, 0]
features_to_drop = features_to_drop[~features_to_drop.str.contains('Year')]

df_dummies_top_features = df_dummies.drop(features_to_drop, axis=1)

X_train = df_dummies_top_features.loc[(df['Year']==2010) | (df['Year']==2011), :].drop('Weekly_Sales', axis=1).values
X_test = df_dummies_top_features.loc[df['Year']==2012, :].drop('Weekly_Sales', axis=1).values
y_train = df_dummies_top_features.loc[(df['Year']==2010) | (df['Year']==2011), 'Weekly_Sales'].values.reshape(-1, 1)
y_test = df_dummies_top_features.loc[df['Year']==2012, 'Weekly_Sales'].values.reshape(-1, 1)

rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print('R2: {:.2f}'.format(metrics.r2_score(y_test, y_pred)))
print('RMSE: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
~~~

R2: 0.71\
RMSE: 11920.49

Random Forest does an excellent job with a max_depth of 10 and only a quarter of the features. Our error is now less than 1/2 the standard deviation of weekly sales. We will try one more class of models to see if we can improve performance even more.

### e. Boosted Trees

Boosting is the process by which we build models sequentially, with each model making adjustments to improve the results of previous models. In our case, we build decision trees one at a time, with each tree (called a base learner) learning from the mistakes of the previous tree.

The best model for our problem is an Extreme Gradient Boosted Regressor, which performs well and trains very quickly. As with Random Forest, we have to set max_depth (ideally, a smaller value so that each tree is shallow) and number of estimators (here: "boosting stages"). We also iterate through possible values for learning rate ("eta") which affects how much each tree contributes to the model, subsamples of rows to use, and subsamples of colunns to use. The code for the final model is shown below. Note that we first convert our data into a special structure called "data matrixes" that are optimized to work in the XGBoost learning API.

~~~
#convert data into DMatrixes
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)

params = {'objective':'reg:squarederror', 'max_depth': 5, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.8}

xgb_model = xgb.train(params=params, dtrain=DM_train, num_boost_round=100)

y_pred = xgb_model.predict(DM_test)

print('R2 with 100 boost rounds: {:.2f}'.format(metrics.r2_score(y_test, y_pred)))
print('RMSE with 100 boost rounds: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
~~~

R2 with 100 boost rounds: 0.83\
RMSE with 100 boost rounds: 9175.80

Here we use all features, but restrict each base learner in terms of depth as well as how much of the rows and columns it is allowed to use. The final RMSE is less than 1/2 of the standard deviation of Weekly Sales. 

## 5. Conclusion

The XGBoost Regressor performs the best on our dataset and trains very quickly. Future work could tune this model further by selecting different features, engineering new ones, or trying out different hyperparameter values. For example, we could add unofficial holidays like the superbowl into our dataset, since they might affect sales that week and their exact date changes from year to year. 

We could also take advantage of additional libraries and functions in Python for dealing with time series. For example, one feature you can use for predicting the sales at a time, t2, is the sales at a previous time, t1. The value at a previous time is called a "lagged value" and can be used in predicting a variable that changes over time such as weather, stock prices, and sales. Below is an "autocorrelation plot" that shows the correlation between weekly sales and previous values of weekly sales.

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

The largest correlation occurs at 52 weeks, which means that the sales for a particular week are related to the sales for the same week last year. There is also some correlation between this weeks sales and last week sales, as well as the sales from 6 weeks ago. Approaches such as autocorrelation models and ARIMA would be worth exploring.
