# A machine learning model to predict the clients' yearly purchases from an E-Commerce website using linear regression

Business Case: There is a clothing store that has its E-Commerce website and mobile application. The clients can book a free appointment with the designers/stylists from the store and have a session of discussion about the kind of clothes that they want. After the discussion the clients can go home and place an order on what the stuff they want to purchase.

Goal of this project: To predict an estimated annual money spent by clients to purchase clothes from the E-Commerce website or application.

We will be using a dataset from kaggle that describes the same data. Link: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXZpVk1kbWc1UlU0WFpDdks2bkltNzUyRUxFQXxBQ3Jtc0tsMkRTZkhJVkZSMjlhMmpaZ3UyLUtVQXNTSzhVME04MUI2WVh4aGFvbFh3Rjc4aTNkV05nbC16UVc5LWdfaS1PTDdxVXZsN3RyT1B1OWxkek04WEFiS3hKWEJyelFEWW4zWFZOOXZuRU1NSjRHVDZ6RQ&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fkolawale%2Ffocusing-on-mobile-app-or-website&v=O2Cw82YR5Bo

About the dataset: The dataset contains the features that help us use them to train the model. The features are:
	• Email, Address of Clients
	• Avatar - Probably the label used for clothes in the website/application
	• Time Spent on App by clients- in minutes
	• Time Spent on website by clients-  in minutes
	• Average Session length by clients-  in minutes
	• Length of membership-  in months
	• Money spent yearly - in dollars
Out of these we will only use the numerical data for understanding the correlations between various features and our target variable i.e. Money spent yearly

EDA (Elaborative Data Analysis)
**Time spent on App vs Yearly amount spent**


The areas of scatter plot that have darker colour say that there are more points in that area.

Correlation measures how strongly two variables are related — in other words, how much one variable tends to change when the other does.

There isn't any correlation between Time on Website and Yearly Amount Spent.

**Time Spent on App vs Yearly Amount Spent**

There is a faint correlation between Time on Website and Yearly Amount Spent. More time spent on App, more likely the client will spend the amount per year.

**Correlation between Yearly Amount Spent vs All the variables**

It can be observed that 
	• faint correlation between Average Session length and Yearly Amount Spent.
	• faint correlation between Time On App and Yearly Amount Spent.
	• definite positive correlation between Length of Membership and Yearly Amount Spent.


Seaborn's lmplot() stands for "linear model plot. It combines a scatter plot and a regression line (fitted using statsmodels or scipy)

Linear Regression: Goal is to create a line which will be close to all the points. This closeness can be measured by a loss function called Mean Squared Error (MSE). The smaller the MSE the smaller will be the loss and we find the line that leads to minimum MSE.

Training data vs Testing Data

We will train model on 70% of data set and test on the remaining 30% . We use the train_test_split utility from the package sklearn.model_selection.

We use code: 
	train_test_split ( Features, Target, test_size=0.3, random_state=42 )
	
Random_state = Without random_state, train_test_split() will give you different results every time (since it shuffles randomly by default). It ensures the same split every time you run your code — which is crucial for reproducibility.

Training the model

We already have python packages that have built in models. One we will use here is LinearRegression from the package of sklearn.linear_model

We just have to initialize the model and the use code:
               lm.fit ( X_training_data, Y_training_data )   
=> This will create a line that is closest to all the points of our dataset


Evaluation of our linear model

We do this to visually compare the model’s predictions against the true (actual) values from the test set.
 If the model is perfect:
	• All points will lie on a 45-degree diagonal line (y = x)
	• Prediction = Actual for every test point
	
 If the model is good but not perfect:
	• Points will cluster around the diagonal line
	• The closer the points to the line, the better the model
	
 If the model is poor:
	• Points will be scattered randomly
	• No visible pattern or trend

Evaluation of our linear model

Mean Absolute Error: MAE is the average of the absolute differences between predicted values and actual values.
Formula:![image](https://github.com/user-attachments/assets/230717f6-bb0b-42e0-bc8f-dfe6532bd507)


Mean Squared Error: MSE is the average of the squared differences between predicted and actual values.![image](https://github.com/user-attachments/assets/8abbab6a-9b1e-4df0-85ef-65fe7f90d35a)

Evaluation of errors/ residuals

It is usually assumed in the linear regression that the distribution of the residuals is a normal distribution. The residuals (errors) are normally distributed with mean zero.

Why is it normally distributed?
Because many statistical inferences you do after fitting a linear model rely on this assumption. Like:
	1. To Make Valid Confidence Intervals and Hypothesis Tests
	2. Model Quality Assessment (R², F-statistics)
	3. Etc.

We can check whether the residuals are normally distributed or not by using 
	residuals = Target_test - predictions
	sns.displot(residuals, bins=30, kde=True)

kde=True: Adds a smooth kernel density estimate curve (like a smoothed version of the histogram)![image](https://github.com/user-attachments/assets/d85aad45-d709-47b7-868e-109787e27414)



