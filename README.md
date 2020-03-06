![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# Final Project Demo

## Overview

This is a dataset for a Portugal bank marketing campaigns results. Conducted campaigns were based mostly on direct phone calls, offering bank's clients to place a term deposit. If after all marking afforts client had agreed to place deposit - target variable marked 'yes', otherwise 'no'.

## The Data

After manipulating the data, as well as inspecting it, we can see that:

1) There is not null values

2) We can see which columns don't contain useful information for our predictions:

* The id_var column contains a unique identifier for each row and will not be useful for prediction.

* The duration column is the last contact duration. This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

3) The column y is categorical, so we have to transform it to numerical.

4) The column pvalue has a value of 999 when the client was not previously contacted, and it's very different from the rest of the values. We don't want to consider 999 as an outlier since there is a lot of rows with this value. In this case we decide to drop this column, since could confuse the ML algorithm.

5) There are some outliers. We have to remove them.

6) The dependent variable is imbalanced, so we have to balance it.

7) We should analyze the correlation of numerical features and delete the columns highly correlated.

8) Transform categorical features to numerical using one-hot encode (get_dummmies).


## Data Modeling

Once we do all of this in our data, we proceeded to prep our data set for modeling. We decided to compare several Supervised Learning classification models via k-fold validation to see which one would best be able to predict if the customer would place the deposit or not. 

https://github.com/yfundorac/final-project/blob/master/main.ipynb

After this, please see the demo of the project.

https://github.com/yfundorac/final-project/blob/master/app.py


## Improvements

* Building a profile of a consumer of banking services (deposits), using unsupervised learning for Customer Segmentation.
* Add some performance metrics to evaluate the models: confusion matrix, precision, recall, F1 score, ROC curve.
