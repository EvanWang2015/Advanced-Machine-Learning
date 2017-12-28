# Overfitting and Regularization

### 1

Select correct statements about overfitting:

Overfitting happens when model is too simple for the problem


> Large model weights can indicate that model is overfitted


> Overfitting is a situation where a model gives lower quality for new data compared to quality on a training sample


Overfitting is a situation where a model gives comparable quality on new data and on a training sample

### 2
What disadvantages do model validation on holdout sample have?


> It can give biased quality estimates for small samples


It requires multiple model fitting


> It is sensitive to the particular split of the sample into training and test parts

### 3
Suppose you are using k-fold cross-validation to assess model quality. How many times should you train the model during this procedure?


1

> k

k(k−1)/2

k^2

### 4
Select correct statements about regularization:


Reducing the training sample size makes data simpler and then leads to better quality


Weight penalty reduces the number of model parameters and leads to faster model training


> Weight penalty drives model parameters closer to zero and prevents the model from being too sensitive to small changes in features


> Regularization restricts model complexity (namely the scale of the coefficients) to reduce overfitting