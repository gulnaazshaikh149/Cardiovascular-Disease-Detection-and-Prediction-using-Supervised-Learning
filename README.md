## Cardiovascular Disease Detection and Prediction using Supervised Learning

## Overview
This repository hosts a sophisticated predictive model designed to assess the likelihood of heart disease based on an individual's comprehensive health metrics. Leveraging logistic regression and a curated dataset, the model incorporates various predictors such as age, sex, chest pain type, blood pressure, and cholesterol levels.

## Libraries
The project relies on several essential R libraries to facilitate robust data manipulation, visualization, and model training:

dplyr for data manipulation
gtools and gmodels for general tools
ggplot2 for data visualization
class for classification and regression training
tidyr for data tidying
lattice for trellis graphics
caret for classification and regression training
rmdformats for formatting R Markdown documents
```R
library(dplyr)
library(gtools)
library(gmodels)
library(ggplot2)
library(class)
library(tidyr)
library(lattice)
library(caret)
library(rmdformats)
```
## Data Import and Wrangling
The heart disease dataset (heart.csv) is meticulously imported and subjected to comprehensive preprocessing. Categorical variables are transformed, and missing values are systematically addressed.

## Logistic Regression
### Data Import 
We will use dataset that record an information about patients with heart disease.
```R
heart <- read.csv("heart.csv")
glimpse(heart)
```
![1](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/5bd32c27-ba0a-43dc-be39-379fabc50dae)
Based on the data above there are several information that maybe useful for our analysis: ï..age : age of respondent sex : 1 = male; 0 = female cp : the severe type of pains trestbps : blood pressure (mm Hg) chol : cholesterol (mg / dl) fbs : blood sugar, 1 = above 120 mg / dl; 0 = below 120 mg / dl restecg : electrocardiograph result thalach : maximum hertbeat recorded exang : exercised include angina, 1 = yes; 0 = no oldspeak : ST Depression, based on exercise to rest slope : Slop of ST Segment ca : Number of the major blood vessels thal : 3 = Normal, 6 = permanently disable, 7 = reversable disable target : 1 = sick; 0 = not sick

```R
head(heart, 3)
```
![2](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/76e7c43f-2cfa-48c2-9ec2-12b3d7e7d6d7)

## Data Wrangling
```R
heart <- heart %>%
  #mutate_if(is.integer, as.factor) %>%
  mutate(cp = as.factor(cp),
         restecg = as.factor(restecg),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(thal),
         sex = factor(sex, levels = c(0,1), labels = c("female", "male")),
         fbs = factor(fbs, levels = c(0,1), labels = c("False", "True")),
         exang = factor(exang, levels = c(0,1), labels = c("No", "Yes")),
         target = factor(target, levels = c(0,1), labels = c("Health", "Not Health")))

glimpse(heart)
```
![3](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/c888dc7f-0d76-41af-8333-7953c3ae5846)
Next, we will check the availability of any missing value in each variables.
```R
colSums(is.na(heart))
```
![7](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/ca7fd4db-06e0-4f13-813f-5bcbffc123ee)
We can see that there are no missing values in each of our variables, so we do not need to drop those variable from our analysis.
## Data Visualization
A thorough exploration of the dataset is conducted, visualizing the distribution of the target variable (sick or not sick) and examining key features, including cholesterol levels and age vs. maximum heart rate.
### Visualization for Distribution of Target Variable
```R
ggplot(heart, aes(x = target, fill = target)) +
  geom_bar() +
  labs(title = "Distribution of Target Variable", x = "Target", y = "Count") +
  theme_minimal()
```
![4](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/7db10dac-224d-476c-8f6b-235caa0caeb8)
### Cholestrol Distribution by Target
```R
ggplot(heart, aes(x = target, y = chol, fill = target)) +
  geom_boxplot() +
  labs(title = "Cholesterol Distribution by Target", x = "Target", y = "Cholesterol") +
  theme_minimal()
```
![5](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/ea39562c-bdbf-4b12-a740-cba3a68b0ee5)
### Scatter Plot of Age vs Maximum Heartbeat (Thalach)
![image](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/7749cbc9-46a9-4971-aebb-c6fc2c2a456f)
## Data Preprocessing
Further data cleaning and preprocessing steps are implemented to ensure the dataset is optimally prepared for subsequent model training.
```R
prop.table(table(heart$target))
```
![8](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/eceb12ca-ae42-4cbc-971a-299510615d27)
These are the actual number of our variable target class.
```R
table(heart$target)
```
![9](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/fabe8bd7-cdf1-47b3-b692-e5835673109e)
## Cross Validation
To guarantee robust model performance, the dataset undergoes a meticulous split into training and testing sets. Cross-validation is employed to ensure a balanced representation of target classes in the training data.
Next step of our analysis will be splitting our data into train and test data. We will use our train data to train our model and use our test data to validate our model when overcoming the unseen data.
```R
set.seed(100)
index <- sample(nrow(heart), nrow(heart)*0.7)

# Data train
train_heart <- heart[index,]

# Data test
test_heart <- heart[-index,]
```
Next, we will check our train data proportion whether the proportion is balance enough to train our model, this need to be done so we can minimize the risk that our models are overfit.
```R
prop.table(table(train_heart$target))
```
![12](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/35c86b38-f9ee-44de-9870-5ac211738b60)
## Data Modelling
Multiple logistic regression models are systematically generated, incorporating various combinations of predictors. Stepwise methods are applied to select the most influential variables, and the final model is chosen based on AIC and Residual Deviance values.
We will create a model based on our train data. we will use several variables that may have a significant effect toward our target variable like sex, cp, fbs, and thal.Then we will also use the stepwise method to see if we can get a better model than our previous one.
```R
# Create a model
model1 <- glm(formula = target ~ sex + cp +  fbs + thal, family = "binomial", data = train_heart)

# Model summary
summary(model1)
```
![12](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/2ff03a7b-dc3d-46eb-831e-dace3447213a)
as we can see, there are only several variables that are significant toward our model. As we have not try other variable that may have been affected our target variable, we will try to use stepwise method to create a better model than our previous one.
```R
# Create a model without predictor
model_none <- glm(target ~ 1, family = "binomial", data = train_heart)

# Create a model with all predictor
model_all <- glm(target ~ ., family = "binomial", data = train_heart)
```
```R
# Stepwise regression backward
model_back <- step(object = model_all, direction = "backward", trace = F)

# Stepwise regression forward
model_forw <- step(object = model_all, scope = list(lower = model_none, upper = model_all), direction = "forward", trace = F)

# Stepwise regression both
model_both <- step(object = model_all, scope = list(lower = model_none, upper = model_all), direction = "both", trace = F)
```
Next we will see our model summary in each of the model that we have been created before.

We will see it based on:

AIC, amount of information lost in the model, lower AIC indicate a good quality of a model.
Residual Deviance, Error of the model when the model have a predictor, lower Residual Deviance means we have a better model
```R
# Model Summary

# Backward

summary(model_back)
```
![13](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/2db52034-8dec-4011-85b0-a5101fb22b89)
```R
# Forward

summary(model_forw)
```
![14](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/6e41895d-d66a-462a-a4f6-457634cc247c)
```R
# Both

summary(model_both)
```
![15](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/70ef53ab-8fb4-4bad-a877-c5bb061cad6a)
Based on the result, we can see that our model_back and model_both have a similar value of AIC & Residual Deviance. Both of those model have the AIC value 459.54 and the Residual deviance value 419.54. while our model_forw have the AIC value 461.96 and the Residual deviance value 415.96.

Although our model_forw have a better Residual Deviance than other model, the number of variables that significant towards the our target variables are not as much as the other two models. Therefore, we will not choose model_forw and proceed to choose either the model_back or model_both.

In this case we will choose model_both for our further analysis.
## Data Prediction
The selected model is employed to predict heart disease in the test dataset. Probability distribution and model evaluation metrics are comprehensively visualized.
```R
test_heart$prediction <-  predict(model_both, type = "response", newdata = test_heart)

# Create Plot

test_heart %>%
  ggplot(aes(x=prediction)) +
  geom_density() +
  labs(title = "Probabilities Distribution of Prediction Data") +
  theme_minimal()
```
![16](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/bc0db804-a2eb-4599-acb3-f28fff67f45d)
```R
pred <- predict(model_both, type = "response", newdata = test_heart)
result_pred <- ifelse(pred >= 0.5, "Not Health", "Health")

# Put our result prediction into our test data

test_heart$prediction <- result_pred
```
## Overview
```R
test_heart %>%
  select(target, prediction) %>%
  head(5)
```
![17](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/204349ae-bb81-47ba-a4bd-63ba93461bfe)
## Model Evaluation
The model's performance is rigorously assessed using key metrics such as accuracy, recall, specificity, and precision. A confusion matrix provides detailed insights into the model's predictive capabilities.

In model evaluation, we will see how good our model based on several different matrix.

Accuracy : How much our prediction able to predict our target variable Recall / Sensitivity : How much our prediction able to predict correctly the positive class Specificity : How much our prediction able to predict correctly the negative class Precision : How much our prediction of positive class correctly predicted
```R
conf_mat <- confusionMatrix(as.factor(result_pred), reference = test_heart$target, positive = "Not Health")

conf_mat
```
![18](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/a65955c5-aa5d-4e8f-9757-399312d0a326)
```R
recall <- round(46/(46+4),3)
specificity <- round(30/(30+11),3)
precision <- round(46/(46+11),3)
accuracy <- round((46+30)/(46+30+11+4),3)

matrix <- cbind.data.frame(accuracy, recall, specificity, precision)

matrix
```
![19](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/53f727b7-b07e-4b35-8592-3cef4c3b2612)
Based on the matrix summary, we can see that the ability of our model to predict our target variable are 83.5%. from entire of our prediction data. Our model ability to correctly predicted the Not Health person is 92%, while it ability to correctly predicted the Health person is 73.2%. Furthermore, from all of Not Health prediction our model able to predict it correctly 80.7%.
## Model Interpretation
In-depth interpretation of the model's coefficients is presented, offering valuable insights into the impact of different variables on the likelihood of heart disease.
```R
# Return the probability value

model_both$coefficients %>%
  inv.logit() %>%
  data.frame()
```
![20](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/b0ba7c88-2b7f-438e-907e-5395cd793a0c)
Some conclusions that we can interprete from these dataframe are:

The probability of male to be diagnosed with heart disease is 11.4%.

People with high level severe type of pain (cp = 3) have a 88% probability to be diagnosed with heart disease.
## K-Nearest Neighbor
Data Wrangling
as we use difference approach for K - Nearest Neighbor method, we need to create a new data frame that consist of dummy variable that we are going to use to predict our target variable.
```R
# Create dummy variable

dummy <- dummyVars("~target + sex +cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal", data = heart)

# Create new data frame

dummy <- data.frame(predict(dummy, newdata = heart))

# Check our data frame structure

str(dummy)
```
![21](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/2ddd65ec-b5c7-4d51-ab1a-893495cffdac)
After we create a new data frame that called dummy, we will remove variables that originally consist of two categories.

Those variables are:

target sex fbs exang
```R
dummy$target.Health <- NULL
dummy$sex.female <- NULL
dummy$fbs.False <- NULL
dummy$exang.No <- NULL
```
```R
head(dummy, 3)
```
![22](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/bffd4db0-c640-4d1d-a1ca-7404daf36110)
## Cross Validation: K - Nearest Method
K - Nearest Method have a different approach for cross validation. We are going to split both of our Predictor and Target variables into train and test data.

The proportion will be 70% for our train data and 30% for our test data.
```R
set.seed(100)

# Predictor

train_x <- dummy[index, -1]
test_x <- dummy[-index, -1]

# Target

train_y <- dummy[index, 1]
test_y <- dummy[-index, 1]
```
## Choose K
```R
sqrt(nrow(train_x))
```
![23](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/a27f7f98-fdc3-4928-8504-15249338ec15)
## Scaling
We will use scale() function to scale both of our predictor train and test data.
```R
train_x <- scale(x = train_x)
test_x <- scale(x = test_x, center = attr(train_x, "scaled:center"), scale = attr(train_x, "scaled:scale"))
```
## Create K - Nearest Neighbor Prediction
Next, we will create a prediction using knn() from library class
```R
pred_knn <- knn(train = train_x, test = test_x, cl = train_y, k = 26)
```
To ease our model summary readibility, we will transform our K - Nearest Neighbor Prediction object into data frame, and rename the levels into their original labels.
0 : Health 1 : Not Health
```R
pred_knn <- pred_knn %>%
  as.data.frame() %>%
  mutate(pred_knn = factor(pred_knn, levels = c(0,1), labels = c("Health", "Not Health"))) %>%
  select(pred_knn)
```
We will do the same way with our target test data which will be use for reference in our confusion matrix
0 : Health 1 : Not Health
```R
test_y <- test_y %>%
  as.data.frame() %>%
  mutate(target = factor(test_y, levels = c(0,1), labels = c("Health", "Not Health"))) %>%
  select(target)
```
## Create Confusion Matrix
```R
conf_mat_knn <- confusionMatrix(pred_knn$pred_knn, reference = test_y$target, positive = "Not Health")

conf_mat_knn
```
![24](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/03bf7560-a067-476f-a985-0c96251ca18e)
```R
recall_knn <- round(48/(48+2),3)
specificity_knn <- round(29/(29+12),3)
precision_knn <- round(48/(48+12),3)
accuracy_knn <- round((48+29)/(48+29+12+2),3)

matrix_knn <- cbind.data.frame(accuracy_knn, recall_knn, specificity_knn, precision_knn)

matrix_knn
```
![25](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/48453e70-bb4c-4840-8493-5fa39ee4c7b8)
Based on the matrix summary, we can see that the ability of our K - Nearest Neighbor Prediction predict our target variable are 84.6%. from entire of our prediction data. Our K - Nearest Neighbor Prediction ability to correctly predicted the Not Health person is 96%, while it ability to correctly predicted the Health person is 70.7%. Furthermore, from all of Not Health prediction our K - Nearest Neighbor Prediction able to predict it correctly 80%.
## Model Comparison: Logistic Regression vs K - Nearest Neighbor
We have two method for our analysis, now we need to choose which of these method output that we will use to predict whether a person is diagnosed with heart disease or vice versa.
```R
# Matrix summary from Logistic Regression

matrix
```
![26](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/4ec93134-dcd7-4f9e-a2b2-04947f40e5ac)
```R
# Matrix summary from K - Nearest Neighbor

matrix_knn
```
![27](https://github.com/faizaligola/Cardiovascular-Disease-Detection-and-Prediction-using-Supervised-Learning-/assets/80847944/9a7e646b-0153-4269-af44-1cb58c8864d8)
We can see from the table above that we have almost similar value for each matrix with K - Nearest Neighbor have a slightly better value in accuracy and recall, while Logistic Regression model have a better value in specificity and precision.

Please feel free to contribute to the project or raise issues for further enhancements.
