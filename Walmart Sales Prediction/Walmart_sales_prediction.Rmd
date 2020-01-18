---
title: "Walmart Sales Prediction in Stormy Weather"
date: "12/15/2019"
output: 
  html_document:
  theme: cerulean
  toc: TRUE
---
<style type="text/css">

body{ /* Normal  */
      font-size: 14px;
      font-family: "Times New Roman", Times, serif;
      line-height: 2em;
  }
td {  /* Table  */
  font-size: 12px;
}
h1.title {
  font-size: 42px;
  color: DarkRed;
}
h1 { /* Header 1 */
  font-size: 30px;
  color: DarkBlue;
}
h2 { /* Header 2 */
    font-size: 24px;
  color: DarkBlue;
}
h3 { /* Header 3 */
  font-size: 20px;
  font-family: "Times New Roman", Times, serif;
  color: DarkBlue;
}
code.r{ /* Code block */
    font-size: 14px;
}
pre { /* Code block - determines code spacing between lines */
    font-size: 16px;
}
</style>

# Group Members

- Aastha Nargas(anargas2)
- Kusum Vanwani(vanwani2)
- Pankaj Sharma(pankajs2)
- Shashi Roshan(sroshan2)

```{r message = FALSE, echo = FALSE}
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(tidyverse)
library(Amelia)
library(mice)
library(naniar)
library(VIM)
library(tibble)
library(randomForest)
library(parallel)
library(kableExtra)
library(reshape2)
library(MASS)
library(gbm)
library(lubridate)
library(glmnet)
library(lmtest)
library(zoo)
library(jtools)
```

# Introduction

Weather plays an important role in the consumer preferences while shopping. During extreme weather events, it would be beneficial for the stores to be stocked with products which are essential to cope up with the event. 
This project aims to predict the sales of 111 products sold in 45 different walmart locations. We are trying to model the effect on the weather conditions on the sales of these products. The 45 locations are covered by 20 weather stations. The task is to predict the amount of each product sold around the time of major weather events. 

We are provided the full observed weather data so we don't need to forecast weather for the sake of this project. More information on the dataset can be found at the kaggle page for this competition, https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data.

Other than the competitive goal of the project, this project is also aimed at solidifying our understanding of the intricacies of the linear model and we will be testing a least squares model on this dataset. We will begin with understanding the datasets and draw insights from the data. By analyzing the data well we are able to create additional variables for our modelling process. Initially, we start with a simple linear model for prediction, then create diagnostics for our model and based on the inferences drawn will make modifications to the model in accordance with the material covered in the class.We will be building different models and testing their score and prediction accuracy. We will also be testing out some advanced models like Random Forest and Gradient Boosting to make the predictions and will compare with the performance of the linear model.

Intuitively, we may expect an increase in the sales of certain products before a big weather event, but it's difficult for supply-chain managers to correctly predict the level of inventory needed to avoid being out-of-stock or overstock during and after that storm. Walmart relies on a variety of vendor tools to predict sales around extreme weather events, but it's an ad-hoc and time-consuming process that lacks a systematic measure of effectiveness. Helping Walmart better predict sales of weather-sensitive products will keep valued customers out of the rain.

## Data

### File Descriptions

- key.csv - the relational mapping between stores and the weather stations that cover them
- train.csv - sales data for all stores & dates in the training set
- test.csv - stores & dates for forecasting (missing 'units', which you must predict)
- weather.csv - a file containing the NOAA weather information for each station and day

Sales data for 111 products sold in 45 different stores whose sales may be impacted by weather such as dairy products, rain essentials etc. are provided. There might be cases where same product is being sold with a different id in a different store. The 45 store locations are covered by 20 weather staions mapping of which is provided in the key file. The full observed weather conditions covering the time period of both training and test data are provided. The training data contains ~ 4.6M rows of data while test data contains ~ 0.52M rows of data. 

```{r loading-data, echo = FALSE}
train = as_tibble(read.csv("D:/Data Science/425 Project/train.csv"))
weather = as_tibble(read.csv("D:/Data Science/425 Project/weather.csv"))
key     = as_tibble(read.csv("D:/Data Science/425 Project/key.csv"))
test = as_tibble(read.csv("D:/Data Science/425 Project/test.csv"))
```

#### A quick look at the datasets.

```{r sneak-peek, echo = FALSE}
head(train, 6) %>% 
   kable(align = "cc", digits = 3, caption = "Table 0.01: A sample of the Training Dataset") %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")
  

head(weather, 6) %>% 
   kable(align = "cc", digits = 3, caption = "Table 0.02: A sample of the Weather Dataset") %>% 
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center") %>% 
  scroll_box(width = "100%", height = "200px")
```

# EDA

We want to start with investigating the data and identify some patterns which can help us the understand the data better. We want to make sure that the data we have does not have missing values since the linear model that we may aim to fit will remove the rows which contain the missing values. We first look at the missing values in train data. **Fig 0.01** is a visualization of the number of records containing missing values by predictor variables. We can see that there are no missing values in the train data.

**Fig 0.02** shows us the missing values in the weather data. We can see that most of the variables have some missing values. We need to think about the implications of these missing values. The missing values need to be imputed because the linear model by default is not very good at handling missing values. We use `MICE` package in `R` to impute the missing values in the weather data. This package helps us to impute the missing values with plausible data values. These plausible values are drawn from a distribution specifically designed for each missing data point. 

Before we do further exploratory analysis on the train data, we need to do some preprocessing. First we need to change the type of the date variable and convert it into a `date` object. We then use this `date` object to extract the day and month as the intuition dictates that sales will depend on the month of the year and the day of the week. People are more likely to go shopping on weekends than weekdays. **Fig .03** confirms this as we see that the total sales are highest on Saturday and Sunday as compared to the rest of the week.

**Fig 0.04** is very similar to the earlier plot but instead of seeing sales by day we instead check the sales by month. We can clearly see that `January` has the highest sales as compared to the rest of the months and `November` and `December` have the least amount of units sold. An insight from this could be drawn, since January is the month with harsher winters, the requirement for weather related products might be more and hence more units maybe sold. Due to holidays and some products' demand being seasonal, we also wanted to see the effect of season on sales. For this, we created a `season` variable in the dataset and plotted it against units sold. **Fig 0.05** plots the relationship between the season and the units sold. We see that the sales are considerably low in *Fall* but for the rest of the seasons there is not much difference.

We want to check the correlation among the predictors in the weather dataset. for this we create a correlation plot in `R`. **Fig 0.06** shows the correlation of all the variables with each other. We can see that time of sunrise and sunset are highly correlated with the temperature variables. Sunset and Sunrise temperatures are highly correlated. As per expectations, average temperature is almost perfectly correlated to maximum and minimum temperatures. Wind parameters are negatively correlated with the temperature variables which is expected as a stronger wind will bring the temperature down. We will be calculating variance inflation factors in this analysis to further assess the correlations and then remove the variables with very high inflation factors. 

Before we start the Exploratory analysis on the weather data, we need to impute the weather data and merge it with the train data so that we can understand the relationship between the weather parameters and units sold. **Fig 0.07** shows that the wind speed isn't necessarily a defining factor but the there no points in the upper right half of the graph showing that large number of units are only sold with low wind speed. With high wind speed, people probably prefer to stay at home instead of going shopping.**Fig 0.08** shows us that with very high and very low temperatures the number of units sold goes down which makes sense as people would try to avoid extreme temperatures.

```{r missing-data-train, echo = FALSE}
train$item_nbr = as.factor(train$item_nbr)
train[train == " "] = NA
train[train == "-"] = NA
apply(is.na(train), 2, sum) %>% 
  kable(align = "cc", digits = 3, caption = "Missing rows in Train Data") %>% 
  kable_styling("striped", full_width = FALSE)
p01 = gg_miss_var(train) + labs(y = "Missing Values in Train Data", title = "Fig 0.02: Missing rows in Training Data")
```

```{r missing-data-weather, echo = FALSE}
weather$codesum = as.character(weather$codesum)
weather$codesum[weather$codesum == " "] = "MO"
weather$codesum = as.factor(weather$codesum)
weather[weather == " "] = NA
weather[weather == "-"] = NA
weather[weather == "M"] = NA
weather[weather == "T"] = NA
apply(is.na(weather), 2, sum) %>% 
  kable(align = "cc", digits = 3, caption = "Missing rows in Weather Data") %>% 
   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")
p02 = gg_miss_var(weather) + labs(y = "Missing Values in Weather Data", title = "Fig 0.02: Missing rows in Weather Data")
```

```{r train-preprocess, echo = FALSE}
train$date    = as.Date(train$date, format = "%Y - %m - %d")
train$day     = as.factor(weekdays(train$date))
train$month   = as.numeric(strftime(train$date, "%m"))
train$season = case_when(
  train$month >= 3 & train$month <=5 ~ "Spring",
  train$month >= 6 & train$month <=8 ~ "Summer",
  train$month >= 9 & train$month <=11 ~ "Fall",
  TRUE ~ "Winter"
)
train$month = as.factor(train$month)
train$season = as.factor(train$season)
```

```{r weekday-units, echo = FALSE, warning = FALSE, message = FALSE}
sum_units_day = train %>%
  group_by(day) %>%
  summarise(sum_units = sum(units))
p03 = sum_units_day %>%
  ggplot(aes(x = day, y = sum_units)) +
  geom_bar(stat = "identity", color = "darkorchid4") +
  labs(
    title = "Fig 0.03: Daily Units Sold",
    subtitle = "Data plotted by Weekday",
    y = "Daily Units sold",
    x = "Weekday"
  ) + theme_bw(base_size = 15)
  
```

```{r month-units, echo = FALSE, warning = FALSE, message = FALSE}
sum_units_month = train %>%
  group_by(month) %>%
  summarise(sum_units = sum(units))
p04 = sum_units_month %>%
  ggplot(aes(x = month, y = sum_units)) +
  geom_bar(stat = "identity", color = "darkorchid4") +
  labs(
    title = "Fig 0.04: Monthly Units Sold",
    subtitle = "Data plotted by Month",
    y = "Monthly Units sold",
    x = "Month"
  ) + theme_bw(base_size = 15)
  
```

```{r season-units, echo = FALSE, warning = FALSE, message = FALSE}
sum_units_season = train %>%
  group_by(season) %>%
  summarise(sum_units = sum(units))
p05 = sum_units_season %>%
  ggplot(aes(x = season, y = sum_units)) +
  geom_bar(stat = "identity", color = "darkorchid4") +
  labs(
    title = "Fig 0.05: Units sold by Season",
    subtitle = "Data plotted by Season",
    y = "Seasonal Units sold",
    x = "Season"
  ) + theme_bw(base_size = 15)
  
```

```{r weather-preprocess, echo = FALSE, warning = FALSE, message = FALSE}
weather$date    = as.Date(weather$date, format = "%Y - %m - %d")
cols = c(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20)

weather[, cols] = apply(weather[, cols], 2, function(x)
  as.numeric(as.factor(x)))

```

```{r weather-cor, fig.width = 24, fig.height = 15, echo = FALSE, warning = FALSE, message = FALSE}
weather_cor = weather[,3:20]
cor_mat = cor(weather_cor[sapply(weather_cor, function(x)
  ! is.factor(x))], use = "complete.obs")
plot <- ggplot(data = melt(cor_mat), aes(x=Var1, y=Var2, fill=value, 
label= value))
p06 <- plot + geom_tile() + ggtitle("Fig 0.06: Correlation plot for weather data")
rm(weather_cor)
```

```{r weather-missingdata, eval = FALSE, echo = FALSE, warning = FALSE, message = FALSE}
apply(is.na(weather), 2, sum)
```

```{r impute-NA, echo = FALSE, warning = FALSE, message = FALSE}
#imputed_data = mice(weather, m = 3, maxit = 4, meth = "rf", seed = 500, printFlag = TRUE)
#weather = complete(imputed_data, 2)
#apply(is.na(weather), 2, sum)
weather = as_tibble(read.csv("D:/Data Science/425 Project/weather_imputed.csv"))
weather$date    = as.Date(weather$date, format = "%Y - %m - %d")
cols = c(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20)
weather[, cols] = apply(weather[, cols], 2, function(x)
  as.numeric(as.factor(x)))
```

```{r train-merge, echo = FALSE, warning = FALSE, message = FALSE}
weather$station_nbr = as.factor(weather$station_nbr)
train$store_nbr = as.factor(train$store_nbr)
key$store_nbr = as.factor(key$store_nbr)
train   = left_join(train, key, by = "store_nbr")
train$station_nbr = as.factor(train$station_nbr)
train = left_join(train, weather, by = c("date", "station_nbr"))
```

```{r test-preprocess, echo = FALSE, warning = FALSE, message = FALSE}
test$date    = as.Date(test$date, format = "%m / %d / %Y")
test$day     = as.factor(weekdays(test$date))
test$month   = as.numeric(strftime(test$date, "%m"))
test$season = case_when(
  test$month >= 3 & test$month <=5 ~ "Spring",
  test$month >= 6 & test$month <=8 ~ "Summer",
  test$month >= 9 & test$month <=11 ~ "Fall",
  TRUE ~ "Winter"
  
  
)
test$month = as.factor(test$month)
test$season = as.factor(test$season)
```

```{r test-merge, echo = FALSE, message = FALSE, warning = FALSE}
test$store_nbr = as.factor(test$store_nbr)
test$item_nbr = as.factor(test$item_nbr)
test   = left_join(test, key, by = "store_nbr")
test$station_nbr = as.factor(test$station_nbr)
test = left_join(test, weather, by = c("date", "station_nbr"))
test$store_nbr = as.factor(test$store_nbr)
test$station_nbr = as.factor(test$station_nbr)
```

```{r, p0708, echo = FALSE, message = FALSE, warning = FALSE}
p07 = ggplot(data = subset(train, units != 0 & units < 2000)) +
  geom_point(aes(
    x = as.numeric(as.character(avgspeed)),
    y = units,
    color = as.numeric(as.character(avgspeed))
  )) +
  labs(title = "Fig 0.07: Units vs Wind Speed",
       y = "Units sold",
       x = "Wind Speed") +
  scale_color_gradient("Wind Speed", low = "blue", high = "yellow")

p08 = ggplot(data = subset(train, units != 0 & units < 2000)) + 
  geom_point(aes(x = as.numeric(as.character(tavg)), y = units, 
                 color = as.numeric(as.character(tavg)))) + 
  labs(title = "Fig 0.08: Average Temperature",
       y = "Units sold",
       x = "Average Temperature") +
  scale_color_gradient("Average Temperature", low="blue", high="red")
```


# Modelling

In the data we notice that there are a large number of items for which sales are zero throughout the time period for which the data is available. Since we are not learning any new information from these observations, we would not consider them to train our model, instead we will just consider the items for which we have atleast some non-zero sales. This will also allow us to reduce the size of the training set quite a bit and significantly reduce the model training time. We will start with a simple linear regression model and then investigate it to assess the quality of the fit and will add additional features and improvements to our base model as necessary. 

We study the summary of the units variable, and observe that there maybe some outliers in the data. There are two observation with units sold greater than 1000, which we have removed to prevent the model from being skewed.

```{r summary_units, echo = FALSE}

summ = summary(train$units)
tibble(
  "Measure" = c("Min", "1st Quantile", "Median", "Mean", "3rd Quantile", "Max"),
  "Value" = c(summ[1], summ[2], summ[3], summ[4], summ[5], summ[6])
) %>% 
  kable(digits = 3,
        caption = "Table 0.03: A summary of the Units") %>% 
  kable_styling("striped", full_width = FALSE)

#length(which(train$units >= 1000))
```


```{r non-zero-sales, echo = FALSE, warning = FALSE, message = FALSE} 
get_store_item_with_nonzero_sales = function(train){
  train$log1p = log(train$units + 1)
  g = train %>% group_by(store_nbr, item_nbr) %>% summarize(mean_sales = mean(log1p))
  g = g %>% filter(mean_sales > 0)
  return (g)
}
g = get_store_item_with_nonzero_sales(train = train)
```

```{r ols-model, echo = FALSE, warning = FALSE, message = FALSE}
train_samp = inner_join(train, g, by = c("store_nbr", "item_nbr"))
write.csv(train_samp, "D:/Data Science/425 Project/train_samp.csv", row.names = FALSE)
train_samp$mean_sales = NULL
train_samp$codesum = NULL
train_samp = train_samp[train_samp$units <= 1000, ]
mod_ols = lm(formula = units ~ . - station_nbr, data = train_samp)
```

**Summary of the Simple Linear Regression is shown below**: 

```{r mod-output, echo = FALSE}
summ(mod_ols) 
```

```{r id-submission-base, echo = FALSE, warning = FALSE, message = FALSE}
id_data_base = test %>% mutate(id = str_c(store_nbr,"_",item_nbr,"_",date))
id_base = id_data_base$id
rm(id_data_base)
#id = paste(test_sort$store_nbr, "_", test_sort$item_nbr, "_", test_completed$date)
```

```{r test-predict, echo = FALSE, warning = FALSE, message = FALSE}
test_samp = inner_join(test, g, by = c("store_nbr", "item_nbr"))
write.csv(test_samp, "D:/Data Science/425 Project/test_samp.csv", row.names = FALSE)
test_samp$store_nbr = as.factor(test_samp$store_nbr)
test_samp$mean_sales = NULL
test_samp$codesum = NULL
pred_model = predict(mod_ols, test_samp)
submission_base = tibble(id = id_base, pred = 0)
```

```{r submission-file, echo = FALSE, warning = FALSE, message = FALSE}
id_data_samp = test_samp %>% mutate(id = str_c(store_nbr,"_",item_nbr,"_",date))
id = id_data_samp$id
rm(id_data_samp)
submission_mod = tibble(id = id, units = pred_model)
submission_final = left_join(submission_base, submission_mod, by = "id")
submission_final$units[is.na(submission_final$units)] = 0
submission_final$units[submission_final$units < 0] = 0
submission_final$pred = NULL
write.csv(submission_final, "D:/Data Science/425 Project/submission_base_ols.csv", row.names = FALSE)
kaggle_base_ols = 0.23442
```

**Simple Linear Regression:** We start with a basic linear model with all the variables at our disposal. 

This model has an $R^{2}$ of 0.72, which implies that the predictors explain the model to certain extent. We also observe that most weather variables having a p-value of greater than 0.05. This implies that they are not significant for the model, and could be removed for further analysis. Variables like store number, item number, days in a week, month are significant in the model. The interpretation of the coefficients can be as follows, if we take a look at the coefficient of `tavg`, a one unit increase in the average temperature, results in decline of units sold by a factor of - 0.06. However, to gauge the performance of the linear model we run some diagnostics to check if the assumptions of the model hold. Based on our assesment of these assumption we make further improvements to our model.

**Assumptions in SLR:**

- Linear relationship
- Multivariate normality
- No or little multicollinearity
- No auto-correlation
- Homoscedasticity(Error terms have constant variance)

For all the item numbers which we did not consider for training the model, we will assign a prediction of zero. We will then merge this with the item numbers for which we made the predictions using our model and get the final submission file. Submitting this file on Kaggle, we were able to achieve a score of `r kaggle_base_ols` which is significantly better than the baseline score of 0.52 which is attainable with all zero predictions. This parameter used to assess the accuracy of the models on kaggle is Root Mean Squared Logarithmic Error(RMSLE).

# Diagnostics

Below the diagnostics plots are shown. 

**Plot 01: Residuals vs Fitted** This plot shows if residuals have non-linear patterns. There could be a non-linear relationship between the response variable and the predictor variables and if it exist then we will see a clear pattern in the plot. However, in our plot we see that there are no obvious patterns in the points plotted and its pretty random. We can conclude with reasonable confidence that the assumption of linear relationship stands corrected. if we find equally spaced residuals around a horizontal line without distinct patterns then it is a good indication that non-linear relationships are not present.  

**Plot 02: Normal QQ** This plot shows if the residuals are normally distributed which test the assumption of normality in the simple linear regression. If the residuals follow a straight line, we can conclude that they satisfy the normality criteria but the curve that we see here in the plot below deviates from the straight line. While the curve may not always follow a straight line perfectly but the deviation is too much to ignore and we can safely conclude that the residuals do not satisfy the normalcy criteria. We will employ transformation techniques like box-cox to correct for this assumption fail.

**Plot 03: Scale Location** This plot is also called spread-location plot. This plot if residuals are spread equally along the ranges of predictions. Using this graph, we can check the assumption of homoscedasticity(equal variance). It is generally considered good if the points are equally distributed on along the horizontal line in the middle. In this case, the residuals appear to spread out more as we move along the horizontal axis. Because the points are not spread equally we don't see a smooth straight horizontal line through the middle.

**Plot 04: Residuals vs Leverage** This plot helps us to find influential points. In Regression, there are two kinds of outliers, influential and non-influential. Even there may exist some extreme points but they may not have enough influence to shift the regression line much. This means that the results wouldn't change much with the inclusion or exclusion of these points. To look for the influential points, we look at the cook's distance which is represented by the red curves in the graph however those are not visible in this graph due to being out of scale as none of our points are even close to being influential. Unlike the first three plots, the pattern in the plot is not relevant here. We watch for outlying values at the upper right and lower right corner.

```{r diagnostics, echo = FALSE, fig.width = 18,fig.height = 24, message = FALSE, warning = FALSE}
par(mfrow = c(2,2))
plot(mod_ols)
```

```{r bptest, echo = FALSE, message = FALSE, warning = FALSE}
bp = bptest(mod_ols)
bp = tibble("Name" = c("BP", "Degrees of Freedom", "P-Value"),
       "Value" =  c(bp$statistic, bp$parameter, bp$p.value))
bp %>% 
  kable(align = "cc", digits = 5, caption = "Studentized Breusch-Pagan Test") %>% 
   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")
```

**Breusch-Pegan Test** THE BP test is a test conducted to check the homoscedasticity(constant variance assumption). Its basically a hypothesis test with the null hypothesis being that the variance of the residuals is constant. If our p-value is sufficiently small, in this case smaller than 0.05 significance level, we reject the null hypothesis and conclude that the variance is not constant. We have already seen through the diagnostic plots that the variance is not constant and the BP test confirms this as our p-value is basically zero. Hence we reject the null hypothesis and conclude that the alternate hypothesis is true i.e the assumption of homoscedasticity is incorrect.

# Improvements 

## Box-Cox

```{r box-cox, echo = FALSE, message = FALSE, warning = FALSE}
mod_ols = lm(formula = units + 1 ~ . - station_nbr, data = train_samp)
boxcox(mod_ols)
```

Linear Regression assumes that the errors follow a normal distribution with mean= 0 and with a constant variance.  This assumption might be false in various ways. One way to fix this is to transform Y to a function h(Y) such that the variance of h(Y) is constant. As we can see from the results of the Breusch-Pegan Test, assumption of the errors being homoskedastic does not hold true. Hence we perform the Box cox transformation. 

Box cox transformation provides a way in which we can solve for the heteroskesticity (non constant variance) and non normality of errors. Suppose that if we know the dependent variable is positive, and consider the model:

&nbsp;
                                                          $y^\lambda = XB + \epsilon$

where $var(\epsilon)$ is constant and $y^\lambda$ is the element-wise power of lambda. When lambda will be equal to 1, we will get the original linear model. 

Box Cox transformation method tries to choose a lambda that aims at maximizing the likelihood of the data and assumes that the erros are normally distributed. We plot the Likelihood on y axis verses the lambda on x axis and choose the lambda for which we get the likelihood as maximum. 

From the above graph, we observe that the lambda for which the likelihood is maximum falls between -1 and 0 and closer to zero. Since the lambda is closer to zero, we can consider the log transformation of Y, to achieve a constant variance of errors.

```{r log-model, echo = FALSE, warning = FALSE, message = FALSE}
mod_ols_log = lm(formula = log(units + 1) ~ ., data = train_samp)
kaggle_log_ols = 0.16624
```

After deciding on the log-transformation, we transformed the units variable to $ln(units + 1)$ and fit the simple linear regression again. With this model we were able to achieve an $R^2$ score of `r summary(mod_ols_log)$r.squared`. We also achieved a kaggle score of `r kaggle_log_ols`.Looking at the diagnostics plot for the log transformed model, we can clearly see that we are closer to linear model assumptions as compared to simple least regression without transformation. One point to note here is that while making predictions we need to convert the $ln(units + 1)$ back to units using the $exp$ function. Since we are decided on the log-transformation, we will consider this model as the base of all further improvements or modification we make.

```{r pred-log, echo = FALSE, warning = FALSE, message = FALSE}
pred_model = predict(mod_ols_log, test_samp)
pred_model = exp(pred_model)
submission_mod = tibble(id = id, units = pred_model)
submission_final = left_join(submission_base, submission_mod, by = "id")
submission_final$units[is.na(submission_final$units)] = 0
submission_final$units[submission_final$units < 0] = 0
submission_final$pred = NULL
write.csv(submission_final, "D:/Data Science/425 Project/submission_ols_log.csv", row.names = FALSE)
```

## Feature Engineering

To facilitate a better understanding of the underlying structure of the data, new predictor variables are created based on the observations through exploratory analysis and observation of the data. It is intuitive that the sales in each day will vary with the position of the day in the month or in a year. Hence variables like month day and day in year were generated in the data. Position of the day in the week was also created as it makes sense that sales might be more on a weekend as compared to weekdays. This is confirmed by our analysis in the EDA section. 

Additionally, from observing the data, we noticed that the sales in each month vary significantly. Hence, we created a monthly average sales for each product which will serve as a new feature. Based on this, we can also analyse of the monthly sales for any product are zero.

Temperature is also another important feature as we noticed in the EDA section that people avoid shopping in extreme cold or hot days. In addition, "feels like" temperature might be more relevant to this particular analysis which is related to the moisture in the air hence two new features were created identifying the moisture in the air. The first predictor calculates the difference between average temperature and dew point temperature as it signifies how far away the moisture in the air is from saturation. The second predictor calculates the difference between the wet bulb temperature and average temperature. This signifies the relative humidity in the air. The larger the difference, the lower the relative humidity is. 

Predictors like precipitation, snowfall and average wind speed are included without any modifications. Later in the analysis we removed the variables with high correlation based on Variance Inflation Factor. The feature `codesum` was ignored for the purpose of this analysis as the data contained too many missing values and most of the info was already covered in the precipitation, snowfall and wind features. 

Because some of the store numbers and if numbers combinations have zero sales throughout the time period covered by the data, we have excluded them from the training data we have used for training the models. Prediction of zero units was assigned to all such cases and merged with the model predictions to prepare the submission file. 

All these features that we have created and modified for the training dataset, we also need to create those for the test dataset to ensure consistency while making predictions with the trained model.

```{r new-features-train, echo = FALSE, warning = FALSE, message = FALSE}
monthly_average = train_samp %>% 
  group_by(month, store_nbr, item_nbr) %>%
  summarise(monthly_average = sum(units))
train_samp = left_join(train_samp, monthly_average, by = c("store_nbr", "item_nbr", "month"))
train_samp$day_in_month = day(train_samp$date)
train_samp$day_in_year = yday(train_samp$date)
train_samp$year = year(train_samp$date)
train_samp$temp_diff_dew = abs(train_samp$tavg - train_samp$dewpoint)
train_samp$temp_diff_wb = abs(train_samp$tavg - train_samp$wetbulb)

```


```{r new-features-test, echo = FALSE, warning = FALSE}
test_samp = left_join(test_samp, monthly_average, by = c("store_nbr", "item_nbr", "month"))

test_samp$day_in_month = day(test_samp$date)
test_samp$day_in_year = yday(test_samp$date)
test_samp$year = year(test_samp$date)

test_samp$temp_diff_dew = abs(test_samp$tavg - test_samp$dewpoint)
test_samp$temp_diff_wb = abs(test_samp$tavg - test_samp$wetbulb)

```

## Multicollinearity

Multicollinearity is difficult to identify. When two or more independent variables are related to each them, we call them multicollinear. This results in unreliable and highly variable regression coefficients. Also, with multicoliinearity. it becomes difficult to interpret the impact of change of one unit in a variable on the change in the response variable. Some of the issues with multicollinearity are:
- If some of the predictors are collinear, then model matrix becomes singular and the inverse of such matrix does not exist.
- The least square estimate of the coefficients will not be unique

One way to check for multicollinearity is to use the Variance Inflation factor(VIF). VIF gives a measure of collinearity of a given predictor with all the other predictors. VIF of a given predictor is calculated by creating a linear regression model with the given predictor as the response variable and all the other predictors as the independent variables. If the coefficient of determination of this model is large, then we can say that the given predictor is collinear with other predictors.

&nbsp;
                                                                 $VIF = 1/ (1- R ^ 2)$


VIF ranges from 1 to infinity. Predictors with the VIF greater than 5, are considered to be highly collinear with the other predictors. 
We have shown the VIF of all the predictors. Before checking for VIF we used the alias function in R to find linearly dependent terms in our model. We remove these terms and then proceed with VIF. We have shown the VIF of all the predictors. Predictor with high VIF per degree of freedom are removed in a stepwise manner. We then build a linear regression model excluding these predictors one at a time, and then finalising the model.

```{r collinearity, echo = FALSE, warning = FALSE, message = FALSE}
alias_variables = alias(mod_ols_log)
library(car)
mod_ols_log_alias = lm(formula = log(units + 1) ~ . - date - station_nbr - month,
                       data = train_samp)
#alias(mod_ols_log_alias)
var_names = row.names(as.data.frame(as.matrix(vif(mod_ols_log_alias))))
vif_before = as.data.frame(as.matrix(vif(mod_ols_log_alias)))
vif_before = tibble(
  "Variable" = var_names,
  "GVIF" = vif_before$GVIF,
  "DF" = vif_before$Df,
  "GVIF^(1/(2*DF))" = vif_before$`GVIF^(1/(2*Df))`
)
vif_before %>%  kable(align = "cc", digits = 5, caption = "Variance Inflation Factor") %>% 
   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")

mod_ols_log_post_vif =  lm(formula = log(units + 1) ~ . - date - station_nbr - month - stnpressure -tavg, 
                data = train_samp)

var_names = row.names(as.data.frame(as.matrix(vif(mod_ols_log_post_vif))))
vif_after = as.data.frame(as.matrix(vif(mod_ols_log_post_vif)))
vif_after = tibble(
  "Variable" = var_names,
  "GVIF" = vif_after$GVIF,
  "DF" = vif_after$Df,
  "GVIF^(1/(2*DF))" = vif_after$`GVIF^(1/(2*Df))`
)

vif_after %>%  kable(align = "cc", digits = 5, caption = "Variance Inflation Factor") %>% 
   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")
kaggle_post_vif = 0.16354
```

We removed the variables with the high variance inflation factor per degree of the freedoms and then for the log transformed linear regression model. We achieved an $R^2$ of `r summary(mod_ols_log_post_vif)$r.squared`. With this model we achieved an kaggle score of `r kaggle_post_vif` which is not much different from the log model that we fit earlier in the analysis which is expected because we knew that since these variables were correlated with other predictors, most of the information that they provided was already provided through other variables so we won't see any big differences in the results.

```{r submission-function, echo = FALSE, warning = FALSE, message = FALSE}
submission = function(mod, name){
pred_model = predict(mod, test_samp)
pred_model = exp(pred_model)
submission_mod = tibble(id = id, units = pred_model)
submission_final = left_join(submission_base, submission_mod, by = "id")
submission_final$units[is.na(submission_final$units)] = 0
submission_final$units[submission_final$units < 0] = 0
submission_final$pred = NULL
write.csv(submission_final, name, row.names = FALSE)
}
```

```{r submission-post-vif, echo = FALSE, warning = FALSE, message = FALSE}
submission(mod_ols_log_post_vif, name = "D:/Data Science/425 Project/submission_post_vif.csv")
```

## Model Selection Methods

### AIC - BIC

In general, adding a new predictor results in increase of the coefficient of determination even if the predictor is not significant. As a result, $R^2$ always tends to favor large models, which results in very complex models and tend to overfit the training data resulting in poor performance on new previously unseen data. To avoid overfitting in such cases, we want to penalize the additional parameters which are increasing the performance on the training data set but are in fact resulting in overfitting. 

To penalize addition of new predictors, we have considered the following two measures:

- *Akaike Information Criteria (AIC)*
AIC is the penalized log likelihood measure.

&nbsp;
                                                 $AIC = 2k - 2ln(L)$

where L is the maximum likelihood and k is the number of parameters. As we increase the number of parameters, the value of AIC will increase. Hence we prefer the model with the lowest AIC value.

- *Bayesian Information Criteria(BIC)*
BIC is another measure for model comparison. It is similar to AIC but it has a higher penalty term as compared to AIC. As a result of this larger penalty, BIC chooses smaller models as compared to AIC.

&nbsp;
                                                 $BIC = 2nln(k) - 2ln(L)$

where L is the maximum likelihood and k is the number of parameters and n is the number of rows in the data. As we increase the number of parameters, the value of BIC will increase. Hence we prefer the model with the lowest BIC value. 


*Feature Subset Selection:*

In the linear regression model, not all the predictors are significant in predicting the response variable. Removing the non significant predictors can result in improving the accuracy of the least square fit. There are several ways to select the significant predictors. 

- Best Subset Selection
- Stepwise Selection

In the best subset selection method, all possible combinations of the predictors are considered. We then consider the best model out of all these possible models. This can be computationally expensive as the number of predictors becomes large. 

Another way of selecting predictors for linear regression is the stepwise Selection. In stepwise selection, we iteratively add or remove variables till we get the subset of predictors that results in the best models with lowest prediction error. 
There are two ways in which we can start adding or removing predictors:
- Forward Selection
- Backward Selection

In forward selection, we start with a null model and keep on adding predictors one at a time till we reach a stopping condition. 

- Let M0 be the null model.
- Consider all the remaining predictors. Add one predictor at a time to the existing model.
- Choose the predictor that results in the lowest AIC or BIC value for the model.
- Repeat step 2 and 3 till we do not see any improvement in the AIC or BIC value.


In Backward selection, we start with a full model and keep on dropping those predictors one at a time till we reach a stopping condition. 

- Let M0 be the full model.
- Consider all remaining predictors. Drop one predictor at a time from the existing model.
- Choose the predictor that results in the lowest AIC or BIC value for the model.
- Repeat step 2 and 3 till we do not see any improvement in the AIC or BIC value.


Since we are starting with the full model, we have chosen to use backward selection method for the purpose of feature selection. We have compared the performance of the models by choosing both AIC and BIC criteria to select the subset of predictors.

```{r aic-bic, echo = FALSE, message = FALSE, warning = FALSE, cache = TRUE, cache.path = "D:/Data Science/425 Project/"}
step_aic_mod_log = step(mod_ols_log_post_vif, direction = c("backward"), trace = 0)
step_bic_mod_log = step(mod_ols_log_post_vif, direction = c("backward"), trace = 0, k = log(nrow(train_samp)))
submission(step_aic_mod_log, name = "D:/Data Science/425 Project/submission_aic.csv")
submission(step_bic_mod_log, name = "D:/Data Science/425 Project/submission_bic.csv")
kaggle_aic = 0.16354
kaggle_bic = 0.16353
```

We made predictions with both the aic and bic based step regression models and received kaggle score of `r kaggle_aic` and `r kaggle_bic` respectively which are not much different from the the last model we submitted after removing all the factors with high variance inflation factor. This shows that most of the features that we are left with post vif analysis are contributing to the predictions. We have stored these variables in the cache of the markdown file so that we don't have to rerun the process everytime we knit the file as the process can be time-consuming.

### Ridge/Lasso Regularization

Ridge and Lasso regression are regularization methods to reduce the complexity of the model and shrink the coefficients so that model is simpler, more interpretative and less prone to overfitting. In ridge regression, the cost function is modified by adding a penalty equivalent to the square of the magnitude of the coefficients. This is saying to minimize the normal RMSE function within the constraint on the coefficients:

&nbsp;
                       $\sum\limits_{j=0}^p w_{i}^2 < c$

The penalty term $\lambda$ regularizes the coefficients such that if the coefficients take large values the optimization function is penalized. This results in the shrinkage of the coefficients and helps to reduce the model complexity and multi-collinearity. A lower constraint will make the model resemble the simple linear regression.

The cost function of Lasso(least absolute shrinkage and selection operator) regression is minimizing the rmse with the constraint on the coefficients in the form below:

&nbsp;
                      $\sum\limits_{j=0}^p \lvert (w_{i}) \rvert < t$
                      
The only difference here is that instead of taking the square of the the coefficients, magnitudes are taken into account. This type of regularization can lead to zero coefficients. This means some of the features are completely removed from the model. So Lasso regression not only helps us avoid overfitting but also helps in feature selection. We have plotted the graphs for both ridge and lasso below and we can see that lasso reduces the number of features significantly. 

```{r ridge-lasso-shrinkage, echo = FALSE, warning = FALSE, message = FALSE}
set.seed(42)
trn_x = model.matrix(log(units + 1) ~ ., data = train_samp)[, -1]
ridge_reg = cv.glmnet(trn_x, log(train_samp$units + 1), alpha = 0, nfolds = 10)
lasso_reg = cv.glmnet(trn_x, log(train_samp$units + 1), alpha = 1, nfolds = 10)
par(mfrow = c(1,2))
plot(ridge_reg)
plot(lasso_reg)
```

### Best Model post Improvements

The best model post improvements is the Stepwise Linear Regression with AIC. As explained above, we first transformed our response variable to log, then removed variables based on VIF and further used the stepwise methodsuing AIC criteria.

This model has a $R^2$ of `r summary(step_aic_mod_log)$r.squared`, which is much better than our initial model. We have `store_nbr`, `item_nbr`, `day`, `season`, and some weather variables as predictors. THe variables are significant with p-values less than 0.05. This model give us a kaggle score of `r kaggle_aic`.



# Advanced Methods

### Additional Features

```{r df-features, echo = FALSE, eval = FALSE, include = FALSE}
# store_item_with_nonzero_sales
df = read.csv('data/train.csv')

get_store_item_with_nonzero_sales = function(df) {
  df$log1p = log(df$units + 1)
  g = df %>% group_by(store_nbr, item_nbr) %>% summarize(mean_sales = mean(log1p))
  g = g %>% filter(mean_sales > 0)
  g1 = g[, c("store_nbr", "item_nbr")]
  g1 = g1 %>% arrange(item_nbr)
  return(g1)
}

store_item_num = get_store_item_with_nonzero_sales(df)
nrow(store_item_num)
write.csv(store_item_num, 'data/store_item_num.csv', row.names = FALSE)

# module 2
# PPR
get_df_fitted_model = function(df) {
  date_range = 0:1034
  
  list_store_items = read.table("data/store_item_num.csv",
                                header = TRUE,
                                sep = ',')
  range_ = 1:nrow(list_store_items)
  
  df$logpplus1 = log(df$units + 1)
  
  origin_date = 15340
  df$date2j =
    as.integer(floor(julian((
      as.POSIXlt(df$date)
    )))) - origin_date
  
  exlude_date =
    16064 - origin_date
  df = df[df$date2j != exlude_date, ]
  
  df_fitted_model = data.frame(date2j = c(),
                               store_num = c(),
                               item_no = c())
  
  for (i in range_) {
    store_num = list_store_items[i, "store_nbr"]
    item_no = list_store_items[i, "item_nbr"]
    df_1 = subset(df, store_nbr == store_num & item_nbr == item_no)
    df_1.ppr =
      ppr(
        max.terms = 4,
        nterms = 4,
        data = df_1,
        logpplus1 ~ date2j
      )
    
    df_2 = data.frame(store_nbr = store_num,
                      item_nbr = item_no,
                      date2j = date_range)
    df_2$ppr_fitted = predict(df_1.ppr, df_2)
    df_fitted_model = rbind(df_fitted_model, df_2)
  }
  return(df_fitted_model)
}

df_fitted_model = get_df_fitted_model(df)

write.table(df_fitted_model,
            "data/baseline.csv")

# module 3
# holidays, holiday_names
key = read.csv("data/key.csv")
wtr = read.csv("data/weather.csv")

get_holidays = function() {
  d = read.table(
    "data/holidays.txt",
    sep = "\n",
    fill = FALSE,
    strip.white = TRUE,
    quote = ""
  )
  # nrow(d)
  d$V1 = as.character(d$V1)
  
  # df_holiday_dates = data.frame(date2=character())
  dt_array = c(as.Date(x = integer(0), origin = "1970-01-01"))
  
  for (idx in 1:nrow(d)) {
    # print(idx)
    # print(d[idx,])
    ss = strsplit(d[idx, ], ' ')
    date_str = paste(ss[[1]][1], '-', ss[[1]][2], '-', ss[[1]][3])
    date_str
    dt = as.Date(paste(ss[[1]][1], '-', ss[[1]][2], '-', ss[[1]][3]),
                 "%Y - %b - %d")
    # print(dt)
    dt_array = c(dt_array, dt)
    # df_holiday_dates = rbind(df_holiday_dates, as.character(dt))
  }
  
  # dt_array
  df_holiday_dates = data.frame(Date = dt_array)
  # nrow(df_holiday_dates)
  return(df_holiday_dates)
}

holidays = get_holidays()
df_holiday_dates = holidays
nrow(holidays)

get_holiday_names = function() {
  d = read.table(
    "data/holiday_names.txt",
    sep = "\n",
    fill = FALSE,
    strip.white = TRUE
  )
  # nrow(d)
  d$V1 = as.character(d$V1)
  
  # df_holiday_dates = data.frame(date2=character())
  dt_array = c(as.Date(x = integer(0), origin = "1970-01-01"))
  holiday_name_array = c()
  
  for (idx in 1:nrow(d)) {
    ss = strsplit(d[idx, ], ' ')
    date_str = paste(ss[[1]][1], '-', ss[[1]][2], '-', ss[[1]][3])
    date_str
    dt = as.Date(paste(ss[[1]][1], '-', ss[[1]][2], '-', ss[[1]][3]),
                 "%Y - %b - %d")
    # print(dt)
    dt_array = c(dt_array, dt)
    
    holiday_name_array = c(holiday_name_array, ss[[1]][4])
    # df_holiday_dates = rbind(df_holiday_dates, as.character(dt))
  }
  
  # dt_array
  # holiday_name_array
  df_holiday_dates2 =
    data.frame(date2 = dt_array, holiday_name = holiday_name_array)
  # df_holiday_dates2
}

holiday_names = get_holiday_names()
df_holiday_dates2 = holiday_names
nrow(holiday_names)

# module 4
# preprocess train and test
to_float = function(series,
                    replace_value_for_M,
                    replace_value_for_T) {
  series = trimws(series)
  series[series == 'M'] = replace_value_for_M
  series[series == 'T'] = replace_value_for_T
  return(series)
  
}

wtr = read.csv("data/weather.csv")

preprocess = function(df, is_train) {
  if (is_train) {
    df$log1p = log(df$units + 1)
  }
  
  df$date2 = as.Date(df$date)
  wtr$date2 = as.Date(wtr$date)
  wtr$preciptotal2 = to_float(as.character(wtr$preciptotal), 0.00, 0.005)
  wtr$preciptotal2 = as.numeric(wtr$preciptotal2)
  wtr$preciptotal_flag =  ifelse(wtr$preciptotal2 > 0.2, 1, 0)
  
  wtr$depart2 = to_float(as.character(wtr$depart), NA, 0.00)
  wtr$depart2 = as.numeric(wtr$depart2)
  
  wtr$depart_flag = 0.0
  wtr$depart_flag = ifelse(wtr$depart2 < -8.0,-1, wtr$depart_flag)
  wtr$depart_flag = ifelse(wtr$depart2 > 8.0 ,  1, wtr$depart_flag)
  
  df = left_join(df, key, on = c('store_nbr'))
  
  df = left_join(df, wtr[, c("date2", "station_nbr", "preciptotal_flag", "depart_flag")],
                 on = c("date2", "station_nbr"))
  
  # weekday
  df$weekday = as.factor(as.POSIXlt(df$date2)$wday)
  df$is_weekend = as.factor(ifelse(((df$weekday == 0) |
                                      (df$weekday == 6)), 1, 0))
  df$is_holiday = as.factor(ifelse(df$date2 %in% df_holiday_dates$Date, 1, 0))
  df$is_holiday_weekday = as.factor((df$is_holiday == 1) &
                                      (df$is_weekend == 0))
  df$is_holiday_weekend = as.factor((df$is_holiday == 1) &
                                      (df$is_weekend == 1))
  
  # day, month, year
  df$day = format(df$date2, format = "%d")
  df$month = format(df$date2, format = "%m")
  df$year = format(df$date2, format = "%Y")
  
  df = left_join(df, df_holiday_dates2, on = c('date2'))
  
  around_BlackFriday = c(
    "BlackFridayM3",
    "BlackFridayM2",
    "ThanksgivingDay",
    "BlackFriday",
    "BlackFriday1",
    "BlackFriday2",
    "BlackFriday3"
  )
  df$around_BlackFriday = ifelse(
    as.character(df$holiday_name) %in% around_BlackFriday,
    as.character(df$holiday_name),
    "Else"
  )
  unique(df$around_BlackFriday)
  return(df)
}

df_train = inner_join(df, store_item_num, by = c('store_nbr', 'item_nbr'))
nrow(df_train)
df_train = preprocess(df_train, T)

df_test = read.csv('data/test.csv')
nrow(df_test)
df_test = preprocess(df_test, F)

# module 5
# rolling mean
library(zoo)

get_rolling_mean = function(df_train) {
  df_list = c()
  
  for (idx in 1:nrow(store_item_num)) {
    store_num_ = store_item_num$store_nbr[idx]
    item_num_ = store_item_num$item_nbr[idx]
    
    df1 = df_train[df_train$store_nbr == store_num_ &
                     df_train$item_nbr == item_num_,]
    
    df1$rolling_mean_10 = rollmean(x = df1$log1p,
                                   k = 10,
                                   fill = 0)
    df1$rolling_mean_15 = rollmean(x = df1$log1p,
                                   k = 15,
                                   fill = 0)
    df1$rolling_mean_20 = rollmean(x = df1$log1p,
                                   k = 20,
                                   fill = 0)
    
    df_list = rbind(df_list, df1)
  }
  
  return(df_list)
  
}

df_rolling_mean = get_rolling_mean(df_train)
df_rolling_mean = df_rolling_mean %>% select(date2,
                                             store_nbr,
                                             item_nbr,
                                             log1p,
                                             rolling_mean_10,
                                             rolling_mean_15,
                                             rolling_mean_20)

head(df_rolling_mean)

write.table(df_rolling_mean,
            "data/df_rolling_mean.csv")
```

Before we move on to some advanced machine learning algorithms, we created a couple of extra features which based on our evaluation of the data so far we think may be useful predictors. We created the features in a separate code file and then created a csv file to be used with the mail analysis. 

The first feature is *ppr_fitted* is calculated using the package `ppr` in `R`. 
*PPR* stands for Pursuit Regression Model. We use this model to make a baseline model for the units. This feature is created by using the projection pursuit regression model in R. The model is fitted by using only date as the predictor and log(units +1) as the response variable. This is sort of like getting a time series version of the projections and treating it as the base case. The predicted values from this model are used as the baseline. Linear Regression assumes the linear relationship between the response and the predictor variables. In the real-world generally, this assumption may not hold true. PPR is a nonparametric approach that overcomes the limitation of linear assumption. These predicted values result in the smoothing of the response variable. 

*rmean* : This is the rolling mean of log(1+units) over a period of 12 days. This is calculated for each unique combination of store and item number. This can be thought of as the moving average of the log(1+units) to understand the impact of units sold on the days before and after a particular day. It is calculated by taking the unweighted mean of the values before and after the particular day. This also results in smoothing of the response variable that is the log(units+1)

```{r additionalfeatures, echo = FALSE, warning = FALSE}
df_features = read.csv("df_features.csv")
df_features$date2 = as.Date(df_features$date2, format = "%m/%d/%y")
df_features[is.na(df_features$rmean), "rmean"] = 0
names(df_features)[names(df_features) == "date2"] <- "date"
```

```{r train_samp_new, echo = FALSE, message = FALSE, warning = FALSE}
df_features$store_nbr = as.factor(df_features$store_nbr)
df_features$item_nbr = as.factor(df_features$item_nbr)
train_samp_join  = left_join(train_samp, df_features, by = c("store_nbr", "item_nbr", "date"))
```

```{r test_samp_new, echo = FALSE, message = FALSE, warning = FALSE}
test_samp_join = left_join(test_samp, df_features, by = c("store_nbr", "item_nbr", "date"))
test_samp_join$mean_sales = NULL
```

Once we have prepared our final training and test datasets, we can move on to testing some advanced machine learning algorithms on our datasets and compare their performance to the linear models we fitted before this. We considered Random Forests and Gradient Boosting Method as the two methods for the purpose of this analysis.

**Random Forest:** Random forest is a supervised machine learning algorithm. It builds a "forest" of decision trees which is generally referred to as the bagging method. Bagging is a machine learning ensembling method designed to increase the stability and accuracy of the statistical algorithms and reduce the variance of the algorithms while having low bias. Not only does it help in reducing variance but it also helps in avoiding overfitting. The general idea is to combine a bunch of learning methods to get better and more accurate results.

In simple words, Random Forests builds multiple decision trees and merges them together to get a more accurate and stable precision. Random forest basically builds on the idea of the wisdom of the crowd. For example: If we ask one person to guess a number between 1 and 1000, the number guessed can be wildly different from person to person meaning high variance in your predictions. But if I ask 1000 people the same question and combine the results to get the average as the final prediction the variation is reduced considerably and the probability of the final prediction being closer to the actual number will be higher.

Random Forest can be used for both classification and regression problems. Random forests select a random subset of data with randomly selected subset of the features to build each tree and then combines them at the end. This allows Random Forest to have a large number of deep uncorrelated trees which increases the effectiveness of the forest of the combined trees. To elaborate more, Instead of searching for the most important feature while splitting a node, it searches for the best feature among a random subset of predictors. This results in wide diversity that generally results in a better model.

Another big advantage of random forest algorithm is that it is very convenient to measure the relative importance of each predictor in making effective predictions for the response variable. This importance score is calculated based on looking at how much the tree nodes that use the feature reduce the impurity across all trees in the forest. This score is calculated for each feature and the results are scaled so the sum of all importance adds up to one. By looking at the feature importance you can decide which features to keep in the model and which features are safe to be dropped as their impact or contribution is not significant to the predictions. This is crucial because the more variables you have in the model, the more the chances of overfitting, so by removing features which don't contribute enough we avoid overfitting. 

Breaking it further down to the individual tree, each internal node in a decision tree represents a function which performs a test on the feature and each branch represents the outcome of that test. Each leaf node represents the final label or value based on the computation of all attributes. One disadvantage of Random Forest is that because a large number of trees are being created and then the subtrees are being combined, it makes the computation slower depending on the number of trees being built. 

Some of the major hyperparameters/tuning parameters of random forest are:

- **n_estimators** is just the number of trees the algorithm builds before combining them to take the average of predictions or taking the maximum voting in case of classification. In general, a higher number of trees increases the performance but slows down the computation. 

- **max_features** is the maximum number of features random forest considers to split a single node. Another important feature is **min_sample_leaf** which determines the minimum number of leafs required to split an internal node.

Another advantage of random forest is that the default hyperparameters it uses often produces very good results and understanding the hyperparameters is very easy and they are relatively less in number. There is a running joke in the data science community that if you are confused fit random forest and if that doesn't work move on to deep learning. I feel that this joke speaks to the versatility and power of the random forests.

```{r randomforest, echo = FALSE, mesage = FALSE, warning = FALSE, cache = TRUE, cache.path = "D:/Data Science/425 Project/"}
set.seed(42)
mod_rf = train(
  form = log(units + 1) ~ . - date - station_nbr - month - stnpressure - tavg,
  data = train_samp_join,
  method = 'ranger',
  metric = 'RMSE',
  trControl = trainControl(method = 'oob'),
  importance = "impurity",
  verbose = FALSE
)
```

```{r varimp, echo = FALSE, warning = FALSE, eval = FALSE}
#varImp(mod_rf)
ggplot(mod_rf$variable.importance, aes(x = reorder(variable, importance), 
                                       y = importance,
                                       fill = importance))+
  geom_bar(stat = "identity", position = "dodge") + coord_flip() +
  ylab("variable importance")+
  xlab("")+
  ggtitle("Information Value Summary")+
  guides(fill = F) +
  scale_fill_gradient(low = "red", high = "blue")
```

```{r submission-rf, echo = FALSE, warning = FALSE, message = FALSE}
pred_model = predict(mod_rf, test_samp_join)
pred_model = exp(pred_model)
submission_mod = tibble(id = id, units = pred_model)
submission_final = left_join(submission_base, submission_mod, by = "id")
submission_final$units[is.na(submission_final$units)] = 0
submission_final$units[submission_final$units < 0] = 0
submission_final$pred = NULL
write.csv(submission_final, "D:/Data Science/425 Project/submission_rf.csv", row.names = FALSE)
```

**Gradient Boosting Method:** GBM is one of the most popular methods used in Data Science competitions. But most developers still treat it as a black box method. We will try to break it down and lay an intuitive framework for this machine learning technique. 

Boosting is a method of converting a series of weak learners into strong learners. In boosting, each new tree is a fit on on a modified version of the original dataset. GBM can be easily explained by first explaining AdaBoost algorithm. The adaboost algorithm begins by training a decision tree in which each observations are equally weighted. Post evaluation of the first tree, the weights of difficult to classify and lower the weights of the observations which are easier to classify. The second tree is therefore grown on the weighted data. The idea is to improve the predictions of the first tree. So the predictions become average of the first and second tree. We then calculate the Root Mean Square error in the case of regression and grow a third tree to predict the revised residuals. This process is repeated for for a specific number of iterations. Subsequent trees will help us make predictions for observations which are not well predicted by the previous trees. Predictions of the final ensemble model is therefore the weighted sum of the predictions made by the previous tree models.

Gradient Boosting trains many models in a gradual, additive and sequential manner. The major difference in gbm as compared to adaboost is in how the two algorithms identify the shortcomings of weak learners. While the adaboost model identifies the shortcomings by using high weight data points, gbm performs the same by using gradients in the loss function. The loss function is a measure indication how good are model's coefficients are at fitting the underlying data. For example, in this case, the loss function would be the error between the true and predicted units. One of the biggest motivations of using a gradient boosting machine is that it allows one to optimize a user define cost function, instead of the default loss functions which may not be suitable to the particular application.

There are two important hyperparameters in the case of gradient boosting, *interaction.depth* and *shrinkage*. Interaction Depth specifies the maximum depth of each tree in the sequence of trees and Shrinkage is considered as the learning rate used for reducing or shrinking the impact of each additional fitted base learner. It reduces the size of incremental steps and thus penalizes the importance of each consecutive iteration. 

Hyperparameter tuning is especially significant for gbm modelling since they are prone to overfitting. The special process of tuning the number of iterations for an algorithm such as gbm and random forest is called "Early Stopping". Early stopping performs model optimization by monitoring the model's performance on a separate test data set and stopping the training procedure once the performance on the test data stops improving beyond a certain number of iterations.

It avoids overfitting by identifying the inflection point where performance starts to stagnate on the test data set while the performance keeps improving on the training dataset. The ideal time to stop training the model is when the validation error has decreased and started to stabilize before it starts increasing due to overfitting. 
```{r, gbm, echo = FALSE, message = FALSE, cache = TRUE, cache.path = "D:/Data Science/425 Project/"}
set.seed(42)
mod_gbm = train(
  form = log(units + 1) ~ . - date - station_nbr - month - stnpressure -tavg,
  data = train_samp_join,
  method = "gbm",
  metric = "RMSE",
  verbose = FALSE
)
```

```{r submission-gbm, echo = FALSE, warning = FALSE, message = FALSE}
pred_model = predict(mod_gbm, test_samp_join)
pred_model = exp(pred_model)
submission_mod = tibble(id = id, units = pred_model)
submission_final = left_join(submission_base, submission_mod, by = "id")
submission_final$units[is.na(submission_final$units)] = 0
submission_final$units[submission_final$units < 0] = 0
submission_final$pred = NULL
write.csv(submission_final, "D:/Data Science/425 Project/submission_gbm.csv", row.names = FALSE)
```

# Conclusion

To conclude, we fitted various models to the walmart sales data in stormy weather and tried to predict the sales of different items in different stores. Random Forest performed the best with a kaggle score of 0.14, the best ever score on kaggle is around 0.10 so this is pretty good performance from our model. We were somewhat limited by the computation power we had as were not able to consider the interactions as it created too big a vector to be handled by the computational power we had at our disposal. Apart from the predictions and the score, we applied most of the concepts learned in the Applied Regression class(STAT 425) and understood them much better after applying all these concepts to real world data. The predictions made from this tool can be used by supply managers to assess the demand of certain products in extreme weather conditions and make supply chain decisions accordingly.

```{r results, echo = FALSE, message = FALSE, warning = FALSE}
results = tibble("Model Name" = c("Simple Linear Regression", 
                                 "SLR with Log Transformation", 
                                 "SLR with Log transformation post VIF Analysis",
                                 "Stepwise Linear Regression with AIC",
                                 "Stepwise Linear Regression with BIC",
                                 "Random Forest",
                                 "Gradient Boosting"),
                "Kaggle Private Score" = c(0.23326,
                                           0.16714,
                                           0.16395,
                                           0.16395,
                                           0.16397,
                                           0.14379,
                                           0.14394),
                "Kaggle Public Score" = c(0.23442,
                                          0.16624,
                                          0.16354,
                                          0.16354,
                                          0.16353,
                                          0.14313,
                                          0.14303))

results %>%  
  kable(align = "ccc", digits = 5, caption = "Kaggle Results for all Models Tested") %>% 
   kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), full_width = FALSE, position = "center")
```

# Appendix

```{r, print-eda-plots, fig.height = 24, fig.width = 24, echo = FALSE}
gridExtra::grid.arrange(p01, p02, p03, p04, p05, p06, p07, p08, ncol = 2)
```











