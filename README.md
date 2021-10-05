# Merger_and_Acquisition_Prediction
#### Video : https://github.com/priyal94/Merger_and_Acquisition_Prediction/blob/master/Video.mp4
## Problem Statement and Description of Dataset
Merger and Acquisition is a critical factor in keeping the business ahead, having that in mind, this project aims to predict the merger and acquisition by a company using supervised machine learning methods considering various global factors of the company to support our hypothesis. The dataset consists of data for 10000 companies and 30 features. Each feature further consisted of more than five categories. We proposed three hypothesis which was to build a predictive model through supervised machine learning to classify which companies will be a part of merger and acquisition activity, whether the companies in loss will be acquired or shut down and what are the potential sectors for a company to participate in M&A activity to maximize its profit.

In this project, we aim to solve 2 very important business cases. 

1. The first business case intends to build a strategy for a company to invest in M&A activity as it is crucial for the financial growth of a business. Our goal is to build a model, such that given a company, it can predict whether a company will acquire, will be merged or will not participate in the activity at all. The application of this model could save thousands of dollars for a company and could help a company study its competitor's actions.
2. The second Business case is to propound a company to expand its business in different sectors for a resource-effective investment. The goal here is to build a recommendation system using machine learning approaches. Maximizing profit not just depend on a single component revenue of the company but more components also. So, keeping that in mind we targeted building recommendation systems which will consider all the important features such as the basic features, financial features and managerial features of companies and will recommend companies is based on them. This is a two-way treat as the bidder company can directly get which companies it can perform the merger with and or get acquired by or even take measures to stop acquisitions. This solution not just increases the possibility of expanding but the scope of expansion to maximize profits of a business.

## Workflow

After generating the hypothesis and deciding all the necessary features, the data was finally prepared to work on. After getting the data, we planned to solve the problem in 5 stages.
### Data Cleaning – 
The goal was to produce consistency in the dataset. The process involved detecting and correcting (or removing) corrupt or inaccurate records from the dataset and identified incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data. Features such as “Exit Date” and “Portfolio Companies” had a lot of missing data and features such as “Active Products” had some outliers. We then imputed the missing values and processed the outliers to produce a more meaningful structure
### Feature Engineering – 
This process involved modifying existing variables to create new variables for analysis. In our dataset, the “Founding year” and “Exit Date” variables were inconsistent as some values consisted of just the year while some had the entire date, we fixed it by just considering just the year. Dependent features were dropped. Many features required grouping operation and binning. The continuous features were converted into categorical and we one hot encoded most of them.
### Exploratory Data Analysis - 
We performed the critical process of performing initial investigations on data to discover patterns, spot anomalies, test hypothesis and to check assumptions with the help of summary statistics and graphical representations. We plotted bar graphs for all the features to observe the balance. Along with observing the balances, we observed the mean, standard deviation along with the minimum and maximum values of the features using describe() function in python
### Data Modelling and Prediction – 
After the EDA, the focus was to solve the business cases. For the first problem, i.e. classification of whether a company would be involved in merger or acquisition, decision tree and ensemble methods were used. For the second business case, clustering was performed using Kmodes and number of optimum cluster were noted and for recommendation of other companies and sectors to merge, we used cosine similarities as a parameter in the nearest neighbor algorithm.


## Data Modelling:
The next stage and the most fundamental part of our project to apply data models. As a solution to our first question, we wanted to predict whether a company would make acquisitions, or will it be a part of a merger, or it will simply not participate in the process. This is a classic example of a multiclass classification problem. 
We initially assumed conditional independence between our features and tried the Naïve Bayes algorithm. Naïve Bayes achieved 80% of 5-fold cross-validation accuracy.
After Naïve Bayes, we applied Decision Trees to overcome the problem of Conditional Dependence. They work by learning a hierarchy of if/else questions and this can force both classes to be addressed. They use a layered splitting process, where at each layer they try to split the data into two or more groups so that data that fall into the same group are most similar to each other which is called as homogeneity, and groups which are as different as possible from each other comes under heterogeneity. Decision Trees apply a top-down approach.
Here we used the ID3 algorithm which uses Entropy and Information Gain as metrics. After using the Decision Tree, we observed 74% of 5-fold cross-validation accuracy which was a significant increase from what we observed in Naïve Bayes.


### ROC Curve
After plotting the ROC curve, we observed the micro and macro average area under the curve to be 0.84 and 0.82 respectively. After getting the accuracy metrics, we tested our model on unobserved data and it predicted the correct class with an accuracy of 78%.

### Precision-Recall Curve:
True positives are data point classified as positive by the model that actually are positive (meaning they are correct), and false negatives are data points the model identifies as negative that actually are positive (incorrect). We now wanted to go beyond accuracy, hence, we calculated precision and recall score. We observed Average Precision Area to be 0.69
The training curve and the model results on completely unobserved data inferred that the model was not overfitting. The problem that we observed in our data was also the high variance of a single estimate, so we now wanted an approach to decrease the variance of that single estimate for higher stability.

## Recommendation System
## Elbow Plot:
We wanted to see the what all companies are tied in one cluster and how many clusters are there in our data-keeping company name as the target, we chose Kmodes clustering as our data were categorical and kmeans deals with numerical data and if we get the dummy variables of our data we get 296 features and complexity will increase and Kmodes directly deals with categorical data. Even after giving 8 iterations, the cost was becoming constant after 3 clusters formation and thus, we achieved the results that the optimum number of cluster formations will be 3. As a result, we can say if we want to classify a new company to which cluster would it belong, a company could be part of any one of these clusters.

## 3-NN:
Furthermore, we wanted to recommend a new company to get the sectors of that company. We decided to solve this problem using content-based recommendation systems that use the K-nearest neighbor’s algorithm that calculates cosine similarity to get the nearest companies for the target company. We chose to predict the top 3 companies based on the cosine distance. We observed that the sector of these recommended top 3 companies has some sectors which were similar and others which were different could be prospective sectors in which the target company could perform Merger and Acquisition
