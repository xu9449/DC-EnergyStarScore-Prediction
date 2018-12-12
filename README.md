# Energy Star Prediction  
This is a Project for my Machine Learning Class.  
We want to build a model that can predict the Energy Star Score of Washington DC's buildings and interpret the result to find the factors which influence the score most.  
  
What is a correlation coefficient?  
how well they are related.  
PPMC Pearson Product Momoent Correlation  
it show the linear relationship between two sets of data    

## Problem Definition

In this project, we work with [2017 Energy and Water Performance Benchmarking Results as of 2018](https://doee.dc.gov/node/1368286) from Washignton DC.  

The data includes the Energy Star Score, which makes this a supervised regression machine learning task.  

We want to develop a model that is both accurate — it can predict the Energy Star Score close to the true value — and interpretable — we can understand the model predictions. Once we know the goal, we can use it to guide our decisions as we dig into the data and build models.  

## Machine Learning Workflow  

1) Data cleaning and formatting
2) Exploratory data analysis
3) Feature engineering and selection
4) Establish a baseline and compare several machine learning models on a performance metric
5) Perform hyperparameter tuning on the best model to optimize it for the problem
6) Evaluate the best model on the testing set
7) Interpret the model results to the extent possible
8) Draw conclusions and write a well-documented report  


### Imports
We will use the standard data science and machine learning libraries: numpy, pandas, and scikit-learn. We also use matplotlib and seaborn for visualization.  
  
```  
# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Matplotlib visualization
import matplotlib.pyplot as plt
%matplotlib inline

# Set default font size
plt.rcParams['font.size'] = 24

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split

```  
### Data Cleaning and Formatting  
