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
**Load in the Data and Examine**     
```
def xlsx_to_csv_pd():
    data_xls = pd.read_excel('2017DC.xlsx', index_col=0)
    data_xls.to_csv('4.csv', encoding='utf-8')  
if __name__ == '__main__':
    xlsx_to_csv_pd()
    data = pd.read_csv('4.csv')
data.head()  
```  
  
![data.head()](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/1_Actual%20Data%20Looklike.png)
      
There are 37 columns and we want to know each's meaning.  Then we found this page, it tell us what each column's term stands for. [Link](https://doee.dc.gov/sites/default/files/dc/sites/ddoe/publication/attachments/Data%20Glossary%20for%20Energy%20and%20Water%20Performance%20Benchmarking%20Data%20Results_3.pdf)  
**Data Types and Missing Values**  
```  
data.info()
```  
![Data Info](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/2.datainfo.png)   
**Missing Values**  
  
 We want to calculates the number of missing values and the percentage of the total values that are misssing for each column.  
  
```  
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # table with the result 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    
    
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    
    
    return mis_val_table_ren_columns  
  ```  
     
  
![missing](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/3_misssing%20value.png)    
   We drop the Columns which missing rate higer than 50%. Normal it depends on the importances of the data set. Most of the people will choose around 85%. In this project, we choose 50%.  
   we will remove 3 columns. 
   
### Exploratory Data Analysis  
**Single Variable Plots**  
![D](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/4_%20energy%20star%20score%20distribution.png)  
We can see that the score 0 is disproporationate.   
Then, we tried to plot the Site EUI(Site Energy Use Intensity)distribution which is the total energy use divided by the square footage of the building.  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/5_Site_eui_before%20move%20out%20outlier.png)
The graph is incredibly skewed because of the presence of a few buldings with very high scores.  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/5.1_find%20outlier.png)  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/5.2_find%20outlier.png)  
One building is clearly far above the rest   
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/5.3%20find%20outlier.png)  
**Removing Outliers**  
```

first_quartile = data['site_eui'].describe()['25%']
third_quartile = data['site_eui'].describe()['75%']
iqr = third_quartile - first_quartile
data = data[(data['site_eui'] > (first_quartile - 3 * iqr)) &
            (data['site_eui'] < (third_quartile + 3 * iqr))]
print(first_quartile)
print(third_quartile)
print(iqr)

```  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/5.4%20site%20eui%20after%20move%20out%20outlier.png)  
### Looking for Relationships  
We create a list of building with more than 50 measurements and plot it.  
```  
# Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['score'])
types = types['primary_ptype_epa'].value_counts()
types = list(types[types.values > 50].index)

# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['primary_ptype_epa'] == b_type]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['score'].dropna(),
               label = b_type, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);

```
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/6_Density%20Plot%20of%20energy%20star%20scores%20by%20building%20type.png)  
  
For the negative part in this image. It is because the Kernel Density Estimation  
Kernel Density Estimation  
Kernel  
A special type of probability density function with the added property that it must be even.  Also, it has the properties:  
non-negative  
real valued  
even  
its definite integral over its support set must equal to 1  
why we get the nagative values ?  
A KDE replaces each observation xi by a little hill of probability density, of area 1/n , and the density estimate at a point x- is the sum of all the little density-hills, evaluated at x-.  
  
**Correlations between Features and Target**  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/7_Values%20of%20Pearson%20Correlation%20Coefficient.png)  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/8_Correlations%20between%20Features%20and%20score.png)   
we take log and square root transformations of the numerical variables to account for possible non-linear relationships. Also, we use one-hot encode the two selected categoriacal variables . 
  
Then we plot the energy star score vs Site EUI  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/9_Energy%20Star%20Score%20vs%20Site%20EUI%20scatterplot.png)  
 
 
  
**Pairs Plot**  
![](https://github.com/xu9449/EnergyStarPrediction/blob/master/Part1_images/10_Pairs%20Plot%20of%20Energy%20Data.png)  
### Feature Engineering and Selection  


