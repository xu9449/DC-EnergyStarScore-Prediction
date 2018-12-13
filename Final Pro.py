
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np 

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('data2017_1.xlsx', index_col=0)
    data_xls.to_csv('2.csv', encoding='utf-8')
    
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize 

import seaborn as sns  
sns.set(font_scale = 2)

from sklearn.model_selection import train_test_split 


# ## Data Cleaning and Formatting  
# ### Load in the Data and Examine

# In[96]:


#transfer XLSX to CSV 
if __name__ == '__main__':
    xlsx_to_csv_pd()
    data = pd.read_csv('2.csv')
data.head()


# ## Data Types and Missing Values

# In[97]:


data.info()


# ### Convert Data to Correct Types 

# In[89]:


data = data.replace({'Not Available': np.nan})


# In[92]:


for col in list(data.columns):
    if('weather_norm_site_eui' in col):
        data[col] = data[col].astype(float)


# In[94]:


data.info()


# In[98]:


data.describe()


# In[99]:


### Missing Values


# In[100]:


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
    


# In[101]:


missing_values_table(data1)


# In[102]:


missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))
                                  


# In[104]:


data = data.drop(columns = list(missing_columns))


# # Exploratory Data Analysis 

# ## Single Variable Plots 

# In[105]:


figsize(8, 8)

# Rename the score 
data = data.rename(columns = {'energy_star_score': 'score'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 
plt.title('Energy Star Score Distribution');


# In[106]:


plt.hist(data['site_eui'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');


# In[107]:


data['site_eui'].describe()


# In[29]:


data['site_eui'].dropna().sort_values().tail(10)


# In[108]:


data.loc[data['site_eui'] == 7619.7, :]


# ### Removing Outliers 

# In[114]:


# I don't know what this meaning,

first_quartile = data['site_eui'].describe()['25%']
third_quartile = data['site_eui'].describe()['75%']
iqr = third_quartile - first_quartile
data = data[(data['site_eui'] > (first_quartile - 3 * iqr)) &
            (data['site_eui'] < (third_quartile + 3 * iqr))]
print(first_quartile)
print(third_quartile)
print(iqr)


# In[116]:


figsize(8, 8)
plt.hist(data['site_eui'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');


# ### Looking for Relationships 

# In[35]:


# Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['score'])
types = types['primary_ptype_self'].value_counts()
types = list(types[types.values > 100].index)


# In[117]:


# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['primary_ptype_self'] == b_type]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['score'].dropna(),
               label = b_type, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);


# ### Correlations between Features and Target 

# In[119]:


# Find all correlations and sort 
correlations_data = data.corr()['score'].sort_values()

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))


# In[120]:


#Select the numeric columns
numeric_subset = data.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = data[['primary_ptype_self']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without an energy star score
features = features.dropna(subset = ['score'])

# Find correlations with the score 
correlations = features.corr()['score'].dropna().sort_values()


# In[121]:


correlations.head(15)


# In[122]:


correlations.tail(15)


# In[123]:


figsize(12, 10)

# Extract the building types
features['primary_ptype_self'] = data.dropna(subset = ['score'])['primary_ptype_self']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['primary_ptype_self'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('site_eui', 'score', 
          hue = 'primary_ptype_self', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("site_eui", size = 28)
plt.ylabel('Energy Star Score', size = 28)
plt.title('Energy Star Score vs Site EUI', size = 36);


# In[125]:


# Extract the columns to  plot
plot_data = features[['score', 'site_eui', 
                      'log_total_ghg_emissions_intensity', 
                      'weather_norm_source_eui']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns 
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
                                        'log_total_ghg_emissions_intensity': 'log GHG emissions intensity',
                                        'weather_norm_source_eui ': 'Weathrer Normal EUI'})

# Drop na values
plot_data = plot_data.dropna()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3)

# Upper is a scatter plot
grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# Bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

# Title for entire plot
plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02);


# In[48]:


# Copy the original data
features = data.copy()

# Select the numeric columns
numeric_subset = data.select_dtypes('number')

# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
        
# Select the categorical columns
categorical_subset = data[['primary_ptype_self']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

features.shape


# In[50]:



plot_data = data[['site_eui', 'total_ghg_emissions_intensity']].dropna()

plt.plot(plot_data['site_eui'], plot_data['total_ghg_emissions_intensity'], 'bo')
plt.xlabel('Site EUI'); plt.ylabel('Weather Norm EUI')
plt.title('GHG Emissions Intensity vs Site EUI, R = %0.4f' % np.corrcoef(data[['site_eui', 'total_ghg_emissions_intensity']].dropna(), rowvar=False)[0][1]);


# In[70]:


no_score = features[features['score'].isna()]
score = features[features['score'].notnull()]

print(no_score.shape)
print(score.shape)


# In[72]:


from sklearn.model_selection import train_test_split
# Separate out the features and targets
features = score.drop(columns='score')
targets = pd.DataFrame(score['score'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)


# In[73]:


def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


# In[74]:



baseline_guess = np.median(y)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))


# In[76]:


# Save the no scores, training, and testing data
no_score.to_csv('no_score.csv', index = False)
X.to_csv('training_features.csv', index = False)
X_test.to_csv('testing_features.csv', index = False)
y.to_csv('training_labels.csv', index = False)
y_test.to_csv('testing_labels.csv', index = False)

