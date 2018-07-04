
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import os


# In[6]:


raw_data_path = os.path.join(os.pardir, 'data', 'raw')
train_file_path = os.path.join(raw_data_path, "train.csv")
test_file_path = os.path.join(raw_data_path, "test.csv")


# In[7]:


train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')
type(train_df)


# In[8]:


df = pd.concat((train_df, test_df))


# In[9]:


df.head(3)


# In[10]:


df.info()


# ## Deal with missing values in the Embarded Column

# In[11]:


df[df.Embarked.isnull()]


# In[12]:


df.Embarked.value_counts()


# Note that in the case of these two passengers, they are first class and paid 80.0.  Instead of mode, let's look at how the class and embarkation point might affect fare

# In[13]:


df.groupby(['Pclass', 'Embarked']).Fare.median().unstack()


# In[14]:


df.Embarked.fillna('C', inplace=True)


# ## Fix one passenger with missing fare

# In[15]:


df[df.Embarked.isnull()]


# In[16]:


df[df.Fare.isnull()]


# In[17]:


median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S')].Fare.median()
print(median_fare)


# In[18]:


df.Fare.fillna(median_fare, inplace=True)


# In[19]:


df[df.Fare.isnull()]


# In[36]:


df.info()


# ## Feature: Age

# In[20]:


df[df.Age.isnull()]


# ## Which feature or features can we use to fill in missing age

# In[22]:


df.Age.plot(kind='hist', bins=20, color='c')


# In[23]:


df.groupby('Sex').Age.median()


# In[24]:


#age_sex_median = df.groupby('Sex').Age.transform('median')
df[df.Age.notnull()].boxplot('Age', 'Pclass')


# In[25]:


df.Name.head()


# ### Try to create a new feature based on the person's "title" embedded in their name

# In[26]:


def extractTitle(name):
    
    title_group = {
        'mr' :'Mr',
        'mrs' :'Mrs',
        'miss' :'Miss',
        'master' :'Master', 
        'don' :'Sir', 
        'rev' :'Sir', 
        'dr' :'Officer',
        'mme' :'Mrs',
        'ms':'Mrs',
        'major' :'Officer',
        'lady' :'Lady',
        'sir' :'Sir', 
        'mlle' :'Miss',
        'col' :'Officer', 
        'capt' :'Officer',
        'the countess':'Lady',
        'jonkheer' :'Sir',
        'dona':'Lady'
    }
    
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    return title_group[title.strip().lower()]


# ### Create a new feature called Title

# In[27]:


df.Name.map(lambda n: extractTitle(n)).unique()


# In[28]:


df['Title'] = df.Name.map(lambda t: extractTitle(t))


# In[29]:


# This looks better
df[df.Age.notnull()].boxplot('Age', 'Title')


# In[30]:


title_median_age = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_median_age, inplace=True)


# In[31]:


df.info()


# # Now deal with outliers

# ![image.png](attachment:image.png)

# In[32]:


# Look for outliers in passenger fare
df.Fare.plot(kind='hist')


# In[40]:


df.boxplot('Fare', 'Pclass')


# In[46]:


df.loc[df.Fare > 500]


# In[44]:


# Note above the common ticket number...
df.loc[df.Fare > 500, ['Age']]


# In[48]:


# You could create a new feature called LogFare
LogFare = np.log(df.Fare+1)


# In[58]:


# You could use quantile cut to create do 'binning'
pd.qcut(df.Fare, 4)


# In[60]:


# You can assign labels to bins to transform a numerical feature to a category feature (this is one technique in the general family
# of discretization techniques)
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])


# In[61]:


pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts()


# In[64]:


pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', rot=45);


# In[65]:


# Create new feature using binning technique above for passenger fare (this is part of feature engineering)
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])


# In[66]:


df


# # Feature Engineering

# ## Age State: (Adult or Child)

# In[69]:


np.where(df.Age >= 18, 'Adult', 'Child')


# In[71]:


df.head(1)


# In[75]:


df['AgeState'] = np.where(df.Age >= 18, 'Adult', 'Child')


# In[80]:


type(df.AgeState)


# In[81]:


type(df['AgeState'])


# In[87]:


df['AgeState'].value_counts()


# In[90]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)


# ## Feature: FamilySize

# In[91]:


df['FamilySize'] = df.Parch + df.SibSp + 1 # parents + siblings + 1 for self


# In[98]:


df.FamilySize.plot(kind='hist', bins=10)


# In[104]:


df.loc[df.FamilySize == max(df.FamilySize), ['Name', 'Survived', 'FamilySize', 'Ticket']]


# In[108]:


pd.crosstab(df[df.Survived != -888].FamilySize, df[df.Survived != -888].Survived).T


# ### IsMother
# (mothers would have been given lifeboat priority)

# In[109]:


df['IsMother'] = np.where((df.Sex == 'female') & (df.Parch > 0) & (df.Age >= 18) & (df.Title != 'Miss'), 1, 0)


# In[112]:


df[df.IsMother == 1].head()


# In[115]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].IsMother)


# ### Deck

# In[116]:


pd.Cabin


# In[119]:


df.Cabin.unique()


# In[122]:


df[df.Cabin == 'T']


# In[125]:


# Assume this is a mistake and set it to unknown
df.loc[df.Cabin == 'T', 'Cabin'] = np.NaN


# In[148]:


def get_deck(cabin):
    return str(cabin)[0].upper() if pd.notnull(cabin) else 'Z'

df['Deck'] = df.Cabin.map(lambda c: get_deck(c))


# In[151]:


pd.crosstab(df.loc[df.Survived != -888, 'Survived'], df.loc[df.Survived != -888, 'Deck'])


# In[153]:


df.info()


# ## Categorical Feature Encoding

# ### The non-numeric categorical data needs to be numeric for many ML algorithms

# Binary Encoding (two classes only -- e.g. Gender) -- is_male (0 or 1)

# If you have more than two...
# 
# Label Encoding - Good for labels that are have an intrinsic order... e.g. Use integers for each level (Low = 1, Medium = 2, High = 3)
# 
# One-Hot Encoding - No intrinsic order (like embarkment point).  You can create a feature for each value and set it to 0 or 1 (e.g. is_A, is_B, is_C, etc.)
# 

# In[159]:


df['IsMale'] = np.where(df['Sex'] == 'male', 1, 0)


# In[163]:


df.loc[0:5,['Sex', 'IsMale']]


# In[171]:


df[0:5][['Sex','IsMale']]


# ## Use the get_dummies pd command (and assign the result back) to create one-hot encoding for our non-ordered categorical features.  Note that we could have done binary encoding for AgeState as there were only two.  The result will be the same.

# In[176]:


df = pd.get_dummies(df, columns = ['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])


# In[177]:


df.info()


# In[178]:


df.head(5)


# ## Drop and reorder features
# ### Drop the columns that won't be used and move the predict column to the front

# In[182]:


df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis = 1, inplace = True)


# In[183]:


df.head()


# In[184]:


df.columns


# In[187]:


columns = ['Survived'] + [column for column in df.columns if column != 'Survived']


# In[189]:


df = df[columns]


# In[193]:


df[df.Survived == -888]


# In[195]:


df.Survived.value_counts()


# In[196]:


df.count()


# # Write the file

# In[206]:


df.loc[pd.isnull(df.Survived), 'Survived'] = -888


# In[207]:


df.info()


# In[208]:


df


# In[211]:


processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
write_train_path = os.path.join(processed_data_path, "train.csv")
write_test_path = os.path.join(processed_data_path, "test.csv")


# In[212]:


df.loc[df.Survived != -888].to_csv(write_train_path)


# In[214]:


columns = [column for column in df.columns if column != 'Survived']
columns


# In[218]:


df.loc[df.Survived == -888, columns].to_csv(write_test_path)

