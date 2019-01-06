
# coding: utf-8

# # Introduction

# Analyzing No-show appointments data set. The goal of this project is to clean the data, analyze and visualize the findings.

# # Load Packages and Dataset

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')


# Ensure the data set is loaded correctly

# In[6]:


df.head(3)


# # Data Wrangling

# Check data types and structure of the data

# In[7]:


df.info()


# In[8]:


df.describe()


# ## Data Cleaning

# ### Change column names to lower case and replace "-" with "_"

# In[9]:


df.rename(columns=lambda x: x.strip().lower().replace("-", "_"), inplace=True)
df.head()


# ### Convert patientid and appointmentid to object

# In[10]:


df['patientid'] =df['patientid'].astype('object')


# In[11]:


df['appointmentid'] =df['appointmentid'].astype('object')


# ### Remove 'Z' and 'T' from scheduleday and appointmentday columns

# In[12]:


df['scheduledday'] = df['scheduledday'].str.replace('Z','').str.replace('T',' ')


# In[13]:


df['appointmentday'] = df['appointmentday'].str.replace('Z','').str.replace('T',' ')


# ### Change datatype of scheduledday and appointmentday to datetime

# In[14]:


df['scheduledday'] = pd.to_datetime(df['scheduledday'])


# In[15]:


df['appointmentday']= pd.to_datetime(df['appointmentday'])


# #### Modify the 'no_show' column to get summary statistics easily. Replace 'No' by 0 and 'Yes' by 1.

# In[16]:


df['no_show'].replace({'No':0,'Yes':1},inplace = True)


# #### Change 'no_show' from 'Object' datatype to 'int'

# In[17]:


df['no_show'] =df['no_show'].astype('int')


# Ensure changes in dataframe

# In[18]:


df.head(3)


# In[19]:


df.info()


# #### Remove rows where age is negative. 

# In[20]:


df_cleaned1 = df.query('age > -1')


# #### As the data dictionary on Kaggle website mentions handcap to be boolean, remove all values greater than 1 

# In[21]:


#keep rows only less than or equal to 1 (0 or 1)
df_cleaned2 = df_cleaned1.query('handcap <= 1')


# #### Write Cleaned data to CSV

# In[22]:


df_cleaned2.to_csv('cleaned_data.csv',index = False)


# # Exploratory Data Analysis

# ### Load cleaned data set, check datatypes 

# In[23]:


df = pd.read_csv('cleaned_data.csv')


# In[24]:


df.info()


# ### Change datatype of required columns

# In[25]:


#Change datatype of 'patientid' and 'appointmentid' to Object. 

df['patientid'] =df['patientid'].astype('object')
df['appointmentid'] =df['appointmentid'].astype('object')

#Change datatype of 'scheduledday' and 'appointmentday' to date_time. 
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])


# In[26]:


df.age.median()


# # Research Questions

# ## 1. Do younger people have higher no_show rate than older people?

# In[27]:


median_age = df['age'].median()
low_age = df.query('age < {}'.format(median_age))
high_age = df.query('age > {}'.format(median_age))

mean_noshow_low = low_age['no_show'].mean()
mean_noshow_high = high_age['no_show'].mean()


# In[28]:


locations = [1,2]
heights = [mean_noshow_low,mean_noshow_high]
labels = ['low_age','high_age']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average no_show rate by Age')
plt.xlabel('Age')
plt.ylabel('no_show rate');


# ## 2. Does gender influence no show rate?

# In[29]:


female = df.query('gender == "F"')
male  = df.query('gender == "M"')

mean_noshow_female = female['no_show'].mean()
mean_noshow_male = male['no_show'].mean()


# In[30]:


locations = [1,2]
heights = [mean_noshow_female,mean_noshow_male]
labels = ['female','male']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average no_show rate by Gender')
plt.xlabel('Gender')
plt.ylabel('no_show rate');


# ## 3. Impact of scholarship on no_show

# In[31]:


no_scholarship = df.query('scholarship == 0')
with_scholarship  = df.query('scholarship == 1')

mean_noshow_no_scholarship = no_scholarship['no_show'].mean()
mean_noshow_with_scholarship = with_scholarship['no_show'].mean()


# In[32]:


locations = [1,2]
heights = [mean_noshow_no_scholarship,mean_noshow_with_scholarship]
labels = ['no_scholarship','with_scholarship']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average no_show rate by Scholarship')
plt.xlabel('Scholarship')
plt.ylabel('no_show rate');


# ## 4. Which days have higher no_show?

# ### Add day of the week columns to the dataframe

# In[33]:


df['scheduled_day_of_week'] = df['scheduledday'].dt.weekday_name
df['appointment_day_of_week'] = df['appointmentday'].dt.weekday_name


# In[34]:


appointmentday_mean = df.groupby('appointment_day_of_week')['no_show'].mean()
appointmentday_mean.plot(kind = 'bar')
plt.title('Average no_show rate by Appointment Day')
plt.xlabel('Appointment Day of Week')
plt.ylabel('no_show rate')


# In[35]:


scheduledday_mean = df.groupby('scheduled_day_of_week')['no_show'].mean()
scheduledday_mean.plot(kind = 'bar')
plt.title('Average no_show rate by Scheduled Day')
plt.xlabel('Scheduled Day of Week')
plt.ylabel('no_show rate')


# ## 5. Which neighbourhood higher no_show rate and which ones are lower?

# create a dataframe with only 'neighbourhood' and 'no_show' values

# In[36]:


#create a new dataframe with just required columns
df_neighbourhood = df[['neighbourhood','no_show']]


# Calculate the average no_show for each neighbourhood.

# In[37]:


#Add mean as a new column to the dataframe
mean_no_show = df_neighbourhood.groupby(['neighbourhood'])['no_show'].mean()

df_neighbourhood = df_neighbourhood.set_index('neighbourhood')
df_neighbourhood['mean_no_show'] = mean_no_show
df_neighbourhood = df_neighbourhood.reset_index()
df_neighbourhood.head(3)


# High 'no-show' neaighbourhoods have high mean as no_show = 1 means the patient did not show up

# ### top 10 neighbourhoods with high 'no_show' rate

# In[38]:


#Group neighbourhood column
df_grouped = df_neighbourhood.groupby('neighbourhood', as_index = False)['mean_no_show'].min()

df_grouped.nlargest(10,'mean_no_show').plot(x = 'neighbourhood',y = 'mean_no_show',kind = 'bar')
plt.title('High no_show rate neighbourhoods')
plt.xlabel('neighbourhood')
plt.ylabel('no_show rate')


# ### top 10 neighbourhoods with low 'no_show' rate

# In[386]:


df_grouped.nsmallest(10,'mean_no_show').plot(x = 'neighbourhood',y = 'mean_no_show',kind = 'bar')
plt.title('Low no_show rate neighbourhoods')
plt.xlabel('neighbourhood')
plt.ylabel('no_show rate')


# In[389]:


df_grouped.nlargest(10,'mean_no_show')


# In[169]:


#df2 = df.groupby(level=0).agg({''})
df3 = df.groupby(['neighbourhood', "no_show"]).size()
#df4 = df3.groupby(level=0).agg({'no_show':['size','sum']})
#print(p)
#df_neighbourhood['proportion'] = (100*p/p.groupby(level=0).sum()
df3.head()


# # Number of people showing up and not showing up based on parameters

# ## Create function to accept parameters and return size

# In[66]:


def groupby_parameters(*argv):
    param_list = []
    for arg in argv:
        param_list.append(arg)
    return ( df.groupby(param_list).size())


# In[69]:


#Group by handcap and no_show
groupby_parameters('handcap','no_show')


# In[70]:


#Group by alcoholism and no_show
groupby_parameters('alcoholism','no_show')


# In[72]:


#Group by diabetes and no_show
groupby_parameters('diabetes','no_show')


# In[73]:


#Group by hipertension and no_show
groupby_parameters('hipertension','no_show')


# In[74]:


#Group by sms_received
groupby_parameters('sms_received','no_show')


# ## Analysis by grouping multiple parameters

# In[75]:


groupby_parameters('gender','sms_received','no_show')


# Determine the no_show rate by the age groups and gender

# In[373]:


# 'gender', 'scheduledday', 'appointmentday', 'age', 'neighbourhood', 
# 'scholarship', 'hipertension', 'diabetes', 'alcoholism', 'handcap', 'sms_received'
#df2 = df
df2['age_group'] = df.age.apply(lambda x: "0-37" if x < 37 else "37+")
df2.groupby(['age_group','gender', "no_show"]).size().apply(lambda x: x / len(df))


# In[81]:


# determine percentage based on the parameters
groupby_parameters('scholarship','sms_received','no_show').apply(lambda x: x / len(df))


# ### Split the data into two data frames based on 'no_show' column. One for 0 and other for 1

# In[59]:


df_yes = df.query('no_show == 0')
df_no  = df.query('no_show == 1')


# In[64]:


df_yes.head()


# #### Plot the distribution of 'age' in both data frames

# In[169]:


df_no['age'].plot(kind = 'hist')


# In[170]:


df_yes['age'].plot(kind = 'hist')


# In[361]:


df.head(3)


# In[362]:


df_yes.head(2)


# # Conclusions

# Here are the conclusions that can be made from the analysis:
#     1. People who are older than 37 years of age have lower no-show rate.
#     2. Men and women have very similar no-show rate.
#     3. Patients without scholarship have lower no-show rate.
#     4. Patients who schedule appointment on saturday have lower no-show rate.
#     5. Patients who have scheduled appointments on satueday have high no-show rate.
#     6. PARQUE INDUSTRIAL has 0% no-show rate.
#     7. ILHAS OCEÂNICAS DE TRINDADE has 100% no-show rate.
#     8. Patients who have no scholarship and not received SMS have lower no-show rate.
#     9. The histogram of age group for the ones who did not show up and actually showed up is similar.
# 
# Limitations:
#     1. The analysis does not include any statistical inferences. Logistic regression and other mathematical models will be developed after progressing in the course.
#     2. The top 10 neighbourhoods with high no show rate have fewer people than others. Example, ILHAS OCEÃ‚NICAS DE TRINDADE which has 100% no show rate has only 2 entries. 
#     3. All the entries where handcapped > 1 is removed, 200 rows are deleted.
# 

# In[1]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

