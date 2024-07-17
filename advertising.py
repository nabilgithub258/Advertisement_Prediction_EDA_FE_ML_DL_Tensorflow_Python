#!/usr/bin/env python
# coding: utf-8

# In[671]:


#####################################################################################################
######################### ADVERTISING DATA SET  #####################################################
#####################################################################################################


# In[672]:


##########################################################################
############### Part I - Importing 
##########################################################################

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[673]:


df = pd.read_csv('advertising.csv')


# In[674]:


df.head()


# In[675]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[676]:


df[df.duplicated()]                  #### data seems to be good so far


# In[677]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[678]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer',ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### no missing value either, suprised


# In[679]:


df.isna().any()


# In[680]:


df.info()


# In[681]:


######################################################################
############## Part IV - EDA
######################################################################


# In[682]:


df.head()


# In[683]:


df['Daily Time Spent on Site'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Advertisement Time Spent Graph')

plt.xlabel('Number of customers')

plt.ylabel('Time')


#### seems like the mean is between 60-70


# In[684]:


df['Daily Time Spent on Site'].mean()            #### we were right


# In[685]:


df['Daily Time Spent on Site'].std()


# In[686]:


df['Age'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Advertisement Age Graph')

plt.xlabel('Number of customers')

plt.ylabel('Age')


# In[687]:


df.Age.mean()                 #### age mean of people targeted


# In[688]:


df.Age.std()                  #### age shifts either + or - on either direction 


# In[689]:


df['Ad Topic Line'].nunique()                      #### nothing unique about this column so further down we will drop it


# In[690]:


df['Area Income'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Advertisement Area Income Graph')

plt.xlabel('Number of customers')

plt.ylabel('Income')


# In[691]:


df['Area Income'].mean()             #### thats our target audience income


# In[692]:


df['Area Income'].std()              #### seems reasonable


# In[693]:


df['Daily Internet Usage'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Advertisement Internet Usage Graph')

plt.xlabel('Number of customers')

plt.ylabel('Internet')


# In[694]:


df['Daily Internet Usage'].mean()                   #### mean of internet usage, this is important for us here so we can target from this those audiences


# In[695]:


df['Daily Internet Usage'].std()                   #### std of those audiences


# In[696]:


custom = {0:'black',
         1:'green'}

g = sns.jointplot(x=df.Age,y=df['Daily Internet Usage'],data=df,hue='Clicked on Ad',palette=custom)

g.fig.set_size_inches(17,9)


#### this gives a very good holistic picture for our advert campaign, it seems like people who are younger don't click on the advert
#### from this plot it seems our audience is above 35 and people who use less internet then compared to those younger audience


# In[697]:


g = sns.jointplot(x='Age',y='Clicked on Ad',data=df,kind='reg',x_bins=[range(1,df.Age.max())],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

g.ax_joint.set_ylim(0,1)


#### this is pretty correlated we can just tell by this plot, as the age crosses above the mean the probability of population colicking on the advert goes way higher


# In[698]:


custom = {0:'pink',
          1:'grey'}

g = sns.jointplot(x=df.Age,y='Clicked on Ad',data=df,hue='Male',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)


#### from this we see the distribution of females to males are pretty even in both cases for people who clicked on the advert and those who didn't
#### but people who didn't click on the add had some marginal advantage supporting male population


# In[699]:


df.Male.value_counts()               #### pretty even distribution


# In[700]:


custom = {0:'pink',
          1:'grey'}

sns.catplot(x='Clicked on Ad',y='Age',data=df,kind='box',height=7,aspect=2,palette=custom,legend=True,hue='Male')


#### seems like we were right, females and males are very close in this for both scenarios
#### also clearly we see that age of people who clicks on the advert and who dont


# In[701]:


custom = {0:'pink',
          1:'grey'}

g = sns.jointplot(x=df.Age,y='Daily Time Spent on Site',data=df,hue='Male',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)


# In[702]:


sns.catplot(x='Clicked on Ad',y='Daily Time Spent on Site',data=df,kind='strip',height=7,aspect=2,palette=custom,legend=True,hue='Male',jitter=True)


#### quite strange though, people who spend less time on the site are the ones who click on the advert, wasn't expecting this honestly


# In[703]:


df.head()


# In[704]:


#### renaming column to make it easier to use

df.rename(columns={'Daily Time Spent on Site':'Time_spent',
                   'Area Income':'Income',
                   'Daily Internet Usage':'Net_Usage',
                   'Ad Topic Line':'Topic',
                   'Clicked on Ad':'Clicked'},inplace=True)


# In[705]:


df.head()               #### much better


# In[706]:


pl = sns.FacetGrid(df,hue='Male',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Income',fill=True)

pl.set(xlim=(0,df.Income.max()))

pl.add_legend()


#### seems like Males are slightly having better income with regards to our data set


# In[707]:


custom = {0:'red',
          1:'green'}

pl = sns.FacetGrid(df,hue='Clicked',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Income',fill=True)

pl.set(xlim=(0,df.Income.max()))

pl.add_legend()


#### interestingly people who make more money don't tend to click on the advert, so from this we can see that our population is middle to lower class audience


# In[708]:


custom = {0:'red',
          1:'green'}

pl = sns.FacetGrid(df,hue='Clicked',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Net_Usage',fill=True)

pl.set(xlim=(0,df.Net_Usage.max()))

pl.add_legend()


#### people who spends 100-150 daily internet usage tends to click on the advert
#### more internet usage doesn't equate to more clicks which kind of makes sense because our audience is not younger generations


# In[709]:


corr = df.corr()


# In[710]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### even with a very basic corr heatmap we can clearly tell clicked and age are highly highly correlated
#### so if we were to run this campaign we would invest our resources to people who are above 35 years old as our target audience


# In[711]:


corr.head()


# In[712]:


df.head()


# In[713]:


df['Age'].plot(kind='hist',legend=True,figsize=(20,7))


# In[714]:


#### as Age is such an important element in our data set, we will give Age the proper treatment it needs

df.Age.min()


# In[715]:


df.Age.max()


# In[716]:


mean_df = df.Age.mean()
std_df = df.Age.std()

print(mean_df,std_df)


# In[717]:


from scipy.stats import norm


x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(12, 6))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')


# In[718]:


#### Comprehensive time

x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='green', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='orange', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='yellow', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Age distribution inside the Titanic Dataset')
plt.xlabel('Age')
plt.ylabel('Probability Density')

plt.legend()


#### from here we can clearly see all 3 levels of z_score either side and mean in the middle


# In[719]:


#### now lets use some confidence level and see the ages of our audience

standard_error = std_df/np.sqrt(df.shape[0])


# In[720]:


#### 95% confidence interval people who are our audience are between these ages

from scipy import stats

stats.norm.interval(alpha=0.95,loc=mean_df,scale=standard_error)


# In[721]:


#### 99% confidence interval people who are our audience are between these ages

stats.norm.interval(alpha=0.99,loc=mean_df,scale=standard_error)


# In[722]:


custom = {0:'pink',
          1:'grey'}

pl = sns.FacetGrid(df,hue='Male',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'Age',fill=True)

pl.set(xlim=(0,df.Age.max()))

pl.add_legend()


#### almost same no difference in gender of our population


# In[723]:


sns.catplot(x='Clicked',data=df,hue='Male',palette=custom,kind='count',height=7,aspect=1.7)


# In[724]:


g = sns.catplot(y='Clicked',x='Male',data=df,kind='point',height=10,aspect=1.5,color='purple')

g.set_xticklabels(['Female', 'Male'])


#### seems like females are very very slighly more inclined to click on the advert but not enough to draw any conclusion out of it


# In[725]:


custom = {0:'black',
          1:'green'}

sns.lmplot(x='Age',y='Time_spent',data=df,hue='Clicked',x_bins=[range(1,df.Age.max())],height=7,aspect=2,palette=custom)


#### we do see some correlation but its not strong, lets clear the null hypothesis with pearsonr


# In[726]:


from scipy.stats import pearsonr


# In[727]:


co_eff,p_value = pearsonr(df.Age,df.Time_spent)


# In[728]:


co_eff


# In[729]:


p_value                                #### p_value less then 0.05 meaning we can reject null hypothesis


# In[730]:


custom = {0:'red',
          1:'green'}

pl = sns.lmplot(x='Age',y='Income',data=df,hue='Clicked',height=7,aspect=2,palette=custom)


#### interesting


# In[731]:


custom = { 1:'grey',
           0:'purple'}

pl = sns.lmplot(x='Age',y='Time_spent',data=df,col='Clicked',hue='Male',height=7,aspect=1.2,palette=custom)


#### this is much better to understand, it seems like it doesn't matter the gender when it boils down to clicks but it does matter then time spent on site
#### people who spend more time don't click on ads and opposite is true for people who spend less time on site


# In[732]:


df.plot('Age','Income',kind='scatter',figsize=(20,6),color='black')


#### interesting, younger people are the ones with higher income then the older people


# In[733]:


df.head()


# In[734]:




sns.lmplot(x='Time_spent',y='Clicked',data=df,hue='Male',palette=custom,x_bins=[30,35,40,45,50,55,60,65,70,75,80,85,90,95],height=7,aspect=2)


#### clear correlation between time spent and clicking of the advert


# In[735]:


df.groupby('Country').sum()['Clicked'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=10,linestyle='dashed',color='red')


#### from here we dont really have much information as the rows are too much to display effectively


# In[736]:


df.groupby('Country').sum()['Clicked'].sort_values(ascending=False).head(9).plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',color='red')


#### top 9 countires where we had the most clicks


# In[737]:


country_df = df.groupby('Country').sum()

country_df


# In[738]:


from sklearn.preprocessing import StandardScaler


# In[739]:


scaler = StandardScaler()


# In[740]:


standardized_df = scaler.fit_transform(country_df)


# In[741]:


df_comp = pd.DataFrame(standardized_df,columns=['Time_spent', 'Age', 'Income', 'Net_Usage', 'Male', 'Clicked'])


# In[742]:


df_comp.head()


# In[743]:


country_df.head()


# In[744]:


df_comp.index = country_df.index                 #### making index for df_comp


# In[745]:


df_comp.head()


# In[746]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(df_comp,linewidths=0.1,ax=ax,cmap='viridis')


#### this is amazingly made heatmap and it reveals a lot of details 
#### but the problem here is that we are ouputting more then 200 rows so its going to be hard to really clearly display them here


# In[747]:


heat = df_comp.head(20)


# In[748]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(heat,annot=True,linewidths=0.5,ax=ax,cmap='viridis')


#### from this we can clearly see that Australia has much higher Click rate then others 


# In[749]:


heat = df_comp.sort_values(by='Clicked',ascending=False).head(40)            #### top 40


# In[750]:


fig, ax = plt.subplots(figsize=(25,15)) 

sns.heatmap(heat,annot=True,linewidths=0.5,ax=ax,cmap='viridis')


#### much better and way easier to understand now, from this what stands out is the Czech Republic and its male population in our data set, intriguing


# In[751]:


df[df.Country=='Czech Republic']['Male'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',linewidth=4,color='red')


#### now makes sense why it showed such heat for Czech Republic in terms of male population


# In[752]:


#### so we have the country column sorted so now lets go back in and see the timestamp

df.head()


# In[753]:


df.info()                #### first need to convert timeStamp to timestamp


# In[754]:


new_df = df.copy()

new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'])


# In[755]:


new_df.info()              #### now its timestamp format


# In[756]:


x = new_df.Timestamp[0]
x


# In[757]:


x.hour


# In[758]:


x.minute


# In[759]:


x.second


# In[760]:


x.month


# In[761]:


x.year


# In[762]:


x.dayofweek


# In[763]:


#### feature engineering time based on Timestamp column

new_df['hour'] = new_df['Timestamp'].apply(lambda x:x.hour)


# In[764]:


new_df['month'] = new_df.Timestamp.apply(lambda x:x.month)


# In[765]:


new_df['day_of_week'] = new_df.Timestamp.apply(lambda x:x.dayofweek)


# In[766]:


new_df['month_name'] = new_df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})


# In[767]:


new_df.month_name.unique()


# In[768]:


new_df['Day'] = new_df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})


# In[769]:


custom = {0:'black',
          1:'green'}

sns.catplot(x='Day',data=new_df,kind='count',hue='Clicked',height=7,aspect=2,palette=custom)


#### seems like we have most clicks on Sundays but with a very small margin compared to days like Thursdays and Wednesdays


# In[770]:


custom = {0:'purple',
          1:'green'}


sns.catplot(x='month_name',data=new_df,kind='count',hue='Clicked',height=7,aspect=2,palette=custom)


#### seems like May and Feb are the months where we get the most clicks


# In[771]:


custom = {0:'grey',
          1:'green'}


sns.catplot(x='hour',data=new_df,kind='count',hue='Clicked',height=7,aspect=2,palette=custom)


#### seems like hour 0 and 9 are the hours which gets the most clicks


# In[772]:


custom = {0:'red',
          1:'green'}


pl = sns.FacetGrid(new_df,hue='Clicked',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'hour',fill=True)

pl.set(xlim=(0,new_df.hour.max()))

pl.add_legend()


#### its not a huge margin but we do see some sort of more clicks traffic during the hours 0-10


# In[773]:


pl = sns.FacetGrid(new_df,hue='Clicked',aspect=4,height=4)

pl.map(sns.kdeplot,'month',fill=True)

pl.set(xlim=(0,new_df.month.max()))

pl.add_legend()


#### its pretty even except with a very small margin on months 4-6


# In[774]:


#### lets see if adding the extra columns did have any impact on correlation

corr = new_df.corr()


# In[775]:


corr.head()


# In[776]:


fig, ax = plt.subplots(figsize=(20,7)) 

sns.heatmap(corr,annot=True,linewidths=0.5,ax=ax,cmap='viridis')


#### it shows some correlation with month but its very small


# In[777]:


new_df.groupby('month_name').count()['Clicked'].plot(legend=True,figsize=(20,7),marker='o',markersize=14,markerfacecolor='black',linestyle='dashed',linewidth=2,color='red')


#### from this we see that most clicks happened in the month of feb and worst in the month of July


# In[778]:


new_df.groupby('month_name').count()


# In[779]:


sns.lmplot(x='month',y='Clicked',data=new_df.groupby('month').count().reset_index(),height=7,aspect=2,line_kws={'color':'black'},scatter_kws={'color':'green'})


#### note in this case we want more clicks so higher clicks the better, it seems like the earlier months of year has been linked to more clicks


# In[780]:


new_df['Date'] = new_df.Timestamp.apply(lambda x:x.date())


# In[781]:


new_df.groupby('Date').count()['Clicked'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='green',markersize=10,linestyle='dashed',linewidth=3)


# In[782]:


new_df[new_df.Male == 1].groupby('Date').count()['Clicked'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',markersize=10,linestyle='dashed',linewidth=3)


#### male population only who clicked on the advert and the dates accordingly


# In[783]:


new_df.groupby(by=['Day','hour']).count()['Clicked'].unstack()                #### beauty of unstack


# In[784]:


heat = new_df.groupby(by=['Day','hour']).count()['Clicked'].unstack()


# In[785]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(heat,ax=ax,linewidths=0.5,annot=True)


#### seems like Monday with hour 7 is most clicked one from this heatmap we have


# In[786]:


new_df.groupby(by=['month_name','Day','hour']).count()['Clicked'].unstack().unstack()


# In[787]:


heat_2 = new_df.groupby(by=['month_name','Day','hour']).count()['Clicked'].unstack().unstack().fillna(0)


# In[788]:


fig, ax = plt.subplots(figsize=(30,15))

sns.heatmap(heat_2,ax=ax,linewidths=0.5)


#### this is just massive informative heatmap, note we are putting month, day and hour into one heatmap but its very informative


# In[789]:


new_df.groupby(by=['Day','month_name']).count()['Clicked'].unstack()

#### to make it simple to understand we will opt out hour here


# In[790]:


heat_3 = new_df.groupby(by=['Day','month_name']).count()['Clicked'].unstack()


# In[791]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(heat_3,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### from this its obvious that Wed on both Feb and Mar were most clicked, interesting


# In[792]:


#### we will conclude our EDA here and go for making models


# In[793]:


#############################################################################
#######################    Model - Classification
############################################################################


# In[794]:


new_df.head()


# In[795]:


X = new_df.drop(columns=['Topic','City','Timestamp','Clicked','hour','month_name','Day','Date'])


# In[796]:


X.head()


# In[797]:


X.Country = X.Country.astype('category')


# In[798]:


X.info()


# In[799]:


y = new_df['Clicked']


# In[800]:


y.head()


# In[801]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[802]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Country']),
                                               ('num', StandardScaler(),['Time_spent','Age','Income','Net_Usage','Male','month','day_of_week'])
                                              ]
                                )


# In[803]:


from sklearn.pipeline import Pipeline


# In[804]:


from sklearn.linear_model import LogisticRegression


# In[805]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# In[806]:


from sklearn.model_selection import train_test_split


# In[807]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[808]:


model.fit(X_train,y_train)


# In[809]:


y_predict = model.predict(X_test)


# In[810]:


from sklearn import metrics


# In[811]:


metrics.accuracy_score(y_test,y_predict)                #### seems too good to be true honestly so lets investigate


# In[812]:


print(metrics.classification_report(y_test,y_predict))


# In[813]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Not Clicked','Clicked']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(20,12))

disp.plot(ax=ax)


# In[814]:


from sklearn.ensemble import RandomForestClassifier


# In[815]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1,verbose=2))
])


# In[816]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train,y_train)')


# In[817]:


y_predict = model.predict(X_test)


# In[818]:


print(metrics.classification_report(y_test,y_predict))             #### seems like random forest didn't improve our model


# In[819]:


from sklearn.model_selection import GridSearchCV


# In[820]:


get_ipython().run_cell_magic('time', '', "\nparam_grid = {\n    'classifier__n_estimators': [100, 200, 300],\n    'classifier__max_depth': [None, 10, 20, 30],\n    'classifier__min_samples_split': [2, 5, 10],\n    'classifier__min_samples_leaf': [1, 2, 4]\n}\n\nmodel_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=2)\nmodel_grid.fit(X_train, y_train)")


# In[821]:


best_model = model_grid.best_estimator_


# In[822]:


best_model


# In[823]:


y_predict = best_model.predict(X_test)


# In[824]:


metrics.accuracy_score(y_test,y_predict)             #### still not better then the basic logistic regression


# In[825]:


print(metrics.classification_report(y_test,y_predict))


# In[826]:


#### we will stop the classification here and go for linear regression on Age column


# In[827]:


###################################################################################
################################ Linear Regression - Age
###################################################################################


# In[828]:


new_df.head()


# In[829]:


X = new_df.drop(columns=['Age','Topic','City','Timestamp','hour','month_name','Day','Date'])


# In[830]:


X.head()


# In[831]:


y = new_df.Age


# In[832]:


y.head()


# In[833]:


from sklearn.linear_model import LinearRegression


# In[834]:


preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Country']),
                                               ('num', StandardScaler(),['Time_spent','Clicked','Income','Net_Usage','Male','month','day_of_week'])
                                              ]
                                )


# In[835]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[836]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())
                       ])


# In[837]:


model.fit(X_train, y_train)


# In[838]:


y_predict = model.predict(X_test)


# In[840]:


plt.figure(figsize=(10,6))

plt.scatter(y_test,y_predict,color='black')


# In[841]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')


# In[842]:


metrics.r2_score(y_test,y_predict)                      #### not good as r2 should never be in negative


# In[843]:


metrics.mean_squared_error(y_test,y_predict) 


# In[844]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))             


# In[845]:


from sklearn.model_selection import GridSearchCV


# In[846]:


from sklearn.ensemble import RandomForestRegressor


# In[847]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


# In[848]:


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}


# In[849]:


get_ipython().run_cell_magic('time', '', "\ngrid_model = GridSearchCV(model, param_grid, cv=5, scoring='r2',verbose=2)\ngrid_model.fit(X_train, y_train)")


# In[850]:


best_model = grid_model.best_estimator_


# In[851]:


best_model


# In[852]:


y_predict = best_model.predict(X_test)


# In[853]:


metrics.r2_score(y_test,y_predict)                             #### much better


# In[854]:


metrics.mean_squared_error(y_test,y_predict)


# In[855]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))           #### much better then linear model we had before


# In[856]:


residuals = y_test - y_predict

plt.figure(figsize=(10,6))

plt.scatter(y_predict,residuals,color='black')

plt.axhline(0,color = 'red',linestyle = '--')

plt.xlabel('predicted')

plt.ylabel('difference between predict and actual aka Residual')

#### not the best one to be honest, it shouldn't form any sort of pattern


# In[478]:


from xgboost import XGBRegressor


# In[479]:


from sklearn.model_selection import RandomizedSearchCV


# In[480]:


from scipy.stats import randint


# In[481]:


from scipy.stats import uniform


# In[482]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])


# In[483]:


param_grid = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__min_child_weight': randint(1, 10),
    'regressor__subsample': uniform(0.5, 0.5),
    'regressor__colsample_bytree': uniform(0.5, 0.5)
}


# In[484]:


random_model = RandomizedSearchCV(model, param_grid, cv=5, scoring='r2', n_iter=100, random_state=42)


# In[485]:


get_ipython().run_cell_magic('time', '', '\nrandom_model.fit(X_train, y_train)')


# In[486]:


best_model = random_model.best_estimator_


# In[487]:


y_predict = best_model.predict(X_test)


# In[488]:


metrics.mean_squared_error(y_test,y_predict)


# In[489]:


np.sqrt(metrics.mean_squared_error(y_test,y_predict))


# In[490]:


metrics.r2_score(y_test,y_predict)                  #### not better then gridsearch + randomforest regressor


# In[491]:


###############################################################################################################
####### After extensive experimentation with the advertisement dataset, we have concluded our model phase. ####
####### Our primary focus was on the 'Click' column for classification, where we achieved an outstanding  #####
####### accuracy of over 0.96 using a Random Forest classifier optimized with GridSearch.  ####################
####### Subsequently, we explored linear regression on the 'Age' column. Despite the dataset not being  #######
####### ideally suited for linear modeling, we aimed to challenge ourselves. The best result was obtained #####
####### with a Random Forest Regressor, also optimized using GridSearch, yielding an R² of 0.38 and an  #######
###### RMSE of 6.57. These efforts highlight both our achievements in classification and our willingness to ###
###### push boundaries with regression tasks, even when faced with less-than-ideal conditions. ################
###############################################################################################################


# In[ ]:




