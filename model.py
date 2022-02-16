#!/usr/bin/env python
# coding: utf-8

# In[2]:


from unittest import result
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from numpy import *
result=[]
def mod(username):
    # In[3]:


    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    pd.set_option('max_colwidth', 5000)


    # In[4]:


    df2=pd.read_csv(r'clean_prod.csv')
    #df2.head()


    df2['Clean_text']=df2['Clean_text'].values.astype('U')

    # ## Feature Extraction

    # In[45]:


    message=[message for message in df2['Clean_text']]

    #Write your code here to initialise the TfidfVectorizer 

    #Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer=TfidfVectorizer(max_df=0.95,min_df=2)

    tfidf_model=vectorizer.fit_transform(message)

    tfidf=pd.DataFrame(tfidf_model.toarray() , columns=vectorizer.get_feature_names())

    tfidf


    # In[46]:


    df2['user_sentiment'].unique()


    # In[47]:


    #df2['user_sentiment']=df2['user_sentiment'].map({'Positive':'1', 'Negative':'0'})


    # In[48]:


    #df2.info()


    # In[49]:


    #df2['user_sentiment']=df2['user_sentiment'].astype('int')


    # In[50]:


    #df2.info()


    # #### Write your code here to transform the word vector to tf-idf

    # In[51]:


    corpus=[]

    for i in df2['Clean_text']:
        
        corpus.append(i)


    # In[52]:




    from sklearn.feature_extraction.text import TfidfVectorizer

    vect=TfidfVectorizer()

    X=vect.fit_transform(corpus).toarray()
    y=df2.iloc[:,1]


    # In[53]:


    #print(X.shape)
    #print(y.shape)


    # ### Class Imbalance

    # In[54]:


    df2['user_sentiment'].value_counts(normalize=True)*100


    # In[55]:


    # class imbalance is vey high, so we need to treat it using SMOTE


    # In[56]:


    from imblearn.over_sampling import SMOTE

    sm=SMOTE(random_state=42)

    X_sm , y_sm=sm.fit_resample(X,y)


    # In[57]:


    y_sm.value_counts(normalize=True)*100


    # In[ ]:





    # In[58]:


    # Perform train test split

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,test_size=0.3,random_state=42)


    # ### <font color=Blue> 1. Logistic Regression

    # In[59]:


    #X_train.shape


    # In[60]:


    #X_test.shape


    # In[61]:


    #y_train.shape


    # In[62]:


    #y_test.shape


    # In[63]:


    from sklearn.linear_model import LogisticRegression

    logit=LogisticRegression()

    logit.fit(X_train,y_train)


    # In[64]:


    y_train_pred=logit.predict(X_train)
    #y_train_pred.shape


    # In[65]:


    from sklearn.metrics import accuracy_score,roc_auc_score

    #print('Train accuracy =' , accuracy_score(y_train,y_train_pred)*100)


    from sklearn.metrics import accuracy_score

    y_test_pred=logit.predict(X_test)


    #print('Test accuracy =' , accuracy_score(y_test,y_test_pred)*100)


    # In[66]:


    from sklearn.metrics import roc_curve , roc_auc_score , confusion_matrix , accuracy_score, classification_report


    # In[67]:


    df2.head(2)


    # In[68]:


    df2['user_sentiment'].shape


    # In[69]:


    #confusion=confusion_matrix(y_train ,y_train_pred)


    # In[70]:


    #TN=confusion[0,0]
    #FP=confusion[0,1]
    #FN=confusion[1,0]
    #TP=confusion[1,1]

    #total=sum(confusion)
        
    #Accuracy_train=  round((TP+TN)/(TP+FP+TN+FN) ,2)*100
        
    #Sensitivity_train=round(float(TP)/(TP+FN) ,2)*100 
        
    #Specificity_train=round(float(TN)/(TN+FP) ,2)*100 

    #Precision_Train=round(float(TP)/(TP+FP) ,2)*100


    #print('Accuracy Train = ',Accuracy_train)
    #print('Sensitivity Train = ',Sensitivity_train)
    #print('Specificity Train= ',Specificity_train)
    #print('Precision Train =',Precision_Train)


    # In[71]:


    # Train Classification report 

    #print(classification_report(y_train, y_train_pred))


    # In[72]:


    # Test

    #confusion1=confusion_matrix(y_test ,y_test_pred)

    #TN=confusion1[0,0]
    #FP=confusion1[0,1]
    #FN=confusion1[1,0]
    #TP=confusion1[1,1]

    #total=sum(confusion1)
        
    #Accuracy_test=  round((TP+TN)/(TP+FP+TN+FN) ,2)*100
        
    #Sensitivity_test=round(float(TP)/(TP+FN) ,2)*100 
        
    #Specificity_test=round(float(TN)/(TN+FP) ,2)*100 

    #Precision_Test=round(float(TP)/(TP+FP) ,2)*100


    #print('Accuracy Test = ',Accuracy_test)
    #print('Sensitivity Test = ',Sensitivity_test)
    #print('Specificity Test= ',Specificity_test)
    #print('Precision Test =',Precision_Test)


    # In[73]:


    # Test Classification Report 

    #print(classification_report(y_test, y_test_pred))


    # In[74]:


    # The accurcay/precison/recall  of train and test are comparable so model performs good 




    # ## Recommendation System

    # In[75]:


    df=pd.read_csv(r'sample30.csv')
    #df.head()


    # In[76]:


    rev=df[['reviews_username','name','reviews_rating']]
    #rev.head()


    # In[77]:


    rev.isnull().sum()/rev.shape[0]*100


    # In[78]:


    # Drop data where username is nit present as it wont help in building recommendation system 

    rev=rev[~rev['reviews_username'].isna()]
    #rev.shape


    # In[79]:


    rev.isnull().sum()


    # In[80]:


    #rev.head()


    # ## Dropping Duplicate Values 

    # In[81]:


    rev.drop_duplicates(keep='last',inplace=True)


    # In[82]:


    #rev.shape


    # In[83]:


    rev1=rev.groupby(['reviews_username', 'name']).agg({'reviews_rating':'max'})


    # In[84]:


    rev1.reset_index(inplace=True)


    # In[85]:


    #rev1.head()


    # In[86]:


    #rev1.shape


    # ### Test and Train split of the dataset.

    # In[87]:



    from sklearn.model_selection import train_test_split
    train, test = train_test_split(rev1, test_size=0.30, random_state=31)


    # In[88]:


    #print(train.shape)
    #print(test.shape)


    # In[89]:


    #rev1.columns


    # In[90]:


    # Pivot the train ratings' dataset into matrix format in which columns are movies and the rows are user IDs.
    rev1_train_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(0)

    #rev1_train_pivot.head(3)


    # In[91]:


    #rev1_train_pivot.shape


    # ### Creating Dummy Train

    # In[92]:


    # Copy the train dataset into dummy_train
    dummy_train = train.copy()


    # In[93]:


    # The movies not rated by user is marked as 1 for prediction. 
    dummy_train['rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(1)


    # In[94]:


    #dummy_train.head()


    # In[95]:


    #dummy_train.shape


    # ### <font color=Blue> User Based Recommendation System

    # In[96]:


    from sklearn.metrics.pairwise import pairwise_distances

    # create cosine similarity matrix using pairwise distance function 

    user_correlation=1-pairwise_distances(rev1_train_pivot , metric='cosine')

    user_correlation[np.isnan(user_correlation)]=0


    # In[97]:


    #print(user_correlation)


    # In[98]:


    #user_correlation.shape


    # ## Using adjusted cosine similarity

    # In[99]:


    # dont replace nan values 

    rev1_train_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    )

    #rev1_train_pivot.head(3)


    # In[100]:


    #rev1_train_pivot.shape


    # ### Normalising the rating of the movie for each user around 0 mean

    # In[101]:


    mean = np.nanmean(rev1_train_pivot, axis=1)
    df_subtracted = (rev1_train_pivot.T-mean).T


    # In[102]:


    #df_subtracted.head()


    # In[103]:


    #df_subtracted.shape


    # ### Finding Cosine similarity

    # In[104]:


    from sklearn.metrics.pairwise import pairwise_distances


    # Creating the User Similarity Matrix using pairwise_distance function.

    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    #print(user_correlation)


    # In[105]:


    #user_correlation.shape


    # ## <font color=Red>Prediction - User User

    # In[ ]:





    # __`Doing the prediction for the users which are positively related with other users, 
    # and not the users which are negatively related as we are interested in the users which are more similar 
    # to the current users. So, ignoring the correlation for values less than 0.`__

    # In[ ]:





    # In[106]:


    user_correlation[user_correlation<0]=0
    #user_correlation


    # In[107]:


    #user_correlation.shape


    # In[ ]:





    # __`Rating predicted by the user (for products rated as well as not rated) is the weighted sum 
    # of correlation with the product rating (as present in the rating dataset).`__

    # In[108]:


    user_predicted_ratings = np.dot(user_correlation, rev1_train_pivot.fillna(0))
    #user_predicted_ratings


    # In[109]:


    #user_predicted_ratings.shape


    # __`Since we are interested only in the products not rated by the user, 
    # we will ignore the products rated by the user by making it zero.`__

    # In[110]:


    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    #user_final_rating.head()


    # In[111]:


    #user_final_rating.shape


    # ### Finding the top 20 recommendation for the *user*

    # In[112]:


    # Take the user ID as input.
    user_input = username
    #print(user_input)
    #00dog3


    # In[113]:


    #user_final_rating.head(2)


    # In[114]:


    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    #print(d)


    # In[ ]:





    # ## <font color=Red>Evaluation - User User 

    # In[ ]:





    # __`Evaluation will we same as  seen above for the prediction. The only difference being, we will evaluate
    # for the product already rated by the user insead of predicting it for the product not rated by the user.`__

    # In[115]:


    # Find out the common users of test and train dataset.
    common = test[test['reviews_username'].isin(train.reviews_username)]
    #common.shape


    # In[116]:


    #common.head()


    # In[117]:


    #common.columns


    # In[118]:


    # convert into the user-movie matrix.
    common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name',
                                                values='reviews_rating')


    # In[119]:


    # Convert the user_correlation matrix into dataframe.
    user_correlation_df = pd.DataFrame(user_correlation)


    # In[120]:


    #df_subtracted.head(1)


    # In[121]:


    user_correlation_df['reviews_username'] = df_subtracted.index
    user_correlation_df.set_index('reviews_username',inplace=True)
    #user_correlation_df.head()


    # In[122]:


    #common.head(1)


    # In[123]:


    # Lets get only user that are common in both train and test and filter them out from our user_correlation matrix


    # In[124]:


    list_name = common.reviews_username.tolist()

    user_correlation_df.columns = df_subtracted.index.tolist()


    user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


    # In[125]:


    #user_correlation_df_1.shape


    # In[126]:


    #user_correlation_df_1


    # In[127]:


    user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


    # In[128]:


    #user_correlation_df_2


    # In[129]:


    # user_correlation_df_3 will contain only common users in both rows and columns 


    # In[130]:


    user_correlation_df_3 = user_correlation_df_2.T


    # In[131]:


    #user_correlation_df_3.shape


    # In[132]:


    #user_correlation_df_3.head()


    # In[133]:


    #common_user_based_matrix.shape


    # In[134]:


    # get ratings for the common users 


    # In[135]:


    user_correlation_df_3[user_correlation_df_3<0]=0

    common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
    #common_user_predicted_ratings


    # __` For Evalutaion we are only concerned about products already rated by the user`__

    # In[136]:


    #common.columns


    # In[137]:


    dummy_test = common.copy()

    dummy_test['rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

    dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)


    # In[138]:


    #dummy_test.shape


    # In[139]:


    # Our model will predict ratings for only those products which user has rated and we will check further how much the 
    # predicted rating is close to our actual rating 


    # In[140]:


    common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


    # In[141]:


    #common_user_predicted_ratings.head(2)


    # ### Calculating the RMSE for only the products rated by user. For RMSE, normalising the rating to (1,5) range.

    # In[142]:


    

    p = common_user_predicted_ratings.copy() 
    p = p[p>0]

    scaler = MinMaxScaler(feature_range=(1, 5))
    print(scaler.fit(p))
    y = (scaler.transform(p))

    #print(y)

    # y will containthe redicted ratings


    # In[143]:


    # These are the actual ratings that we have and we storeit in common_

    common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


    # In[144]:


    # Finding total non-NaN value

    total_non_nan = np.count_nonzero(~np.isnan(y))


    # In[145]:


    rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
    #print(rmse)


    # __` This seems as decent RMSE value`__



    #d


    # In[148]:


    # Convert the user based recommended movie into a dictionary 


    recomm_movies=pd.DataFrame(d)
    #recomm_movies.head(20)


    recomm_movies.reset_index(inplace=True)

    recomm_movies['name']

    #recomm_movies.head(20)


    # ## Use Sentiment analysis model to Recommend top 5 movies from top 20 movies

    # In[149]:


    #df.columns


    # In[150]:


    #recomm_movies.head(20)


    # In[151]:


    #df2.head()


    # In[152]:


    # Make our trained model predict the sentiment for all values in dataset


    # In[153]:


    corpus1=[]

    for i in df2['Clean_text']:
        
        corpus1.append(i)


    # In[154]:


    from sklearn.feature_extraction.text import TfidfVectorizer

    vect=TfidfVectorizer()

    X=vect.fit_transform(corpus1).toarray()

    y_pred=logit.predict(X)


    # In[155]:


    #y_pred.shape


    # In[156]:


    df2['sent_pred']=y_pred


    # In[157]:


    #df2.head()


    # In[ ]:





    # In[158]:


    df=df[~df['user_sentiment'].isna()]


    # In[159]:


    #df.shape


    # In[160]:


    df2['prod_name']=df['name']


    # In[161]:


    #df2.head()


    # In[162]:


    df3=df2.groupby(['prod_name','sent_pred']).count()


    # In[163]:


    df3.reset_index(inplace=True)


    # In[164]:


    df3=df3[['prod_name','sent_pred','user_sentiment']]
    #df3.head()


    # In[165]:


    l=[]

    for i in recomm_movies['name']:
        
        l.append(i)
        
    


    # In[166]:


    df4=df3.groupby('prod_name').sum()
    df4.reset_index(inplace=True)
    df4=df4[['prod_name','user_sentiment']]
    #df4.head()


    # In[167]:


    df3=df3.merge(df4,how='inner',on='prod_name')


    # In[168]:


    df3['percent_val']=(df3['user_sentiment_x']/df3['user_sentiment_y'])*100


    # In[169]:


    #df3.head()


    # In[170]:


    df5=df3[df3['prod_name'].isin(l)]


    # In[171]:


    # These are the top five recommended movies for this particular user 


    # In[172]:


    df6=df5.groupby('prod_name').max()
    df6.reset_index(inplace=True)


    # In[173]:


    df6[df6['sent_pred']==1].sort_values('percent_val',ascending=False)[0:5]['prod_name']

    for i in df6[df6['sent_pred']==1].sort_values('percent_val',ascending=False)[0:5]['prod_name']:
        print(i)
        result.append(i)
    return result
     
#mode = mod('00dog3')        
#print('return : ',mode[0],mode[1],mode[3])  
