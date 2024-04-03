import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,log_loss
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer,LancasterStemmer,WordNetLemmatizer
from wordcloud import WordCloud
import pickle
import nltk
nltk.download("punkt")
nltk.download("stopwords")

data=pd.read_csv("fakenews.csv")
fv=data.iloc[:,0]
cv=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=1,stratify=cv)

with st.sidebar:
    radio_button=st.radio('FakeNews Detector ',['Prediction of Text','Problem statement','Simple EDA','Preprocessing','EDA','Model selection'])


if(radio_button=='Problem statement'):
    st.subheader('Problem Statement: The challenge is to determine if the story being reported is genuine or fraud')
    st.write('The spread of false information has become a serious issue with the rapid growth of social media and digital news outlets. Rapid spread of fake news can cause false information, disturbances in society, and a decline in public confidence in the media. Preventing the negative impacts of fake news and preserving the credibility of information sources depend on the ability to identify it.The goal of this research is to create a machine learning model that, using information from various relevant elements and the content of news stories, accurately recognises them as fake or real.')
    st.write('The dataset is collected from Kaggle website.https://www.kaggle.com/datasets/iamrahulthorat/fakenews-csv?resource=download')
    st.write('The dataset is made up of a number of news stories that have been classified as false (1) or true (0).There are 4986 distinct variables in the "fakenews.csv" dataset that have been classified as fake or true news. Every entry in the collection probably corresponds to a news story or fact item, coupled with a label designating whether it is true or untrue. Of the 4986 news articles, 2014 are classified as fake news, and 2972 as real news.')
    st.write('Based on the problem statement, I have implemented Supervised Machine Learning techniques in Classification.')
    st.write('KNN(Bag of words, Binary Bag of words, TFIDF vectorizer)')
    st.write('Bernoulli Naive bayes using Binary Bag of  words')
    st.write('Multinomial Naive Bayes(Bag of words and TFIDF)')


if(radio_button=='Simple EDA'):
    def eda3(data,column):
        lower=' '.join(data[column]).islower()
        html=data[column].apply(lambda x: True if re.search('<.*?>',x) else False).sum()
        urls=data[column].apply(lambda x: True if re.search('http[s]?://.+?\S+',x) else False).sum()
        hasht=data[column].apply(lambda x: True if re.search('#\S+',x) else False).sum()
        mentions=data[column].apply(lambda x: True if re.search('@\S+',x) else False).sum()
        un_c=data[column].apply(lambda x: True if re.search("[]\.\*'\-#@$%^?~`!&,(0-9)]",x) else False).sum()
        emojiss=data[column].apply(lambda x: True if emoji.emoji_count(x) else False).sum()
        if(lower==False):
            st.write('your data contains lower and upper case')
        if(html>0):
            st.write("Your data contains html tags")
        if(urls>0):
            st.write("Your data contains urls")
        if(hasht>0):
            st.write("Your data contains hashtags")
        if(mentions>0):
            st.write("Your data contains mentions")
        if(un_c):
            st.write("Your data contains unwanted chars")
        if(emojiss):
            st.write("Your data contains emojis")

    eda3(data,'text')


if(radio_button=='Preprocessing'):
    st.image('https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/613749410d056eb67ec4b11f_model-building.png')
    st.write('The dataset "fakenews.csv" contained mixed case, HTML tags, URLs, hashtags, mentions, unwanted characters, and emojis. To standardize the text for analysis, preprocessing steps included lowercasing, HTML tag, URL, mention, and hashtag removal, elimination of unwanted characters, and conversion of emojis to text. These steps ensured data consistency and prepared the text for analysis tasks like fake news detection and natural language processing.')
    st.write('The dataset underwent additional preprocessing steps where stop words were removed, and each word was lemmatized to its root form. These enhancements further refined the text data, improving the quality of analysis tasks.')

def basic_pp(x,emoj="F"):
    if(emoj=="T"):
        x=emoji.demojize(x)
    x=x.lower()
    x=re.sub('<.*?>',' ',x)
    x=re.sub('http[s]?://.+?\S+',' ',x)
    x=re.sub('#\S+',' ',x)
    x=re.sub('@\S+',' ',x)
    x=re.sub("[]\.\*'’‘_—,:{}\-#@$%^?~`!&(0-9)]",' ',x)
    
    return x

x_train_p=x_train.apply(basic_pp,args=("T"))
x_test_p=x_test.apply(basic_pp,args=('T'))

## Stop words removal

stp=stopwords.words('english')
stp.remove('not')



def stop_words(x):
    sent=[]
    for word in word_tokenize(x):
        if word in stp:
            pass
        else:
            sent.append(word)
    return ' '.join(sent)

x_train_p=x_train_p.apply(stop_words)
x_test_p=x_test_p.apply(stop_words)

def stem(x):
    sent=[]
    ls=LancasterStemmer()
    for word in word_tokenize(x):
        sent.append(ls.stem(word))
    return " ".join(sent)

x_train_p=x_train_p.apply(stem)
x_test_p=x_test_p.apply(stem)


if(radio_button == 'EDA'):
    data1 = pd.DataFrame(x_train_p)
    data1['label'] = y_train
    data2 = data1.loc[data1['label'] == 1, 'text']
    wc = WordCloud(background_color='black', width=1600, height=800).generate(' '.join(data2))
    wc_image_path = 'wordcloud.png'
    wc.to_file(wc_image_path)
    st.image(wc_image_path, caption='Word Cloud for Fake news')

    data2 = data1.loc[data1['label'] == 0, 'text']
    wc = WordCloud(background_color='black', width=1600, height=800).generate(' '.join(data2))
    wc_image_path = 'wordcloud.png'
    wc.to_file(wc_image_path)
    st.image(wc_image_path, caption='Word Cloud for Real News')


if(radio_button=='Model selection'):
    st.write('I used supervised machine learning methods, such as Naive Bayes and KNN, to categorise news items as authentic or fraudulent. I developed six models using these approaches, each with a unique vectorizer. This method made it possible to thoroughly investigate feature representation strategies and how they affect classification performance.')
    ## KNN with Bag of words
    st.subheader('KNN with Bag of words')
    st.write('Following preprocessing, I utilized the Bag of Words method to convert the text data into numerical vectors.')
    st.write('I utilized the Stratified K-Fold technique with 5 splits and visualized the training F1 score against the cross-validation F1 score for different values of K. Remarkably, both F1 scores reached their highest point at K=1.')
    st.write('The selected final model is the 1-Nearest Neighbors (1NN) classifier, which attained a Generalized F1 score of 0.57.')
    
    ## KNN with BBOW
    st.subheader('KNN with Binary Bag of words')
    st.write('Subsequently, I utilized the Binary Bag of Words technique to convert the preprocessed text data into numerical vectors.')
    st.write('I utilized the Stratified K-Fold technique with 5 splits and plotted the training F1 score against the cross-validation F1 score for different values of K. Interestingly, both F1 scores reached their peak when K was equal to 1.')
    st.write('The chosen final model is the 1-Nearest Neighbors (1NN) classifier, which achieved a Generalized F1 score of 0.513.')

    ## KNN with TFIDF
    st.subheader('KNN with TFIDF')
    st.write('Following that, I utilized the TF-IDF vectorizer to convert the text data into numerical vectors.')
    st.write('Using the Stratified K-Fold technique with 5 splits, I visualized the training F1 score against the cross-validation F1 score for various values of K. Remarkably, both F1 scores reached their highest point at K=1.')
    st.write('The finalized model remains the 1-Nearest Neighbors (1NN) classifier, achieving a Generalized F1 score of 0.66.')

    ## Bernoulli Naive Bayes
    st.subheader('Bernoulli Naive Bayes')
    st.write('I switched to Bernoulli Naive Bayes as the preferred algorithm, employing the Binary Bag of Words vectorizer.')
    st.write('I utilized the cross-validation score method with 5 folds to explore different values of alpha and plotted the corresponding cross-validation F1 scores. The optimal alpha value, yielding the highest F1 score, was determined to be 1.')
    st.write('In the end, the finalized model, configured with alpha=1, attained a generalized F1 score of 0.56 on the test dataset.')

    ## Multinomial Naive Bayes using Bag of words
    st.subheader('Multinomial Naive Bayes with Bag of words')
    st.write('I switched to using Multinomial Naive Bayes as the selected algorithm, utilizing the Bag of Words vectorizer.')
    st.write('I examined various alpha values and their corresponding cross-validation F1 scores using the 5-fold cross-validation scoring method. The optimal alpha, which resulted in the highest F1 score, was identified as 1.')
    st.write('In the end, the finalized model, configured with alpha=1, attained a generalized F1 score of 0.65 on the test dataset.')

    ## Multinomial Naive Bayes with TFIDF Vectorizer
    st.subheader('Multinomial using TFIDF Vectorizer')
    st.write('I transitioned to using the TF-IDF vectorizer for feature extraction.')
    st.write('Employing the cross-validation score method with 5 folds, I explored various values of alpha and plotted the cross-validation F1 scores. The optimal alpha value, resulting in the highest F1 score, was identified as 1.')
    st.write('Ultimately, the final model, configured with alpha=1, achieved a generalized F1 score of 0.39 on the test data.')

    st.subheader("Selecting the best model")
    st.write("Out of the six models constructed employing different algorithms and vectorizers, the K-Nearest Neighbors (KNN) model with the TF-IDF vectorizer emerges as the most effective in distinguishing between fake and real news articles.")
model=pickle.load(open('KNNTFIDF.pkl','rb'))
tfidf=pickle.load(open('TFIDF.pkl','rb'))
if(radio_button=="Prediction of Text"):
    text_button= st.text_input("Enter the Text")
    predict_button= st.button("predict")
    if(text_button and predict_button):
        def predict_news(x,tfidf,model):
            preprocessed_review=basic_pp(x,emoj='T')
            preprocessed_review=stop_words(preprocessed_review)
            preprocessed_review=stem(preprocessed_review)
            preprocessed_review=[preprocessed_review]
            preprocessed_review=tfidf.transform(preprocessed_review)
            prediction=model.predict(preprocessed_review)[0]
            return prediction
        prediction=predict_news('Its official: Facebook will start charging user fees.',tfidf,model)

        if(prediction==1):
            st.write('The entered news is fake')
        else:
            st.write('The entered news is real')
