# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:29:02 2017

@author: svech
"""



import pandas as pd
import re, string
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import networkx as nx
import re, string
import nltk
nltk.download()
from collections import Counter
#from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
from subprocess import check_output
import warnings
from textblob import TextBlob
data=pd.read_csv("C:/Users/sbtrader/downloads/data_elonmusk.csv",encoding = "ISO-8859-1")
data_musk_frame=pd.DataFrame(data)
data_musk_frame['Time'] = pd.to_datetime(data_musk_frame['Time'])
data_musk_frame['Time'].hist(label="Frequency",alpha=0.7 ,color="blue")
plt.legend()
plt.title("Tweet Activty Over The Years")
plt.show()

time=data_musk_frame['Time'] 
time2=pd.to_datetime(time,format='%y')
time3=pd.DataFrame(time2)
time3.year


#cleaning data#
data_musk_frame
data_musk_frame_1 = data_musk_frame.drop('row ID',axis='columns')
data_musk_frame_1
#creating a training and test set 
from sklearn.cross_validation import train_test_split
train, test = train_test_split(data_musk_frame_1, test_size=0.2)
tweet = train['Tweet'].astype(str).tolist()
tweet_test = test['Tweet'].astype(str).tolist()
# Building a corpus and pre-processing
corpus=[]
a=[]
for i in range(len(data['Tweet'])):
        a=data['Tweet'][i]
        corpus.append(a)
corpus = [i.replace('@', ' ') for i in corpus]
list1 = ['RT','rt']
from string import punctuation
stoplist = stopwords.words('english') + list(punctuation) + list1
texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]
#Lemmatization
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma_list_of_words = []
for i in range(0, len(texts)):
     l1 = texts[i]
     l2 = ' '.join([lemmatizer.lemmatize(word) for word in l1])
     lemma_list_of_words.append(l2)
lemma_list_of_words
#Sentiment Analysis using textblob#
clean_lemma = pd.DataFrame(lemma_list_of_words)
clean_lemma['polarity']=clean_lemma[0].apply(lambda x: TextBlob(x).polarity)
#Basic Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
for sentence in lemma_list_of_words:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence,str(vs)))
    
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
analyzer = SentimentIntensityAnalyzer()
for sentence in lemma_list_of_words:
    
    ss = analyzer.polarity_scores(sentence)
    for k in sorted(ss):
       y= print('{0}: {1}, '.format(k, ss[k]), )
       print (y)
       
compound = [analyzer.polarity_scores(s)['compound'] for s in lemma_list_of_words]
pos = [analyzer.polarity_scores(s)['pos'] for s in lemma_list_of_words]
neg = [analyzer.polarity_scores(s)['neg'] for s in lemma_list_of_words]
neu = [analyzer.polarity_scores(s)['neu'] for s in lemma_list_of_words]  

pos_pd=pd.DataFrame({'pos':pos})
neg_pd=pd.DataFrame({'neg':neg})
neu_pd=pd.DataFrame({'neu':neu})
comp_pd=pd.DataFrame({'comp':compound})
lemma_pd=pd.DataFrame({'tweet':lemma_list_of_words})
time=data_musk_frame['Time']
time_pd=pd.DataFrame({'time':time})
concat_final=pd.concat([pos_pd,neg_pd,neu_pd,comp_pd,lemma_pd,time_pd], axis=1)

#exporting files to excel
writer = pd.ExcelWriter('output.xlsx')
concat_final.to_excel(writer,'Sheet1')
writer.save()
#plotting sentiments 
import matplotlib.pyplot as plt
plt.scatter(concat_final['pos'], concat_final['time'])
plt.show()

for key, value in y:
    temp = [key,value]
    dictlist.append(temp)        
    print(temp)    
from collections import defaultdict
d = defaultdict(list)
for k, v in ss.items():
    d[v].append(k)
    print dict(d)
import nltk.data
from nltk import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    
sentences = tokenizer.tokenize(lemmastr)   
for sentence in sentences:
        print(sentence)
        ss = analyzer.polarity_scores(sentence)
        for k in sorted(ss):
                print('{0}: {1}, '.format(k, ss[k]), end='')
        print()


pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for word in lemma_list_of_words:
    if (analyzer.polarity_scores(word)['compound']) >= 0.7:
        pos_word_list.append(word)
    elif (analyzer.polarity_scores(word)['compound']) <= -0.3:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)                

print('Positive :',pos_word_list)        
print('Neutral :',neu_word_list)    
print('Negative :',neg_word_list)



 
message = lemmastr



lemmastr=str(lemma_list_of_words)
lemmalist=lemmastr.split()   #dictionary for all intents and purposes, obatined after lemmatization
#Topic Modelling    
import gensim

import logging
import tempfile

from gensim import corpora

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))  
#converting dataframe to string
#lemmalist_1=lemmastr.to_string() 
#Tokenization#
nltk.download('averaged_perceptron_tagger')
lemmadf=pd.DataFrame(lemmalist)
lemmadf_str=lemmadf.to_string()
from nltk import word_tokenize
tokens= nltk.word_tokenize(lemmadf_str)
tokens
len(tokens)
 #ngrams#
from nltk.util import ngrams
from collections import Counter
bigrams = ngrams(tokens,2)
trigrams = ngrams(tokens,3)
fourgrams = ngrams(tokens,4)
fivegrams = ngrams(tokens,5)
print(Counter(bigrams))
#STEP2 : TRANSFORMATIONS : Calculating document frequencies : Simple Tlfdf model#
from gensim import corpora, models, similarities
#Initializing a model#
dictionary = corpora.Dictionary(texts)
import os
dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict')) 
print(dictionary)
#convert tokenized documents to vectors#
corpus_l = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'elon.mm'), corpus_l)
print(corpus_l)

#STEP2 : TRANSFORMATIONS : Calculating document frequencies : Simple Tfidf model#
from gensim import corpora, models, similarities
#Initializing a model#
tfidf = models.TfidfModel(corpus_l)
corpus_tfidf = tfidf[corpus_l]
for doc in corpus_tfidf:
            print(doc)
#Serializing transformations#
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(5)


 
 #LDA model#
lda = models.LdaModel(corpus_l, id2word=dictionary, num_topics=5)
corpus_lda=lda[corpus_l]

for doc2 in corpus_lda:
         print (doc2)
#Metric Evaluation : Perplexity
#computes perplexity of the unigram model on a testset  
def perplexity(tweet_test, corpus_lsi):
    
    perplexity = 1
    N = 0
    for word in tweet_test:
        N += 1
        perplexity = perplexity * (1/corpus_lsi[word])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

   
  
perplexity(tweet_test,corpus_lsi)
    




