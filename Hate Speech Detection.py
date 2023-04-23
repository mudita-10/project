#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

data = pd.read_csv("sample.csv")
data.head()


# In[12]:


data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data.head()


# In[13]:


data = data[["tweet", "labels"]]
data.head()


# In[14]:


nltk.download('stopwords')
stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
data.head()


# In[15]:


x = np.array(data["tweet"])
y = np.array(data["labels"])
count_vector = CountVectorizer()
X = count_vector.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[16]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)
user_input = input("Enter a text to check:")
data = count_vector.transform([user_input]).toarray()
output = classifier.predict(data)
print(output)


# In[ ]:




