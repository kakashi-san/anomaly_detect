#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import regex as re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# import category_encoders as ce


# In[2]:


options = {'vector_size' : 5, 
          'window' : 3,
          'min_count' : 1,
          'epochs' : 100}
punc_set = ['/', '%','\\', '-', ":"]




# In[3]:


def isParameter(word, punc_set):

    letters = set([letter for letter in word])

    return bool(letters.intersection(punc_set))


def splitLogsByDate(path):
    f = open(path)
    log_data = ""
    for line in f:
        log_data = log_data + " " + line
    f.close()
    
    return log_data

def removePunctuations(string):
  
    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, " ")
  
    # Print string without punctuation
    string = string.lower().replace("  ", " ")
    return string

def formatLogData(log_data, form):
    
    formatted =  re.findall(r'{0}.*?(?=\s*{0}|$)'.format(form), log_data, re.DOTALL)

    structured_data = {"Date-time": [],
                     "server"   : [],
                     "ISP"      : [],
                     "Details"  : []}

    for line in formatted:
        listOfWords = line.split(" ")


        structured_data["Date-time"].append(listOfWords[0] + " " + listOfWords[1])
        structured_data["server"]   .append(listOfWords[2])
        structured_data["ISP"]      .append(listOfWords[3])
        structured_data["Details"]  .append(" ".join(listOfWords[4:]))


    return structured_data



d = r'\d{4}-\d?\d-\d?\d (?:2[0-3]|[01]?[0-9]):[0-5]?[0-9]:[0-5]?[0-9]'
log_unstruct = splitLogsByDate(path = 'rest.log')
dic = formatLogData(log_data = log_unstruct, form  = d)
log_dataframe = pd.DataFrame(dic)
log_dataframe


# In[4]:


def formatDateTime(dates):
    
    return (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
    



def getDiffOfTime(time_series):
    diff_ser        = time_series.values[1:] - time_series.values[:-1]
    padded_diff_ser = np.pad(diff_ser, (1,0), 'constant')
    return pd.Series(padded_diff_ser)

log_dataframe["Date-time"] = formatDateTime(dates = pd.to_datetime(log_dataframe["Date-time"]))
time_series     = log_dataframe["Date-time"]
getDiffOfTime(time_series).value_counts()


# In[5]:


log_dataframe['server'].value_counts()


# In[6]:


log_dataframe['ISP'] = log_dataframe['ISP'].str[1:-1]


# In[7]:


# ce_bin = ce.BinaryEncoder()
# ce_bin.fit_transform(log_dataframe['ISP'], verbosity =1)

log_dataframe['ISP'].value_counts()
pd.get_dummies(log_dataframe['ISP'])


# In[8]:


log_dataframe['Details'].apply(removePunctuations)


# In[9]:


sentences =  log_dataframe['Details'].apply(removePunctuations)
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))


# In[10]:


tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]


# In[11]:


vector_size = options['vector_size']
window      = options['window']
min_count   = options['min_count']
epochs      = options['epochs']


# In[12]:


model = Doc2Vec(tagged_data, vector_size = vector_size, window = window, min_count = min_count, epochs = epochs)


# In[13]:


tokenized_sents = sentences.apply(word_tokenize)
vectorized_sents= tokenized_sents.apply(model.infer_vector)

vectorized_sents


# In[34]:


word_embed_vecs = pd.DataFrame(vectorized_sents.tolist())


# In[41]:


pd.concat([log_dataframe['Date-time'], getDiffOfTime(time_series), word_embed_vecs, pd.get_dummies(log_dataframe['ISP'])], axis = 1).to_csv('cons.csv', index = False)


# In[ ]:




