#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import re 
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


        
def clean_tag(tag_list):
    processed_tags =[]
    for t in tag_list:
        t = re.sub('[^A-Za-z0-9 ]+'," " ,t)
        t = re.sub(r'[0-9]+', ' ', t)
        t = re.sub(' +', ' ', t)
        t = t.lower()
        t = re.sub('\s+', ' ', t)
        t = " ".join(t.split())
        
        processed_tags.append(t)
    #good_tags = remove_stopwords(pd.Series(processed_tags)).tolist()
    return processed_tags

def expan_tags(tag, model, n):
    similar_words = ""
    try:
        similar_words = model.wv.most_similar(positive=tag, topn=n)
        ##print ('similar_words', similar_words)
    except:
        return similar_words
    return ' '.join(str(e[0]) for e in [*similar_words] )
    
# In[1]:

#nlp = spacy.load('en', disable=['parser', 'ner'])
def find_topic(df_topic_keywords, text, vectorizer, best_lda_model):
   # global sent_to_words
    #global lemmatization
# Step 1: Clean with simple_preprocess
   # mytext_2 = list(sent_to_words(text))
# Step 2: Lemmatize
    #mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Step 1: Vectorize transform
    mytext_4 = vectorizer.transform(text)
    #tfidf_transformer.transform(vectorizer.transform([doc]))
# Step 4: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), 1:14].values.tolist()
    
    # Step 5: Infer Topic
    infer_topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), -1]
    
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topics]
    return infer_topic, topic, topic_probability_scores

def get_model():
    # Load Google's pre-trained Word2Vec model.
    model =gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
    words = list(model.wv.vocab)
    return model,words

def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


def print_top_words(model, feature_names, n_top_words):
    topics_words = []
    
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic %d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics_words.append(message)
        print(message)
    print()
    return topics_words


def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
    return frozenset(stop_set)


# In[2]:


def data_clean(df, columns):
    for col in columns:
        #df.fillna('')
        df[col] = df[col].str.replace("http\S+|www.\S+", " ",case=False)
        df[col] = df[col].str.replace(r"http\S+", " ",case=False)
        df[col] = df[col].str.replace(r'\b\w{1,3}\b',' ',case=False)
        df[col] = df[col].str.replace(r"[^a-zA-Z0-9]+", ' ',case=False)
        df[col] = df[col].str.replace(r'[0-9]+', ' ',case=False)
        df[col] = df[col].str.strip().replace(' +', ' ')
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip().replace('\s+', ' ', regex=True)
        #df[col] = re.sub(r'\s{2,}', ' ', df[col])
    return df


# In[3]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


# In[4]:


def extract_topn_from_vector(feature_names, sorted_items, topn=20):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# In[5]:


import nltk
from nltk.stem import WordNetLemmatizer
def do_lemmaization(text_data):
    doc_clean = []    
    wordnet_lemmatizer = WordNetLemmatizer()
    for text in text_data:
        filtered_doc = ""
        tokenization = nltk.word_tokenize(text)
        for w in tokenization: 
                filtered_doc += wordnet_lemmatizer.lemmatize(w)+" "
        doc_clean.append(filtered_doc.strip())
    return doc_clean


# In[6]:



def remove_stopwords(text_data, stoplist):
    doc_clean = []
    for d in text_data:
        filtered_doc = "" # []
        for w in d.split(" "):
            #print(w)
            if  w.lower() not in stoplist: #type(w) is str and
                filtered_doc += w.lower()+" "
                #filtered_doc.append(w.lower())
       # print(filtered_doc)
        doc_clean.append(filtered_doc.strip())
    return doc_clean

#remove_stopwords(['text'])


# In[7]:


def clean_tag(tag_list):
    processed_tags =[]
    for t in tag_list:
        t = re.sub('[^A-Za-z0-9 ]+'," " ,t)
        t = re.sub(r'[0-9]+', ' ', t)
        t = re.sub(' +', ' ', t)
        t = t.lower()
        t = re.sub('\s+', ' ', t)
        t = " ".join(t.split())
        
        processed_tags.append(t)
    #good_tags = remove_stopwords(pd.Series(processed_tags)).tolist()
    return processed_tags


# In[8]:


## word2vec google model.
import gensim
def get_model():
    # Load Google's pre-trained Word2Vec model.
    model =gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
    words = list(model.wv.vocab)
    return model,words


# In[9]:


def read_concept_tags():
    #file = open('/Users/dewet/Downloads/hdx-work/tag_list.txt', 'r')
    #print(file.read())
    lineList = [line.rstrip('\n') for line in open('tag_list.txt', 'r')]
    #print(lineList)
    #tag_list = file.read()
    return lineList


# In[10]:


from nltk.corpus import wordnet as wn
def syn(word, lch_threshold=2.26):
    for net1 in wn.synsets(word):
        for net2 in wn.all_synsets():
            try:
                lch = net1.lch_similarity(net2)
            except:
                continue
            # The value to compare the LCH to was found empirically.
            # (The value is very application dependent. Experiment!)
            if lch >= lch_threshold:
                yield (net1, net2, lch)


# In[11]:


def normalize(df, feature_names):
    result = df.copy()
    for feature_name in feature_names:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[12]:


def vectorize( doc, model , index2word_set):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = doc.strip().split(" ") 
        word_vecs = []
        for word in words:
            try:
                vec = model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                vec = np.zeros((300, ), dtype='float32')
                word_vecs.append(vec)
        vector = np.mean(word_vecs, axis=0)
        return vector


# In[13]:


def cosine_distance_list( input ,target_list, model, index2word_set, num) :
    cosine_dict ={}
    word_list = []
    tags = []
    doc = vectorize(input, model , index2word_set)
    for item in target_list :
        tag = vectorize(item, model , index2word_set)
        #cos_sim = np.dot(doc, tag)/(np.linalg.norm(doc)*np.linalg.norm(tag))
        #cos_sim =spatial.distance.cosine(doc,tag)
        cos_sim = np.dot(doc, tag)/(np.linalg.norm(doc)*np.linalg.norm(tag))
        #print(item , cos_sim)
        if np.isnan(cos_sim): #float('NaN'):
            cos_sim=0
        cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        #print(type(item))
        word_list.append(item[0]) #, item[1]))
        #print(item[0])
        #tags.append(item[0])
        
    return word_list[0:num]


# In[16]:


def cosine_distance ( input ,target_list, model, index2word_set, num) :
    cosine_dict ={}
    word_list = []
    doc = vectorize(input, model , index2word_set)
    for item in target_list :
        tag = vectorize(item, model , index2word_set)
        #cos_sim = np.dot(doc, tag)/(np.linalg.norm(doc)*np.linalg.norm(tag))
        #cos_sim =spatial.distance.cosine(doc,tag)
        cos_sim = np.dot(doc, tag)/(np.linalg.norm(doc)*np.linalg.norm(tag))
        #print(item , cos_sim)
        if np.isnan(cos_sim):# == NaN:
            cos_sim= 0
        cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]




# In[15]:


########clean evaluation set:

#### map the current tags of the document to the clean set of tags
#for i, row in test_set_final.iterrows():
 #   perfect_tags = []
  #  tag_list = test_set_final.ix[i]['tag_list']
  #  for tag in tag_list:
   #     try:
    #        new_tag = cleanuplist[cleanuplist['Current Tag'] == tag.lower()]['compare'].values[0].lower()
     #       if (new_tag !=''):
      #          perfect_tags.append(new_tag)
       # except:
            #print("Excep" ,tag)
        #    continue
        #print(perfect_tags)
        #test_set_final.at[i,'perfect_tags'] = perfect_tags


# In[ ]:

#### test lookup
#import numpy as np
#from sklearn.neighbors import KDTree
#np.random.seed(0)
#X = np.random.random((5, 2))  # 5 points in 2 dimensions
#tree = KDTree(TagsArray2, leaf_size=3) ## the list of tags
#nearest_dist, nearest_ind = tree.query(np.reshape(doc2doc2vec, (1, -1)), k=5)  # k=2 nearest neighbors where k1 = identity
#print(nearest_dist[:, 1])    # drop id; assumes sorted -> see args!
#for i in np.nditer(nearest_ind):
    #print("i" ,i)
 #   print("tag: ", tag_list[i])


####if i want to do tag expansion using wordnet
from nltk.corpus import wordnet as wn
def tag_expansion_wordnet(df_tags):
    df_tags['tag_expansion'] = ""
    for i, row in df_tags.iterrows():
        tag_scentence = df_tags.ix[i]['clean_tags'].split(' ')
        print(tag_scentence)
        synonyms = []
        for tag in tag_scentence:
            print(tag)
            for syn in wn.synsets(tag): 
            #print("lem names" ,syn.lemma_names())
                for l in syn.lemmas() :
                   # print(l)
                    lemma_name = l.name().lower()
                    if  lemma_name not in synonyms:            
                        synonyms.append(lemma_name) 
            print(synonyms)#syn.definition())
        df_tags.at[i,'tag_expansion'] = ' '.join(str(e) for e in [*synonyms] )
    return df_tags


