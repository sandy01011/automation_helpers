
# Parts Of Speech Tagging

#Importing libraries
import nltk, re, pprint
#nltk.download('treebank')
#nltk.download('universal_tagset')
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    
    return (count_w_given_tag, count_tag)

# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability

def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))




# Viterbi Heuristic
def Viterbi(words, train_bag):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
        print(key)
    return list(zip(words, state))

## Modification -1: all unknown words are replaced by Noun (Noun being the most common tag)

def Viterbi_modf1(test_words, train_bag = train_tagged_words):
    tagged_seq = Viterbi(test_words)
    V = list(set([pair[0] for pair in train_bag]))
    
    words = [pair[0] for pair in tagged_seq]
    Viterbi_tags = [pair[1] for pair in tagged_seq]
    
    for key, word in enumerate(words):
        if word not in V:
            Viterbi_tags[key] = 'NOUN'
            
    
    return list(zip(words, Viterbi_tags))  

## Viterbi Modification -2: 
## 1. all unknown words with first letter capital/ all letters capitals are tagged as Noun, numbers are tagged as NUM, words ending with '-ous' as ADJ, and rest as Noun

def Viterbi_modf2(test_words, train_bag = train_tagged_words):
    tagged_seq = Viterbi(test_words)
    V = list(set([pair[0] for pair in train_bag]))
    
    words = [pair[0] for pair in tagged_seq]
    Viterbi_tags = [pair[1] for pair in tagged_seq]
    
    for key, word in enumerate(words):
        if word not in V:
            ## word ending with '-ous'
            if word[-3:] == 'ous':
                Viterbi_tags[key] = 'ADJ'
            
            ## if word is number
            elif (word.isdigit() == True or word[:-2].isdigit() == True):
                Viterbi_tags[key] = 'NUM'
                
            ## all letters capitalised
            elif word.upper() == word:
                Viterbi_tags[key] = 'NOUN'
                
            ## first letter is capitalised:
            elif word[0].upper() == word[0]:
                Viterbi_tags[key] = 'NOUN' 
                
            else: 
                Viterbi_tags[key] = 'NOUN'
    
    return list(zip(words, Viterbi_tags)) 

# Viterbi Modification -3: state probability is dependent only on transition probability
def Viterbi_modf3(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            
            if word in V:
                state_probability = transition_p * emission_p              
            else:
                state_probability = transition_p
            
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
        print(key)
    return list(zip(words, state))



    # IOB labels

    # function to create (word, pos_tag, iob_label) tuples for a given dataset
def create_word_pos_label(pos_tagged_data, labels):
    iob_labels = []         # initialize the list of 3-tuples to be returned
    
    for sent in list(zip(pos_tagged_data, labels)):
        pos = sent[0]       
        labels = sent[1]    
        zipped_list = list(zip(pos, labels)) # [(word, pos), label]
        
        # create (word, pos, label) tuples from zipped list
        tuple_3 = [(word_pos_tuple[0], word_pos_tuple[1], id_to_labels[label]) 
                   for word_pos_tuple, label in zipped_list]
        iob_labels.append(tuple_3)
    return iob_labels


# unigram chunker

from nltk import ChunkParserI

class UnigramChunker(ChunkParserI):    
    def __init__(self, train_sents):
        # convert train sents from tree format to tags
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] 
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
        
    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        
        # convert to tree again
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

# bigram tagger

class BigramChunker(ChunkParserI):    
    def __init__(self, train_sents):
        # convert train sents from tree format to tags
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] 
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)
        
    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        
        # convert to tree again
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


# define a function to look up a given word in cities, states, county
def gazetteer_lookup(word):
    return (word in cities, word in states, word in counties)

# extracts features for the word at index i in a sentence 
def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    
    # the first word has both previous word and previous tag undefined
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]

    # gazetteer lookup features (see section below)
    gazetteer = gazetteer_lookup(word)

    return {"pos": pos, "prevpos": prevpos, 'word':word,
           'word_is_city': gazetteer[0],
           'word_is_state': gazetteer[1],
           'word_is_county': gazetteer[2]}

class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            # compute features for each word
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

    
    # CRF
    # import relevant libraries
from itertools import chain
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# pip/conda install sklearn_crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
# extract features from a given sentence
def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    
    # first word
    if i==0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]
    
    # last word
    if i == len(sent)-1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]
    
    # word is in gazetteer
    gazetteer = gazetteer_lookup(word)
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
    
    return {'word':word,
            'pos': pos, 
            'prevword': prevword,
            'prevpos': prevpos,  
            'nextword': nextword, 
            'nextpos': nextpos,
            'word_is_city': gazetteer[0],
            'word_is_state': gazetteer[1],
            'word_is_county': gazetteer[2],
            'word_is_digit': word in 'DIGITDIGITDIGIT',
            'suff_1': suff_1,  
            'suff_2': suff_2,  
            'suff_3': suff_3,  
            'suff_4': suff_4, 
            'pref_1': pref_1,  
            'pref_2': pref_2,  
            'pref_3': pref_3, 
            'pref_4': pref_4 }
# defining a few more functions to extract featrues, labels, words from sentences

def sent2features(sent):
    return [word_features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]  

# extract integer dates etc from query tree
def extract_entities_from_tree(tree):
    query_dict = tree_to_dict(tree)
    entities = {}
    
    for key, val in query_dict.items():
        
        # get airport codes from city names as a list
        if key == "fromloc.city_name" or key == "toloc.city_name":
            entities[key] = city_to_airport_code(val)
            
        # strip the last 's' e.g. tuesdays from day of week
        if key == "depart_date.day_name":
            query_dict[key] = val[:-1] if val.endswith("s") else val
            
            # get year, month, day of the next day_name
            day_num = day2int[query_dict[key]]
            today = datetime.date.today()
            next_day = next_weekday(today, day_num) # 0 = Monday, 1=Tuesday, 2=Wednesday...
            entities['day'] = next_day.day
            entities['month'] = next_day.month
            entities['year'] = next_day.year

        # day number explicitly mentioned
        if key == "depart_date.day_number":
            entities['day'] = text2int(val)
            today = datetime.date.today()
            entities['month'] = today.month
            entities['year'] = today.year
            
        # month explicitly mentioned
        if key == "depart_date.month_name":
            entities['month'] = month2int[val]
            # assume today's date and year
            today = datetime.date.today()
            entities['day'] = today.day
            entities['year'] = today.year
        
        # if day/month/year still not in dict, show tomorrow's flights
        if ('day' not in entities.keys() or 
            'month' not in entities.keys() or 
            'year' not in entities.keys()):
            today = datetime.date.today()
            tom = today + datetime.timedelta(days=1)
            entities['day'] = tom.day
            entities['month'] = tom.month
            entities['year'] = tom.year 
        
    return query_dict, entities