# This file contain functions helpful to perform lexical processing on text corpous


# Scope A: Perform Lexical Text Processing and EDA

#Step 1: EDA: analyse word frequency of the document
import seaborn as sns
from nltk import FreqDist
from nltk.corpus import stopwords

# plot word frequency with or without stop words
def plot_word_frequency(words, top_n=10, stop_words=True):         # plot top words
    # remove stop words
    if stop_words:
        pass
    else:
        words = [word for word in words if word not in stopwords.words("english")] 
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plot = sns.barplot(labels, counts)
    return plot

#Step 2: Feature Extracton: extract features from the document including term features along with word features
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

# Preprocess document and convert it to its stem or lemma
def preprocess(document, stem=True):
    'changes document to lower case and removes stopwords'
    # change sentence to lower case
    document = document.lower()
    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    
    if stem:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

    # join words to make sentence
    document = " ".join(words)
    
    return document


#Step 3: apply simple vectoriser to analyse document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.feature_extraction.text import CountVectorizer
def count_vectors(document, bow=True, tfidf=True):
    if bow ==True and tfidf==True:
        vectorizer = CountVectorizer()
        bow_model = vectorizer.fit_transform(document)
        vectorizer = TfidfVectorizer()
        tfidf_model = vectorizer.fit_transform(document)
        #tfidf_model.to_array()
        return (bow_model, tfidf_model)
    else:
        vectorizer = TfidfVectorizer()
        tfidf_model = vectorizer.fit_transform(document)
        return tfidf_model

# Step 4: Advance Lexical Processing

  # soundex
def get_soundex(token, dictionary=None):
    """Get the soundex code for the string"""
    token = token.upper()

    soundex = ""
    
    # first letter of input is always the first letter of soundex
    soundex += token[0]
    
    # create a dictionary which maps letters to respective soundex codes. Vowels and 'H', 'W' and 'Y' will be represented by '.'
    if dictionary !=None:
        pass
    else: 
        dictionary = {"BFPV": "1", "CGJKQSXZ":"2", "DT":"3", "L":"4", "MN":"5", "R":"6", "AEIOUHWY":"."}

    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != soundex[-1]:
                    soundex += code

    # remove vowels and 'H', 'W' and 'Y' from soundex
    soundex = soundex.replace(".", "")
    
    # trim or pad to make soundex a 4-character code
    soundex = soundex[:4].ljust(4, "0")
        
    return soundex

   # Edit Distance
def lev_distance(source='', target=''):
    """Make a Levenshtein Distances Matrix"""
    
    # get length of both strings
    n1, n2 = len(source), len(target)
    
    # create matrix using length of both strings - source string sits on columns, target string sits on rows
    matrix = [ [ 0 for i1 in range(n1 + 1) ] for i2 in range(n2 + 1) ]
    
    # fill the first row - (0 to n1-1)
    for i1 in range(1, n1 + 1):
        matrix[0][i1] = i1
    
    # fill the first column - (0 to n2-1)
    for i2 in range(1, n2 + 1):
        matrix[i2][0] = i2
    
    # fill the matrix
    for i2 in range(1, n2 + 1):
        for i1 in range(1, n1 + 1):
            
            # check whether letters being compared are same
            if (source[i1-1] == target[i2-1]):
                value = matrix[i2-1][i1-1]               # top-left cell value
            else:
                value = min(matrix[i2-1][i1]   + 1,      # left cell value     + 1
                            matrix[i2][i1-1]   + 1,      # top cell  value     + 1
                            matrix[i2-1][i1-1] + 1)      # top-left cell value + 1
            
            matrix[i2][i1] = value
    
    # return bottom-right cell value
    return matrix[-1][-1]
# import library
#from nltk.metrics.distance import edit_distance
#edit_distance("apple", "appel")
#edit_distance("apple", "appel", transpositions=False, )
#The Damerau-Levenshtein distance allows transpositions (swap of two letters which are adjacent to each other) as well