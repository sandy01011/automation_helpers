# This file contain functions helpful to perform lexical processing on text corpous


# Scope A: Perform Lexical Text Processing EDA
#Step 1: analyse word frequency of the document
import seaborn as sns
from nltk import FreqDist
from nltk.corpus import stopwords

# plot word frequency 
def plot_word_frequency(words, top_n=10):         # plot top words 
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plot = sns.barplot(labels, counts)
    return plot

#Step 2: extract features from the document including term features along with word features
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

    # spell check 

    #Step 3: apply simple vectoriser to analyse document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  
def count_vectors(document, bow=True, tfidf=True):
    if bow=True and tfidf=True:
        pass
    else:
        vectorizer = TfidfVectorizer()
        tfidf_model = vectorizer.fit_transform(document)
        return tfidf_model

