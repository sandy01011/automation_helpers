# This file contain functions helpful to perform lexical processing on text corpous

import seaborn as sns
from nltk import FreqDist
from nltk.corpus import stopwords


def plot_word_frequency(words, top_n=10):         # plot top words 
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plot = sns.barplot(labels, counts)
    return plot