from NLP.lexicalProcessing import plot_word_frequency, preprocess
import requests


url = "https://www.gutenberg.org/files/11/11-0.txt"
alice = requests.get(url)
print(alice.text)
alice_words = alice.text.split()
plot_word_frequency(alice_words, 10)