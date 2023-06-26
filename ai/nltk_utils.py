# nltk_utils.py
import numpy as np
import nltk

# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

from spellchecker import SpellChecker

stemmer = PorterStemmer()
spell = SpellChecker(language="fr")

# TODO Modify with data preparation from this tutorial https://pytorch.org/tutorials/beginner/chatbot_tutorial.html


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    print(nltk.word_tokenize(sentence))
    return nltk.word_tokenize(sentence, language="french")


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    print(word)
    return stemmer.stem(word.lower())


def correct(sentence):
    """
    Autocorrect words
    "bnojour -> "bonjour"
    """

    # for i in range(0, len(sentence)):
    #     sentence[i] = spell.correction(sentence[i])
    # print(sentence)
    return sentence


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    print(words)
    print(tokenized_sentence)
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
