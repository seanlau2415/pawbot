import random
import json
import pickle
import numpy as np
import nltk
import re

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from textblob import TextBlob
from spellchecker import SpellChecker

lemmatizer = WordNetLemmatizer()

ignoredSymbols = ['?', '!', ',', '.']
stop_words = set(stopwords.words('english'))

sentence = "rabies rabie"

ignoredWords = []

with open('ignoredWords.txt') as f:
    ignoredWords = f.readlines()
    ignoredWords = [x.strip() for x in ignoredWords]


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


removedWords = ""
recombinedWords = ""
sentence_three = ""
sentence_two = nltk.word_tokenize(str(sentence.lower()))

for i in sentence_two:
    if i not in ignoredWords:
        sentence_three += (i + " ")
    else:
        removedWords += (i + " ")

str1 = re.findall("[a-zA-Z,.]+", sentence_three)
new_sentence = (" ".join(str1))

spell = SpellChecker()

misspelled = spell.unknown(str1)
print(misspelled)

fixed_sentence = TextBlob(new_sentence)

print(fixed_sentence)

result = fixed_sentence.correct()
print(result)

recombinedWords = result + " " + removedWords
print(recombinedWords)

sentence_final = []
sentence_words = nltk.pos_tag(nltk.word_tokenize(str(recombinedWords.lower())))
sentence_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), sentence_words))
for word, tag in sentence_tagged:
    if tag is None:
        sentence_final.append(word)
    else:
        sentence_final.append(word)
        sentence_final = [lemmatizer.lemmatize(word, tag) for word in sentence_final if
                            word not in stop_words and word not in ignoredSymbols]
print(sentence_final)

#def get_response(intents_list, intents_json):
    #if not intents_list:
        #tag = 'unsure'
    #else:
        #tag = intents_list[0]['intent']

    #json_intents = intents_json['intents']
    #for i in json_intents:
        #if i['tag'] == tag:
            #response = random.choice(i['responses'])
            #print(intents_list)
            #break
    #return response


