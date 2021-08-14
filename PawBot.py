import random
import json
import pickle
import numpy as np
import nltk
import re
import pyttsx3

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from textblob import TextBlob
from spellchecker import SpellChecker

lemmatizer = WordNetLemmatizer()

ignoredSymbols = ['?', '!', ',', '.']

ignoredWords = []
with open('ignoredWords.txt') as f:
    ignoredWords = f.readlines()
    ignoredWords = [x.strip() for x in ignoredWords]

diseasesList = []
with open('diseasesList.txt') as f:
    diseasesList = f.readlines()
    diseasesList = [x.strip() for x in diseasesList]

stop_words = set(stopwords.words('english'))

intents = None
wordsList = None
classesList = None
model = None


def chose_dog():
    global intents, wordsList, classesList, model
    intents = json.loads(open('intentsDog.json', encoding="utf8").read())
    wordsList = pickle.load(open('wordsListDog.pkl', 'rb'))
    classesList = pickle.load(open('classesListDog.pkl', 'rb'))
    model = load_model('pawbotModelD.h5')
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say("welcome to paw bot")
    engine.runAndWait()


def chose_cat():
    global intents, wordsList, classesList, model
    intents = json.loads(open('intentsCat.json', encoding="utf8").read())
    wordsList = pickle.load(open('wordsListCat.pkl', 'rb'))
    classesList = pickle.load(open('classesListCat.pkl', 'rb'))
    model = load_model('pawbotModelC.h5')
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say("welcome to paw bot")
    engine.runAndWait()


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


def input_cleaner(sentence):
    filtered_sentence = ""
    removed_words = ""
    sentence_two = nltk.word_tokenize(str(sentence.lower()))
    for i in sentence_two:
        if i not in ignoredWords:
            filtered_sentence += (i + " ")
        else:
            removed_words += (i + " ")
    unchecked = re.findall("[a-zA-Z,.]+", filtered_sentence)
    new_sentence = (" ".join(unchecked))
    checked_sentence = TextBlob(new_sentence)
    fixed_sentence = checked_sentence.correct()
    recombined_sentence = fixed_sentence + " " + removed_words
    sentence_final = []
    sentence_words = nltk.pos_tag(nltk.word_tokenize(str(recombined_sentence.lower())))
    sentence_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), sentence_words))
    for word, tag in sentence_tagged:
        if tag is None:
            sentence_final.append(word)
        else:
            sentence_final.append(word)
            sentence_final = [lemmatizer.lemmatize(word, tag) for word in sentence_final if
                              word not in stop_words and word not in ignoredSymbols]
    print(sentence_final)
    return sentence_final


def bag_of_words(sentence):
    sentence_words = input_cleaner(sentence)
    bag = [0] * len(wordsList)
    for w in sentence_words:
        for i, word in enumerate(wordsList):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_tag(sentence):
    final_bag = bag_of_words(sentence)
    out = model.predict(np.array([final_bag]))[0]
    error_threshold = 0.15
    result = [[i, r] for i, r in enumerate(out) if r > error_threshold]
    not_result = [[i, r] for i, r in enumerate(out) if r < error_threshold]
    result.sort(key=lambda x: x[1], reverse=True)
    not_result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    not_result_list = []
    for r in result:
        return_list.append({'intent': classesList[r[0]], 'probability': str(r[1])})
    for r in not_result:
        not_result_list.append({'intent': classesList[r[0]], 'probability': str(r[1])})
    print(return_list)
    print(not_result_list)
    return return_list


def get_response(intents_list, intents_json):
    tag_list = []
    prob_list = []
    response_list = []
    response_prob = []
    if not intents_list:
        tag_list.append('unsure')
        prob_list.append('1')
    else:
        for i in intents_list:
            tag_list.append(i['intent'])
            prob_list.append(i['probability'])
    print(tag_list)

    json_intents = intents_json['intents']
    for i, k in zip(tag_list, prob_list):
        for j in json_intents:
            if j['tag'] == i:
                response_list.append(random.choice(j['responses']))
                response_prob.append(" (Probability: " + k + ")")
                break

    print(response_list)

    if len(response_list) > 1:
        if response_list[0] in diseasesList:
            for i in response_list:
                if i not in diseasesList:
                    response_list.remove(i)
            response = "<br>According to the symptoms given, you pet may have:<br><br>" + " <br>".join([a + str(b) for a, b in zip(response_list,response_prob)]) + \
                       "<br><br>You can ask me more about its/their definition(s), symptoms or treatment(s). " + \
                       "A low probability score indicates that the symptom(s) you have provided are too general, " \
                       "providing me with more symptoms may result in a better diagnosis. " \
                       "Only diagnoses above 15% probability are displayed."
        else:
            response = response_list[0]
    else:
        if response_list[0] in diseasesList:
            response = "<br>According to the symptoms given, you pet may have:<br><br>" + " <br>".join([a + str(b) for a, b in zip(response_list, response_prob)]) + \
                       "<br><br>You can ask me more about its/their definition(s), symptoms or treatment(s). " + \
                       "A low probability score indicates that the symptom(s) you have provided are too general, " \
                       "providing me with more symptoms may result in a better diagnosis. " \
                       "Only diagnoses above 15% probability are displayed."
        else:
            response = " <br>".join(response_list)

    return response


def pawbot_response(user_input):
    predicted = predict_tag(user_input)
    bot_output = get_response(predicted, intents)
    return bot_output





