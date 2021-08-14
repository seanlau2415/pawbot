import random
import json
import pickle
import numpy as np
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intentsCat.json', encoding="utf8").read())
ignoredSymbols = ['?', '!', ',', '.']
stop_words = set(stopwords.words('english'))

wordsList = []
classesList = []
combinedList = []


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


for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = nltk.pos_tag(nltk.word_tokenize(pattern.lower()))
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), words))
        for word, tag in wordnet_tagged:
            if tag is None:
                wordsList.append(word)
            else:
                wordsList.append(word)
                wordsList = [lemmatizer.lemmatize(word, tag) for word in wordsList if
                             word not in stop_words and word not in ignoredSymbols]
        combinedList.append((words, intent['tag']))
        if intent['tag'] not in classesList:
            classesList.append(intent['tag'])

wordsList = sorted(set(wordsList))
classesList = sorted(set(classesList))

pickle.dump(wordsList, open('wordsListCat.pkl', 'wb'))
pickle.dump(classesList, open('classesListCat.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classesList)

for combined in combinedList:
    bag = []
    wordPatternsFinal = []
    wordPatterns = combined[0]
    wordPatterns_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), wordPatterns))
    for word, tag in wordPatterns_tagged:
        if tag is None:
            wordPatternsFinal.append(word)
        else:
            wordPatternsFinal.append(word)
            wordPatternsFinal = [lemmatizer.lemmatize(word, tag) for word in wordPatternsFinal if
                                 word not in stop_words and word not in ignoredSymbols]
    for words in wordsList:
        bag.append(1) if words in wordPatternsFinal else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classesList.index(combined[1])] = 1
    training.append([bag, outputRow])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()

model.add(Dense(500, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
paw = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
model.save('pawbotModelC.h5', paw)
print("Training is complete!")

