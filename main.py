import tensorflow as tf
import json
import os
import numpy as np
import sys
import csv
import gensim
import itertools

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import Word2Vec

"""
To download the movie corpus
"""
#from convokit import Corpus, download
TRAIN_WORD_MODEL = True

EPOCHS = 10
TEST_SiZE = 0.4

VECTOR_SIZE = 5
LINE_LENGTH = 20

corpusName = "movie-corpus"
dataPath = "data"
word2vecmodels = "Word2Vec-Models"

ignoreCharacters = ".,!?"

y_train = []

def main():
    if len(sys.argv) not in [3]:
        sys.exit("Usage: python main.py [bool: Train model] [model_name (to save to or to read from)]")

    """
    Get model, either previously trained or train new one
    """
    filename = sys.argv[2]
    trainModel = inp_to_bool(sys.argv[1])
    model = None
    if (trainModel): #If input is true, train new model
        model = getModel()
        """
        Get training data to train the model
        """
        questions, answers = getData()

        """
        Convert questions and answers from dataset to trainable bytes
        """
        trainModel = TRAIN_WORD_MODEL
        try:
            Word2Vec.load(os.path.join(word2vecmodels, f"word2vec{VECTOR_SIZE}.model"))
        except FileNotFoundError:
            trainModel = True

        vectorQuestions, vectorAnswers = encodeList(trainModel, questions, answers)

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(vectorQuestions), np.array(vectorAnswers), test_size=TEST_SiZE
        )

        model.fit(x_train, y_train, epochs=EPOCHS)
        model.evaluate(x_test,  y_test, verbose=2)

        #Save trained model to file
        tf.keras.models.save_model(model, filename, save_format="tf")
        print(f"Model saved to {filename}.")
    else: #If input is false, just use saved model
        model = tf.keras.models.load_model(filename)

    print("\nTo exit, press 'Ctrl + C'\n------------------------")
    aiInteract(model)

"""
Interact with the AI
"""
def aiInteract(model):
    try:
        usrInput = input("Question to AI: ")
        print(f"Answer from AI: {getOutput(model, usrInput)}\n")

        aiInteract(model)
    except KeyboardInterrupt:
        exit

"""
Get an actual output from trained neural network
"""
def getOutput(model, input):
    wordModel = Word2Vec.load(os.path.join(word2vecmodels, f"word2vec{VECTOR_SIZE}.model"))
    data = []
    for word in word_tokenize(input):
        try:
            data.append(wordModel.wv[word.lower()])
        except KeyError:
            return "Could you please rephrase that question?"

    data = encodeList(False, [input])[0]

    output = model.predict(data)[0]

    outputWords = []
    for vector in output:
        outputWords.append(wordModel.wv.most_similar(positive=vector, topn=1)[0][0])

    strOutput = ' '.join(outputWords).capitalize()
    return strOutput

def getData():
    #data = dataSetScraper.scrapeData('https://hellobio.com/blog/interviews-with-scientists-mohammed-alawami.html')
    corpus = os.path.join(dataPath, corpusName)
    utterances = os.path.join(corpus, "utterances.jsonl")

    #convoPairs = []
    conversations = {}
    with open(utterances) as f:
        for line in f:
            lineJson = json.loads(line)

            lineConvID = lineJson["conversation_id"]
            if lineConvID not in conversations.keys():
                conversations[lineConvID] = [lineJson["text"]]
            else:
                conversations[lineConvID].append(lineJson["text"])

    questions = []
    answers = []
    for conversation in conversations:
        for i in range(len(conversations[conversation]) - 1):
            qLine = conversations[conversation][i]
            aLine = conversations[conversation][i+1]

            questions.append(qLine)
            answers.append(aLine)

    return questions, answers

def getModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(LINE_LENGTH, VECTOR_SIZE)),
        tf.keras.layers.Embedding(VECTOR_SIZE, 128),
        tf.keras.layers.Reshape((LINE_LENGTH, -1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(LINE_LENGTH*VECTOR_SIZE, activation="sigmoid"),
        tf.keras.layers.Reshape((LINE_LENGTH, VECTOR_SIZE))
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def stripString(input):
    output = ""
    for i in input:
        if i not in ignoreCharacters:
            output += i
    return output

"""
Convert list of strings to list of vectors
"""
def encodeList(trainModel, *inputLists):
    inputList = list(itertools.chain.from_iterable(inputLists))
    data = []
    
    """
    Only if you want to train a new word model
    """
    if trainModel:
        for sentence in inputList:
            temp = []
            
            for word in word_tokenize(sentence):
                temp.append(stripString(word.lower()))
        
            data.append(temp)
        wordModel = gensim.models.Word2Vec(data, min_count=1, vector_size=VECTOR_SIZE, window=5)
        wordModel.save(os.path.join(word2vecmodels, f"word2vec{VECTOR_SIZE}.model"))
    else:
        wordModel = Word2Vec.load(os.path.join(word2vecmodels, f"word2vec{VECTOR_SIZE}.model"))

    """
    Get vector values for words
    """
    outputList = []
    for category in inputLists:
        catList = []
        for sentence in category:
            temp = []
            tokenizedSent = word_tokenize(sentence)
            for i in range(LINE_LENGTH):
                try:
                    temp.append(wordModel.wv[stripString(tokenizedSent[i].lower())].tolist())
                except IndexError:
                    temp.append([0.0] * VECTOR_SIZE)
            catList.append(temp)

        outputList.append(catList)

    return outputList

def inp_to_bool(inp):
    return inp.capitalize() in ["True", "1", "true"]

if __name__ == "__main__":
    main()