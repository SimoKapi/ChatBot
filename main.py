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

# import gensim
# from gensim.models import Word2Vec

"""
To download the movie corpus
"""
# from convokit import Corpus, download
TRAIN_WORD_MODEL = False

EPOCHS = 20
TEST_SiZE = 0.4
WORD_INDEX_SIZE = 57629 #Number of lines in indexPath

LINE_LENGTH = 20

corpusName = "movie-corpus"
dataPath = "data"
indexPath = "wordindex.csv"

ignoreCharacters = ".,!?"

wordDict = {}

def main():
    if len(sys.argv) not in [3]:
        sys.exit(
            "Usage: python main.py [bool: Train model] [model_name (to save to or to read from)]")

    """
    Get model, either previously trained or train new one
    """
    filename = sys.argv[2]
    trainModel = inp_to_bool(sys.argv[1])
    model = None
    if (trainModel):  # If input is true, train new model
        """
        Get training data to train the model
        """
        questions, answers = getData()

        """
        Convert questions and answers from dataset to trainable bytes
        """
        vectorQuestions, vectorAnswers = encodeList(
            TRAIN_WORD_MODEL, questions, answers)

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(vectorQuestions), np.array(vectorAnswers), test_size=TEST_SiZE
        )
        model = getModel()
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32)
        model.evaluate(x_test,  y_test, verbose=2)

        # Save trained model to file
        tf.keras.models.save_model(model, filename, save_format="tf")
        print(f"Model saved to {filename}.")
    else:  # If input is false, just use saved model
        model = tf.keras.models.load_model(filename)

    with open(indexPath) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            wordDict.update({row[1]: row[0]})
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
    try:
        data = encodeList(False, [input])[0]
    except KeyError:
        return ("I'm sorry, could you rephrase that?")
    output = model.predict(data)[0]

    outputWords = []
    for index in output:
        newIndex = str(int(index))
        if newIndex in wordDict.keys():
            outputWords.append(wordDict[newIndex])
        else:
            outputWords.append("")

    strOutput = ' '.join(outputWords).capitalize()
    return strOutput

def getData():
    # data = dataSetScraper.scrapeData('https://hellobio.com/blog/interviews-with-scientists-mohammed-alawami.html')
    corpus = os.path.join(dataPath, corpusName)
    utterances = os.path.join(corpus, "utterances.jsonl")

    # convoPairs = []
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

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=WORD_INDEX_SIZE, output_dim=128, input_shape=(LINE_LENGTH,)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(LINE_LENGTH, activation="softmax")
    ])

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    # history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=(LINE_LENGTH,)),
    #     tf.keras.layers.Embedding(WORD_INDEX_SIZE, 128),
    #     tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(64, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(LINE_LENGTH, activation="softmax")
    # ])

    # model.compile(
    #     optimizer="adam",
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"]
    # )

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
def encodeList(newData, *inputLists):
    inputList = list(itertools.chain.from_iterable(inputLists))
    data = []

    """
    Only if new dataset
    """
    if newData:
        words = []
        indexes = []
        for sentence in inputList:
            for word in word_tokenize(sentence):
                strippedString = stripString(word.lower())
                if (strippedString not in words and len(strippedString) != 0):
                    words.append(strippedString)
                    if (len(indexes) > 0):
                        indexes.append(indexes[-1] + 1)
                    else:
                        indexes.append(1)

        with open(indexPath, 'w') as f:
            csvwriter = csv.writer(f)
            for i in range(len(words)):
                csvwriter.writerow((words[i], indexes[i]))
                    
    """
    Get values for words
    """
    outputList = []
    with open(indexPath, 'r') as f:
        csvreader = csv.reader(f)
        wordDict = {}
        for row in csvreader:
            wordDict.update({row[0]: row[1]})
        for category in inputLists:
            catList = []
            for sentence in category:
                temp = []

                tokenizedSent = word_tokenize(sentence)
                updatedTokSent = []
                for word in tokenizedSent:
                    newStr = stripString(word.lower())
                    if len(newStr) > 0:
                        updatedTokSent.append(newStr)

                for i in range(LINE_LENGTH):
                    try:
                        inp = updatedTokSent[i]
                        temp.append(int(wordDict[inp]))
                    except IndexError:
                        temp.append(0)
                catList.append(temp)
            outputList.append(catList)
    return outputList

def inp_to_bool(inp):
    return inp.capitalize() in ["True", "1", "true"]

if __name__ == "__main__":
    main()