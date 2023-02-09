#https://github.com/pytorch/tutorials/blob/master/beginner_source/chatbot_tutorial.py
import tensorflow as tf
import json
import os
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

"""
To download the movie corpus
"""
#from convokit import Corpus, download

EPOCHS = 10
TEST_SiZE = 0.4
LINE_LENGTH = 30

corpusName = "movie-corpus"
dataPath = "data"

def main():
    if len(sys.argv) not in [3]:
        sys.exit("Usage: python main.py [bool: Train model] [model_name (to save to or to read from)]")

    """
    Get training data to train the model
    """
    questions, answers = getData()

    """
    Convert questions and answers from dataset to readable bytes
    """
    byteQuestions = encodeList(questions)
    byteAnswers = encodeList(answers)

    #byteAnswers = tf.keras.utils.to_categorical(byteAnswers)
    #byteAnswers = tf.keras.utils.to_categorical([tf.keras.utils.to_categorical(np.array(i)) for i in byteAnswers])

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(byteQuestions), np.array(byteAnswers), test_size=TEST_SiZE
    )

    """
    Get model, either previously trained or train new one
    """
    model = None
    filename = sys.argv[2]
    trainModel = inp_to_bool(sys.argv[1])
    if (trainModel): #If input is true, train new model
        model = getModel()
        model.fit(x_train, y_train, epochs=EPOCHS)
        model.evaluate(x_test,  y_test, verbose=2)

        #Save trained model to file
        model.save(filename)
        print(f"Model saved to {filename}.")
    else: #If input is false, just use saved model
        model = load_model(filename)

    input = "How are you?"
    output = model(bytes(input, "ascii"))
    #output = encoder.inverse_transform(output)
    print(bytes(output))

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
            #convoPairs.append((qLine, aLine))

    #return convoPairs
    return questions, answers

def getModel():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=(LINE_LENGTH, 1, 1), output_dim=64),
            tf.keras.layers.Dense(10)
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def encodeList(inputList):
    outputList = []
    for i in inputList:
        iList = []
        for j in bytes(i, "ascii"):
            iList.append(j)

        """
        Resize iList array to a standardized length of LINE_LENGTH
        """
        npArrayiList = np.array(iList)
        npArrayiList.resize((LINE_LENGTH,))

        outputList.append(npArrayiList)

    return outputList

def inp_to_bool(inp):
    return inp.capitalize() in ["True", "1", "true"]

if __name__ == "__main__":
    main()