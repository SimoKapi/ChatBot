import gensim
from gensim.models import Word2Vec
import os
import json

from nltk.tokenize import sent_tokenize, word_tokenize

corpusName = "movie-corpus"
dataPath = "data"

trainModel = False

def main():
    corpus = os.path.join(dataPath, corpusName)
    utterances = os.path.join(corpus, "utterances.jsonl")

    conversations = {}
    with open(utterances) as f:
        for line in f:
            lineJson = json.loads(line)

            lineConvID = lineJson["conversation_id"]
            if lineConvID not in conversations.keys():
                conversations[lineConvID] = [lineJson["text"]]
            else:
                conversations[lineConvID].append(lineJson["text"])

    words = []
    for conversation in conversations:
        for i in range(len(conversations[conversation]) - 1):
            for i in word_tokenize(conversations[conversation][i]):
                words.append(i.lower())

    if trainModel:
        wordModel = gensim.models.Word2Vec([words], min_count=1, vector_size=4, window=5)
        wordModel.save("testmodel.model")
    else:
        wordModel = Word2Vec.load("testmodel.model")

    input1 = input("First word to transform: ")
    input2 = input("Word to subtract: ")
    input3 = input("Word to add: ")

    input1Vec = wordModel.wv[input1.lower()]
    input2Vec = wordModel.wv[input2.lower()]
    input3Vec = wordModel.wv[input3.lower()]

    outputVec = input1Vec - input2Vec + input3Vec
    output = wordModel.wv.most_similar(outputVec, topn=1)[0][0]
    print(f"Most probable output: {output}")

if __name__ == "__main__":
    main()