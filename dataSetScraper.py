import requests
from bs4 import BeautifulSoup
import csv
import copy
from collections import Counter
import os

dataPath = "data"

qType = "strong"
aType = "p"

punctuation = "!.,?"

def main():
    scrapeData('https://hellobio.com/blog/interviews-with-scientists-mohammed-alawami.html')

def scrapeData(link, dataSetName = "dataSet.csv"):
    # Send an HTTP request to the website
    response = requests.get(link)

    # Parse the HTML or XML response
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the data you want to extract
    questions = soup.find_all(qType)

    qaPairs = []
    questionsVerified = []
    for question in questions:
        if "?" in question.text:
            questionsVerified.append(question.text)

    previousQuestion = questionsVerified[0]
    previousAnswers = soup.find(qType, string=previousQuestion).find_all_next(aType)

    for question in questionsVerified[1:]:
        #Get all of the answers coming after a question
        nextAnswers = soup.find(qType, string=question).find_all_next(aType)

        #Update the previous list of answers, keeping only the ones not present in the new list
        previousAnswers = list((Counter(previousAnswers) - Counter((nextAnswers + questionsVerified))).elements())

        #Finally add the previous question and previous answers to that question to qaPairs
        finalPreviousAnswers = str()
        for ans in previousAnswers:
            ansSoup = BeautifulSoup(str(ans), "html.parser").get_text()
            if any(punc in ansSoup for punc in punctuation) and ansSoup not in questionsVerified:
                finalPreviousAnswers += ansSoup

        qaPairs.append((previousQuestion, finalPreviousAnswers))

        #Save new values as previous values
        previousAnswers = copy.copy(nextAnswers)
        previousQuestion = copy.copy(question)

    # Write the data to a CSV file
    with open(os.path.join(dataPath, dataSetName), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(qaPairs)
        return(dataSetName)
    
if __name__ == "__main__":
    main()