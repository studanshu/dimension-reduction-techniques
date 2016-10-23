import operator
from sys import argv
import os
import numpy as np
trainFolder = "../data/20news-bydate-train/"
testFolder = "../data/20news-bydate-test/"
vocabFile = "../data/vocabulary.txt"

wordList = None
categoryList = None

def getDictionary():
	global wordList
	if wordList != None:
		return wordList
	with open(vocabFile) as f:
		wordList = f.readlines()
	for i in range(0, len(wordList)):
		wordList[i] = normalize(wordList[i])
	return wordList

def getCategoriesList():
	global categoryList
	if categoryList != None:
		return categoryList
	categoryList = []
	for folderName in os.listdir(trainFolder):
		if folderName != ".DS_Store":
			categoryList += [folderName]
	return categoryList

def removeStopWord(data, isString = False):
	dataArr = data
	if isString == True:
		dataArr = data.split(' ')
	from nltk.corpus import stopwords
	filteredWords = [word for word in dataArr if word not in stopwords.words('english')]
	if isString == True:
		return ' '.join(word for word in filteredWords);
	return filteredWords

def lemmatize(data, isString = False):
	dataArr = data
	if isString == True:
		dataArr = data.split(' ')
	from nltk.stem import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	modifiedData = [lemmatizer.lemmatize(word) for word in dataArr]
	if isString == True:
		return ' '.join(word for word in modifiedData);
	return modifiedData


def stemming(data, isString = False):
	dataStr = data
	if isString == False:
		dataStr = ' '.join(word for word in data)
	from nltk.stem.porter import *
	stemmer = PorterStemmer()
	modifiedData = stemmer.stem(dataStr)
	if isString == False:
		return modifiedData
	return modifiedData.split(' ')


def normalize(data, isString = False):
	dataArr = data
	if isString == True:
		dataArr = data.lower().strip('\n\t').split(' ')
	modifiedData1 = removeStopWord(dataArr, False)
	modifiedData2 = lemmatize(modifiedData1, False)
	# data3 = stemming(modifiedData1, isString)
	if isString == True:
		return ' '.join(word for word in modifiedData2);
	return modifiedData2

#data can be path to file also.
def getFeatureVector(data, isString = False):
	dataStr = data
	if isString == False:
		with open(data) as f:
			dataStr = f.readlines()
		dataStr = ' '.join(line.strip() for line in dataStr)
	
	normalizedStr = normalize(removeStopWord(dataStr, isString = True), isString = True)

