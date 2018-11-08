#!/usr/bin/python
# -*- coding: utf-8 -*-

# Load CSV (using python)

import csv
import numpy
import sys
from time import time
import datetime
sys.path.append('../tools/')
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import RandomizedPCA


def getUnique(list1):

    # intilize a null list

    unique_list = []

    # traverse for all elements

    for x in list1:

        # check if exists in unique_list or not

        if x not in unique_list:
            unique_list.append(x)

    return unique_list



def prepocess(filePassed):
	
	filename = filePassed
	raw_data = open(filename, 'rt')
	reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
	x = list(reader)
	data = numpy.array(x)
	features_list = [
		'booking_Date',
		'origin',
		'destination',
		'dep_date',
		'dep_time',
		'pax',
		]
	classLabels = []
	featureList = []
	Origin = []
	org = []
	des = []
	dep_time = []
	pax = []
	bookingDateWeek = []
	departureDateWeek = []
	id=[]
		
	for i in range(1, len(data)):
		if(filename=="train.csv"):
			classLabels.append(data[i][7])
		dep_time.append(int(data[i][5]))
		pax.append(int(data[i][6]))
		Origin.append(data[i][2])
		bookingDateWeek.append(datetime.date(int((data[i][1])[:4]),
							   int((data[i][1])[4:6]),
							   int((data[i][1])[6:8])).isocalendar()[1])
		departureDateWeek.append(datetime.date(int((data[i][4])[:4]),
								 int((data[i][4])[4:6]),
								 int((data[i][4])[6:8])).isocalendar()[1])
		

	uniqueList = getUnique(Origin)
	for i in range(1, len(data)):
		if data[i][2] == 'MAA':
			org.append(1)
		elif data[i][3] == 'MAA':
			des.append(1)
		if data[i][2] == 'DEL':
			org.append(2)
		elif data[i][3] == 'DEL':
			des.append(2)
		if data[i][2] == 'BOM':
			org.append(3)
		elif data[i][3] == 'BOM':
			des.append(3)
		if data[i][2] == 'BLR':
			org.append(4)
		elif data[i][3] == 'BLR':
			des.append(4)
		if data[i][2] == 'CCU':
			org.append(5)
		elif data[i][3] == 'CCU':
			des.append(5)
		if data[i][2] == 'HYD':
			org.append(6)
		elif data[i][3] == 'HYD':
			des.append(6)
		if data[i][2] == 'GOI':
			org.append(7)
		elif data[i][3] == 'GOI':
			des.append(7)


	for i in range(0, len(data)-1):
		tempList = []
		for j in range(i, len(data)):
			tempList.append(bookingDateWeek[j])
			tempList.append(org[j])
			tempList.append(des[j])
			tempList.append(departureDateWeek[j])
			tempList.append(dep_time[j])
			tempList.append(pax[j])
			break
		featureList.append(tempList)
	print (" First tuple -> " ,featureList[0])
	return featureList,classLabels


def fittingModel(featureList,classLabels):
	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(featureList, classLabels, test_size=0.2, random_state=42)
	#pca=pcaAnalysis(features_train)
	#pca_featureTrain=pca.transform(features_train)
	#pca_featureTest=pca.transform(features_test)
	print ("Total number of Test lables " ,len(labels_test))
	model = LogisticRegression(solver='newton-cg',max_iter=200,multi_class='multinomial')
	model.fit(features_train, labels_train)
	#print feature_test[98]
	#print pred[98]
	return model,features_test

#def calculatingAccuracy(pred,labels_test):
#	print(accuracy_score(pred,labels_test))

	
def getProbabilities(model,feature_test):
		print (model.classes_)
		pred=model.predict_proba(feature_test)[:,0]
		pred1=model.predict_proba(feature_test)[:,1]
		pred2=model.predict_proba(feature_test)[:,2]
		pred3=model.predict_proba(feature_test)[:,3]
		print ("Classic ->" ,pred[0], " Deal -> " ,pred1[0], " Flex -> "  ,pred2[0], " Saver ->" ,pred3[0])



#	print "probability of Count Classic", countClassic/float(countFlex+countSaver+countDeal+countClassic)
#	print "Count Flex", countFlex/(float)(countFlex+countSaver+countDeal+countClassic)
#	print "Count Saver", countSaver/(float)(countFlex+countSaver+countDeal+countClassic)
#	print "Count Deal", countDeal/(float)(countFlex+countSaver+countDeal+countClassic)
#	print(accuracy_score(pred,labels_test))#
#def pcaAnalysis(X_train):
#	pca = RandomizedPCA(n_components=4, whiten=True).fit(X_train)
#	return pca
		
	
def main():
	trainName="train.csv"
	testName="test.csv"
	featureDataset,labelsDataset=prepocess(trainName)
	#print featureDataset[0],labelsDataset[0]
	model,feature_test=fittingModel(featureDataset,labelsDataset)
	featureDataset,labelsDataset=prepocess(testName)
	
	#calculatingAccuracy(prediction,true_labels)
	#getProbabilities(model,feature_test)
	getProbabilities(model,featureDataset)
	#clf=GaussianNB()
	#t0 = time()
	#clf.fit(features_train,labels_train)
	#print "training time:", round(time()-t0, 3), "s"
	#t0 = time()
	#y_true=clf.predict(features_test)
	#print "training time:", round(time()-t0, 3), "s"


main()






			