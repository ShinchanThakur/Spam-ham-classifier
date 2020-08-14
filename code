#SPAM - HAM predictor using Navie Bayes Classifier
#Trained on 0.8 * SMS_SPAM_COLLECTION
#Tested on 0.2 * SMS_SPAM_COLLECTION
#Predictions: Satisfactory, as error rate is 2-4%
#Suggestions: Override the Vectorizers analyzer for better text preprocessing


import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
def loadCorpus(fileName):
	fh = open(fileName)
	pattern = re.compile(r'(.+)(\t)(.+)')
	labels = []
	messages = []
	for msg in fh:
		x = re.search(pattern, msg)
		if x:
			labels.append(x.group(1))
			messages.append(x.group(3))

	fh.close()
	return labels, messages

#Load the corpus in memory
labels, messages = loadCorpus('d:/SMSSpamCollection')

#Split the corpus into training and testing sets
train_M, test_M, train_L, test_L = train_test_split(messages, labels, test_size= 0.2)
print(len(messages), len(train_M), len(test_M))
print(len(labels), len(train_L), len(test_L))

#create a vectorizer for feature extraction
vectorizer = TfidfVectorizer(stop_words='english')

#build a vocabulary from the training set
vectorizer.fit(train_M)

# a look inside
#for x, y in zip(vectorizer.get_feature_names(), vectorizer.idf_):
#	print(x, y , sep=' : ')

#transform the training set into a bag of words
bow = vectorizer.transform(train_M)

#instantiate the algorithm
algo = MultinomialNB() #navie_bayes classifier

#train it
algo.fit(bow, train_L)
print('I am trained to predict')

#prediction time
tot = len(test_M)
err = 0
for lbl,msg in zip(test_L, test_M):
	#transform the message to be processed
	msg_bow = vectorizer.transform([msg])
	#predict
	prediction =  algo.predict(msg_bow)
	#show up
	print(prediction[0], lbl, sep= ' : ')
	if prediction[0] != lbl:
		err+=1
print('Failure Rate :', err, '/', tot)
