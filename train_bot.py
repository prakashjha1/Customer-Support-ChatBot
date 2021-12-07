import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 


words = []
classes = []
documents = []
ignore_words = ['?','!',',','.','_']
data_file = open("intents.json").read()
intents = json.loads(data_file)

for intent in intents['intents']:
	for pattern in intent['patterns']:
		# Tokenize here
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		documents.append((w,intent['tag']))
		# Adding tag to classes list
		if intent['tag'] not in classes:
			classes.append(intent['tag'])

	
    
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0]*len(classes)

for doc in documents:
	bag = []
	pattern_words = doc[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
	

	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)
	

	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1
	

	training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# compiling the model & define an Optimizer function
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

mfit = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('ChatBot_model.h5',mfit)

print('ChatBot is created!!')