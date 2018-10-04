import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline



data = pd.read_csv("mbti_1.csv")


train_size = int(len(data) * .8)
train_posts = data['posts'][:train_size]
train_types = data['type'][:train_size]
test_posts = data['posts'][train_size:]
test_types = data['type'][train_size:]
vocab_size = 1000
num_labels = 16
batch_size = 32
tokenize = Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_posts)

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)
encoder = LabelBinarizer()
encoder.fit(train_types)
y_train = encoder.transform(train_types)
y_test = encoder.transform(test_types)
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))                      
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=100, 
                    verbose=1, 
                    validation_split=0.1)
score = model.evaluate(x_test, y_test, 
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
"""
----------
x = dataset[:0,4].astype(float)
y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
#convert int to dummy variables 
dummy_y = np_utils.to_categorical(encoded_y)

def baseline_model():
	model = Sequential()
	model.add(Dense(8, input_dim = 4, activation= 'relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
	return model 
	
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""