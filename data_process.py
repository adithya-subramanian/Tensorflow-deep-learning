import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import Pickle
from collections import Counter
import random
import numpy as np
import tensorflow as tf

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

lexicon = []
for fi in ['positive.txt','negative.txt']:
	with open (fi,'r') as f:
	    contents = f.readlines()
	    for c in contents:
	        all_words = word_tokenize(c.lower())
	        lexicon += all_words

lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
lexicon = list(lexicon)
w_count = Counter(lexicon)
l2 = []
for w in w_count:
    if 500 >  w > 50:
    	l2.append(w)

final_feature = []
feature_set_positive = []
with open('positive.txt','r') as f:
	content = f.readlines()
	for c in content:
		words = word_tokenize(c.lower())
		words = [lemmatizer.lemmatize(i) for i in words]
		features = np.zeros(len(lexicon))
		for w1 in lexicon:
			if w1 in lexicon:
				features[lexicon.index(w1)] += 1
		features = list(features)
		feature_set_positive.append([features,[1 0]])

final_feature += feature_set_positive

feature_set_negative = []
with open('negative.txt','r') as f:
	content = f.readlines()
	for c in content:
		words = word_tokenize(c.lower())
		words = [lemmatizer.lemmatize(i) for i in words]
		features = np.zeros(len(lexicon))
		for w1 in lexicon:
			if w1 in lexicon:
				features[lexicon.index(w1)] += 1
		features = list(features)
		feature_set_negative.append([features,[0 1]])

final_feature += feature_set_negative
final_feature = np.array(final_feature)
train_size = int(0.5*len(final_feature))	
train_X = final_feature[:,0][:-train_size]
train_y = final_feature[:.1][:-train_size]

test_X = final_feature[:,0][-train_size:]
test_y = final_feature[:,1][-train_size:]

with open('sentiment_sent.pickle','wb') as f:
	pickle.dump([train_X,train_y,test_X,test_y],f)

number_of_layers = input('enter the number of layers(excluding the output layer):')
num_neurons = []
for i in range(num_of_layers):
	num_neurons.append(500)
num_neurons.append(2)
hidden_layer = []
batch_size = 100
for i in range(len(num_neurons)):
	hidden_layer.append(['hey':'hey'])

for i in range(len(num_neurons)):
	if i == 0:
	    hidden_layer[i] = {'weights':tf.Variable(tf.random_normal([None,len(train_X[0])])),'biases':tf.Variable(tf.random_normal([num_neurons[0]]))}
	else :
		hidden_layer[i] = {'weights':tf.Variable(tf.random_normal([num_neurons[i],num_neurons[i+1]])),'biases':tf.Variable(tf.random_normal([num_neurons[i]]))}

l = []
for i in range(len(num_neurons)):
	l.append([])

for i in range(len(num_neurons)):
	if i == 0:
		l[i] = tf.nn.relu(tf.add(tf.matmul(data,hidden_layer[i]['weights']),hidden_layer[i]['biases']))
	elif i != len(num_neurons)-1 and i != 0:
	    l[i] = tf.nn.relu(tf.add(tf.matmul(l[i-1],hidden_layer[i]['weights']),hidden_layer[i]['biases']))
    elif i == len(num_neurons)-1:
    	l[i] = (tf.add(tf.matmul(data,hidden_layer[i]['weights']),hidden_layer[i]['biases'])) 

prediction = l[len(num_neurons)-1]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))	
optimizer = tf.train.AdamOptimizer().minimize(cost)
min_epochs = 10

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch in range(min_epochs):
		epoch_loss = 0
		i = 0
		while i < len(train_X):
			start = i
			end = i + batch_size
			epoch_x = np.array(train_X[start:end])
			epoch_y = np.array(train_y[start:end])
			i,c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
			epoch_loss = epoch_loss + c
		print epoch

	correct = tf.equal(tf.argmax(prediction,axis = 1),tf.argmax(y,axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correct,'float'))
print accuracy