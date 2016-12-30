import tensorflow as tf
from tensorflow.examples.tutroials.mnit import input_data
mnist = input_data.read_data_sets('data/',one_hot = True)

num_neurons = [500,500,500,10]
hidden_layer = []
batch_size = 100
for i in range(len(num_neurons)):
	hidden_layer.append(['hey':'hey'])

for i in range(len(num_neurons)):
	if i == 0:
	    hidden_layer[i] = {'weights':tf.Variable(tf.random_normal([None,784])),'biases':tf.Variable(tf.random_normal([num_neurons[0]]))}
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
		for i in range(int(mnist.train.num_examples/batch_size)):
			x,y = mnist.train.next_batch(batch_size)
			i,c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
			epoch_loss = epoch_loss + c
		print epoch

	correct = tf.equal(tf.argmax(prediction,axis = 1),tf.argmax(y,axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correct,'float'))
	print accuracy

