import theano
import theano.tensor as T
import numpy as np

class HiddenLayer(object):
	def __init__(self,n_in,n_out,input,output,activation = T.tanh,rng,W = None,b = None):
		self.rng = rng
		self.n_in = n_in
		self.n_out = n_out
		self.input = input

		if W is None:
			W_values =  np.asarray(self.rng.uniform( low = -np.sqrt(6./(n_in+n_out)),high = np.sqrt(6./(n_in+n_out)),size = (n_in,n_out)),dtype = theano.config.floatX)
			W = theano.shared(W_values)

		if b is None:
			b_values = np.zeros(size = (nout,) , dtype = theano.config.floatX)
			b = theano.shared.floatX

		self.W = W
		self.b = b

		lin_output = T.dot(x,W)+b
		if activation is None:
			self.output = lin_output
		else :
		    self.output = activation(lin_output)

class softmax(object):
	def __init__(self,n_in,n_out,input,output,activation = T.tanh,rng,W = None,b = None):
		self.rng = rng
		self.n_in = n_in
		self.n_out = n_out
		self.input = input

		if W is None:
			W_values =  np.asarray(self.rng.uniform( low = -np.sqrt(6./(n_in+n_out)),high = np.sqrt(6./(n_in+n_out)),size = (n_in,n_out)),dtype = theano.config.floatX)
			W = theano.shared(W_values)

		if b is None:
			b_values = np.zeros(size = (nout,) , dtype = theano.config.floatX)
			b = theano.shared.floatX

		self.W = W
		self.b = b

		self.p_y_given_x = T.nnet.softmax(T.dot(x,W)+b)
		self.y_pred = T.argmax(self.p_y_given_x,axis = 1)

    def negative_log_likelihood(self,y):
    	T.log(-self.)



