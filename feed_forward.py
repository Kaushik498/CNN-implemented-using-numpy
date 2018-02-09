import numpy as np 
from required_functions import image_preprocessor

class feedforward:
	def __init__(self, batch, kernal_size, kernal_count):
		
		self.data = image_preprocessor(batch).Reshape()

		# defining variables related to the filter
		self.kernal_size = kernal_size
		self.kernal_count = kernal_count


	def load_weights(self , Wt ):
		self.features = Wt
		

	def predict(self):
		X = self.data
		# Creating all the layers
		#first CNN layer
		h1 = self.cnn_layer(X , layer_i=0)
		x1 = self.relu(h1)
		# Second CNN layer
		h2 = self.cnn_layer(x1 , layer_i=1)
		x2 = self.relu(h2)
		# Max pooling layer
		p1 = self.maxpool(x2)
		# Dropout
		d1 = self.dropout(p1, 0.25)
		# fully connected/dense layer
		d2 = self.dense_layer(d1 , layer_i=2)
		# calculating loss (losss function : softmax)
		s1 = self.softmax(d2)
		# Prediction :
		res = self.classify(s1)
		return res[0]

	def cnn_layer(self, X , layer_i):
		# defining a couple of important values
		cnn_weights = self.features['weights'][layer_i]
		cnn_bias = self.features['bias'][layer_i]

		image_dim = X.shape[3]
		num_channels = X.shape[1]
		num_images = X.shape[0]

		conv_dim = image_dim + self.kernal_size - 1
		conv_features = np.zeros((num_images , self.kernal_count , conv_dim , conv_dim)) ;

		for i_im in xrange(num_images):
			for i_ker in xrange(self.kernal_count):
				conv_image = np.zeros((conv_dim , conv_dim))
				for i_chn in xrange(num_channels):
					weights = cnn_weights[i_ker , i_chn , : , :]
					image = X[i_im , i_chn , : , :]
					conv_image += self.convolve2d(image , weights)
				
				conv_image += cnn_bias[i_ker]
				conv_features[i_im , i_ker , : , :] =  conv_image
		return conv_features

	def maxpool(self, conv_feature):
		num_image = conv_feature.shape[0]
		num_feature = conv_feature.shape[1]
		conv_dim = conv_feature.shape[2]
		pool_dim = int(conv_dim / 2)

		maxpooled_features = np.empty(shape=(num_image , num_feature , pool_dim , pool_dim))
		pooled_feature = np.zeros((num_image, num_feature, pool_dim, pool_dim))
		for i_im in xrange(num_image):
			for i_ft in xrange(num_feature):
				for r_pool in xrange(pool_dim):
					row_sart = r_pool*2
					row_end = row_sart + 2 

					for c_pool in xrange(pool_dim):
						col_start = c_pool*2
						col_end = col_start + 2

						patch = conv_feature[i_im, i_ft, row_start:row_end , col_start:col_end]
						maxpooled_features[i_im, i_ft, r_pool, c_pool] = np.max(patch)
		return maxpooled_features

	def dense_layer(self , X, layer_i):
		X = np.asarray(X).reshape(-1)
		W = self.features['weights'][layer_i]
		B = self.features['bias'][layer_i]
		result = np.dot(X, W) + B
		return result


	@staticmethod
	def convolve2d(image , feature):
		image_dim = image.shape[0]
		feature_dim = feature.shape[0]
		target_dim = image_dim + feature_dim -1

		fft_result = np.fft.fft2(image, target_dim)*np,fft,fft2(feature, target_dim)
		return np.fft.ifft2(fft_result).real

	@staticmethod
	def relu(X):
		Zeroes = np.zeros(X.shape)
		return np.where(X>Zeroes , X , Zeroes)

	@staticmethod
	def dropout(X, p):
		X *= (1.00-p)
		return X

	@staticmethod
	def softmax(X):
		"""Compute softmax values for each sets of scores in x."""
		e_x = np.exp(X - np.max(X))
		return e_x / e_x.sum()

	@staticmethod
	def classify(X):
		return X.argmax(axis=-1)
