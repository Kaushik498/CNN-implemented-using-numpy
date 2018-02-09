import pickle as pk
import numpy as np
from PIL import Image

mat_dict = {}

class preprocessor:
	
	def __init__(self , file):
		with open(file, 'rb') as fo:
			self.dict = pk.load(fo)


	def Reshape(self):
		return self.dict['data'].reshape(10000,3,32,32)
		

X = preprocessor('/media/kirito/New Volume1/data/cifar-10/data_batch_1').Reshape()

print X.shape
