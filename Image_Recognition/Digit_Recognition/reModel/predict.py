import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='4'
import sys
import time
from django.conf import settings
from Image_upload.models import Pictures
import tensorflow as tf
from PIL import Image,ImageFilter
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import time


class Test_predict(object):
	def __init__(self):      
		tf.reset_default_graph()  
		self.create_module()
		self.restore_module()
	def pool(self,x,stride):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, stride,stride, 1], padding='SAME')
	def imageprepare(self,argv): 
		"""
		This function returns the pixel values.
		The imput is a png file location.
		"""
		im = Image.open(argv).convert('L')
		width = float(im.size[0])
		height = float(im.size[1])
		newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

		if width > height:  # check which dimension is bigger
			# Width is bigger. Width becomes 20 pixels.
			nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
			if nheight == 0:  # rare case but minimum is 1 pixel
				nheight = 1
				# resize and sharpen
			img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
			newImage.paste(img, (4, wtop))  # paste resized image on white canvas
		else:
			# Height is bigger. Heigth becomes 20 pixels.
			nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
			if (nwidth == 0):  # rare case but minimum is 1 pixel
				nwidth = 1
				# resize and sharpen
			img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
			newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

		# newImage.save("sample.png")

		tv = list(newImage.getdata())  # get pixel values

		# normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
		tva = [(255 - x) * 1.0 / 255.0 for x in tv]
		return tva
				
	def create_module(self):	
		self.X_input = tf.placeholder(tf.float32, [None,784])
		"""Step 1: load the model"""
		# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
		X = tf.reshape(self.X_input, shape = [-1, 28, 28, 1])
		# correct answers will go here
		Y_ = tf.placeholder(tf.float32, [None, 10])
		# variable learning rate
		lr = tf.placeholder(tf.float32)
		# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
		self.pkeep = tf.placeholder(tf.float32)

		# three convolutional layers with their channel counts, and a
		# fully connected layer (the last layer has 10 softmax neurons)
		K = 64    # first convolutional layer 
		L = 128   # second convolutional layer output depth
		M = 256   # third convolutional layer
		Z = 512	# forth convolutional layer
		O = 1024	# fifth convolutional layer
		N = 1024  # fully connected layer
	#	K = 2  # first convolutional layer
	#	L = 4  # second convolutional layer output depth
	#	M = 8  # third convolutional layer
	#	Z = 16  # forth convolutional layer
	#	O = 200  # fifth convolutional layer
	#	N = 200  # fully connected layer
        
		W1 = tf.Variable(tf.truncated_normal([11, 11, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
		B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
		W2 = tf.Variable(tf.truncated_normal([7, 7, K, L], stddev=0.1))
		B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
		W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
		B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
		W4 = tf.Variable(tf.truncated_normal([3, 3, M, Z], stddev=0.1))
		B4 = tf.Variable(tf.constant(0.1, tf.float32, [Z]))
		W5 = tf.Variable(tf.truncated_normal([3, 3, Z, O], stddev=0.1))
		B5 = tf.Variable(tf.constant(0.1, tf.float32, [O]))

		W6 = tf.Variable(tf.truncated_normal([1 * 1 * O, N], stddev=0.1))
		B6 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
		W7 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
		B7 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

		# The model
		stride = 1  
		Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
		Y1_POOL = self.pool(Y1,2)
		stride = 2  
		Y2 = tf.nn.relu(tf.nn.conv2d(Y1_POOL, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
		Y2_POOL = self.pool(Y2,1)
		stride = 2  
		Y3 = tf.nn.relu(tf.nn.conv2d(Y2_POOL, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
		Y3_POOL = self.pool(Y3,1)
		stride = 1  
		Y4 = tf.nn.relu(tf.nn.conv2d(Y3_POOL, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)
		Y4_POOL = self.pool(Y4,2)
		stride = 1  
		Y5 = tf.nn.relu(tf.nn.conv2d(Y4_POOL, W5, strides=[1, stride, stride, 1], padding='SAME') + B5)
		Y5_POOL = self.pool(Y5,2)

		# reshape the output from the third convolution for the fully connected layer
		
		YY = tf.reshape(Y5_POOL, shape=[-1, 1 * 1 * O])
		DropConnect_W6 = tf.nn.dropout(W6,self.pkeep)*self.pkeep#drop the intpu weight.
		Y6 = tf.nn.relu(tf.matmul(YY, DropConnect_W6) + B6)
		DropConnect_W7 = tf.nn.dropout(W7,self.pkeep)*self.pkeep#drop the intpu weight.
		self.Ylogits = tf.matmul(Y6, DropConnect_W7) + B7
		Y = tf.nn.softmax(self.Ylogits)
		
	def predict(self,filename):
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)			
			"""Step 2.0: test the input"""						
			array=self.imageprepare(filename)
			Y = tf.nn.softmax(self.Ylogits)
			Y = Y.eval(feed_dict={self.X_input:[array],self.pkeep:1.0},session=self.sess)			
		return Y,self.sess.run(tf.argmax(Y, 1), feed_dict={self.X_input:[array],self.pkeep:1.0})[0]
	def restore_module(self):
		self.sess = tf.Session()
		#saver = tf.train.import_meta_graph('./ckpt/mnist.ckpt-2500.meta')
		model_file=tf.train.latest_checkpoint(settings.MEDIA_ROOT + "/Digit_Recognition/ckpt")
		saver = tf.train.Saver()
		saver.restore(self.sess,model_file)
if __name__ == '__main__':
	Test_predict()
