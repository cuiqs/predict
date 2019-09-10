#实现由生产数据到产品质量的训练过程

import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import time


#为Tensorflow会话创建占位符
def create_placeholders(n_pvs,n_sams):
	X=tf.placeholder(tf.float32,[n_pvs,None],name="X")
	Y=tf.placeholder(tf.float32,[n_sams,None],name="Y")

	return X,Y

#初始化神经网络的参数
def initialize_parameters(w1x,w1y,w2x,w3x):
	
	tf.set_random_seed(1)
	W1=tf.get_variable("W1",[w1x,w1y],initializer=tf.contrib.layers.xavier_initializer(seed=1))
#	W1=tf.get_variable("W1",[w1x,w1y],initializer=tf.contrib.layers.variance_scaling_initializer())
	b1=tf.get_variable("b1",[w1x,1],initializer=tf.zeros_initializer())
	W2=tf.get_variable("W2",[w2x,w1x],initializer=tf.contrib.layers.xavier_initializer(seed=1))
#	W2=tf.get_variable("W2",[w2x,w1x],initializer=tf.contrib.layers.variance_scaling_initializer())
	b2=tf.get_variable("b2",[w2x,1],initializer=tf.zeros_initializer())
	W3=tf.get_variable("W3",[w3x,w2x],initializer=tf.contrib.layers.xavier_initializer(seed=1))
#	W3=tf.get_variable("W3",[w3x,w2x],initializer=tf.contrib.layers.variance_scaling_initializer())
	b3=tf.get_variable("b3",[w3x,1],initializer=tf.zeros_initializer())

	
	parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
	
	return parameters


#实现前向传播，返回最后一个节点的输出
def forward_propagation(X,parameters):
	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]
	W3=parameters["W3"]
	b3=parameters["b3"]
	
	
	Z1=tf.matmul(W1,X)+b1
	A1=tf.nn.relu(Z1)
#	A1=tf.nn.sigmoid(Z1)
	Z2=tf.matmul(W2,A1)+b2
	A2=tf.nn.relu(Z2)
#	A2=tf.nn.sigmoid(Z2)
	Z3=tf.matmul(W3,A2)+b3

	return Z3

#计算损失
def compute_cost(Z3,Y):

	logits=tf.transpose(Z3)
	labels=tf.reshape(Y,[-1,1])

#	cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
	cost=tf.reduce_mean(tf.square(Y-Z3)/2)
	return cost


#实现三层神经网络
def model(X_train,Y_train,net_weight,learning_rate=0.015,num_epochs=80000):
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed=3

	(n_x,m)=X_train.shape
	n_y=Y_train.shape[0]

	costs=[]

	X,Y=create_placeholders(n_x,n_y)

	parameters=initialize_parameters(net_weight[0],net_weight[1],net_weight[2],net_weight[3])

	Z3=forward_propagation(X,parameters)

	cost=compute_cost(Z3,Y)

	optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init=tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		
		for epoch in range(num_epochs):
			epoch_cost=0
			_,epoch_cost=sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train})
			
			if epoch%5==0:
				costs.append(epoch_cost)
				if epoch%500==0:
					print("eopch"+str(epoch)+"cost"+str(epoch_cost))

	
		parameters=sess.run(parameters)
#		np.savez("trained_parameters.npz",parameters["W1"],parameters["b1"],parameters["W2"],parameters["b2"],parameters["W3"],parameters["b3"])
	
	return parameters

#读取已经训练后的参数文件
def get_parameters(para_file):
	w=np.load(para_file)
	W1=w["arr_0"]
	b1=w["arr_1"]
	W2=w["arr_2"]
	b2=w["arr_3"]
	W3=w["arr_4"]
	b3=w["arr_5"]

	parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}

	return parameters


#r=np.load("vapor_press_1801_1907_tmspan2_train.npz")
#X_train=r["arr_0"].reshape(r["arr_0"].shape[0],-1).T
#Y_train=r["arr_1"].reshape(1,-1)
#net_weight=[11,X_train.shape[0],8,1]
#parameters=model(X_train,Y_train,net_weight)

#np.savez("vapor_press_1801_1907_tmspan2_trained_parameters.npz",parameters["W1"],parameters["b1"],parameters["W2"],parameters["b2"],parameters["W3"],parameters["b3"])
parameters=get_parameters("vapor_press_1801_1907_tmspan2_trained_parameters.npz")
def test(parameters,testfile,scale,low):
#	parameters=get_parameters("stagasdry_trained_parameters.npz")
#	print(parameters["W1"].shape)
	t=np.load(testfile)
	X_test=t["arr_0"].reshape(t["arr_0"].shape[0],-1).T
	print(X_test.shape)
	Y_test=t["arr_1"].reshape(1,-1)
	Z=forward_propagation(X_test,parameters)
	sess=tf.Session()
	Z3=sess.run(compute_cost(Z,Y_test))
	Zn=sess.run(Z)
	print(Z3)
	for i in range(0,Zn.shape[1]):
		print('{0:.3f} {1:.3f}'.format(Y_test[0][i]*scale+low,Zn[0][i]*scale+low))

	print(np.mean(abs(Y_test-Zn))*scale)

def diesel_test():
	parameters=get_parameters("trained_parameters.npz")
	t=np.load("diesel_sam_pv81.npz")
	X_test=t["arr_0"].reshape(t["arr_0"].shape[0],-1).T
	Y_test=t["arr_1"].reshape(1,-1)
	Z=forward_propagation(X_test,parameters)
	sess=tf.Session()
	Z3=sess.run(compute_cost(Z,Y_test))
	print(Z3)
	Zn=sess.run(Z)
	print(Y_test*60-30)
	print(Zn*60-30)
	print((Zn-Y_test)*60)

#diesel_test()
test(parameters,"vapor_press_1907_tmspan2.npz",40,40)
