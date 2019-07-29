import csv
from datetime import datetime,timedelta
import time
import numpy as np



files=["FCC3_FI1308A_PV.csv","FCC3_FI1314_PV.csv","FCC3_FI1822_PV.csv","FCC3_FIC1105_PV.csv","FCC3_FIC1116_PV.csv","FCC3_FIC1201_PV.csv","FCC3_FIC1204_PV.csv","FCC3_FIC1205_PV.csv","FCC3_FIC1206_PV.csv","FCC3_FIC1214_PV.csv","FCC3_FIC1219_PV.csv","FCC3_FIC1227_PV.csv","FCC3_FIC1301_PV.csv","FCC3_FIC1309_PV.csv","FCC3_FIC1835_PV.csv","FCC3_PI1112_PV.csv","FCC3_PI1201_PV.csv","FCC3_PI1305_PV.csv","FCC3_PI1306_PV.csv","FCC3_PIC1301_PV.csv","FCC3_PIC1303_PV.csv","FCC3_TI1219_PV.csv","FCC3_TI1231_PV.csv","FCC3_TI1305_PV.csv","FCC3_TI1313_PV.csv","FCC3_TI1315_PV.csv","FCC3_TI1324_PV.csv","FCC3_TI1325A_PV.csv","FCC3_TI1326B_PV.csv","FCC3_TIC1201_PV.csv","FCC3_TIC1203_PV.csv","FCC3_TIC1217_PV.csv","FCC3_TIC1301_PV.csv"]
3
maxrange={"FCC3_FI1308A_PV":200,"FCC3_FI1314_PV":35000,"FCC3_FI1822_PV":200,"FCC3_FIC1105_PV":10,"FCC3_FIC1116_PV":6,"FCC3_FIC1201_PV":250,"FCC3_FIC1204_PV":50,"FCC3_FIC1205_PV":750,"FCC3_FIC1206_PV":300,"FCC3_FIC1214_PV":100,"FCC3_FIC1219_PV":50,"FCC3_FIC1227_PV":50,"FCC3_FIC1301_PV":80,"FCC3_FIC1309_PV":45,"FCC3_FIC1835_PV":75,"FCC3_PI1112_PV":400,"FCC3_PI1201_PV":250,"FCC3_PI1305_PV":2.5,"FCC3_PI1306_PV":2.5,"FCC3_PIC1301_PV":2.5,"FCC3_PIC1303_PV":1.6,"FCC3_TI1219_PV":150,"FCC3_TI1231_PV":300,"FCC3_TI1305_PV":100,"FCC3_TI1313_PV":100,"FCC3_TI1315_PV":200,"FCC3_TI1324_PV":300,"FCC3_TI1325A_PV":100,"FCC3_TI1326B_PV":200,"FCC3_TIC1201_PV":200,"FCC3_TIC1203_PV":400,"FCC3_TIC1217_PV":500,"FCC3_TIC1301_PV":200}

diesel_pvs=["FCC3_FI1822_PV","FCC3_FIC1105_PV","FCC3_FIC1116_PV","FCC3_FIC1201_PV","FCC3_FIC1204_PV","FCC3_FIC1205_PV","FCC3_FIC1206_PV","FCC3_FIC1214_PV","FCC3_FIC1219_PV","FCC3_FIC1227_PV","FCC3_FIC1835_PV","FCC3_PI1112_PV","FCC3_PI1201_PV","FCC3_TI1219_PV","FCC3_TI1231_PV","FCC3_TIC1201_PV","FCC3_TIC1203_PV","FCC3_TIC1217_PV"]
sample_span={"dieselconl":-30,"dieselconh":30,"stagasdryl":170,"stagasdryh":210,"stagaspresl":50,"stagaspresh":100,"lpgc2l":0,"lpgc2h":6,"lpgc5l":0,"lpgc5h":6}



class Predict:

#定义需要使用的生产数据
	pvs=list(maxrange.keys())
	pvs.sort()

#定义取PV时间间隔，及时间跨度，单位是分钟
	tmdelay=timedelta(minutes=10)
	tmspan=timedelta(minutes=480)

#生产数据字典，键为时间值，值为包含多个pv值的列表
	fcc3={}

#dt_from=datetime.strptime("2018-09-27 00:00","%Y-%m-%d %H:%M")
#dt_end=datetime.strptime("2019-06-05 00:00","%Y-%m-%d %H:%M")
#定义提取生产数据的起止时间
	dt_from=datetime.strptime("2018-01-01 00:00","%Y-%m-%d %H:%M")
	dt_end=datetime.strptime("2018-07-01 00:00","%Y-%m-%d %H:%M")

#需要预测的产品质量高低值
	saml=0
	samh=0

	def __init__(self,pvs,tmspan,tmdelay,dt_from,dt_end,sam):
		self.pvs=pvs
		self.tmspan=tmspan
		self.tmdelay=tmdelay
		self.dt_from=dt_from
		self.dt_end=dt_end
		
		self.saml=sample_span[sam+"l"]
		self.samh=sample_span[sam+"h"]
		
		self.fcc3={}
		self.timeout()


	def addfile(self,filename):
		mdelay=timedelta(minutes=1)
		n=2
		with open(filename) as csvfile:
			csv_reader=csv.reader(csvfile)
			header=next(csv_reader)  #skip table header
#			last=next(csv_reader)
#			if not ":" in last[1]:
#				last[1]=last[1]+" 00:00"
			is_first=True
			for row in csv_reader:
				if not ":" in row[1]:
					row[1]=row[1]+" 00:00"
				if is_first:
					is_first=False
					dt_l=datetime.strptime(row[1],"%Y-%m-%d %H:%M")
					self.fcc3[dt_l].append(float(row[2]))
					last=row
					print(row)
					continue
				else:
					dt_l=datetime.strptime(last[1],"%Y-%m-%d %H:%M")
				dt_n=datetime.strptime(row[1],"%Y-%m-%d %H:%M")
				if dt_l>dt_from and dt_n<dt_end:
					if dt_n==dt_l+mdelay:
						try:
							self.fcc3[dt_n].append(float(row[2]))
						except:
							self.fcc3[dt_n].append(float(last[2]))
							row[2]=last[2]
					else:
						while dt_n>dt_l+mdelay:
							try:
								self.fcc3[dt_l+mdelay].append(float(last[2]))
								dt_l+=mdelay
							except:
								raise
						try:
							self.fcc3[dt_n].append(float(row[2]))
						except:
							self.fcc3[dt_n].append(float(last[2]))
							row[2]=last[2]
				n+=1
				last=row

#产生时间序列及空列表
	def timeout(self):
		mdelay=timedelta(minutes=1)
	
		curtime=dt_from
		while curtime<=dt_end:
			self.fcc3[curtime]=[]	
			curtime+=mdelay

#将生产数据集中到一个文件中
	def gather(produce_data):
		for f in files:
			addfile(f)
			print(f)
		with open(produce_data,"w") as csvfile:
			csv_writer=csv.writer(csvfile)
			for moment in fcc3.keys():
				row=[]
				row.append(str(moment))
				for item in fcc3[moment]:
					row.append(str(item))
				csv_writer.writerow(row)


#用PV值除以各自量程，使得数据都小于1
	def uniform(from_f,to_f):
		count=len(maxrange)
		with open(from_f,"r+") as csvrd_file:
			with open(to_f,"w") as csvwr_file:
				csv_writer=csv.writer(csvwr_file)
				csv_reader=csv.reader(csvrd_file)
				for row in csv_reader:
#					print(row)
					for i in range(1,count+1):
						row[i]=str(float(row[i])/maxrange[files[i-1][0:-4]])#测量值比量程，使数据都<1
					csv_writer.writerow(row)

#从生产数据文件中提取需要的列数据
	def get_pv(pvfile,curpvs):
	
		fcc3_pv={}
		with open(pvfile) as csv_file:
			reader=csv.reader(csv_file)
			for row in reader:
				if len(row[0])>16:
					tm=datetime.strptime(row[0][0:-3],"%Y-%m-%d %H:%M")
				else:
					tm=datetime.strptime(row[0],"%Y-%m-%d %H:%M")
				fcc3_pv[tm]=[]
				for pv in curpvs:
					fcc3_pv[tm].append(float(row[pvs.index(pv)+1])) #row[0] is time

#		print(fcc3_pv.keys())
		return fcc3_pv



#从化验文件中提取数据
	def get_sample(samfile,sam):
		samples={}
		if sam=="chaiyou":
			saml=sample_span["dieselconl"]
			samspan=sample_span["dieselconh"]-saml
		if sam=="stagas":
			saml=sample_span["stagasdryl"]
			samspan=sample_span["stagasdryh"]-saml

		with open(samfile) as csv_file:
			csv_reader=csv.reader(csv_file)
			for row in csv_reader:
#				print(row)
				if len(row[0])>16:
					tm_sam=datetime.strptime(row[0][0:-3],"%Y-%m-%d %H:%M")
				else:
					tm_sam=datetime.strptime(row[0],"%Y-%m-%d %H:%M")
				samples[tm_sam]=(float(row[1])-saml)/samspan
		return samples

#生产数据文件与化验文件联系到一起
	def sample_add_pv(pvfile,diesel_pvs,samplefile,sample):
		samples=get_sample(samplefile,sample)
		fcc3_pvs=get_pv(pvfile,diesel_pvs)

		sam_num=len(samples)
		na_sam=np.empty([sam_num],float)
		na_pv=np.empty([sam_num,int(tmspan/tmdelay),len(diesel_pvs)],float)

		i=0
		for end_tm,val in samples.items():
			na_sam[i]=val
			tm=end_tm
			begin_tm=end_tm-tmspan
			j=0
			print(end_tm)
			while(tm>begin_tm):
				na_pv[i][j]=fcc3_pvs[tm]
				tm=tm-tmdelay
				j+=1
			print("na_sam"+str(i)+":")
			print(na_sam[i])
			print("na_pv"+str(i)+":")
			print(na_pv[i])
			i+=1
#		np.savez("stagas_sam_pv",na_sam,na_pv)
		return na_sam,na_pv	



tmdelay=timedelta(minutes=10)
tmspan=timedelta(minutes=240)
tvs=["FCC3_FI1822_PV"]
dt_end=datetime.strptime("2018-07-01 00:00","%Y-%m-%d %H:%M")
dt_from=datetime.strptime("2018-01-01 00:00","%Y-%m-%d %H:%M")

p=Predict(tvs,tmspan,tmdelay,dt_from,dt_end,'lpgc2')
p.addfile("FCC3_FI1822_PV.csv")
"""
#从npz文件中提取出生产数据及化验数据，按照8：2的比例分成训练集和测试集
	def random_and_classes(npzfile,train_set_file,test_set_file):
		r=np.load(npzfile)
#		print(r.files)
		pvs=r["arr_1"]
		sams=r["arr_0"]
		np.random.seed(1)
		pm=list(np.random.permutation(len(pvs)))
		train_size=round(len(pvs)*0.8)
		test_size=len(pvs)-train_size
	
		train_pvs=np.empty([train_size,pvs.shape[1],pvs.shape[2]],np.float32)
		train_sams=np.empty([train_size],np.float32)
		for i in range(train_size):
			train_pvs[i]=pvs[pm[i]]#完成随机化，打乱时间顺序
			train_sams[i]=sams[pm[i]]

		test_pvs=np.empty([test_size,pvs.shape[1],pvs.shape[2]],np.float32)
		test_sams=np.empty([test_size],np.float32)
		for i in range(test_size):
			test_pvs[i]=pvs[pm[i+train_size]]
			test_sams[i]=sams[pm[i+train_size]]
	
		np.savez(train_set_file,train_pvs,train_sams)
		np.savez(test_set_file,test_pvs,test_sams)
		
#random_and_classes("sam_pv.npz")

#sample_add_pv("produce3.csv","chaiyou.csv","chaiyou")
#timeout()
#gather("produce.csv")
#uniform("produce19.csv","produce3.csv")
#get_sample("chaiyou.csv","chaiyou")
na_sams,na_pvs=sample_add_pv("produce3.csv",pvs,"wenqigandian.csv","stagas")
np.savez("stagas_sam_pv",na_sams,na_pvs)
random_and_classes("stagas_sam_pv.npz","stagas_train.npz","stagas_test.npz")
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
	cost=tf.reduce_mean(tf.square(Y-Z3))
	return cost


#实现三层神经网络
def model(X_train,Y_train,net_weight,learning_rate=0.01,num_epochs=40000):
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


r=np.load("stagas_train.npz")
X_train=r["arr_0"].reshape(r["arr_0"].shape[0],-1).T
Y_train=r["arr_1"].reshape(1,-1)
net_weight=[20,X_train.shape[0],8,1]
parameters=model(X_train,Y_train,net_weight)

np.savez("stagasdry_trained_parameters.npz",parameters["W1"],parameters["b1"],parameters["W2"],parameters["b2"],parameters["W3"],parameters["b3"])
#parameters=get_parameters("trained_parameters.npz")
t=np.load("stagas_test.npz")
X_test=t["arr_0"].reshape(t["arr_0"].shape[0],-1).T
Y_test=t["arr_1"].reshape(1,-1)
Z=forward_propagation(X_test,parameters)
sess=tf.Session()
Z3=sess.run(compute_cost(Z,Y_test))
Zn=sess.run(Z)
print(Z3)
#print(Z3[0,0:20])
#print(Y_test[0,0:20])
print(Y_test*40+170)
print(Zn*40+170)	
"""
