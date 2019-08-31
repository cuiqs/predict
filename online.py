import train
import numpy as np
import time
import tensorflow as tf


def get_a_line():
	t=np.load("vapor_press_test.npz")
	pvs=t["arr_0"].reshape(-1,19)
	for item in pvs:
		yield item

def predict():
	cur_pvs=np.zeros([24,19],dtype=np.float32)
	paras=train.get_parameters("vapor_press_trained_parameters.npz")
	pvs=get_a_line()
	while True:
		time.sleep(5)
		line=next(pvs)
		cur_pvs=np.delete(cur_pvs,-1,axis=0)
		cur_pvs=np.insert(cur_pvs,0,line,axis=0)
		cur_pvs_trans=cur_pvs.reshape(-1,1)
		pre_val=train.forward_propagation(cur_pvs_trans,paras)
		sess=tf.Session()
		val=sess.run(pre_val)
		val=val*50+50
		print(val)

predict()

