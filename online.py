import train
import gettrain_x
import numpy as np
import time

import tensorflow as tf
from xml.dom.minidom import parse
from xml.dom.minidom import Document
import xml.dom.minidom

vapor_press_pvs=gettrain_x.vapor_press_pvs

def get_a_line():
	t=np.load("vapor_press_test.npz")
	pvs=t["arr_0"].reshape(-1,19)
	for item in pvs:
		yield item

def get_a_line_fromxml(file):
    maxrange=gettrain_x.maxrange
    DOMTree=xml.dom.minidom.parse(file)
    dom_pvs=DOMTree.documentElement
    pvlist=[]
    for pv in vapor_press_pvs:
        one_pv=dom_pvs.getElementsByTagName(pv)
       
        pvlist.append(one_pv[0].firstChild.data)
#        val=float(one_pv[0].firstChild.data)
#        val=val.astype(np.float32)
#        line[0,index]=
    pvarr=np.array(pvlist)    
    line=pvarr.astype(dtype=np.float32)
    for i in range(len(pvlist)):
        item=vapor_press_pvs[i]
        line[i]=(line[i]-maxrange[item][0])/maxrange[item][1]
    line=line.reshape(1,len(vapor_press_pvs))    
    return line
        
def write_to_xml(fore_value,filename):
    doc = Document()
    root = doc.createElement("prediction")
    doc.appendChild(root)
 
    
    tnode = doc.createElement("time")
    cur_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    txnode=doc.createTextNode(cur_time)
    tnode.appendChild(txnode)
    
    valnode=doc.createElement("predict value")
    val= doc.createTextNode(str(fore_value))
    valnode.appendChild(val)
    root.appendChild(tnode)
    root.appendChild(valnode)
  
    with open(filename, 'w') as f:
         f.write(doc.toprettyxml(indent='\t'))

def predict():
    cur_pvs=np.zeros([24,19],dtype=np.float32)
    paras=train.get_parameters("vapor_press_trained_parameters.npz")
#	pvs=get_a_line()
#	while True:
#		time.sleep(600000)
    line=get_a_line_fromxml("Book.xml")
    cur_pvs=np.delete(cur_pvs,-1,axis=0)
    cur_pvs=np.insert(cur_pvs,0,line,axis=0)
    cur_pvs_trans=cur_pvs.reshape(-1,1)
    pre_val=train.forward_propagation(cur_pvs_trans,paras)
    sess=tf.Session()
    val=sess.run(pre_val)
    val=val*50+50
    write_to_xml(val,"predict.xml")
    
    
if __name__ == '__main__':
     predict()
#     

