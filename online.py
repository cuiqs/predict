from xml.dom.minidom import Document

import train
import numpy as np
import time
import tensorflow as tf
import gettrain_x
import xml


vapor_press_pvs=gettrain_x.vapor_press_pvs

def get_a_line():
    t=np.load("vapor_press_test.npz")
    pvs=t["arr_0"].reshape(-1,19)
    for item in pvs:
        yield item


def get_a_line_fromxml(file):
    maxrange = gettrain_x.maxrange
    DOMTree = xml.dom.minidom.parse(file)
    dom_pvs = DOMTree.documentElement
    pvlist = []
    for pv in vapor_press_pvs:
        one_pv = dom_pvs.getElementsByTagName(pv)

        pvlist.append(one_pv[0].firstChild.data)
    #        val=float(one_pv[0].firstChild.data)
    #        val=val.astype(np.float32)
    #        line[0,index]=
    pvarr = np.array(pvlist)
    line = pvarr.astype(dtype=np.float32)
    for i in range(len(pvlist)):
        item = vapor_press_pvs[i]
        line[i] = (line[i] - maxrange[item][0]) / maxrange[item][1]
    line = line.reshape(1, len(vapor_press_pvs))
    return line

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

#把预测值写入xml文件
def write_to_xml(val_dict, filename):
    doc = Document()
    root = doc.createElement("prediction")
    doc.appendChild(root)

    tnode = doc.createElement("time")
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    tnode.setAttribute("time",cur_time)


    for item in val_dict.keys():
        snode=doc.createElement(item)
        stnode=doc.createTextNode(val_dict[str(item)])
        snode.appendChild(stnode)
        tnode.appendChild(snode)

    root.appendChild(tnode)


    with open(filename, 'w') as f:
        f.write(doc.toprettyxml(indent='\t'))

        
def predict2(parameters,cur_pvs,line,scale,base):
    """
    if sample=="lpgc5":
        cur_pvs=np.load("lpgc5_pvs.npy")
        paras=train.get_parameters("lpgc5_trained_parameters.npz")
        line=get_a_line_fromxml("book.xml")
        base=0
        scale=4
    if sample=="vapor":
        cur_pvs=np.load("vapor_press_pvs.npz")  
        paras=train.get_parameters("vapor_press_trained_parameters.npz")
        line=get_a_line_fromxml("book.xml")
        base=50
        scale=50
     """
    cur_pvs=np.delete(cur_pvs,-1,axis=0)
    cur_pvs=np.insert(cur_pvs,0,line,axis=0)
    cur_pvs_trans=cur_pvs.reshape(-1,1)
    pre_val=train.forward_propagation(cur_pvs_trans,parameters)
    sess=tf.Session()
    val=sess.run(pre_val)

       
    val=val*scale+base
    return val
  #  write_to_xml(val,"predict.xml")


if __name__=="__main__":

    val_dict={}
    #print(len(vapor_press_pvs))
    pa_vapor=train.get_parameters("vapor_press_trained_parameters.npz")
    cur_pvs=np.load("pvs.npy")
    line=get_a_line_fromxml("book.xml")

    scale_vapor=gettrain_x.sample_span["stagaspresh"]-gettrain_x.sample_span["stagaspresl"]
    base_vapor=gettrain_x.sample_span["stagaspresl"]
    vapor_predict_val=predict2(pa_vapor,cur_pvs,line,scale_vapor,base_vapor);
    val_dict["vapor_press"]="{:.3f}".format(vapor_predict_val[0][0])

    pa_lpg=train.get_parameters("lpgc51801-1906_head2_trained_parameters.npz")
    scale_lpgc5 = gettrain_x.sample_span["lpgc5h"]-gettrain_x.sample_span["lpgc5l"]
    base_lpgc5 = gettrain_x.sample_span["lpgc5l"]
    lpg_predict_value=predict2(pa_lpg,cur_pvs,line,scale_lpgc5,base_lpgc5);
    if lpg_predict_value[0][0]<0:
        lpg_predict_value[0][0]=0.000

    val_dict["lpgc5"]="{:.3f}".format(lpg_predict_value[0][0])

    write_to_xml(val_dict,"predict.xml")


