import csv
from datetime import datetime,timedelta
import time
import numpy as np


fcc3={}

files=["FCC3_FI1308A_PV.csv","FCC3_FI1314_PV.csv","FCC3_FI1822_PV.csv","FCC3_FIC1105_PV.csv","FCC3_FIC1116_PV.csv","FCC3_FIC1201_PV.csv","FCC3_FIC1204_PV.csv","FCC3_FIC1205_PV.csv","FCC3_FIC1206_PV.csv","FCC3_FIC1214_PV.csv","FCC3_FIC1219_PV.csv","FCC3_FIC1227_PV.csv","FCC3_FIC1301_PV.csv","FCC3_FIC1309_PV.csv","FCC3_FIC1835_PV.csv","FCC3_PI1112_PV.csv","FCC3_PI1201_PV.csv","FCC3_PI1305_PV.csv","FCC3_PI1306_PV.csv","FCC3_PIC1301_PV.csv","FCC3_PIC1303_PV.csv","FCC3_TI1219_PV.csv","FCC3_TI1231_PV.csv","FCC3_TI1305_PV.csv","FCC3_TI1313_PV.csv","FCC3_TI1315_PV.csv","FCC3_TI1324_PV.csv","FCC3_TI1325A_PV.csv","FCC3_TI1326B_PV.csv","FCC3_TIC1201_PV.csv","FCC3_TIC1203_PV.csv","FCC3_TIC1217_PV.csv","FCC3_TIC1301_PV.csv"]

maxrange={"FCC3_FI1308A_PV":[80,70],"FCC3_FI1314_PV":[0,35000],"FCC3_FI1822_PV":[40,60],"FCC3_FIC1105_PV":[5,5],"FCC3_FIC1116_PV":[0,6],"FCC3_FIC1201_PV":[0,250],"FCC3_FIC1204_PV":[0,40],"FCC3_FIC1205_PV":[0,500],"FCC3_FIC1206_PV":[0,300],"FCC3_FIC1214_PV":[0,100],"FCC3_FIC1219_PV":[0,50],"FCC3_FIC1227_PV":[0,30],"FCC3_FIC1301_PV":[0,80],"FCC3_FIC1309_PV":[0,45],"FCC3_FIC1835_PV":[0,75],"FCC3_PI1112_PV":[100,200],"FCC3_PI1201_PV":[50,200],"FCC3_PI1305_PV":[0.5,1],"FCC3_PI1306_PV":[0.5,1],"FCC3_PIC1301_PV":[0.5,1],"FCC3_PIC1303_PV":[0.5,1],"FCC3_TI1219_PV":[50,100],"FCC3_TI1231_PV":[150,100],"FCC3_TI1305_PV":[20,50],"FCC3_TI1313_PV":[20,50],"FCC3_TI1315_PV":[50,100],"FCC3_TI1324_PV":[100,100],"FCC3_TI1325A_PV":[30,50],"FCC3_TI1326B_PV":[100,100],"FCC3_TIC1201_PV":[80,70],"FCC3_TIC1203_PV":[180,100],"FCC3_TIC1217_PV":[280,100],"FCC3_TIC1301_PV":[100,80]}

pvs_mean_var={"FCC3_FI1308A_PV":[80,70],"FCC3_FI1314_PV":[0,35000],"FCC3_FI1822_PV":[40,60],"FCC3_FIC1105_PV":[5,5],"FCC3_FIC1116_PV":[0,6],"FCC3_FIC1201_PV":[0,250],"FCC3_FIC1204_PV":[0,40],"FCC3_FIC1205_PV":[0,500],"FCC3_FIC1206_PV":[0,300],"FCC3_FIC1214_PV":[0,100],"FCC3_FIC1219_PV":[0,50],"FCC3_FIC1227_PV":[0,30],"FCC3_FIC1301_PV":[0,80],"FCC3_FIC1309_PV":[0,45],"FCC3_FIC1835_PV":[0,75],"FCC3_PI1112_PV":[100,200],"FCC3_PI1201_PV":[50,200],"FCC3_PI1305_PV":[0.5,1],"FCC3_PI1306_PV":[0.5,1],"FCC3_PIC1301_PV":[0.5,1],"FCC3_PIC1303_PV":[0.5,1],"FCC3_TI1219_PV":[50,100],"FCC3_TI1231_PV":[150,100],"FCC3_TI1305_PV":[20,50],"FCC3_TI1313_PV":[20,50],"FCC3_TI1315_PV":[50,100],"FCC3_TI1324_PV":[100,100],"FCC3_TI1325A_PV":[30,50],"FCC3_TI1326B_PV":[100,100],"FCC3_TIC1201_PV":[80,70],"FCC3_TIC1203_PV":[180,100],"FCC3_TIC1217_PV":[280,100],"FCC3_TIC1301_PV":[100,80]}
#vapor_press_pvs=["FCC3_FI1308A_PV","FCC3_FI1314_PV","FCC3_FIC1201_PV","FCC3_FIC1214_PV","FCC3_FIC1301_PV","FCC3_FIC1309_PV","FCC3_PI1201_PV","FCC3_PI1305_PV","FCC3_PI1306_PV","FCC3_PIC1301_PV","FCC3_PIC1303_PV","FCC3_TI1305_PV","FCC3_TI1313_PV","FCC3_TI1315_PV","FCC3_TI1324_PV","FCC3_TI1325A_PV","FCC3_TI1326B_PV","FCC3_TIC1201_PV","FCC3_TIC1301_PV"]
vapor_press_pvs=["FCC3_FI1308A_PV","FCC3_FI1314_PV","FCC3_FIC1201_PV","FCC3_FIC1214_PV","FCC3_FIC1301_PV","FCC3_FIC1309_PV","FCC3_PI1201_PV","FCC3_PI1305_PV","FCC3_PI1306_PV","FCC3_PIC1301_PV","FCC3_PIC1303_PV","FCC3_TI1305_PV","FCC3_TI1313_PV","FCC3_TI1315_PV","FCC3_TI1324_PV","FCC3_TI1325A_PV","FCC3_TI1326B_PV","FCC3_TIC1201_PV","FCC3_TIC1301_PV"]
diesel_pvs=["FCC3_FI1822_PV","FCC3_FIC1105_PV","FCC3_FIC1116_PV","FCC3_FIC1201_PV","FCC3_FIC1204_PV","FCC3_FIC1205_PV","FCC3_FIC1206_PV","FCC3_FIC1214_PV","FCC3_FIC1219_PV","FCC3_FIC1227_PV","FCC3_FIC1835_PV","FCC3_PI1112_PV","FCC3_PI1201_PV","FCC3_TI1219_PV","FCC3_TI1231_PV","FCC3_TIC1201_PV","FCC3_TIC1203_PV","FCC3_TIC1217_PV"]
pvs=list(maxrange.keys())
pvs.sort()
vapor_press_pvs.sort()
filename=["FCC3_FI1308A_PV.csv"]
#定义取PV时间间隔，及时间跨度，单位是分钟
tmdelay=timedelta(minutes=2)
tmspan=timedelta(minutes=120)
tmhead=timedelta(minutes=30)

sample_span={"dieselconl":-30,"dieselconh":30,"stagasdryl":170,"stagasdryh":210,"stagaspresl":40,"stagaspresh":80,"lpgc2l":0,"lpgc2h":6,"lpgc5l":0,"lpgc5h":4}

#dt_from=datetime.strptime("2018-09-27 00:00","%Y-%m-%d %H:%M")
#dt_end=datetime.strptime("2019-06-05 00:00","%Y-%m-%d %H:%M")

dt_from=datetime.strptime("2019-07-01 00:00","%Y-%m-%d %H:%M")
dt_end=datetime.strptime("2019-07-30 00:00","%Y-%m-%d %H:%M")


def addfile(filename):
	mdelay=timedelta(minutes=1)
	n=2
	with open(filename) as csvfile:
		csv_reader=csv.reader(csvfile)
		header=next(csv_reader)
		last=next(csv_reader)
		if not ":" in last[1]:
			last[1]=last[1]+" 00:00"
		for row in csv_reader:
			dt_l=datetime.strptime(last[1],"%Y/%m/%d %H:%M")
			if not ":" in row[1]:
				row[1]=row[1]+" 00:00"
			dt_n=datetime.strptime(row[1],"%Y/%m/%d %H:%M")
			if dt_l>dt_from and dt_l<dt_end:
				if dt_n==dt_l+mdelay:
					try:
						fcc3[dt_n].append(float(row[2]))
					except:
						fcc3[dt_n].append(float(last[2]))
						row[2]=last[2]
				else:
					while dt_n>dt_l+mdelay:
						try:
							fcc3[dt_l+mdelay].append(float(last[2]))
							dt_l+=mdelay
						except:
							raise
					try:
						fcc3[dt_n].append(float(row[2]))
					except:
						fcc3[dt_n].append(float(last[2]))
						row[2]=last[2]
			n+=1
			last=row

#产生时间序列及空列表
def timeout():
	mdelay=timedelta(minutes=1)

	curtime=dt_from
	while curtime<=dt_end:
		fcc3[curtime]=[]	
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

#计算pv值各自的平均值及方差
def get_pvs_mean_var(from_f):
	count=len(maxrange)
	table=[]
	with open(from_f,"r+") as csvrd_file:
		csv_reader=csv.reader(csvrd_file)
		for row in csv_reader:
			line=[]
			for i in range(1,count+1):
				line.append(float(row[i]))
			table.append(line)
	
	pvs_np=np.asarray(table,dtype=np.float32)
	pvs_mean=np.mean(pvs_np,0)
	print(pvs_mean.shape)
	pvs_var=np.var(pvs_np,0)
	print(pvs_var)
	for i in range(count):
		pvs_mean_var[pvs[i]][0]=pvs_mean[i]
		pvs_mean_var[pvs[i]][1]=pvs_var[i]
	
	print(pvs_mean_var)

#用PV值除以各自量程，使得数据都小于1
def uniform(from_f,to_f):
	count=len(maxrange)
	with open(from_f,"r+") as csvrd_file:
		with open(to_f,"w") as csvwr_file:
			csv_writer=csv.writer(csvwr_file)
			csv_reader=csv.reader(csvrd_file)
			for row in csv_reader:
#				print(row)
				for i in range(1,count+1):
					row[i]=str((float(row[i])-pvs_mean_var[pvs[i-1]][0])/pvs_mean_var[pvs[i-1]][1])#测量值比量程，使数据都<1
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

#	print(fcc3_pv.keys())
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
	if sam=="vapor_press":
		saml=sample_span["stagaspresl"]
		samspan=sample_span["stagaspresh"]-saml
	if sam=="lpgc5":
		saml=sample_span["lpgc5l"]
		samspan=sample_span["lpgc5h"]-saml
	with open(samfile) as csv_file:
		csv_reader=csv.reader(csv_file)
		for row in csv_reader:
			print(row)
			if len(row[0])>16:
				tm_sam=datetime.strptime(row[0][0:-3],"%Y/%m/%d %H:%M")
			else:
				tm_sam=datetime.strptime(row[0],"%Y/%m/%d %H:%M")
			samples[tm_sam]=(float(row[1])-saml)/samspan
			print(samples[tm_sam])
	return samples

#生产数据文件与化验文件联系到一起
def sample_add_pv(pvfile,c_pvs,samplefile,sample):
	samples=get_sample(samplefile,sample)
	fcc3_pvs=get_pv(pvfile,c_pvs)

	sam_num=len(samples)
	na_sam=np.empty([sam_num],np.float32)
	na_pv=np.empty([sam_num,int(tmspan/tmdelay),len(c_pvs)],np.float32)

	i=0
	for end_tm,val in samples.items():
		na_sam[i]=val
		tm=end_tm-tmhead	#采样时间提前1小时
		begin_tm=end_tm-tmspan-tmhead
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
#	np.savez("stagas_sam_pv81",na_sam,na_pv)
	return na_sam,na_pv	

#从npz文件中提取出生产数据及化验数据，按照8：2的比例分成训练集和测试集
def random_and_classes(npzfile,train_set_file,test_set_file):
	r=np.load(npzfile)
#	print(r.files)
	pvs=r["arr_0"]
	sams=r["arr_1"]
	np.random.seed(2)
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

#timeout()
#gather("produce730d.csv")
#uniform("produce730.csv","produce730u.csv")
#get_sample("chaiyou.csv","chaiyou")

#sample_add_pv("produce730du.csv","chaiyou.csv",diesel_pvs,"stagas")
#na_sams,na_pvs=sample_add_pv("produce730u.csv",vapor_press_pvs,"vapor19-07.csv","vapor_press")
#np.savez("vapor_press_1907_tmspan2.npz",na_pvs,na_sams)
#random_and_classes("vapor_press_1801_1906_tmspan2.npz","vapor_press_1801_1907_tmspan2_train.npz","vapor_press_1801_1907_tmspan2_test.npz")

#改变数据预处理方法
"""
get_pvs_mean_var("produce1810-1906.csv")
uniform("produce1810-1906.csv","produce1810-1906_mean_var.csv")
na_sams,na_pvs=sample_add_pv("produce1810-1906_mean_var.csv",diesel_pvs,"chaiyou.csv","chaiyou")
np.savez("diesel_mean_var.npz",na_pvs,na_sams)
random_and_classes("diesel_mean_var.npz","diesel_mean_var_train.npz","diesel_mean_var_test.npz")
"""
