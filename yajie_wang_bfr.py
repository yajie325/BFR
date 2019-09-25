import sys
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
import  time
from sklearn.metrics.cluster import normalized_mutual_info_score
#读数据
start = time.time()

path = sys.argv[1]
s = int(sys.argv[2])
output_f = sys.argv[3]

raw_data = []
f = open(path)
for v in f:
    raw_data.append(v.split(','))
f.close()
raw_data1 = np.array(raw_data)

raw_data  =raw_data1.astype(float)
np.random.shuffle(raw_data)
d = float((np.sqrt(int(np.size(raw_data,1))-2))*2)
d1 = int(np.size(raw_data, 1))-2 #列数

len_data = int(len(raw_data))
n1 = round(len_data * 0.2)
#print(n1)
n_star = n1
n_end = n_star + n1
# print(n)
a = np.arange(raw_data.shape[0])
#np.random.shuffle(a)
random_data = raw_data[0:n_star]  # 随机选择前%20
rest_data = raw_data[n_star:-1]  # 剩下的后80%
n_sample = len(raw_data)
k = s

#.................................................................初始化字典函数
def bulild_rawdata_dic(label_pred, random_data, raw_data_dic):
    # 建立字典label对应的cluster的点
    for i in range(len(label_pred)):
        if label_pred[i] in raw_data_dic:
            if isinstance(raw_data_dic[label_pred[i]], list):
                thelist = raw_data_dic[label_pred[i]]
                thelist.append(random_data[i])
                raw_data_dic[label_pred[i]] = thelist
            else:
                thelist = raw_data_dic[label_pred[i]]
                raw_data_dic[label_pred[i]] = list()
                raw_data_dic[label_pred[i]].append(thelist)
                raw_data_dic[label_pred[i]].append(random_data[i])
        else:
            raw_data_dic[label_pred[i]] = random_data[i]

#.stap2 run kmeans...........................................

km = KMeans(n_clusters=10*k, random_state=0)
kmeans = km.fit(random_data[:, 2:])

label_pred1 = kmeans.labels_  # 获取聚类标签

dic_counts1 = defaultdict(int)  # 建立标签和个数的字典
for key in label_pred1:
    dic_counts1[key] += 1
count1=[] #记录单个点的
mul_count1=[]#记录多个点的
for i in dic_counts1:
    if dic_counts1[i] == 1:
        count1.append(i)
    else:
        mul_count1.append(i)
'''raw_data={}
bulild_rawdata_dic(label_pred1, random_data, raw_data)
rs = []
for i in count1:
   rs.append(raw_data[i])#rs only keep dot
still_use_dot=[]
for j in mul_count1:
 still_use_dot.append(raw_data[j])# use to build ds'''
count =Counter(label_pred1)
rs_index=[]
left_index=[]
for i in range(n1):
    if count[label_pred1[i]]==1:
        rs_index.append(i)
    else:
        left_index.append(i)
rs=[]
for i in rs_index:
    rs.append(random_data[i])

still_use_dot=[]
for i in left_index:
    still_use_dot.append(random_data[i])


#.step4. run keams to generate ds....................................

km = KMeans(n_clusters=k, random_state=0)
kmeans = km.fit(np.array(still_use_dot)[:, 2:])

label_pred2 = kmeans.labels_

dic_counts2 = defaultdict(int)  # 建立标签和个数的字典
for key in label_pred2:
    dic_counts2[key] += 1
count2=[] #记录单个点的
mul_count2=[]#记录多个点的

for i in dic_counts2:
    if dic_counts2[i] == 1:
        count2.append(i)
    else:
        mul_count2.append(i)

raw_data_dic={}
bulild_rawdata_dic(label_pred2,still_use_dot,raw_data_dic)

p_ds_data_dic = {}  # 找到初始化的ds 第四步存放点
for i in mul_count2:
    p_ds_data_dic[i] = raw_data_dic[i]

#print('多个点',p_ds_data_dic)
#print('初始数据',raw_data_dic)
#print('单个点格式',count2)


#算多个点的数据ds基础数据
ds_data_dic = defaultdict(list)#初始化需要的数据
for i in p_ds_data_dic:
    n = len(p_ds_data_dic[i])
    s = np.array([0] * d1, dtype='float64')
    sq = np.array([0] * d1,dtype='float64')
    for p in p_ds_data_dic[i]:
        s += p[2:]
        sq += (p[2:]) ** 2
    ds_data_dic[i].append(len(p_ds_data_dic[i]))
    ds_data_dic[i].append(s)
    ds_data_dic[i].append(sq)
#print('算好的数据',ds_data_dic)
#...............................................step6 算cs

k_cs =round(len(rs)*0.8)


km = KMeans(n_clusters=k_cs, random_state=0)
kmeans = km.fit(np.array(rs)[:, 2:])

label_pred3 = kmeans.labels_  # 获取聚类标签

dic_counts3 = defaultdict(int)  # 建立标签和个数的字典
for key in label_pred3:
    dic_counts3[key] += 1
count3=[] #记录单个点的
mul_count3=[]#记录多个点的

for i in dic_counts3:
    if dic_counts3[i] == 1:
        count3.append(i)
    else:
        mul_count3.append(i)

raw_data_dic1={}
bulild_rawdata_dic(label_pred3, rs, raw_data_dic1)

p_cs_list=[]
for i in mul_count3:
    p_cs_list.append(raw_data_dic1[i])
#print('cs的初始',p_cs_list)

#算多个点的数据rs基础数据
cs_data_dic = defaultdict(list)#初始化需要的数据
for i in mul_count3:
    n = len(raw_data_dic1[i])
    s = np.array([0] * d1, dtype='float64')
    sq = np.array([0] * d1,dtype='float64')
    for p in raw_data_dic1[i]:
        s += p[2:]
        sq += (p[2:]) ** 2
    cs_data_dic[i].append(len(raw_data_dic1[i]))
    cs_data_dic[i].append(s)
    cs_data_dic[i].append(sq)

cs_data_list =[]
for i in cs_data_dic:
    cs_data_list.append(cs_data_dic[i])
rs_new=[]
for i in count3:
    rs_new.append(raw_data_dic1[i])

rs = np.array(rs_new)

output1=[]
output1.append([sum(len(p_ds_data_dic[k]) for k in p_ds_data_dic),len(p_cs_list),\
        sum(len(p_cs_list[k]) for k in range(len(p_cs_list))),len(rs)])

#print('最新rs',rs)
#......................................step 7-10
while n_star<len_data:
    if n_end>len_data:
        break
    else:
        circle_data = raw_data[n_star:n_end]#循环的点
        #print('起始点',n_star,n_end,len_data)
        new_rs =[]
        for i in range(len(circle_data)): #for i in range(len(circle_data)):
            test_dot = circle_data[i][2:]
            ds_centroid=np.array([np.divide(ds_data_dic[k][1], ds_data_dic[k][0]) for k in ds_data_dic])#ds中心点
            ds_sumsq=np.array([np.divide(ds_data_dic[k][2], ds_data_dic[k][0]) for k in ds_data_dic])
            ds_standard =np.sqrt(ds_sumsq - np.square(ds_centroid))

            ma=np.sqrt(np.square(np.array(test_dot - ds_centroid) / ds_standard).sum(axis=1))
            min_md_dis = np.min(ma)
            if min_md_dis<d:
                best_key = np.argmin(ma)
                p_ds_data_dic[mul_count2[best_key]].append(circle_data[i])
                add_new_sum = [1,test_dot,np.square(test_dot)]
                ds_data_dic[mul_count2[best_key]]=[ds_data_dic[mul_count2[best_key]][i]+add_new_sum[i] for i in range(3)]
            else:
                cs_centroid = np.array([np.divide(cs_data_list[k][1], cs_data_list[k][0]) for k in range (len(cs_data_list))])  # ds中心点
                cs_sumsq = np.array([np.divide(cs_data_list[k][2], cs_data_list[k][0]) for k in range (len(cs_data_list))])

                cs_standard = np.sqrt(cs_sumsq - np.square(cs_centroid))
                ma_cs = np.sqrt(np.square(np.array(test_dot - cs_centroid) / cs_standard).sum(axis=1))
                min_cs_md_dis = np.min(ma_cs)
                if min_cs_md_dis<d:
                    best_key = np.argmin(ma_cs)
                    p_cs_list[best_key].append(circle_data[i])
                    add_new_sum = [1, test_dot, np.square(test_dot)]
                    cs_data_list[best_key] = [cs_data_list[best_key][i] + add_new_sum[i] for i in range(3)]

                else:
                    #rs = np.vstack((rs, circle_data[i]))
                    new_rs.append(circle_data[i])
                   # print('................newrs',new_rs)

        rs = np.vstack((rs, new_rs))

        k_cs = round(len(rs) * 0.8)

        km = KMeans(n_clusters=k_cs, random_state=0)  # kmean算法
        # print('更新后',rs_raw)
        kmeans = km.fit(rs[:, 2:])
        label_pred4 = kmeans.labels_  # 获取cs标签
        # print('rs的label',label_pred3)
        dic_counts4 = defaultdict(int)  # 建立标签和个数的字典
        for key in label_pred4:
            dic_counts4[key] += 1

        count4 = []
        mul_count4 = []

        # find rc
        for i in dic_counts4:
            if dic_counts4[i] == 1:
                count4.append(i)
            else:
                mul_count4.append(i)
        # print('geshu',mul_count4)

        raw_cs_data_dic1 = {}
        bulild_rawdata_dic(label_pred4, rs, raw_cs_data_dic1)

        rs_data_dic4 = {}  # 初始化的rs新的rs
        for i in count4:
            rs_data_dic4[i] = raw_cs_data_dic1[i]

        rs = []  # 先把rs里面数据取出来
        for i in count4:
            rs.append(raw_cs_data_dic1[i])

        p_cs_data_list4 = []  # ..............多个点的合并为初始化的cs
        for i in mul_count4:
            p_cs_data_list4.append(raw_cs_data_dic1[i])



        # ...............................................对cs也要计算一遍距离

        cs_data_dic1 = defaultdict(list)  # 找到的cs的字典存储了
        for i in mul_count4:
            n = len(raw_cs_data_dic1[i])
            s = np.array([0] * d1, dtype='float64')
            sq = np.array([0] * d1, dtype='float64')
            for p in raw_cs_data_dic1[i]:
                s += p[2:]
                sq += (p[2:]) ** 2
            cs_data_dic1[i].append(len(raw_cs_data_dic1[i]))
            cs_data_dic1[i].append(s)
            cs_data_dic1[i].append(sq)

        cs_data_list1 = []
        for i in cs_data_dic1:
            cs_data_list1.append(cs_data_dic1[i])
        # print('cs最开始初始数据........',cs_data_dic)

        cs_data_list = cs_data_list + cs_data_list1
        p_cs_list = p_cs_list + p_cs_data_list4

        n_star = n_end
        n_end = n_star + n1


        output1.append([sum(len(p_ds_data_dic[k]) for k in p_ds_data_dic),len(p_cs_list),\
        sum(len(p_cs_list[k]) for k in range(len(p_cs_list))),len(rs)])

#..............................................cs合并ds


cs_dot = np.array([cs_data_list[i][1] / cs_data_list[i][0] for i in range(len(cs_data_list))])
for i_m in range(len(cs_dot)):  # for i in range(len(circle_data)):
    p = cs_dot[i_m]
    ds_centroid = np.array([np.divide(ds_data_dic[k][1], ds_data_dic[k][0]) for k in ds_data_dic])  # ds中心点
    ds_sumsq = np.array([np.divide(ds_data_dic[k][2], ds_data_dic[k][0]) for k in ds_data_dic])
    ds_standard = np.sqrt(ds_sumsq - np.square(ds_centroid))
    ma = np.sqrt(np.square((p - ds_centroid) / ds_standard).sum(axis=1))
    min_md_dis = np.min(ma)
    if min_md_dis < d:
        best_key = np.argmin(ma)
        p_ds_data_dic[mul_count2[best_key]]+p_cs_list[i_m]
        ds_data_dic[mul_count2[best_key]] = [ds_data_dic[mul_count2[best_key]][i] + cs_data_list[i_m][i] for i in range(3)]

    else:
        rs=np.vstack((rs,p_cs_list[i_m]))
        # print(i_cs[i_m])
        #for v in range(len(p_cs_list[i_m])):
            #rs.append(p_cs_list[i_m][v])

output1.append([sum(len(p_ds_data_dic[k]) for k in p_ds_data_dic),0,0,len(rs)])
#print(rs)

#print(len(rs))
#rs = np.array(rs).astype(str)
#p_ds_data_dic=np.array(p_ds_data_dic).astype(str)


n_rs = len(rs)
lable_rs =[-1]*n_rs
result = np.c_[lable_rs,rs]
#print(result.shape)
for i in range(k):
    ds_n=len(p_ds_data_dic[i])
    lable = [i]*ds_n
    #f_result1 = np.c_[lable_ds,np.array(p_ds_data_dic[i])]
    result =np.vstack((result,np.c_[lable,np.array(p_ds_data_dic[i])]))
#print(result)

f_output =result[:,[1,0]]
#print(f_output.shape)
#sort_index=f_output[f_output[:,0].np.all.argsort()]
sort_index = np.argsort((f_output[:,0]))
f_output = f_output[sort_index].tolist()

#print(f_result)
true= result[:,2].tolist()
pred =result[:,0].tolist()
#print(true)
#print(output1)
#print(normalized_mutual_info_score(true,pred))
#print(output1)
#print(len(output1))
output = open(output_f ,'w')
output.write('The intermediate results:'+'\n')
for i in range (len(output1)):
    write ='Round {number}: {n_ds},{c_cs},{cs},{n_rs}'\
    .format(number = i+1,n_ds=output1[i][0],c_cs =output1[i][1],cs =output1[i][2],n_rs=output1[i][3])
    output.write(write+'\n')
output.write('\n'+'The clustering results'+'\n')
for i in f_output:
    output.write('{a},{b}'.format(a=int(i[0]),b=int(i[1]))+'\n')
output.closed


#end =time.time()
#print(end-start)


