#!/usr/bin/env python
# coding: utf-8

# ## 导入所用的包
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import decomposition
# from IPython.core.interactiveshell import InteractiveShell

warnings.filterwarnings('ignore')
# InteractiveShell.ast_node_interactivity = "all"


# ## 数据的读取与预处理（此处采用归一化为min-max标准化）

# In[2]:


data = pd.read_csv(r"D:\Jupyter_Notebook\shujuwajue\TheThree\L1-train.csv")
print('初始状态下的df文件***********')
print(data)

load = data['LOAD']
load = load.values.reshape(-1,24)

hangtime = pd.date_range('2005-01-01','2010-09-30')

lietime =[]
for i in range(1,25):
    lietime.append(i)

df = pd.DataFrame(load,index=hangtime,columns=lietime)
print('构建N×24的特征矩阵的df文件***********')
print(df)


df_norm = (df - df.min()) / (df.max() - df.min())
print('使用min-max标准化后状态下的df文件***********')
print(df_norm)


# ## 相关关系分析

# In[3]:


# 相关关系图
sns.pairplot(df_norm)


# In[4]:


# 相关关系热力图
figure, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df_norm.corr(), square=True,linecolor='white',annot=True,cmap="coolwarm", ax=ax)


# ## 归一化数据可视化

# In[5]:


a = df_norm
t = np.arange(1, 25)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=100, figsize=[40, 40])
for num in range(0,a.shape[0]-1):
    
    plt.plot(t,np.array(a[num:num+1]).reshape(24,1))
        
plt.title('原始数据归一化可视图', fontsize=50)
plt.grid(True)
plt.xlabel('时间', fontsize=50)
plt.ylabel("数值", fontsize=50)
plt.xticks(t,fontsize=50)
plt.yticks(fontsize=50)
plt.show()


# ## 分别使用Silhouette系数，Calinski-Harabaz指数和Davies-Bouldin Index来评估模型
# 

# ### 搜索使用Kmean++的情况下较优k值

# In[6]:


scores1 = []
scores2 = []
scores3 = []
x = np.arange(2,200)
for i in x:
    model = KMeans(n_clusters=i,init='k-means++', random_state=520)
    yhat = model.fit_predict(df_norm)
    #new1查看各个类数量
    #print('当k='+ str(i) +'的时候分类及其各个分类数量')
    #print(np.unique(yhat,return_counts=True))
    labels = model.labels_
    score1 = metrics.silhouette_score(df_norm, labels, metric='euclidean')
    score2 = metrics.calinski_harabasz_score(df_norm, labels)
    score3 = metrics.davies_bouldin_score(df_norm, labels)  
    scores1.append(score1)
    scores2.append(score2)
    scores3.append(score3)
#scores1
#scores2
#scores3


# ### 可视化k值与各项指标的关系

# In[24]:


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[20, 15])
plt.plot(x, scores1)
plt.title("Silhouette系数与k取值关系曲线", fontsize=30)
plt.xlabel("k的取值", fontsize=30)
plt.ylabel("Silhouette系数得分", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[20, 15])
plt.plot(x, scores2)
plt.title("Calinski-Harabaz指数与k取值关系曲线", fontsize=30)
plt.xlabel("k的取值", fontsize=30)
plt.ylabel("Calinski-Harabaz指数", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[20, 15])
plt.plot(x, scores3)
plt.title("Davies-Bouldin Index与k取值关系曲线", fontsize=30)
plt.xlabel("k的取值", fontsize=30)
plt.ylabel("Davies-Bouldin Index得分", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### 查看使用三个指标情况下分类的情况

# In[25]:


model = KMeans(n_clusters=scores1.index(max(scores1))+3,init='k-means++', random_state=520)
yhat = model.fit_predict(df_norm)
    #new1查看各个类数量
print('当k='+ str(scores1.index(max(scores1))+3) +'的时候分类及其各个分类数量')
print(np.unique(yhat,return_counts=True))
labels1 = model.labels_

model = KMeans(n_clusters=scores2.index(max(scores2))+3,init='k-means++', random_state=520)
yhat = model.fit_predict(df_norm)
    #new1查看各个类数量
print('当k='+ str(scores2.index(max(scores2))+3) +'的时候分类及其各个分类数量')
print(np.unique(yhat,return_counts=True))
labels2 = model.labels_

model = KMeans(n_clusters=scores3.index(max(scores3))+3,init='k-means++', random_state=520)
yhat = model.fit_predict(df_norm)
    #new1查看各个类数量
print('当k='+ str(scores3.index(max(scores3))+3) +'的时候分类及其各个分类数量')
print(np.unique(yhat,return_counts=True))
labels3 = model.labels_



# ### 绘制模式图

# In[26]:


def draw(i,labels,df_norm):
    a = df_norm.loc[hangtime[labels==i]]
    t = np.arange(1, 25)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(dpi=100, figsize=[40, 40])
    for num in range(0,a.shape[0]-1):
    
        plt.plot(t,np.array(a[num:num+1]).reshape(24,1))
        
    plt.title("第"+str(i)+'类中数据模式曲线图', fontsize=50)
    plt.grid(True)
    plt.xlabel('时间', fontsize=50)
    plt.ylabel("数值", fontsize=50)
    plt.xticks(t,fontsize=50)
    plt.yticks(fontsize=50)
    plt.show()


# In[ ]:


for i in range(0,max(labels1)+1):
    draw(i,labels1,df_norm)
print('********************************') 

for i in range(0,max(labels2)+1):
    draw(i,labels2,df_norm)
print('********************************')    

for i in range(0,max(labels3)+1):
    draw(i,labels3,df_norm)
print('********************************')


# ### 可视化聚类随时间分布

# In[ ]:


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.scatter(hangtime, labels1)
plt.title("使用Silhouette系数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.scatter(hangtime, labels2)
plt.title("使用Calinski-Harabaz指数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[40, 30])
plt.scatter(hangtime, labels3)
plt.title("使用Davies-Bouldin Index分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()


# In[ ]:





# ## 搜索较优DBSCAN参数（可使用PCA降维之后的数据。。。）

# In[ ]:


res = []
epss = np.arange(0.001,0.51,0.005)
min_sampless = np.arange(2,100)
for eps in epss:
    for min_samples in min_sampless:
        dbscan = cluster.DBSCAN(eps = eps, min_samples = min_samples)
        # 模型拟合
        yhat = dbscan.fit(df_norm)
        # 统计各参数组合下的聚类个数（-1表示异常点）
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # 异常点的个数
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # 统计每个簇的样本个数
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        labels = dbscan.labels_
        if n_clusters>=2:
            score1 = metrics.silhouette_score(df_norm, labels, metric='euclidean')
            score2 = metrics.calinski_harabasz_score(df_norm, labels)
            score3 = metrics.davies_bouldin_score(df_norm, labels)  
        else:
            score1 = 0
            score2 = 0
            score3 = 0
            
        #score1 = metrics.silhouette_score(pcadf, labels, metric='euclidean')
        #score2 = metrics.calinski_harabasz_score(pcadf, labels)
        #score3 = metrics.davies_bouldin_score(pcadf, labels)  
        res.append({'eps':eps,
                    'min_samples':min_samples,
                    'n_clusters':n_clusters,
                    'score1':score1,
                    'score2':score2,
                    'score3':score3,
                    'outliners':outliners,
                    'stats':stats})
# 将迭代后的结果存储到数据框中        
df = pd.DataFrame(res)
#df

df = df.loc[df.n_clusters>=3,:]    
print(df)


# In[ ]:


a1 = df['score1'].max()
a2 = df['score2'].max()
a3 = df['score3'].max()
print('选用Silhouette系数来选择参数')
df.loc[df.score1 == a1, :]
print('选用Calinski-Harabaz指数来选择参数')
df.loc[df.score2 == a2, :]
print('选用Davies-Bouldin Index来选择参数')
df.loc[df.score3 == a3, :]


# In[ ]:


eps = [0.241,0.211,0.126] 
min_samples = [5,61,9]
name = ['Silhouette系数','Calinski-Harabaz指数','Davies-Bouldin Index']
for j in range(0,3):
    
    dbscan = cluster.DBSCAN(eps = eps[j], min_samples = min_samples[j])
    # 模型拟合
    # dbscan.fit(df_norm)
    yhat = dbscan.fit_predict(df_norm)
    #new1查看各个类数量
    print('当选用'+name[j] +'的时候分类及其各个分类数量')
    print(np.unique(yhat,return_counts=True))
    if j==0:
        labels1 = dbscan.labels_
    elif j==1:
        labels2 = dbscan.labels_
    else:
        labels3 = dbscan.labels_


# ### 可视化模式图

# In[ ]:


for i in range(0,max(labels1)+1):
    draw(i,labels1,df_norm)
    
for i in range(0,max(labels2)+1):
    draw(i,labels2,df_norm)
    
for i in range(0,max(labels3)+1):
    draw(i,labels3,df_norm)


# ### 可视化分类与时间关系图

# In[ ]:


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels1)
plt.title("使用Silhouette系数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels2)
plt.title("使用Calinski-Harabaz指数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[40, 30])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels3)
plt.title("使用Davies-Bouldin Index分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()


# In[ ]:





# ## 使用PCA降维之后的数据进行实验

# ### PCA降维

# In[ ]:


pca = decomposition.PCA()
pca.fit(df_norm)
print('24维数据经过PCA计算后的数值')
print(pca.explained_variance_)  
# 选择较为重要的8维数据
pca.n_components = 8
pcadf = pca.fit_transform(df_norm)
print('经过PCA降维得到的八维数据')
print(pcadf)


# In[ ]:


res = []
epss = np.arange(0.001,0.51,0.005)
min_sampless = np.arange(2,100)
for eps in epss:
    for min_samples in min_sampless:
        dbscan = cluster.DBSCAN(eps = eps, min_samples = min_samples)
        # 模型拟合
        yhat = dbscan.fit(pcadf)
        # 统计各参数组合下的聚类个数（-1表示异常点）
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # 异常点的个数
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # 统计每个簇的样本个数
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        labels = dbscan.labels_
        if n_clusters>=2:
            score1 = metrics.silhouette_score(pcadf, labels, metric='euclidean')
            score2 = metrics.calinski_harabasz_score(pcadf, labels)
            score3 = metrics.davies_bouldin_score(pcadf, labels)  
        else:
            score1 = 0
            score2 = 0
            score3 = 0
            
        #score1 = metrics.silhouette_score(pcadf, labels, metric='euclidean')
        #score2 = metrics.calinski_harabasz_score(pcadf, labels)
        #score3 = metrics.davies_bouldin_score(pcadf, labels)  
        res.append({'eps':eps,
                    'min_samples':min_samples,
                    'n_clusters':n_clusters,
                    'score1':score1,
                    'score2':score2,
                    'score3':score3,
                    'outliners':outliners,
                    'stats':stats})
# 将迭代后的结果存储到数据框中        
df2 = pd.DataFrame(res)
print(df2)

# df2 = df2.loc[df.n_clusters>=3,:]    
# df2


# In[ ]:


df2 = df2.loc[df2.n_clusters>=2,:]    
print(df2)


# In[ ]:


a1 = df2['score1'].max()
a2 = df2['score2'].max()
a3 = df2['score3'].max()
print('选用Silhouette系数来选择参数')
df2.loc[df2.score1 == a1, :]
print('选用Calinski-Harabaz指数来选择参数')
df2.loc[df2.score2 == a2, :]
print('选用Davies-Bouldin Index来选择参数')
df2.loc[df2.score3 == a3, :]


# In[ ]:


eps = [0.311,0.241,0.126] 
min_samples = [10,94,10]
name = ['Silhouette系数','Calinski-Harabaz指数','Davies-Bouldin Index']
for j in range(0,3):
    
    dbscan = cluster.DBSCAN(eps = eps[j], min_samples = min_samples[j])
    # 模型拟合
    # dbscan.fit(df_norm)
    yhat = dbscan.fit_predict(pcadf)
    #new1查看各个类数量
    print('当选用'+name[j] +'的时候分类及其各个分类数量')
    print(np.unique(yhat,return_counts=True))
    if j==0:
        labels1 = dbscan.labels_
    elif j==1:
        labels2 = dbscan.labels_
    else:
        labels3 = dbscan.labels_


# ### 可视化模式图

# In[ ]:


for i in range(0,max(labels1)+1):
    draw(i,labels1,df_norm)
    
for i in range(0,max(labels2)+1):
    draw(i,labels2,df_norm)
    
for i in range(0,max(labels3)+1):
    draw(i,labels3,df_norm)


# In[ ]:


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels1)
plt.title("使用Silhouette系数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[30, 15])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels2)
plt.title("使用Calinski-Harabaz指数分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(dpi=200, figsize=[40, 30])
plt.rcParams['axes.unicode_minus']=False 
plt.scatter(hangtime, labels3)
plt.title("使用Davies-Bouldin Index分类关系图", fontsize=50)
plt.xlabel("时间", fontsize=50)
plt.ylabel("分类结果", fontsize=50)
plt.xticks(rotation=90)
plt.grid(True)
plt.xticks(pd.date_range('2005-01-01','2010-09-30',freq='1m'),fontsize=30)
plt.yticks(fontsize=50)
plt.show()


# In[ ]:





# In[ ]:




