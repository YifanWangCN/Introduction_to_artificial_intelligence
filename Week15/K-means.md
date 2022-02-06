# K-means算法简单介绍
其实这个算法很容易理解，即在各个特征维度上找到最能代表某个类别的中心点的方式。

其中该算法的损失函数定义为各个样本距离所属簇中心点的误差平方的和，即欧式距离。

该算法的步骤如下所示：
###
    1：初始化，即随机先选取K个簇的中心点。
    2：计算每个样本到各个中心点的距离，将距离最小的样本归类到其对应的簇里。
    3：重新计算中心点，根据已有的簇里的样本，计算其到其对应中心点的平均值（各个维度分别计算）
    4：进行迭代，直到收敛

具体代码如下所示
###
    def kmeans(k, X, iters=10):
    
    # 首先对样本个数进行统计
    m = X.shape[0]  
    
    # 损失函数
    Cost = []

    # 接下来先要随机设置每个clusters的中心点center的位置，即对中心点做初始化
    centers = {}
    for index, i in enumerate(random.sample(range(m),k)):
      centers[index] = X[i]

    # 接下来就是对数据进行迭代
    for i in range(iters):

      #对簇进行初始化，一共有K个簇,每个簇都初始化一个空列表
      clusters = {}
      for j in range(k):
        clusters[j] = [] 

      #记录损失函数
      loss = 0

      # 首先计算每一个样本到每个簇的中心点的距离,选择距离最小的值，并将该样本分配给对应的中心点所在的簇
      for sample in X:

        #用来存储每一个点到各簇中心点的距离，并返回最小值的索引
        distances = []  

        for center_point in centers:
          distances.append(Euclidean_distance(sample,centers[center_point]))

        #遍历查询每一个样本距离各个簇中心距离中最小的点
        min_index = distances.index(min(distances))
        
        #遍历每一个点的损失，并累加,损失函数是 sum of squared distances
        loss = loss + (min(distances)**2) 

        #该索引对应的其实就是簇的索引，将该样本输入至该簇
        clusters[min_index].append(sample)
      
      # 记录损失函数
      Cost.append(loss)

      # 更新每一个簇的center
      for c in range(k):
        centers[c] = np.mean(clusters[c],axis=0)

    # return the centres and the labels.
    return centers, clusters, Cost