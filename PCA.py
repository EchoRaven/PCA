import numpy as np

class PCA:
    def __init__(self):
        self.data = []
        self.avg = []
        #转换矩阵
        self.Transformer = None
        #数据维度
        self.dim = 0

    def Train(self, data = [], threshold = 0.99):
        self.data = data
        #对于传入的数据要做去中心化
        #计算每一个属性的均值
        for index in range(len(data[0])):
            self.avg.append(0)
            for d in self.data:
                self.avg[index] += d[index]
            self.avg[index] /= len(data)
        #去中心化
        for d in self.data:
            for index in range(len(d)):
                d[index] -= self.avg[index]
        dataMatrix = np.array(self.data, dtype="float64")
        dataSigma = np.dot(dataMatrix.T, dataMatrix)
        #获取特征值
        eigenvalue, featurevector = np.linalg.eig(dataSigma)
        #将特征值和特征向量排序(获取特征向量排序下标从小到大)
        sort_indices = np.argsort(eigenvalue)
        totValue = eigenvalue.sum()
        pos = 0
        dec = 0
        for index in range(len(sort_indices)):
            dec += eigenvalue[sort_indices[index]]
            pos = index
            if (totValue-dec)/totValue < threshold:
                break
        #取得的特征向量是从
        matrix = []
        for index in range(pos, len(sort_indices)):
            matrix.append(list(featurevector[index]))
        #Transformer是self.dim行，d列的矩阵
        self.Transformer = np.array(matrix, dtype="float64")
        self.dim = len(sort_indices) - pos

    #根据转换矩阵，变形数据
    def Transform(self, data = []):
        res = []
        for index in range(len(data)):
            res.append(data[index]-self.avg[index])
        res = np.array(res, dtype="float64").reshape([len(res), 1])
        matrix = np.dot(self.Transformer, res)
        res = []
        for i in matrix:
            res.extend(i)
        return res

if __name__ == "__main__":
    pca = PCA()
    datas = [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
             [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
             [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
             [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
             [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
             [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437],
             [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376],
             [0.725, 0.445], [0.446, 0.459]]
    pca.Train(data=datas)
    print(pca.Transform([0.697, 0.460]))