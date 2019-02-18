
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import os

class naive_bayes:
    def __init__(self, train_path, test_path, q=4):
        self.quantization = q
        self.train_path = train_path
        self.test_path = test_path
        self.raw_train_data = None
        self.raw_train_data_shape = None
        self.raw_train_data_size = None
        self.raw_train_labels = None
        self.train_label_map = None
        self.train_label_num = None
        self.class_train_labels = None
        self.eigen_values = np.array([])
        self.eigen_vectors = np.array([])
        self.class_boundary = {}
        return
    
    def read_train_data(self):
        self.raw_train_data = np.array([])
        self.raw_train_labels = np.array([])
        self.train_label_map = {}
        self.train_label_imap = {}
        self.train_label_num = 0
        with open(self.train_path) as fp:
            line = fp.readline().strip('\n')
            elem_shape = []
            num_elem = 0
            while line:
                line = line.split(' ')
                num_elem += 1
                img = np.asarray(Image.open(line[0].strip('\n')).convert('L'))
                img_shape = np.shape(img)
                img = np.reshape(img, [img_shape[0]*img_shape[1], 1])
                elem_shape = img_shape
                if not line[1].strip('\n') in self.train_label_map:
                    self.train_label_map[line[1].rstrip('\n')] = self.train_label_num
                    self.train_label_imap[self.train_label_num] = line[1].strip('\n')
                    self.train_label_num += 1
                if self.raw_train_data.size == 0:
                    self.raw_train_data = img
                    self.raw_train_labels = np.array([line[1].strip('\n')])
                else:
                    self.raw_train_data = np.concatenate([self.raw_train_data, img], axis=1)
                    self.raw_train_labels = np.concatenate([self.raw_train_labels, np.array([line[1].strip('\n')])])
                line = fp.readline().strip('\n')
            self.raw_train_data_shape = elem_shape
            self.raw_train_data_size = num_elem
        return
    
    def pca(self):
        sz = np.shape(self.raw_train_data)
        M = np.mean(self.raw_train_data, axis=0)
        CX = self.raw_train_data - M
        COV = CX.T.dot(CX)
        eigen_values, eigen_vecs = np.linalg.eig(COV)
        pseudo_eigen_vecs = CX.dot(eigen_vecs)
        data = []
        for idx, val in enumerate(eigen_values):
            data.append((np.array(val), np.reshape(pseudo_eigen_vecs[:, idx], [sz[0], 1])))
        data.sort(key=lambda pair: pair[0], reverse=True)
        for val in data:
            if self.eigen_values.size == 0:
                self.eigen_values = np.array([val[0]])
                self.eigen_vectors = np.array(val[1])
            else:
                self.eigen_values = np.concatenate([self.eigen_values, np.array([val[0]])])
                self.eigen_vectors = np.concatenate([self.eigen_vectors, val[1]], axis=1)
        return

    def project_samples(self, num_comp=-1):
        if num_comp == -1:
            num_comp = np.shape(self.raw_train_data)[1]
        reduced_samples = np.array([])
        M = np.mean(self.raw_train_data, axis=0)
        C_values = self.raw_train_data - M
        self.raw_train_data_mean = np.copy(M)
        for v in self.eigen_vectors.T[0:num_comp]:
            abs_v = np.sum(np.sqrt(np.square(v)))
            v = v / abs_v
            alpha = np.array([v]).dot(C_values)
            if reduced_samples.size == 0:
                reduced_samples = alpha
            else:
                reduced_samples = np.concatenate([reduced_samples, alpha])
        self.transformed_data = np.copy(reduced_samples)
        return reduced_samples
    
    def divide_class(self, num_comp=-1):
        if num_comp == -1:
            num_comp = np.shape(self.raw_train_data)[1]
        sorted_values = self.project_samples(num_comp)
        sorted_values.sort(axis=1)
        index = int(self.raw_train_data_size // self.quantization)
        num_rows = self.raw_train_data_shape[0]
        for i in range(0, num_comp):
            self.class_boundary[i] = []
            for j in range(1, self.quantization):
                self.class_boundary[i].append((sorted_values[i,index*j] + sorted_values[i,index*j-1])/2)
        return

    def relabel_train_data(self, num_comp=-1):
        if num_comp == -1:
            num_comp = np.shape(self.raw_train_data)[1]
        self.class_train_labels = np.zeros_like(self.raw_train_data[0:num_comp, :])
        for i in range(0, num_comp):
            for j in range(0, self.raw_train_data_size):
                for k in range(0, self.quantization-1):
                    if self.transformed_data[i, j] < self.class_boundary[i][k]:
                        self.class_train_labels[i,j] = k+1
                        break
                if not self.transformed_data[i, j] < self.class_boundary[i][self.quantization-2]:
                    self.class_train_labels[i, j] = self.quantization
        return
    
    def calculate_probabilities(self):
        self.class_probabilities = np.zeros([np.shape(self.class_train_labels)[0], self.train_label_num, self.quantization+1])
        self.class_samples = np.zeros([np.shape(self.class_train_labels)[0], self.train_label_num])
        for i, row in enumerate(self.class_train_labels):
            for j, col in enumerate(row):
                self.class_probabilities[i, self.train_label_map[self.raw_train_labels[j]], self.class_train_labels[i,j]] += 1
                self.class_samples[i, self.train_label_map[self.raw_train_labels[j]]] += 1
        
        for i in range(1, self.quantization+1):
            self.class_probabilities[:, :, i] = self.class_probabilities[:, :, i] / self.class_samples
        return
    
    def read_test_data(self, path):
        X = np.array([])
        with open(path) as fp:
            line = fp.readline().strip('\n')
            while line:
                line = line.split(' ')
                img = np.asarray(Image.open(line[0]).convert('L'))
                img_shape = np.shape(img)
                img = np.reshape(img, [img_shape[0]*img_shape[1], 1])
                elem_shape = img_shape
                if X.size == 0:
                    X = img
                else:
                    X = np.concatenate([X, img], axis=1)
                line = fp.readline().strip('\n')
        return X
    
    def center_data(self, X):
        CX = X - np.mean(X, axis=0)
        return CX
    
    def reduce_test_samples(self, CX, num_comp):
        reduced_sample = np.array([]) 
        for v in self.eigen_vectors.T[0:num_comp]:
            abs_v = np.sum(np.sqrt(np.square(v)))
            v = v / abs_v
            alpha = np.array([v]).dot(CX)
            if reduced_sample.size == 0:
                reduced_sample = alpha
            else:
                reduced_sample = np.concatenate([reduced_sample, alpha])
        return reduced_sample

    def relabel_data(self, X, num_comp):
        r_X = np.zeros_like(X)
        for i in range(0, num_comp):
            for j in range(0, np.shape(X)[1]):
                for k in range(0, self.quantization-1):
                    if X[i,j] < self.class_boundary[i][k]:
                        r_X[i,j] = k+1
                        break
                if not X[i,j] < self.class_boundary[i][self.quantization-2]:
                    r_X[i,j] = self.quantization
        return r_X

    def get_predicted_label(self, X, num_comp):
        probabilities = np.ones([self.train_label_num, np.shape(X)[1]])
        for i in range(0, num_comp):
            for j in range(0, np.shape(X)[1]):
                for k in range(0, self.train_label_num):
                    probabilities[k, j] = probabilities[k, j]*self.class_probabilities[i, k, int(X[i, j])]
        
        label = np.argmax(probabilities, axis = 0)
        rev_label = []
        for i in label:
            rev_label.append(self.train_label_imap[i])
        return rev_label
   
    def predict(self, path):
        X = self.read_test_data(path)
        CX = self.center_data(X)
        RX = self.reduce_test_samples(CX, 32)
        LX = self.relabel_data(RX, 32)
        P = self.get_predicted_label(LX, 32)
        return P
    


# In[3]:


import sys

# if __name__ == '__main__':
train_file = sys.argv[1]
test_file = sys.argv[2]
#     Remove comments when actually running
#     train_file = './problem_statement/sample_train.txt'
# train_file = './train_sample.txt'
# test_file = './problem_statement/sample_test.txt'
model = naive_bayes(train_file, test_file, 10)
model.read_train_data()
model.pca()
model.divide_class(32)
model.relabel_train_data(32)
model.calculate_probabilities()
pred_labels = model.predict(test_file)

for i in pred_labels:
    print(i)

