
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import os

class linear_classifier:
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
        self.train_weights = None
        self.weights = np.array([])
        self.eigen_values = np.array([])
        self.eigen_vectors = np.array([])
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
     
    def soft_max(self, X):
        stable_x = X - np.max(X)
        exp_x = np.exp(stable_x)
        return exp_x / np.sum(exp_x, axis = 0)
    
    def regression(self, eta, num_iter, num_comp=32, max_err=0):
        C = 0.00002
        self.weights = np.random.rand(self.train_label_num, num_comp)
        prev_wts = np.zeros_like(self.weights)
        labels = np.array([self.train_label_map[x] for x in self.raw_train_labels])
        diff_wts = self.weights - prev_wts
        itr = 0
        while itr < num_iter and max_err < np.sqrt(np.sum(diff_wts*diff_wts)):
            weight_labels = self.soft_max(np.matmul(self.weights, self.transformed_data))
            weight_labels[labels[:], np.arange(self.raw_train_data_size)] -= 1.0
            J = np.matmul(weight_labels, self.transformed_data.T) / self.raw_train_data_size + C * self.weights
            prev_wts = np.copy(self.weights)
            self.weights -= eta * J
            diff_wts = self.weights - prev_wts
            itr += 1
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
    
    def predict(self, test_file_path, num_comp):
        X = self.read_test_data(test_file_path)
        CX = self.center_data(X)
        RX = self.reduce_test_samples(CX, num_comp)
        P = []
        for val in RX.T:
            P.append(self.train_label_imap[np.argmax(self.soft_max(np.matmul(self.weights, np.array(val))))])
        return P


# In[2]:


import sys

# if __name__ == '__main__':
train_file = sys.argv[1]
test_file = sys.argv[2]
#     Remove comments when actually running
# train_file = './train_sample.txt'
# test_file = './train_sample3.txt'
model = linear_classifier(train_file, test_file, 10)
model.read_train_data()
model.pca()
_ = model.project_samples(32)
model.regression(0.003, 500000, 32, 0.000002)
pred_labels = model.predict(test_file, 32)

for i in pred_labels:
    print(i)

