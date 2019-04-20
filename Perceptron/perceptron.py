"""
CMSC 25400 - Homework 3
Instructor: Prof. Kondor
Author: Ken Chen
"""

import numpy as np
import matplotlib.pyplot as plt

class perceptron:
    
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.w = np.zeros(train_x.shape[1])
        
    def __online(self):
        l = self.train_x.shape[0]
        m = 0
        mis = []; label = []
        for i in range(l):
            pred_y = 1 if np.dot(self.w, self.train_x[i])>=0 else -1
            label.append(pred_y)
            if pred_y != self.train_y[i]:
                if pred_y==1: self.w -= self.train_x[i]
                else: self.w += self.train_x[i]
                m +=1
            mis.append(m)
        return label, mis
    
    def batch(self, M):
        mistakes = []; labels = [];
        for i in range(M):
            lab, mis = self.__online()
            mistakes.append(mis)
            labels.append(lab)
        return mistakes, labels
    
    def predict(self, x):
        return (np.dot(x, self.w) >= 0)*1.0 - (np.dot(x, self.w) < 0)*1.0
    
    def test(self, x, y):
        pred = self.predict(x)
        error_rate = (pred != y).sum()/len(y)
        return error_rate
        
        
# loads the training data and the testing data
raw_train_x = np.loadtxt("train35.digits")
train_y = np.loadtxt("train35.labels")
raw_test_x = np.loadtxt("test35.digits")
train_x = raw_train_x/np.linalg.norm(raw_train_x, axis = 1).reshape(-1,1)
test_x = raw_test_x/np.linalg.norm(raw_test_x, axis = 1).reshape(-1,1)
## debug
#x_temp = train_x[:200]
#y_temp = train_y[:200]
#p = perceptron(x_temp, y_temp)
#m,l = p.batch(5)
#test = train_x[200:300]
#p.predict(test)
#p.test(test, train_y[200:300])

# Cross Validation 
np.random.seed(12345)
idx = np.random.choice(5, 2000)
t_x, h_x, t_y, h_y = [], [], [], []
for i in range(5):
    t_x.append(train_x[idx != i]); t_y.append(train_y[idx != i])
    h_x.append(train_x[idx == i]); h_y.append(train_y[idx == i])

error_rates = []
for M in range(1, 10):
    error = []
    for i in range(5):
        p = perceptron(t_x[i], t_y[i])
        l, m = p.batch(M)
        error.append(p.test(h_x[i], h_y[i]))
    error_rates.append(sum(error)/5)

x = range(1, 10)
plt.figure()
plt.plot(x, error_rates, color = 'orange')
plt.xlabel('M'); plt.ylabel('Average Error Rate')
plt.title("Cross-Validation Mean Average Error Rates: M from 1 to 9")
plt.grid()

#Implement the algorithm with optimal M
M_opt = np.argmin(error_rates)+1
p = perceptron(train_x, train_y)
mis, lab = p.batch(M_opt)
pred = p.predict(test_x)
np.savetxt('test35.predictions', pred, fmt = "%d")

# Plot the cumulative mistakes
mis_num = np.array(mis)
mis_tot = mis_num[0]
for i in range(1, M_opt):
    mis_tot = np.hstack((mis_tot, (mis_tot[-1]+mis_num[i])))
plt.figure()
plt.plot(range(1, len(mis_tot)+1), mis_tot, color = 'orange')
plt.xlabel("# of examples"); plt.ylabel("# of mistakes")
plt.title("Cumulative Number of Mistakes")
plt.grid()
plt.show()