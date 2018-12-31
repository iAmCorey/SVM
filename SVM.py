# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    SVM.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Corey <390583019@qq.com>                   +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/12/19 19:43:57 by Corey             #+#    #+#              #
#    Updated: 2018/12/31 21:46:34 by Corey            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import time
import random
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

TRAIN_DATA_PATH = ''
TEST_DATA_PATH = ''
TIME_BUDGET = 0

def main(training_set, test_set):
    X_train, y_train = training_set[:,:-1], training_set[:,-1]
    X_test = test_set

    # split train set and test set
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)
    
    # predict and score
    # Gradient descent method
    #t1 = time.time()
    gd_classifier = GD(X_train, y_train,max_iter=500)
    gd_classifier.train()
    y_pred = gd_classifier.predict(X_test)
    for i in range(len(y_pred)):
        print(int(y_pred[i]))
    #t2 = time.time()
    #print(gd_classifier.score(X_test,y_test))
    #print(t2-t1)

    # smo method
    #smo_classifier = SMO(X_train, y_train, 0.6, 0.001, 2)
    #smo_classifier.train()
    #t3 = time.time()
    #print(smo_classifier.score(X_test, y_test))
    #print(t3-t2)


class GD():
    def __init__(self,x,y,max_iter=200,learning_rate=0.1):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size = np.shape(self.x)[1],)
    
    def get_loss(self, x, y):
        loss = max(0, 1-y*np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, w):
        if y*np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        return w

    def train(self):
        for epoch in range(self.max_iter):
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
        # print("training done")

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.sign(np.dot(x_test, self.w))

    def score(self, X_test, y_test):
        right_count = 0
        y_pred = self.predict(X_test)
        for i in range(len(X_test)):
            if y_pred[i] == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

class SMO():
    """
    Steps:
    1. calculate the E (error)
    2. calculate the L and H
    3. calculate the η
    4. update the alpha(j)
    5. clip the alpha(j)
    6. update the alpha(i)
    7. update b1 and b2
    8. update b
    """
    def __init__(self, x, y, C, tolerate, max_iter):
        self.x = np.mat(x) 
        self.y = np.mat(y).transpose()
        self.C = C
        self.tolerate = tolerate
        self.max_iter = max_iter 
        self.w = 0
        self.b = 0
        self.m, self.n = np.shape(self.x)
        self.alpha = np.mat(np.zeros((self.m,1)))
        pass
    
    def train(self):
        iter_num = 1
        while(iter_num <= self.max_iter):
            alpha_opt = 0 # record the optimization number of alpha
            for i in range(self.m):
                # step 1
                f_xi =  float(np.multiply(self.alpha, self.y).T * (self.x * self.x[i,:].T)) + self.b 
                Ei = f_xi - self.y[i]
                if ((self.y[i]*Ei < -self.tolerate) and (self.alpha[i] < self.C)) or ((self.y[i]*Ei > self.tolerate) and (self.alpha[i] > 0)):
                    # choose a alpha j
                    j = self.select_alpha_j(i)
                    f_xj =  float(np.multiply(self.alpha, self.y).T * (self.x * self.x[j,:].T)) + self.b 
                    Ej = f_xj - self.y[j]
                    alpha_i_old, alpha_j_old = self.alpha[i].copy(), self.alpha[j].copy()
                    # step 2
                    if (self.y[i] != self.y[j]):
                        L, H = max(0, self.alpha[j]-self.alpha[i]), min(self.C, self.C+self.alpha[j]-self.alpha[i])
                    else:
                        L, H = max(0, self.alpha[j]+self.alpha[i]-self.C), min(self.C, self.alpha[j]-self.alpha[i])
                    # step 3
                    eta =  self.x[i,:]*self.x[i,:].T + self.x[j,:]*self.x[j,:].T - 2*self.x[j,:]*self.x[i,:].T
                    # step 4
                    self.alpha[j] += (self.y[j]*(Ei-Ej))/eta
                    # step 5
                    self.alpha[j] = self.clip(self.alpha[j], H, L)
                    # step 6
                    self.alpha[i] += self.y[i]*self.y[j]*(alpha_j_old-self.alpha[j])
                    # step 7
                    b1 = self.b - Ei - self.y[i]*(self.alpha[i]-alpha_i_old)*self.x[i,:]*self.x[i,:].T - self.y[j]*(self.alpha[j]-alpha_j_old)*self.x[i,:]*self.x[j,:].T
                    b2 = self.b - Ej - self.y[i]*(self.alpha[i]-alpha_i_old)*self.x[i,:]*self.x[j,:].T - self.y[j]*(self.alpha[j]-alpha_j_old)*self.x[j,:]*self.x[j,:].T
                    # step 8
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2)/2
                    alpha_opt += 1
                    # print('iterate time: ',iter_num,' alpha optimization time: ',alpha_opt)
            # if (alpha_opt == 0):
            #     self.iter_num += 1 
            # else:
            #     self.iter_num = 0
            # print('totally iterate times: ', self.iter_num)
            iter_num += 1 
        # update w
        yx = np.array(self.y).reshape(-1, 1)*np.array(self.x)
        self.w = np.dot(yx.T, self.alpha)
        # print('training done')
        return

    def select_alpha_j(self, i):
        j = i
        while(j==i):
            j = int(random.uniform(0,self.m))
        return j

    # step 5
    def clip(self,alpha_j, H, L): 
        if alpha_j >= H:
            return H
        if alpha_j <= L:
            return L
        else:
            return alpha_j

    def predict(self,x):
        y_pred = []
        #print(np.mat(x[0]).T.shape)
        for i in range(len(x)):
            if self.w.T * np.mat(x[i]).T + self.b < 0:
                y_pred.append(-1)
            else:
                y_pred.append(1)
        return y_pred

    def score(self,x_test, y_test):
        y_pred =  self.predict(x_test)
        cnt = len(y_pred)
        right_cnt = 0
        for i in range(cnt):
            if y_pred[i] == y_test[i]:
                right_cnt +=1 
        return right_cnt/cnt

    

if __name__ == "__main__":
    TRAIN_DATA_PATH, TEST_DATA_PATH, TIME_BUDGET = sys.argv[1], sys.argv[2], int(sys.argv[4])
    try:
        training_data = np.loadtxt(TRAIN_DATA_PATH)
        test_data = np.loadtxt(TEST_DATA_PATH)
        main(training_data, test_data)
    except IOError:
        sys.exit('File Not Exist!')
