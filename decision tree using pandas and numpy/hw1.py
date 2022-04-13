import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('figure', figsize=[12,8])  #set the default figure size

import timeit


class Node(object):
    def __init__(self):
        self.name = None
        self.node_type = None
        self.predicted_class = None
        self.X = None
        self.test_attribute = None
        self.test_value = None
        self.children = []
        #added instance variable "test_attribute_cont_flag" which is helpful in .predict method
        self.test_attribute_cont_flag = None
    def __repr__(self):
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {self.X.shape[0]} examples, "
                 f"tests attribute {self.test_attribute} at {self.test_value}")
           
        else:
            s = (f"{self.name} Leaf with {self.X.shape[0]} examples, predicts"
                 f" {self.predicted_class}")
        return s
    
class DecisionTree(object):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X, y):
        '''
        Fit a tree on data, in which X is a 2-d numpy array
        of inputs, and y is a 1-d numpy array of outputs.
        '''
        self.root = self.recursive_build_tree(
            X, y, curr_depth=0, name='0')
            
    def recursive_build_tree(self, X, y, curr_depth, name):

        # WRITE YOUR CODE HERE
        #print(name)
        #print(X.shape)
        node = Node()
        node.name = name
        node.X = X

        if (len(np.unique(y)) == 1) or (curr_depth == self.max_depth) or (len(np.unique(X, axis = 0))==1):
            node.node_type = 'leaf'
            node.predicted_class = scipy.stats.mode(y)[0][0]
            #print(node.predicted_class)
        else:
            node.node_type = "internal node"
            node.test_attribute, node.test_attribute_cont_flag, node.test_value = self.importance(X,y)
            #print(node.test_attribute, node.test_attribute_cont_flag, node.test_value)
            curr_depth += 1
            if node.test_attribute_cont_flag == 1:
                node.children = [self.recursive_build_tree(X[X[:,node.test_attribute] <= node.test_value], 
                                                      y[X[:,node.test_attribute] <= node.test_value],curr_depth, name +".0"),
                                self.recursive_build_tree(X[X[:,node.test_attribute] > node.test_value], 
                                                      y[X[:,node.test_attribute] > node.test_value],curr_depth, name +".1")]
            else:
                node.children = [self.recursive_build_tree(X[X[:,node.test_attribute] == 0], 
                                                      y[X[:,node.test_attribute] == 0],curr_depth, name +".0"),
                                self.recursive_build_tree(X[X[:,node.test_attribute] == 1], 
                                                      y[X[:,node.test_attribute] == 1],curr_depth, name +".1")]
        
        return node
                        


    
    def predict(self, testset):
        if testset.ndim == 1:
            return self.find_class(testset, self.root)
        elif len(testset) == 0:
            print("Data has no records")
        else:
            prediction = np.empty(testset.shape[0])
            for i in range(0,testset.shape[0]):
                prediction[i] = self.find_class(testset[i], self.root)
        return prediction
            
    def find_class(self, row, node):
        if node.node_type == "leaf":
            return node.predicted_class        
        elif node.test_attribute_cont_flag == 0:
            if row[node.test_attribute] == 0:
                prediction = self.find_class(row, node.children[0])
            else:
                prediction = self.find_class(row, node.children[1])
        else:
            if row[node.test_attribute] <= node.test_value:
                prediction = self.find_class(row, node.children[0])
            else:
                prediction = self.find_class(row, node.children[1])
        return prediction


    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)
            
    def importance(self, X, y):
        columns = X.shape[1]
        min_rem = np.inf
        min_rem_var = None
        min_var_cont_flag = 0
        tot_rows = X[:,0].shape[0]
        split_value = None
        
        for i in range(columns):
            cont_flag = 0
            rem = 0

            if ((X[:,i]==0) | (X[:,i]==1)).all():
                for v in [0,1]:
                    rem += self.entropy(y[X[:,i] == v])*(y[X[:,i] == v].shape[0])/tot_rows
            else:
                cont_flag = 1
                rem, rem_split = self.best_split(X[:,i],y)                
            
            if rem < min_rem:
                min_rem = rem
                min_rem_var = i
                min_var_cont_flag = cont_flag
                if min_var_cont_flag == 1:
                    split_value = rem_split
        
        return min_rem_var, min_var_cont_flag, split_value
            
    
    def best_split(self, X, y):
        min_rem = np.inf
        min_rem_split = None
        tot_rows = y.shape[0]
        data = pd.DataFrame(np.hstack((X.reshape(-1,1), y.reshape(-1,1))))
        data_sort = data.sort_values(0, kind = "mergesort").reset_index(drop = True)
        indexes = data_sort[data_sort[1].diff() != 0].index.values
        for i in range(1, len(indexes)):
            v = (data_sort.iloc[indexes[i], 0] + data_sort.iloc[indexes[i]-1, 0])/2
            rem = (self.entropy(y[X <= v])*(y[X <= v].shape[0])/tot_rows) + (self.entropy(y[X > v])*(y[X > v].shape[0])/tot_rows)
            if rem < min_rem:
                min_rem = rem
                min_rem_split = v
        return min_rem, min_rem_split
    
                    
                
            
    def entropy(self, y):
        'Return the information entropy in 1-d array y'
        
        _, counts = np.unique(y, return_counts = True)
        probs = counts/counts.sum()
        return -(np.log2(probs) * probs).sum()
            


def validation_curve(filepath = './arrhythmia.csv'):     
    
    # WRITE YOUR CODE HERE
    # Read the data into a pandas dataframe
    df = pd.read_csv(filepath, header=None, na_values="?")
    # Replace each missing value with the mode
    
    for i in range(280):
        if df[i].isnull().sum() > 0:
            #print(f'Column {i} has missing values.')
            df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)
    
    #randomly shuffle the data        
    df = df.sample(frac=1).reset_index(drop=True)
    
    num_rows = int(df.shape[0]/3)
    print("Size of each partition = ", num_rows)
    X1 = df.loc[:num_rows, :278]
    Y1 = df.loc[:num_rows, 279]
    X2 = df.loc[num_rows+1:2*num_rows,:278]
    Y2 = df.loc[num_rows+1:2*num_rows, 279]
    X3 = df.loc[2*num_rows+1:, :278]
    Y3 = df.loc[2*num_rows+1:, 279]
    
    max_depths = np.array([i for i in range(2,17,2)])
    
    train_acc = np.empty(max_depths.shape)
    test_acc = np.empty(max_depths.shape)
    
    for i, max_depth in enumerate(max_depths):
        tree = DecisionTree(max_depth)
        #1
        Xtrain = np.vstack((X1.values, X2.values))
        Ytrain = np.hstack((Y1.values, Y2.values))
        
        Xtest = X3.values
        Ytest = Y3.values
        
        tree.fit(Xtrain, Ytrain)
        
        predict_train = tree.predict(Xtrain)
        predict_test = tree.predict(Xtest)
        
        train_acc1 = (predict_train == Ytrain).sum()/float(predict_train.size)
        test_acc1 = (predict_test == Ytest).sum()/float(predict_test.size)
        
        #2
        Xtrain = np.vstack((X1.values, X3.values))
        Ytrain = np.hstack((Y1.values, Y3.values))
        
        Xtest = X2.values
        Ytest = Y2.values
        
        tree.fit(Xtrain, Ytrain)
        
        predict_train = tree.predict(Xtrain)
        predict_test = tree.predict(Xtest)
        
        train_acc2 = (predict_train == Ytrain).sum()/float(predict_train.size)
        test_acc2 = (predict_test == Ytest).sum()/float(predict_test.size)
        
        #3
        Xtrain = np.vstack((X2.values, X3.values))
        Ytrain = np.hstack((Y2.values, Y3.values))
        
        Xtest = X1.values
        Ytest = Y1.values
        
        tree.fit(Xtrain, Ytrain)
        
        predict_train = tree.predict(Xtrain)
        predict_test = tree.predict(Xtest)
        
        train_acc3 = (predict_train == Ytrain).sum()/float(predict_train.size)
        test_acc3 = (predict_test == Ytest).sum()/float(predict_test.size)
        
        train_acc[i] = (train_acc1 + train_acc2 + train_acc3)/3
        test_acc[i] = (test_acc1 + test_acc2 + test_acc3)/3
    
    plt.plot(max_depths, train_acc, label = "Training accuracy")
    plt.plot(max_depths, test_acc, label = "Testing accuracy")
    plt.legend()
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.savefig('validation.pdf')
    plt.show()
    
if __name__ == "__main__":
    
    start = timeit.default_timer()
    
    validation_curve()
    
    print("Time:", timeit.default_timer()-start)


