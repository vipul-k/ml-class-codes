import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def prediction(X, theta):
    'Return yhat for given inputs X and parameters theta.'
    return (1/(1 + np.exp(-np.matmul(X, theta))))

def loss(X, y, theta, lamb):
    '''
    Return the loss for given inputs X, targets y, parameters theta,
    and regularization coefficient lamb.
    '''
    yhat = prediction(X, theta)
    return (-((y * np.log(yhat)) + 
              ((1 - y) * np.log(1-yhat))
             ).mean(axis=0) + 
            (lamb) * (theta**2).sum(axis=0)
           )

def gradient_descent(X, y, alpha, lamb, T, theta_init=None):
    
    theta = theta_init.copy()
    
    ### YOUR CODE HERE ###
    
    cost_list = []
    epoch_list = []
    
    m = X.shape[0]
    X0 = np.ones(X.shape[0])
    X = np.hstack((X0.reshape(-1,1), X))
    n = X.shape[1]
    for i in range(0, T):
        y_pred = prediction(X, theta)
        grad = ((y - y_pred)*X.T).mean(axis = 1) - (2*lamb*theta)
        theta = theta + (alpha*grad)
        cost = loss(X, y, theta, lamb)
        epoch_list.append(i)
        cost_list.append(cost)
    

    plt.plot(epoch_list, cost_list)
    plt.title("Cost vs Epochs")
    plt.xlabel('iteration number')
    plt.ylabel('cost')
    plt.show()
    plt.savefig("Cost vs epochs.pdf")
    return(theta)

def test_gradient_descent():
    X = np.array([[-0.31885351, -1.60298056],
       [-1.53521787, -0.57040089],
       [-0.2167283 ,  0.2548743 ],
       [-0.14944994,  2.01078257],
       [-0.09678416,  0.42220166],
       [-0.22546156, -0.63794309],
       [-0.0162863 ,  1.04421678],
       [-1.08488033, -2.20592483],
       [-0.95121901,  0.83297319],
       [-1.00020817,  0.34346274]])
    y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    alpha = 0.1
    lamb = 1
    theta_init = np.zeros(X.shape[1]+1)
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 1, theta_init),
        np.array([-0.03     ,  0.0189148,  0.0256793]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 2, theta_init),
        np.array([-0.05325034,  0.0333282 ,  0.04540004]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 3, theta_init),
        np.array([-0.07127091,  0.04431147,  0.06054757]))
    print('test_gradient_descent passed')
    
    
def load_data():
    df = pd.read_csv('wdbc.data', header=None)
    base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                     'conpoints', 'symmetry', 'fracdim']
    names = ['m' + name for name in base_names]
    names += ['s' + name for name in base_names]
    names += ['e' + name for name in base_names]
    columns = ['id', 'class'] + names
    df.columns = columns
    df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
    
    return df, names
    
    
    
def run_prob_5(df, names):

    X = df[names].values
    y = df['color'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    alpha = 0.1
    lamb = 1
    theta_init = np.zeros(X.shape[1]+1)
    
    theta = gradient_descent(X, y, alpha, lamb, 100, theta_init)
    
    print(theta)
    
    
def run_prob_6(df, names):

    
    X = df[names].values
    y = df['color'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    


    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)
    
    print(clf.intercept_, clf.coef_)
    
def run_prob_7(df, names):
    
    my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')
    c1 = 'mradius'
    c2 = 'mtexture'
    
    c1_col = df[c1]
    c2_col = df[c2]
    X = np.vstack((c1_col,c2_col,np.power(c1_col,2),np.power(c2_col,2),(c1_col*c2_col),np.power(c1_col,3),np.power(c2_col,3),(c1_col*np.power(c2_col,2)),(c2_col*np.power(c1_col,2)))).T
    
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, df['color'])
    
    plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
    plt.xlabel(c1)
    plt.ylabel(c2)
    
    x = np.linspace(df[c1].min(), df[c1].max(), 1000)
    y = np.linspace(df[c2].min(), df[c2].max(), 1000)
    xx, yy = np.meshgrid(x,y)
    
    predicted_prob = clf.predict_proba(
            np.hstack((xx.ravel().reshape(-1,1), 
            yy.ravel().reshape(-1,1),
            xx.ravel().reshape(-1,1)**2,
            yy.ravel().reshape(-1,1)**2,
            (xx.ravel()*yy.ravel()).reshape(-1,1),
            xx.ravel().reshape(-1,1)**3,
                       yy.ravel().reshape(-1,1)**3,
                       (((xx.ravel())**2)*yy.ravel()).reshape(-1,1),
                       (((yy.ravel())**2)*xx.ravel()).reshape(-1,1))))[:,1]
    predicted_prob = predicted_prob.reshape(xx.shape)
    
    plt.contour(xx, yy, predicted_prob, [0.5], colors=['b'])
    plt.title("Decision BOundary")
    
    
if __name__ == "__main__":
    df, names = load_data()
    
    #test for gradient descent
    test_gradient_descent()
    
    #q5
    run_prob_5(df, names)
    
    #q6
    run_prob_6(df, names)
    
    #q7
    run_prob_7(df, names)
    