import numpy as np
import matplotlib.pyplot as plt
class logistic_regression():
  def __init__(self,n_iter=100,l_rate=0.001):
    self.n_iter = n_iter
    self.l_rate = l_rate
    self.costs = []      #i like doing this
    self.params = {}
    self.grads = {}
  
  def dimensions(self,X,Y):
    n_x = X.shape[1]
    n_y = Y.shape[1]

    return (n_x,n_y)
  
  def ini_params(self,n_x,n_y):
    W = np.random.randn(n_x,n_y)*0.01 # it's going to be XW instead of WX
    b = 0 #brodcasting will take care of itself!

    params = {
        "W": W,
        "b": b
    }

    self.params = params

  def sigmoid(self,calcs):
    return 1 / (1+np.exp(-calcs))
  
  def forward_prop(self,X,Y):
    W = self.params["W"]
    b = self.params["b"]

    res = np.dot(X,W) + b
    y_pred = self.sigmoid(res)
    return y_pred
  
  def cost_func(self,y_pred,Y):
    p1 = np.multiply(Y,np.log(y_pred))
    p2 = np.multiply((1-Y),np.log(1-y_pred))

    cost = np.sum(-p1 - p2) #summing up errors on each example is necessary!!!

    return cost

  def grad_des(self,X, Y, y_pred):
    m = Y.shape[0]
    dW = (1/m) * np.dot(X.T, (y_pred-Y))
    db = (1/m) * np.sum((y_pred-Y))

    grads = {
        "dW": dW,
        "db": db
    }

    self.grads = grads
    
  def fit(self,X,Y):
    n_iter = self.n_iter
    l_rate = self.l_rate
    n_x,n_y = self.dimensions(X,Y)
    self.ini_params(n_x,n_y)

    for i in range(n_iter):
      y_pred = self.forward_prop(X,Y)
      cost = self.cost_func(y_pred,Y)
      self.costs.append(cost)
      self.grad_des(X,Y,y_pred)
      
      dW = self.grads['dW']
      db = self.grads['db']

      self.params["W"] = self.params['W'] - (l_rate*dW) #thought of making code more readable
      self.params["b"] = self.params['b'] - (l_rate*db)


      if i%100 == 0:
        print(f"Cost of the iteration {i} is {cost}")
    
    final_vals = {
        "W": self.params['W'],
        "b": self.params["b"] 
    }
    
    return self.costs, final_vals
  
  def predict(self,X):
    model_params = self.params
    W = model_params['W']
    b = model_params['b']

    calcs = np.dot(X,W) + b
    preds = self.sigmoid(calcs)

    return preds