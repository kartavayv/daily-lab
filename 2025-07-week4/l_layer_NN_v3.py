class l_layer_NN():

  def __init__(self,final_activ="sigmoid", layer_units = [7,4,3,1], l_rate=0.01, n_iter=1000, mini_batch = True, batch_size=128):
    self.layer_units = layer_units
    self.l_rate = l_rate
    self.params = {}
    self.forward_cache = {}
    self.grads = {}
    self.n_iter = n_iter
    self.costs = []
    self.final_activ = final_activ
    self.epsilon = 1e-15 #to get human like values
    self.mini_batch = mini_batch
    self.batch_size = batch_size  # exponent to divide mini batch based on
    # self.batches = {} # storing mini batches here, USELESS!
    # self.time_note = {} useless!

  def params_ini(self,X):

    for i in range(len(self.layer_units)):
      if i == 0:
        self.params[f"W{i+1}"] = np.random.randn(self.layer_units[0],X.shape[0]) * np.sqrt(2/X.shape[0]).astype(np.float32) #He intialization; good for sigmoid!!!
        self.params[f"b{i+1}"] = np.zeros((self.layer_units[i],1)).astype(np.float32)

      else:
        self.params[f"W{i+1}"] = np.random.randn(self.layer_units[i], self.layer_units[i-1])  * np.sqrt(2/self.layer_units[i-1]).astype(np.float32)
        self.params[f"b{i+1}"] = np.zeros((self.layer_units[i],1)).astype(np.float32)



  def forward_prop(self, X):

    out = X
    self.forward_cache["A0"] = X
    L = len(self.layer_units)

    for i in range(1,L+1):
      W = self.params[f"W{i}"]
      b = self.params[f"b{i}"]
      Z = W @ out + b
      A = (self.hidden_activ(Z) if i!=len(self.layer_units) else self.final_l_activ(Z))

      self.forward_cache[f"Z{i}"] = Z
      self.forward_cache[f"A{i}"] = A

      out = A

    return out



  def back_prop(self, y): #calculates gradient for a single layer

    m = y.shape[1] # !0 because y is shaped like this > (1,n) where n are the number of examples in the dataset

    dA = self.forward_cache[f"A{int(len(self.forward_cache)/2)}"] - y #yes i was also thinking that will become a problem as A[l] changes according to each layer l
    self.grads[f"dA"] = dA

    for l in range(int(len(self.forward_cache)/2),0,-1):

      if l == int(len(self.forward_cache)/2): #this thing needs to be done only one time as it's related to the last layer, i was doing it multiple times and that's not cool
        if self.final_activ == "sigmoid":
          dZ = dA * self.forward_cache[f"A{l}"] * (1-self.forward_cache[f"A{l}"])
        else:
          dZ = dA  # if it's softmax

      else:
        # dZ = dA * (self.forward_cache[f"A{l}"] > 0)
        dZ = dA * (self.forward_cache[f"Z{l}"] > 0)

      dW = (1/m) * dZ @ self.forward_cache[f"A{l-1}"].T
      db = (1/m) * np.sum(dZ,axis=1, keepdims=True)
      dA_prev = self.params[f"W{l}"].T @ dZ #dA_prev = da[l-1]

      self.grads[f"dW{l}"] = dW
      self.grads[f"db{l}"] = db
      # self.grads[f"dA{l-1}"] = dA_prev # why do we even need this if we are going to calculate it in the next iteration of the loop? broooooo?
      self.grads["dA"] = dA_prev #this saves RAM!!!



  def gradient_descent(self):

    for l in range(1,int(len(self.params)/2)+1):
      self.params[f"W{l}"] -= self.l_rate * self.grads[f"dW{l}"]
      self.params[f"b{l}"] -= self.l_rate * self.grads[f"db{l}"]


  def create_mini_batches(self,X,Y):

    m = X.shape[1]
    batch_size = self.batch_size
    mini_batches = []  #better to have a list than a demon looking dictionary

    for i in range(0, m, batch_size):

      X_mini = X[:,i:i+batch_size]
      Y_mini = Y[:,i:i+batch_size]

      mini_batches.append((X_mini,Y_mini))

    return mini_batches



  def softmax_l_layer_mini(self,prev_output):

    prev_output = prev_output - np.max(prev_output, axis=0, keepdims = True) # prevents numerical instability by making the max value in the array 0
    prev_output = np.clip(prev_output, -500, 500)
    exps= np.exp(prev_output)

    return exps / np.sum(exps, axis=0, keepdims = True)



  def hidden_activ(self,Z):

      return self.relu(Z) # it's just relu for now we will make others later



  def final_l_activ(self,Z):

    if self.final_activ == "sigmoid":
      return 1/(1+np.exp(-np.clip(Z,-500,500)))
    else: #pressuming for now it's softmax
      return self.softmax_l_layer_mini(Z)



  def relu(self,Z):
    return np.maximum(0, Z)



  def loss_compute(self,y_pred,y):

    m = y.shape[1]
    if self.final_activ=="sigmoid":
      y_pred = np.clip(y_pred,self.epsilon,1-self.epsilon) #we just want to prevent the values from exploding and getting very low so yeah bet 0 and 1
      cost = -np.sum(y*np.log(y_pred + self.epsilon) + (1-y)*np.log(1 - y_pred + self.epsilon)) / m #for binary output, BCE
    else:
      cost = - np.sum((y * np.log(y_pred))) # CE for loss, softmax, learn it, assuming it's softmax the CF is CE

    return cost



  def train_mini_batch(self,X,Y):

    m = X.shape[1]
    perm = np.random.permutation(m)

    X_shuffled = X[:, perm] #shuffling is important baby {edit: but just frequently}
    Y_shuffled = Y[:, perm]

    batches = self.create_mini_batches(X_shuffled,Y_shuffled)

    for i in range(len(batches)):

      # global cost
      out = self.forward_prop(batches[i][0]) #tuples inside dictionaries suits for this structure well
      self.back_prop(batches[i][1])
      self.gradient_descent()

      # cost = self.loss_compute(out,batches[i][1]) #for the purpose of storing cost for the last batch, yeah but it does not look like a good idea prob will store ocst of eah batch later

    # self.costs.append(cost)

    #up until here we have made batch * epoch updates to our perameters


  def train(self,X,Y):

    self.params_ini(X)
    a = time.time()
    for epoch in range(self.n_iter): #just for now

      if self.mini_batch:
          self.train_mini_batch(X,Y)

          if epoch%100==0:
            print(f"I am epoch {epoch}")
          # if (epoch+1)%100 ==0 :
          #   b = time.time()
          #   print(f"Cost for iteration {epoch+1} is", self.costs[epoch+1], f"Time it took for the last 100 iterations is {b-a}")
          #   a = time.time()

      else:
        y_preds = self.forward_prop(X)
        cost = self.loss_compute(y_preds, Y)
        self.costs.append(cost)
        self.back_prop(Y)
        self.gradient_descent()

        if (epoch+1) %100 ==0 :
          b = time.time()
          print(f"Cost for iteration {epoch+1} is", cost, f"Time it took for the last 100 iterations is {b-a}")
          a = time.time()


  def predict(self, X):

    m = X.shape[0]
    X = X.T.reshape(-1,m) # now it matches with the expected number of features by the first layer NN

    return self.forward_prop(X) #dimension mismatch fixed hahahhahhaha!
