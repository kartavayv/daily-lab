class l_layer_NN():

  def __init__(self,final_activ="sigmoid", layer_units = [7,4,3,1], l_rate=0.01, n_iter=1000, mini_batch = True, batch_size=128, use_adam = True):
    self.layer_units = layer_units
    self.l_rate = l_rate
    self.params = {}
    self.forward_cache = {}
    self.grads = {}
    self.n_iter = n_iter
    self.costs = []
    self.final_activ = final_activ
    self.epsilon = 1e-15 #to get human like values and prevent div by 0!
    self.mini_batch = mini_batch
    self.batch_size = batch_size  # exponent to divide mini batch based on
    # self.batches = {} # storing mini batches here, USELESS!
    # self.time_note = {} useless!
    self.beta1 = 0.9
    self.beta2 = 0.99
    self.use_adam = use_adam
    self.t = 0 # actually calculates

    #ADAM needs these:
    self.m_m = {}
    self.v_m = {}
    self.m_b = {}
    self.v_b = {}


  def params_ini(self,X):

    for i in range(len(self.layer_units)):
      if i == 0:
        self.params[f"W{i+1}"] = np.random.randn(self.layer_units[0],X.shape[0]) * np.sqrt(2/X.shape[0]).astype(np.float32) #He intialization; good for sigmoid!!!
        self.params[f"b{i+1}"] = np.zeros((self.layer_units[i],1)).astype(np.float32)

      else:
        self.params[f"W{i+1}"] = np.random.randn(self.layer_units[i], self.layer_units[i-1])  * np.sqrt(2/self.layer_units[i-1]).astype(np.float32)
        self.params[f"b{i+1}"] = np.zeros((self.layer_units[i],1)).astype(np.float32)



    #HELPS IN ADAM LATER
      if self.use_adam:
          self.m_m[i+1] = np.zeros_like(self.params[f"W{i+1}"])
          self.v_m[i+1] = np.zeros_like(self.params[f"W{i+1}"])
          self.m_b[i+1] = np.zeros_like(self.params[f"b{i+1}"])
          self.v_b[i+1] = np.zeros_like(self.params[f"b{i+1}"])


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
      # print(Z.shape)
      self.forward_cache[f"A{i}"] = A
      # print(A.shape)
      out = A

    # print(self.forward_cache["Z1"].shape, self.forward_cache["A1"].shape, self.forward_cache["Z2"].shape, self.forward_cache["A2"].shape, self.forward_cache["Z3"].shape,self.forward_cache["A3"].shape)
    return out



  def back_prop(self, y): #calculates gradient for a single layer

    m = y.shape[1] # !0 because y is shaped like this > (1,n) where n are the number of examples in the dataset

    dA = self.forward_cache[f"A{int(len(self.forward_cache)/2)}"] - y #yes i was also thinking that will become a problem as A[l] changes according to each layer l
    self.grads[f"dA"] = dA

    for l in range(int(len(self.forward_cache)/2),0,-1):
      # print(l)
      if l == int(len(self.forward_cache)/2): #this thing needs to be done only one time as it's related to the last layer, i was doing it multiple times and that's not cool
        if self.final_activ == "sigmoid":
          dZ = dA * self.forward_cache[f"A{l}"] * (1-self.forward_cache[f"A{l}"])
        else:
          print(dA.shape)
          dZ = dA  # if it's softmax

      else:
        # print(l)
        print(self.forward_cache[f"Z{l}"].shape)
        # print(int(len(self.forward_cache)/2))
        # dZ = dA * (self.forward_cache[f"A{l}"] > 0)
        dZ = dA * (self.forward_cache[f"Z{l}"] > 0)

      dW = (1/m) * dZ @ self.forward_cache[f"A{l-1}"].T
      db = (1/m) * np.sum(dZ,axis=1, keepdims=True)
      dA_prev = self.params[f"W{l}"].T @ dZ #dA_prev = da[l-1]
      print(dA_prev.shape)

      self.grads[f"dW{l}"] = dW
      self.grads[f"db{l}"] = db
      # self.grads[f"dA{l-1}"] = dA_prev # why do we even need this if we are going to calculate it in the next iteration of the loop? broooooo?
      self.grads["dA"] = dA_prev #this saves RAM!!!



  def gradient_descent(self):

    self.t += 1 #we want it to be 1 at iteration 0 to prevent div by 0 in adam
    if self.use_adam: #ADAM hai to mumkin hai!
        adam_res = self.adam()

    for l in range(1,int(len(self.params)/2)+1):

      if self.use_adam:       
        self.params[f"W{l}"] -= (self.l_rate * adam_res[f"WFM{l}"]) / (adam_res[f"WSM{l}"] + self.epsilon)
        self.params[f"b{l}"] -= (self.l_rate * adam_res[f"BFM{l}"]) / (adam_res[f"BSM{l}"] + self.epsilon)

      else:
        self.params[f"W{l}"] -= (self.l_rate*self.grads[f"dW{l}"])
        self.params[f"b{l}"] -= (self.l_rate*self.grads[f"db{l}"])

  def adam(self):

    #Currently working with just mini batch + ADAM

    n_layers = len(self.layer_units)
    t = self.t  # ie how many times the we have made parameter updates but iteration+1 is veyr important here very very important!

    adam_res = {}

    for i in range(n_layers):
      dW,db = self.grads[f"dW{i+1}"], self.grads[f"db{i+1}"]

      self.m_m[i+1] = self.beta1 * (self.m_m[i+1]) + (1-self.beta1) * (dW)
      self.v_m[i+1] = self.beta2 * (self.v_m[i+1]) + (1-self.beta2) * (dW**2)

      self.m_b[i+1]  = self.beta1 * (self.m_b[i+1]) + (1-self.beta1) * (db)
      self.v_b[i+1] = self.beta2 * (self.v_b[i+1]) + (1-self.beta2) * (db**2)

      m_m_hat = self.m_m[i+1] / (1-self.beta1**t) #prevents that initial value loq stage
      v_m_hat = self.v_m[i+1] / (1-self.beta2**t)

      m_b_hat = self.m_b[i+1] / (1-self.beta1**t)
      v_b_hat = self.v_b[i+1] / (1-self.beta2**t)

      #helps in making update in ADAM, no need to store too long
      adam_res[f"WFM{i+1}"] =  m_m_hat
      adam_res[f"WSM{i+1}"] =  v_m_hat
      adam_res[f"BFM{i+1}"] =  m_b_hat
      adam_res[f"BSM{i+1}"] =  v_b_hat

      # print(adam_res[f"WFM{i+1}"].shape,adam_res[f"WSM{i+1}"].shape,adam_res[f"BFM{i+1}"].shape,adam_res[f"BSM{i+1}"].shape)

    return adam_res



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
    exps = np.exp(prev_output)
    out = exps / np.sum(exps, axis=0, keepdims = True)

    return out



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



  def train_mini_batch(self,X,Y, epoch, even_shuffle = False):

    m = X.shape[1]

    if even_shuffle and epoch%2==0:
      perm = np.random.permutation(m)

      X_shuffled = X[:, perm] #shuffling is important baby {edit: but just frequently}
      Y_shuffled = Y[:, perm]
      batches = self.create_mini_batches(X_shuffled,Y_shuffled)

    else:
      batches = self.create_mini_batches(X,Y)

    for i in range(len(batches)):

      out = self.forward_prop(batches[i][0]) #tuples inside dictionaries suits for this structure well
      self.back_prop(batches[i][1])
      self.gradient_descent()

      # cost = self.loss_compute(out,batches[i][1]) #for the purpose of storing cost for the last batch, yeah but it does not look like a good idea prob will store ocst of eah batch later. to compare it with SGD it's not a good way to compare the cost on the last batch so comparing it based on the whole dataset
      
      if epoch % 100 == 0:
          y_preds = self.forward_prop(X)
          cost = self.loss_compute(y_preds, Y)
          self.costs.append(cost)
          b = time.time()
          print(f"Cost for iteration {epoch} is", cost, f"Time it took for the last 100 iterations is {b-a}")
          a = time.time()

    #up until here we have made batch * epoch updates to our perameters


  def train(self,X,Y):

    self.params_ini(X)
    a = time.time()
    for epoch in range(self.n_iter): #just for now

      if self.mini_batch:
          self.train_mini_batch(X,Y, epoch, True)

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

        if epoch % 100 ==0 :
          b = time.time()
          print(f"Cost for iteration {epoch} is", cost, f"Time it took for the last 100 iterations is {b-a}")
          a = time.time()


  def predict(self, X):

    m = X.shape[0]
    X = X.T.reshape(-1,m) # now it matches with the expected number of features by the first layer NN

    return self.forward_prop(X) #dimension mismatch fixed hahahhahhaha!
