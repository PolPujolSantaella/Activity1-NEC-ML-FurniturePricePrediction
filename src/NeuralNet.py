import numpy as np

# Neural Network Class
class NeuralNet:
  def __init__(self, layers, epochs=100, learning_rate=0.01, momentum=0.9, fact="relu", validation_split=0.2):
    self.L = len(layers) # Number of Layers (length of the layers array) 
    self.n = layers.copy() # Number of units in each layer (array layers)

    self.h = []  # Fields (output values before activation function)
    self.xi = [] # Activations (output values of each unit)
    self.theta = [] # Thresholds  
    self.delta = [] # Gradient (Propagation error)
    self.d_theta = [] # Changes of thresholds
    self.d_theta_prev = [] # Previous changes of thresholds (momentum)
    
    # Initialize to 0 (Activation, Fields, Thresholds, Gradient) for each unit 
    # Vectors of n[i] x 1
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))
      self.h.append(np.zeros(layers[lay]))
      self.theta.append(np.zeros(layers[lay]))
      self.delta.append(np.zeros(layers[lay]))
      self.d_theta.append(np.zeros(layers[lay]))
      self.d_theta_prev.append(np.zeros(layers[lay]))

    self.w = [] # Weights of each unit
    self.w.append(np.zeros((1, 1))) # Initialize to 0 w[0] (relation layer 0 -> 1)

    self.d_w = [] # Changes of weights
    self.d_w.append(np.zeros((1,1))) # Initialize to 0 d_w[0] (relation layer 0 -> 1)

    self.d_w_prev = [] # Previous changes of weigths
    self.d_w_prev.append(np.zeros((1,1))) # Initialize to 0 d_w[0] (relation layer 0 -> 1)

    # Initialize (Weigths, Changes of weigths, Previous changes of weigths) each relation between previous layer units to next layer units
    # Matrices of n[i] x n[i - 1]
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
      self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

    # Initialize name of the activation function (sigmoid, relu, linear, tanh)
    activation_functions = {
      "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
      "relu": lambda x: np.maximum(0, x),
      "linear": lambda x: x,
      "tanh": np.tanh
    }

    # Derivates of activation functions
    activation_derivatives_functions = {
      "sigmoid": lambda x: activation_functions["sigmoid"](x) * (1 - activation_functions["sigmoid"](x)),
      "relu": lambda x: (x > 0).astype(float),
      "linear": lambda x: np.ones_like(x),
      "tanh": lambda x: 1 - np.tanh(x)**2
    }

    # I fact is not in activation_functions raise Error
    if fact not in activation_functions:
      raise ValueError(f"Unknown activation fucntion '{fact}'. Must be one of: {list(activation_functions.keys())}")
    
    # Activation function
    self.fact_name = fact 
    self.fact = activation_functions[fact]

    # Derivative of activation function
    self.fact_der = activation_derivatives_functions[fact]

    #Other input parameters
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.validation_split = validation_split

    # History of the error in trainning & validation
    self.train_loss_history = []
    self.val_loss_history = []


  def _forward(self, x_pattern):

    self.xi[0] = x_pattern # The first input layer contains the random pattern
    
    # For each layer 
    for l in range(1, self.L - 1):
      self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]
      
      # Compute Activation (output of the unit)
      self.xi[l] = self.fact(self.h[l])

    # The output layer must be Linear function
    l_out = self.L - 1
    self.h[l_out] = np.dot(self.w[l_out], self.xi[l_out - 1]) - self.theta[l_out]
    self.xi[l_out] = self.h[l_out]

    return self.xi[l_out]
  
  
  def _backPropagation(self, y_pattern):

    # Error (prediction - real)
    error = self.xi[self.L - 1] - y_pattern

    # Gradient = error * fact'(h[L - 1]])
    self.delta[self.L - 1] = self.fact_der(self.h[self.L - 1]) * error

    # Back Propagate to the rest of the network
    for l in range(self.L - 1, 1, -1):
      # Previous Gradient = fact'(h[l - 1]) * SUM(self.delta[l] * self.weigths)
      # TRANSPOSE WEIGHTS because delta is (n[l] x 1) and weigths (n[l] x n[l + 1])
      delta_sum = np.dot(self.w[l].T, self.delta[l])
      self.delta[l - 1] = self.fact_der(self.h[l - 1]) * delta_sum

  def _update(self):
    # Change Weights & Thresholds with descent gradient
    # New weight = - learning_rate * delta[l] * xi[l-1] + momentum * previous change d_w_prev[l]
    for l in range(1, self.L):
      dw_grad = np.outer(self.delta[l], self.xi[l - 1])
      self.d_w[l] = -self.learning_rate * dw_grad + self.momentum * self.d_w_prev[l]

      # Update
      self.w[l] += self.d_w[l]

      # Thresholds
      # New threshold = learning_rate * delta[l] + momentum * previous change d_t_prev
      self.d_theta[l] = self.learning_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]

      #Update
      self.theta[l] += self.d_theta[l]

      # Store previous changes of weights & thresholds
      self.d_w_prev[l] = self.d_w[l].copy()
      self.d_theta_prev[l] = self.d_theta[l].copy()

  def _score(self, X, y):
    X = X.to_numpy() if not isinstance(X, np.ndarray) else X
    y = y.to_numpy() if not isinstance(y, np.ndarray) else y

    num_samples = X.shape[0]
    total_squared_error = 0.0

    # For every pattern
    for i in range(num_samples):
      x_pattern = X[i]
      y_pattern = y[i]

      # Feed-Forward
      prediction = self._forward(x_pattern)

      # Quadratic Error
      squared_error = np.sum((prediction - y_pattern)**2)
      total_squared_error += squared_error

    # Mean Quadratic Error
    mse = total_squared_error / num_samples
    return mse
  
  def loss_epochs(self):
    return np.array(self.train_loss_history), np.array(self.val_loss_history)
  
  def fit(self, X, y):
    """
    Train the neural network using backpropagation
    Params:
      X: np.ndarray
        Training input data (n_samples x n_features)
      y: np.ndarray
        Target output data of shape (n_samples x output_size)
    """

    print(f"{self.n}")

    X = X.to_numpy() if not isinstance(X, np.ndarray) else X
    y = y.to_numpy() if not isinstance(y, np.ndarray) else y

    # STEP 1: Divide data into trainning and validation
    val_size = int(X.shape[0] * self.validation_split)

    X_val = X[:val_size]
    y_val = y[:val_size]
    X_train = X[val_size:]
    y_train = y[val_size:]

    num_train_patterns = X_train.shape[0]

    # STEP 2: Initialize Weigths & Thresholds randomly
    for lay in range(1, self.L):
      if self.fact_name == "relu":
        # He Initialization for Relu
        std_dev = np.sqrt(2.0/self.n[lay - 1])
        self.w[lay] = np.random.normal(0, std_dev, (self.n[lay], self.n[lay - 1]))
        
      else:
        # Random values from -1 to 1
        self.w[lay] = np.random.uniform(-1, 1, (self.n[lay], self.n[lay-1]))
        self.theta[lay] = np.random.uniform(-1, 1, self.n[lay])


    # STEP 3: For each epoch
    for epoch in range(self.epochs):
      epoch_errors = []
      # STEP 4: For each patter in trainning
      for _ in range(num_train_patterns):

        # STEP 5: Choose one random
        random_pattern = np.random.randint(0, num_train_patterns)
        x_pattern = X_train[random_pattern]
        y_pattern = y_train[random_pattern]

        # STEP 6: Feed-Forward
        self._forward(x_pattern)

        # STEP 7: Back-Propagation
        self._backPropagation(y_pattern)

        # STEP 8: Update-Weights
        self._update()

      # STEP 9: Feed-Forward Trainning & Quadratic Error
      mse_train = self._score(X_train, y_train)

      # STEP 10: Feed-Forward Validation & Quadratic Error
      mse_val = self._score(X_val, y_val)

      # Add to loss history
      self.train_loss_history.append(mse_train)
      self.val_loss_history.append(mse_val)

      if epoch % 10 == 0 or epoch == self.epochs - 1:
        print(f"Epoch {epoch + 1}/{self.epochs}, Train MSE: {mse_train:.6f}, Validation MSE: {mse_val:.6f}")

  def predict(self, X):
    X = X.to_numpy() if not isinstance(X, np.ndarray) else X

    predictions = []
    for x_pattern in X:
      # Feed Forward to obtain prediction
      prediction = self._forward(x_pattern)
      predictions.append(prediction)
    return np.array(predictions)




if __name__ == "__main__":
  # MAIN
  layers = [18, 32, 16, 1]
  epochs = 100
  learning_rate = 0.01
  momentum = 0.9
  fact = "relu"
  validation_split = 0.2
  nn = NeuralNet(layers, epochs, learning_rate, momentum, fact, validation_split)

  print("L =", nn.L)
  print("n =", nn.n)
  print("\nxi =", nn.xi)
  print("xi[0] =", nn.xi[0])
  print("xi[1] =", nn.xi[1])
  print("\nw =", nn.w)
  print("w[1] =", nn.w[1])
  print("\nd_w =", nn.d_w)
  print("\nd_w_prev =", nn.d_w_prev)
  print("\ntheta =", nn.theta)
  print("\nd_theta =", nn.d_theta)
  print("\nd_theta_prev =", nn.d_theta_prev)
  print("\nActivation function:", nn.fact_name)
