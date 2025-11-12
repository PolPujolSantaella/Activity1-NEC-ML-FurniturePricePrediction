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
    # Vectors of 1 x n[i]
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))
      self.h.append(np.zeros(layers[lay]))
      self.theta.append(np.zeros(layers[lay]))
      self.delta.append(np.zeros(layers[lay]))
      self.d_theta.append(np.zeros(layers[lay]))
      self.d_theta_prev.append(np.zeros(layers[lay]))

    self.w = [] # Weights of each unit
    self.w.append(np.zeros((1, 1))) # Initialize to 0 w[0] (relation layer 0 -> 1)

    self.d_w = []
    self.d_w.append(np.zeros((1,1))) # Initialize to 0 d_w[0] (relation layer 0 -> 1)

    self.d_w_prev = []
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

    # I fact is not in activation_functions raise Error
    if fact not in activation_functions:
      raise ValueError(f"Unknown activation fucntion '{fact}'. Must be one of: {list(activation_functions.keys())}")
    
    self.fact_name = fact 
    self.fact = activation_functions[fact]

    #Other input parameters
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.validation_split = validation_split

  def fit(self, X, y):
    """
    Train the neural network using backpropagation
    Params:
      X: np.ndarray
        Training input data (n_samples x n_features)
      y: np.ndarray
        Target output data of shape (n_samples x output_size)
    """

    # STEP 1: Divide data into trainning and validation
    val_size = int(X.shape[0] * self.validation_split)
    X_val = X[:val_size]
    y_val = y[:val_size]
    X_train = X[val_size:]
    y_train = y[val_size:]

    X_val = X_val.to_numpy() if not isinstance(X_val, np.ndarray) else X_val
    y_val = y_val.to_numpy() if not isinstance(y_val, np.ndarray) else y_val
    X_train = X_train.to_numpy() if not isinstance(X_train, np.ndarray) else X_train
    y_train = y_train.to_numpy() if not isinstance(y_train, np.ndarray) else y_train

    # STEP 2: Initialize Weigths & Thresholds randomly
    for lay in range(1, self.L):
      # Random values from -1 to 1
      self.w[lay] = np.random.uniform(-1, 1, (self.n[lay], self.n[lay - 1]))
      self.theta[lay] = np.random.uniform(-1, 1, self.n[lay])


    # STEP 3: For each epoch
    for epoch in range(self.epochs):
      # STEP 4: For each patter in trainning
      for _ in range(X_train.shape[0]):

        # STEP 5: Choose one random
        random_pattern = np.random.randint(0, X_train.shape[0])
        x_pattern = X_train[random_pattern]
        y_pattern = y_train[random_pattern]

        # STEP 6: Feed-Forward
        self.xi[0] = x_pattern # The first input layer contains the random pattern

        # For each layer
        for l in range(1, self.L):
          # First compute Sum of weights
          # For each unit
          for i in range(self.n[l]):
            sum = 0.0
            # For each unit of previous layer
            for j in range(self.n[l - 1]):
              # Sum = Weight Product * previous activation
              sum += self.w[l][i][j] * self.xi[l - 1][j] 

            # Compute field h(x)
            self.h[l][i] = sum - self.theta[l][i] 

            # Compute Activation (output of the unit)
            self.xi[l][i] = self.fact(self.h[l][i])

        # DEBUG FEED FORWARD
        if epoch % 10 == 0 and _ == 0:
          print(f"Epoch {epoch}: Feed-forward CHECKING")
          print(f"Input 3 samples: {x_pattern[:3]}")
          print(f"Input layer activation: {self.xi[0]}")
          print(f"Output layer activation: {self.xi[self.L - 1]}")

if __name__ == "__main__":
  # MAIN
  layers = [19, 32, 16, 1]
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
