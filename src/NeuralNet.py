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

    if fact not in activation_functions:
      raise ValueError(f"Unknown activation fucntion '{fact}'. Must be one of: {list(activation_functions.keys())}")
    
    self.fact_name = fact
    self.fact = activation_functions[fact]

    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.validation_split = validation_split

# MAIN

layers = [4, 9, 5, 1]
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
