# Ted's Deep Neural Network Framework
I have no intentions to make this a great framework. But since I believe implementation is the quickest way to apply what you learnt. This framework is built to help me understand some of the details in implementing and practicing DNNs.

# Usage
Example
```
# importing the function
from dnn import dnn

# initialize object with X and Y training labels
'''
must be in the form of X.shape = (nx, m) and Y.shape = (1, m) dimensions.
'''
obj = dnn.dnn(X, Y)

# manually set alpha learning rate
'''
a must tune hyper parameter, the faster the learning rate, the faster the loss convergence may be, but it might cause negative effect if already close to minima.
'''
obj.learning_rate = 0.05

# output cost in console
'''
if want to view the cost output, enable verbose and choose a verbose_int, the verbose_int will determine how many times of output will be displayed in the console. 100 means show 100 cost outputs, if want to see more, choose 500 instead. BUT this number cannot be more than the training iteration epochs.
'''
obj.verbose = True
obj.verbose_int = 100

# momentum, rms or adam optimization methods
'''
momentum, rms or adam optimization methods, adam enalbes momentum and rms together.
'''
obj.adam = True

# normalization
'''
enabling normalization, normalization enables normalization at each hidden layer; normalization makes each hidden layer less reliant on the previous layer as well as has a small regularization effect.
'''
obj.normalization = True
obj.norm_momentum = 0.9

# regularization
'''
there is the normal regularization method and the dropout method.
if regularization method, can tune the lambd to adjust the regularization effect.
if dropout method, can tune the keep_probs = {'D1': 0.9, 'D2': 0.6}
'''
obj.regularization = True

# mini batch
'''
if training set is large, can utilize mini batch to speed up the training process. adjust the batch size for each iteration.
'''
obj.mini_batch = True
obj.batch_size = 64

# start adding linear layer and activation function
'''
needs to add the linear layer and the activation layer by each layer's node size as well as the non linear activation function. this is required to define the forward pro and the backward prop procedures.
'''
obj.add_layer(node_size=5, activation='leaky_relu')
obj.add_layer(node_size=3, activation='softmax')

# start training! params will be the trained parameters, 1000 epochs...etc
params = obj.train(1000)

# costs
costs = obj.costs
#if you want to plot out the costs, adjust verbose_int if needed.
# plt.plot(obj.costs.keys(), obj.costs.values())

# predict:
result = obj.predict(X)
```

# Disclaimer
This software is not extensively tested, please use at your own risk.
