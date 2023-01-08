# Ted Deep Neural Network Framework
I have no intentions to make this a great framework. But since I believe implementation is the quickest way to apply what you learnt. This framework is built to help me understand some of the details in implementing and practicing DNNs.

# Usage
Example
```
# importing the function
from dnn import dnn

# initialize object with X and Y training labels
obj = dnn.dnn(X, Y)
# manually set alpha learning rate
obj.learning_rate = 0.05
# output cost in console
obj.verbose = True
obj.verbose_int = 100
# momentum, rms or adam
obj.adam = True
# start adding linear layer and activation function
obj.add_layer(node_size=5, activation='leaky_relu')
obj.add_layer(node_size=3, activation='softmax')
# start training! params will be the trained parameters
params = obj.train(1000)
# predict:
result = obj.predict(X)
```

# Disclaimer
This software is not extensively tested, please use at your own risk.
