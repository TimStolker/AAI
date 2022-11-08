import numpy as np
import random

random.seed(0)

data = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
flower = np.genfromtxt('iris.data',dtype=str, delimiter=',', usecols=[4])

# 80% train data
X_train = np.concatenate((data[:40], data[50:90], data[100:140]))
y_train = np.concatenate((flower[:40], flower[50:90], flower[100:140]))

# 20% test data
X_test = np.concatenate((data[40:50], data[90:100], data[140:]))
y_test = np.concatenate((flower[40:50], flower[90:100], flower[140:]))

# Convert flowers to array of desired output
y_trainConverted = []
for sample in y_train:
    if sample == "Iris-setosa":
        y_trainConverted.append([1,0,0])
    elif sample == "Iris-versicolor":
        y_trainConverted.append([0,1,0])
    elif sample == "Iris-virginica":
        y_trainConverted.append([0,0,1])
    else:
        print("Unknown sample")
        break

class Neuron:
    """"This class specifies a neuron of a neural network"""
    def __init__(self, weightCount):
        self.weights = []
        # Make a list of random weights
        for weight in range(weightCount):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(0, 1)
        self.Z = 0
        self.delta = 0

    def sigmoid(self, x):
        """"Returns the sigmoid output with a given number"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """"Returns the derivative sigmoid output with a given number"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, inputs):
        """"Calculates the output of a neuron and returns the sigmoid of it"""
        result = self.bias
        # Get the sum of every output times the weight
        for numberIndex, inputNumber in enumerate(inputs):
            result += inputNumber*self.weights[numberIndex]
        self.Z = result
        return self.sigmoid(result)

    def backpropHidden(self, sums):
        """"Calculates the delta of a neuron in a hidden layer"""
        result = np.sum(sums)
        derivative = self.sigmoid_derivative(self.Z)
        self.delta = derivative * result

    def backpropLast(self, desiredOutput):
        """"Calculates the error together with the delta of a neuron in the last layer"""
        error = (desiredOutput - self.sigmoid(self.Z))
        self.delta = self.sigmoid_derivative(self.Z) * error

    def getSum(self, weightIndex):
        """"Returns the delta times the neuron's weight"""
        return self.delta * self.weights[weightIndex]

    def update(self, prevLayerOutputs, learningRate):
        """"Updates the neuron's weights and bias"""
        # Update weights
        for weightIndex, weight in enumerate(self.weights):
            self.weights[weightIndex] += (learningRate * self.delta * prevLayerOutputs[weightIndex])
        # Update Bias
        self.bias += learningRate * self.delta


def createNetwork(neuronStructure):
    """"Function to build the structure of the neuron network"""
    network = []
    for layer in range(len(neuronStructure)):
        newLayer = []
        for neuron in range(neuronStructure[layer]):
            if len(network) > 0:
                newLayer.append(Neuron(len(network[layer - 1])))
            else:
                newLayer.append(Neuron(0))
        network.append(newLayer)
    return network

def trainNetwork(network, X_train, y_train):
    """"Function to train the neural network.
    Returns the output list"""
    # fill outputs with empty numbers
    outputs = []
    for layerIndex, layer in enumerate(network):
        outputs.append([0] * len(layer))

    # Loop through all the data 'epochs' times
    for epoch in range(epochs):
        # Loop through all the samples from the train data
        for sampleIndex, sample in enumerate(X_train):
            # Set the inputs
            outputs[0] = sample

            # Forward propagation
            for layerIndex, layer in enumerate(network):
                # Skip first layer
                if layerIndex:
                    # Perform forward propagation on every neuron in the layer and get the new outputs
                    for neuronIndex, neuron in enumerate(network[layerIndex]):
                        outputs[layerIndex][neuronIndex] = neuron.forward(outputs[layerIndex - 1])

            # Backward propagation
            for layerIndex, layer in reversed(list(enumerate(network))):
                # Skip first layer
                if layerIndex:
                    # Last layer
                    if layerIndex == len(network) - 1:
                        # Perform backpropagation on every neuron in the layer
                        for neuronIndex, neuron in enumerate(network[layerIndex]):
                            neuron.backpropLast(y_trainConverted[sampleIndex][neuronIndex])
                    # Hidden layer
                    else:
                        # Perform backpropagation on every neuron in the layer
                        for neuronIndex, neuron in enumerate(network[layerIndex]):
                            sums = []
                            for sumNeuron in network[layerIndex + 1]:
                                sums.append(sumNeuron.getSum(neuronIndex))
                            neuron.backpropHidden(sums)

            # Weights and bias update for every neuron in every layer
            for layerIndex, layer in enumerate(network):
                if layerIndex:
                    for neuronIndex, neuron in enumerate(network[layerIndex]):
                        neuron.update(outputs[layerIndex - 1], learningRate)
    return outputs

def testNetwork(network, output, X_test, y_test):
    """"Function to test the network and print the accuracy"""
    accuracy = 0

    # Loop though all the test samples
    for sampleIndex, sample in enumerate(X_test):
        # Set the inputs
        outputs[0] = sample

        # Forward propagation for test data
        for layerIndex, layer in enumerate(network):
            if layerIndex:
                for neuronIndex, neuron in enumerate(network[layerIndex]):
                    outputs[layerIndex][neuronIndex] = neuron.forward(outputs[layerIndex - 1])

        # get the index of the highest number from the outputs
        GuessedFlowerIndex = outputs[len(outputs)-1].index(max(outputs[len(outputs)-1]))

        # Check if the guess is correct
        if flowerList[GuessedFlowerIndex] == y_test[sampleIndex]:
            accuracy += 1
    print(f'Accuracy: {accuracy/len(X_test)*100}%')

if __name__ == '__main__':
    flowerList = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    # Create the network
    network = createNetwork([4, 4, 3])

    learningRate = 0.1
    epochs = 300

    # Train network
    outputs = trainNetwork(network, X_train, y_train)

    # Test network
    testNetwork(network, outputs, X_test, y_test)
