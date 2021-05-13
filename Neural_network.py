from softmax import Activation_Softmax


class NeuralNetwork:
    def __init__(self, input, output, hidden_layer_size, l_layers):
        self.hidden_layer_size = hidden_layer_size
        self.l_layers = l_layers
        self.layers = []
        if l_layers >= 2:
            self.layers.extend([Activation_Softmax(input, hidden_layer_size, 1)])
            for i in range(l_layers - 2):
                self.layers.extend([Activation_Softmax(hidden_layer_size, hidden_layer_size, 1)])
            self.last_layer = Activation_Softmax(hidden_layer_size, output, 0)
        else:
            self.last_layer = Activation_Softmax(input, output, 0)

    def loss(self, predicted, actual):
        return self.last_layer.loss(predicted, actual)

    def forward(self, inputs):
        curr_input = inputs
        for layer in self.layers:
            curr_input = layer.forward(curr_input)
        return curr_input

    def backward(self):
        gradient = self.last_layer.backward()
        for layer in reversed(self.layers):
            gradient = layer.Linearbackward(gradient)
        return gradient
