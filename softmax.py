import numpy as np


class Layer_Dense:
    def __init__(self):
        #self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)#W
        #self.biases = np.zeros((1, n_neurons))#b
        self.grad_X = None
        self.grad_W = []
        self.grad_b = None
        self.rng = np.random.default_rng()
        self.inputs = None


#class Activation_ReLu:
#    def forward(self, inputs):
#        self.output = np.maximum(0, inputs)

class Activation_Softmax(Layer_Dense):
    def __init__(self, n_inputs, n_neurons , type):
        super().__init__()
        std = np.sqrt(2 / (n_inputs + n_neurons))
        self.weightslinear = self.rng.normal(0, std, size=(n_neurons, n_inputs))
        self.weights = self.rng.normal(0, std, size=(n_neurons, n_inputs))

        bound = 1 / np.sqrt(n_inputs)
        if type ==1 :
            self.biases = self.rng.uniform(-bound, bound, size=(n_neurons, 1))
        if type ==0:
            self.logits = None

        #self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)#W
        #self.biases = np.zeros((1, n_neurons))#b
        self.actual = None
        self.type = type

    def loss(self, logits, actual):
        self.actual = actual
        self.logits = logits
        l, m = actual.shape
        # print(inputs)
        mechane = np.sum(np.exp(np.dot(self.weights, logits)), axis=0)
        loss =0

        for k in range(l):

            mone = np.exp(np.dot(logits.T, self.weights[k]))
            temp = mone/mechane
            # print(np.log(temp))
            # print(actual[k])
            loss = loss + np.dot(actual[k], np.log(temp))

        # loss = -loss/m
        # print(loss)
        return -loss/m


    def backward(self):
        self.gradient()
        res = self.grad_X
        return res


    def forward(self, inputs):
        if self.type == 0:
            z = self.weights @ inputs
            exps = np.exp(z - np.max(z))
            return exps / np.sum(exps, axis=0)

            # exp_values = np.exp(np.dot( self.weights, inputs) - np.max(np.dot(self.weights, inputs)))
            # probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            # return probabilities

        else:
            self.inputs = inputs
            res = np.tanh(np.dot(self.weightslinear, inputs) + self.biases)  # tanh(Wx +b)
            return res

    def Linearbackward(self, v):
        self.grad_W = self.jacobian_test_transpose(self.inputs, v, "weights")
        self.grad_b = self.jacobian_test_transpose(self.inputs, v, "biases")
        res = self.jacobian_test_transpose(self.inputs, v, "inputs")
       #ה print(res)
        # print()

        return res

    # def gradient(self):
    #     l, m = self.actual.shape
    #     sumexp = np.sum(np.exp(self.weights @ self.inputs), axis=0)
    #     self.grad_W = np.array([(self.inputs @ ((np.exp(self.inputs.T @ self.weights[p]) / sumexp) - self.actual[p])) / m for p in range(l)])
    #     self.grad_X = (self.weights.T @ ((np.exp(self.weights @ self.inputs) / sumexp) - self.actual)) / m


    # def gradient(self):
    #     self.grad_W =[]
    #     l, m = self.actual.shape
    #     mechane = np.sum(np.exp(np.dot( self.weights, self.logits)), axis=0)
    #     for p in range(l):
    #         mone = np.exp(np.dot(self.logits.T, self.weights[p]))
    #         res = np.dot(self.logits, ((mone/mechane) - self.actual[p]))
    #         res = res / m
    #         self.grad_W.extend([res])
    #     mone2 = np.dot(self.weights, self.logits)
    #     mechane2 = np.sum(np.exp(np.dot(self.weights, self.logits)))
    #
    #     res2 = (mone2/mechane2) - self.actual
    #     res2 = np.dot(self.weights.T, res2)/m
    #     self.grad_X = res2

    def gradient(self):
        l, m = self.actual.shape
        sumexp = np.sum(np.exp(self.weights @ self.logits), axis=0)
        self.grad_W = np.array([(self.logits @ ((np.exp(self.logits.T @ self.weights[p]) / sumexp) - self.actual[p])) / m for p in range(l)])
        self.grad_X = (self.weights.T @ ((np.exp(self.weights @ self.logits) / sumexp) - self.actual)) / m



    def deriv(self, input):
        return 1 - (np.tanh(input) ** 2)

    def jacobian_test(self, x,  v , case):
        nigzeret = np.diag(np.ravel(Activation_Softmax.deriv(self, np.dot(self.weightslinear, x) + self.biases)))

        if(case == "weights"):
            return np.dot(np.dot(nigzeret, np.kron(x.T, np.eye(nigzeret.shape[0]))), np.expand_dims(np.ravel(v, order="F"), axis=1))

        elif(case == "biases"):
            return np.dot(nigzeret, v)

        elif(case == "inputs"):
            return np.dot(np.dot(nigzeret, self.weightslinear), v)

    # def jac_m_v(self, x, v, param):
    #     z = self.activation.deriv(self.params["W"] @ x + self.params["b"])
    #     diag = np.diag(np.ravel(z))
    #     if param == "W":
    #         return diag @ np.kron(x.T, np.eye(z.shape[0])) @ np.expand_dims(np.ravel(v, order="F"), axis=1)
    #     if param == "b":
    #         return diag @ v
    #     if param == "x":
    #         return diag @ self.params["W"] @ v


    def jacobian_test_transpose(self, x, v, case):
        wxb = np.dot(self.weightslinear, x) + self.biases
        # nigzeret = np.diag(Activation_Softmax.deriv(x))
        nigzeret = Activation_Softmax.deriv(self, wxb)
        if(case == "weights"):
            xt = x.T
            return np.dot(nigzeret* v, xt)
            # return np.dot(np.dot(np.dot(x.T, np.eye(nigzeret.shape[0])).T , nigzeret) , v) #NOT WORK => ravel to v

        elif(case == "biases"):
            # return nigzeret * v
            return np.sum(nigzeret * v, axis=1, keepdims=True)

        elif(case == "inputs"):
            return np.dot(self.weightslinear.T, nigzeret * v)

# class SGD:
#     def __init__(self, lr, model):
#         self.name = "SGD"
#         self.model = model
#         self.lr = lr
#
#     def step_size(self):
#         for Layer_Dense in self.model.layers:
#             # print(layer.grads["W"])
#             Layer_Dense.weights -= self.lr * Layer_Dense.grad_W
#             Layer_Dense.biases -= self.lr * Layer_Dense.grad_b
#
#         arr = np.array(self.model.last_layer.grad_W)
#         print(arr)
#         self.model.last_layer.weights -= self.lr * arr


class SGD:
    def __init__(self, learn_rate, network):
        self.leran_rate = learn_rate
        self.network = network
        # self.tolerance = 1e-06

    def step_size(self):
        for Layer_Dense in self.network.layers:

            # print(Layer_Dense.grad_W)
            step = self.leran_rate * Layer_Dense.grad_W
            new_weight = Layer_Dense.weightslinear - step
            Layer_Dense.weightslinear = new_weight

            step2 = self.leran_rate * Layer_Dense.grad_b
            new_bias = Layer_Dense.biases - step2
            Layer_Dense.biases = new_bias

        arr = np.array(self.network.last_layer.grad_W)


        # print(self.network.last_layer.grad_W)
        step =  self.leran_rate * arr
        new_weight = self.network.last_layer.weights - step
        self.network.last_layer.weights = new_weight




# import numpy as np
#
#
#
# class Layer_Dense:
#     def __init__(self):
#         self.rng = np.random.default_rng()
#         self.inputs = None
#         self.grad_X = None
#         self.grad_W = []
#         self.grad_b = None
#
#
# class Activation_Softmax(Layer_Dense):
#     def __init__(self, in_features, out_features, type):
#         super().__init__()
#         std = np.sqrt(2 / (in_features + out_features))
#         self.weights = self.rng.normal(0, std, size=(out_features, in_features))
#         self.weights2 = self.rng.normal(0, std, size=(out_features, in_features))
#
#         bound = 1 / np.sqrt(in_features)
#         if type ==1 :
#             self.biases = self.rng.uniform(-bound, bound, size=(out_features, 1))
#         if type ==0 :
#             self.actual = None # ????????????????????????
#         self.type = type
#
#
#         std = np.sqrt(2 / (in_features + out_features))
#         self.params["W"] = self.rng.normal(0, std, size=(out_features, in_features))
#         self.logits = None
#         self.actual = None
#
#
#     def deriv(self, input):
#         return 1 - (np.tanh(input) ** 2)
#
#     # def _activation_deriv(self, x):
#     #     return self.activation.deriv(self.params["W"] @ x + self.params["b"])
#
#     def forward(self, inputs):
#
#         if self.type == 0:
#             z = self.weights @ inputs
#             exps = np.exp(z - np.max(z))
#             return exps / np.sum(exps, axis=0)
#
#             # exp_values = np.exp(np.dot( self.weights, inputs) - np.max(np.dot(self.weights, inputs)))
#             # probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#             # return probabilities
#
#         else:
#             self.inputs = inputs
#             res = np.tanh(np.dot(self.weights, inputs) + self.biases)  # tanh(Wx +b)
#             return res
#
#     def Linearbackward(self, v):
#         self.grad_W = self.jacobian_test_transpose(self.inputs, v, "weights")
#         self.grad_b = self.jacobian_test_transpose(self.inputs, v, "biases")
#         res = self.jacobian_test_transpose(self.inputs, v, "inputs")
#        #ה print(res)
#         # print()
#
#     def backward(self):
#             self.gradient()
#             res = self.grad_X
#             return res
#
#     def jacobian_test(self, x,  v , case):
#         nigzeret = np.diag(Activation_Softmax.deriv(np.dot(self.weights, x) + self.biases))
#
#         if(case == "weights"):
#             return np.dot(np.dot(nigzeret, np.dot(x.T, np.eye(nigzeret.shape[0]))), v)
#
#         elif(case == "biases"):
#             return np.dot(nigzeret, v)
#
#         elif(case == "inputs"):
#             return np.dot(np.dot(nigzeret, self.weights), v)
#
#
#
#     def jacobian_test_transpose(self, x, v, case):
#         wxb = np.dot(self.weights, x) + self.biases
#         # nigzeret = np.diag(Activation_Softmax.deriv(x))
#         nigzeret = Activation_Softmax.deriv(self, wxb)
#         if(case == "weights"):
#             xt = x.T
#             return np.dot(nigzeret* v, xt)
#             # return np.dot(np.dot(np.dot(x.T, np.eye(nigzeret.shape[0])).T , nigzeret) , v) #NOT WORK => ravel to v
#
#         elif(case == "biases"):
#             # return nigzeret * v
#             return np.sum(nigzeret * v, axis=1, keepdims=True)
#
#         elif(case == "inputs"):
#             return np.dot(self.weights.T, nigzeret * v)
#
#
#
# class Softmax(Layer):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         std = np.sqrt(2 / (in_features + out_features))
#         self.params["W"] = self.rng.normal(0, std, size=(out_features, in_features))
#         self.logits = None
#         self.actual = None
#
#     def loss(self, logits, actual):
#         self.logits = logits
#         # print(logits)
#         self.actual = actual
#         l, m = actual.shape
#         # print(self.params["W"])
#         loss = 0
#         sumexp = np.sum(np.exp(self.params["W"] @ logits), axis=0)
#
#         for k in range(l):
#
#             loss += actual[k] @ np.log(np.exp(logits.T @ self.params["W"][k]) / sumexp)
#         # print(-loss / m)
#         return -loss / m
#
#     def loss(self, logits, actual):
#         self.actual = actual
#         self.logits = logits
#         l, m = actual.shape
#         # print(inputs)
#         mechane = np.sum(np.exp(np.dot(self.weights, logits)), axis=0)
#         loss =0
#
#         for k in range(l):
#
#             mone = np.exp(np.dot(logits.T, self.weights[k]))
#             temp = mone/mechane
#             # print(np.log(temp))
#             # print(actual[k])
#             loss = loss + np.dot(actual[k], np.log(temp))
#
#         loss = -loss/m
#         # print(loss)
#         return loss
#
#     def _grad(self):
#         l, m = self.actual.shape
#         sumexp = np.sum(np.exp(self.params["W"] @ self.logits), axis=0)
#         self.grads["W"] = np.array([(self.logits @ ((np.exp(self.logits.T @ self.params["W"][p]) / sumexp) - self.actual[p])) / m for p in range(l)])
#         self.grads["X"] = (self.params["W"].T @ ((np.exp(self.params["W"] @ self.logits) / sumexp) - self.actual)) / m
#







