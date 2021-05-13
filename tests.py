import numpy as np
from Neural_network import NeuralNetwork
import copy
from collections import defaultdict

class MainTest:
    def __init__(self, l_layers, hidden_layer_size, inputs, outputs):
        self.network = NeuralNetwork(inputs.shape[0], outputs.shape[0] ,hidden_layer_size, l_layers)
        self.inputs = inputs
        self.outputs = outputs
        self.rng = np.random.default_rng()

    def gradient_test(self):
        activation_softmax = self.network.last_layer
        n, m = activation_softmax.weights.shape
        tags = self.rng.integers(low=0, high=n, size=10)
        x = self.rng.standard_normal(size=(n, m))
        Ygt = self.rng.standard_normal(size=(m, 10))
        Cgt = np.zeros((n, 10))
        Cgt[tags, np.arange(10)] = 1

        d = self.rng.standard_normal(size=x.shape)
        d = d/ np.linalg.norm(d)
        d_flat_transpossed = np.ravel(d).T
        results_1 = []
        results_2 = []
        activation_softmax.weights = x
        fx = activation_softmax.loss(Ygt, Cgt)
        activation_softmax.gradient()
        grad_w = activation_softmax.grad_W
        d_grad_x = np.dot(d_flat_transpossed, np.ravel(grad_w)) #check if right
        init_epsilon = 0.5
        epsilon = 0.5
        for i in range (20):
            activation_softmax.weights = x + epsilon * d
            fx_plus_epsilon = activation_softmax.loss(Ygt, Cgt)
            results_1.extend([abs(fx_plus_epsilon - fx)])
            results_2.extend([abs(fx_plus_epsilon-fx-epsilon*d_grad_x)])
            epsilon =(init_epsilon * (0.5 ** i))

        return [results_1,results_2]


    def gradient_test_wholeNetwork(self):
        weights = []
        shapes = []
        d = []
        d_flat_transposseds = []
        x = []
        index =0
        for layer in self.network.layers:
            weights.extend([layer.weights])
            shapes.extend([weights[index].shape])
            temp_d = self.rng.standard_normal(size=shapes[index])
            temp_d = temp_d / np.linalg.norm(temp_d)
            d.extend([temp_d])
            d_flat_transpossed = np.ravel(temp_d).T
            d_flat_transposseds.extend([d_flat_transpossed])
            x.extend([self.rng.standard_normal(size=(shapes[index]))])
            index = index +1
        weights.extend([self.network.last_layer.weights])
        shapes.extend([weights[index].shape])
        temp_d = self.rng.standard_normal(size=shapes[index])
        temp_d = temp_d / np.linalg.norm(temp_d)
        d.extend([temp_d])
        d_flat_transpossed = np.ravel(temp_d).T
        d_flat_transposseds.extend([d_flat_transpossed])

        x.extend([self.rng.standard_normal(size=(shapes[index]))])
        Ygt = self.inputs
        Cgt = self.outputs

        for i in range (index):
            self.network.layers[i].weights = x[i]

        self.network.last_layer.weights = x[index]
        fx = self.network.loss(self.network.forward(Ygt), Cgt)
        self.network.backward()
        grads_w = []
        for i in range (index):
            grads_w.extend([np.ravel(self.network.layers[i].grad_W)])
        grads_w.extend([self.network.last_layer.grad_W])
        grads_x= []
        # print(index)
        # print(len(d_flat_transposseds))
        # exit(0)

        for i in range (index):
            grads_x.extend([np.dot(d_flat_transposseds[i], np.ravel(grads_w[i]))])
        grads_x.extend([np.dot(d_flat_transposseds[index], np.ravel(grads_w[index]))])
        arr = np.array(grads_x)
        results_1 = []
        results_2 = []
        init_epsilon = 0.5
        epsilon = 0.5
        for i in range (20):
            x_plus_epsilon = []
            for j in range(index+1):
                x_plus_epsilon.extend([x[j]+ epsilon * d[j]])

            for j in range(index):
                    self.network.layers[j].weights = x_plus_epsilon[j]

            self.network.last_layer.weights = x_plus_epsilon[index]
            fxs_plus_eplison = self.network.loss(self.network.forward(Ygt), Cgt)

            results_1.extend([abs(fxs_plus_eplison - fx)])
            results_2.extend([abs(fxs_plus_eplison-fx-(epsilon*arr))])
            epsilon =(init_epsilon * (0.5 ** i))

        return [results_1,results_2]

    def jaacobian_test(self):
        init_layer = self.network.layers[0]
        n, m = init_layer.weightslinear.shape
        dx = self.rng.standard_normal(size=(m, 1))
        dx = dx / np.linalg.norm(dx)
        dw = self.rng.standard_normal(size=(n, m))
        dw = dw / np.linalg.norm(dw)
        db = self.rng.standard_normal(size=(n, 1))
        db = db / np.linalg.norm(db)
        init_layer_1 = copy.deepcopy(init_layer)
        init_layer_2 = copy.deepcopy(init_layer)
        #models = [copy.deepcopy(init_layer) for _ in range(2)]
        #ds = [dw, db]
        x = self.rng.standard_normal(size=(m, 1))
        fx = init_layer.forward(x)
        init_epsilon = 0.5
        epsilon = 0.5

        #results = defaultdict(lambda: defaultdict(list))

        res = defaultdict(lambda: defaultdict(list))


        for i in range (20):
            init_layer_1.weightslinear = init_layer.weightslinear + epsilon * dw
            fx_plus_epsilon = init_layer_1.forward(x)

            res[0][0].append(np.linalg.norm(fx_plus_epsilon - fx))
            jaacobian = init_layer_1.jacobian_test(x, epsilon * dw, "weights")
            res[0][1].append(np.linalg.norm(fx_plus_epsilon - fx - jaacobian))

            init_layer_2.biases = init_layer.biases + epsilon * db
            fx_plus_epsilon = init_layer_2.forward(x)
            res[1][0].append(np.linalg.norm(fx_plus_epsilon - fx))
            jaacobian2 = init_layer_2.jacobian_test(x, epsilon * db, "biases")
            res[1][1].append(np.linalg.norm(fx_plus_epsilon - fx - jaacobian2))

            fx_plus_epsilon = init_layer.forward(x + epsilon * dx)
            jaacobian3 = init_layer.jacobian_test(x, epsilon * dx, "inputs")
            res[2][0].append(np.linalg.norm(fx_plus_epsilon - fx))
            res[2][1].append(np.linalg.norm(fx_plus_epsilon - fx - jaacobian3))

            epsilon =(init_epsilon * (0.5 ** i))

        return res

    def jaacobian_test_transpossed(self):
        init_layer = self.model.layers[0]
        n, m = init_layer.weights.shape
        x = self.rng.standard_normal(size=(m, 1))
        u = self.rng.standard_normal(size=(n, 1))
        vw = self.rng.standard_normal(size=(n, m))
        vb = self.rng.standard_normal(size=(n, 1))
        vx = self.rng.standard_normal(size=(m, 1))
        res = []

        jaacobian = init_layer.jacobian_test(x, vw, "weights")
        jaacobian_transposed = init_layer.jacobian_test_transpose(x, u, "weights")
        res[0] = abs(np.dot(u.T, jaacobian) - np.dot(np.ravel(vw).T , np.ravel(jaacobian_transposed)))

        jaacobian2 = init_layer.jacobian_test(x, vb, "biases")
        jaacobian_transposed2 = init_layer.jacobian_test_transpose(x, u, "biases")
        res[1] = abs(np.dot(u.T, jaacobian2) - np.dot(np.ravel(vw).T , np.ravel(jaacobian_transposed2)))

        jaacobian3 = init_layer.jacobian_test(x, vx,  "inputs")
        jaacobian_transposed3 = init_layer.jacobian_test_transpose(x, u, "inputs")
        res[2] = abs(np.dot(u.T, jaacobian3) - np.dot(np.ravel(vw).T , np.ravel(jaacobian_transposed3)))

        return res
