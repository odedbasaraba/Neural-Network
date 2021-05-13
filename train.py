import numpy as np
from sklearn.metrics import accuracy_score

class Train:
    def __init__(self,network, input_train, target_train, input_valid, target_valid , mini_batch , epochs, sgd1):
        self.network = network
        self.input_train = input_train
        self.target_train = target_train
        self.input_valid = input_valid
        self.target_valid = target_valid
        self.mini_batch = mini_batch
        self.epochs = epochs
        self.all_loss = []
        self.validation_success = []
        self.training_success = []
        self.sgd = sgd1

    def accuracy(self, our_output, real_output):
        max_args = np.argmax(our_output, axis=0)
        counter = 0
        x = np.where(real_output.T == 1)[1]
        for i in range(0,len(x)):
            if max_args[i] == x[i]:
                counter += 1
        return counter/len(x)



    def training(self):
        num_features = self.input_train.shape[0]
        m = self.input_train.shape[1]
        rng = np.random.default_rng(None)
        inputs_and_targets = np.vstack([self.input_train, self.target_train])
        for epochs in range(1, self.epochs+1):
            current_loss = 0
            rng.shuffle(inputs_and_targets, axis=1)
            for i in range(0, m, self.mini_batch):
                cur_input = inputs_and_targets[:num_features]
                cur_target = inputs_and_targets[num_features:]
                inputs_batch = cur_input[:, i:i + self.mini_batch]
                targets_batch = cur_target[:, i:i + self.mini_batch]
                new_input = self.network.forward(inputs_batch)
                current_loss = current_loss + self.network.loss(new_input, targets_batch)
                self.network.backward()
                self.sgd.step_size()
            current_loss = current_loss / self.mini_batch
            self.all_loss.append(current_loss)
            train_preds = self.network.last_layer.forward(self.network.forward(self.input_train))
            train_acc = self.accuracy(train_preds, self.target_train) # HANDLE THE ACCURACY
            val_preds = self.network.last_layer.forward(self.network.forward(self.input_valid))
            val_acc = self.accuracy(val_preds, self.target_valid)  # HANDLE THE ACCURACY
            print("~~~~~~~~~~~~~~~~~")
            print(f" Epoch {epochs}\n Loss: {current_loss:.3f}\n Train Accuracy: {train_acc:.3f}\n Validation Accuracy: {val_acc:.3f}")
            self.training_success.append(train_acc)
            self.validation_success.append(val_acc)
        return [self.mini_batch, self.network.hidden_layer_size , self.network.l_layers, self.all_loss, self.training_success, self.validation_success]


