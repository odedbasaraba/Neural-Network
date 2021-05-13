
import numpy as np
# import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

from tests import MainTest

class Plotter:
    def __init__(self):
        self.s = None
        # sns.set_context("paper", font_scale=3, rc={"font.size": 8, "axes.labelsize": 5})

        # plt.rcParams["figure.figsize"] = 6.4, 4.8
        # plt.rcParams["font.size"] = 14
        # plt.rcParams["text.usetex"] = True
        # plt.rcParams["font.family"] = "monospace"

class ProjectPlotter(Plotter):
    def __init__(self, l_layers, hidden_layer_size, input, output):
        super().__init__()
        self.test = MainTest(l_layers, hidden_layer_size, input, output)

    def gradient_test_plot(self):
        res = self.test.gradient_test()
        plt.title("2.1.1 : Gradient Test")
        plt.xlabel("epochs")
        plt.xticks(np.arange(20), np.arange(1, 21))
        plt.plot(res[0],
             label="$\\left|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x})\\right|$")
        plt.plot(res[1],
             label="$\\left|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x}) - \\epsilon \\mathbf{d}^{\\top} \\nabla f \\right|$")
        plt.legend(loc="lower left")
        plt.yscale("log")
        plt.show()
        plt.clf()

    def gradient_test_plot2(self):
        res = self.test.gradient_test_wholeNetwork()
        plt.title("2.2.3 : Gradient Test Network")
        plt.xlabel("epochs")
        plt.xticks(np.arange(20), np.arange(1, 21))
        plt.plot(res[0],
             label="$\\left|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x})\\right|$")
        plt.plot(res[1],
             label="$\\left|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x}) - \\epsilon \\mathbf{d}^{\\top} \\nabla f \\right|$")
        plt.legend(loc="lower left")
        plt.yscale("log")
        plt.show()
        plt.clf()



    def plot_jacobian_tests(self):
        results = self.test.jaacobian_test()

        def plot_jacobian_test(param_for_title, param):
            plt.title(f"Jacobian Test w.r.t {param_for_title}")
            plt.xlabel("\\# iterations")
            plt.xticks(np.arange(20), np.arange(1, 21))
            plt.plot(results[param][0],
                     label="$\\|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x})\\|$")
            plt.plot(results[param][1],
                     label="$\\|f(\\mathbf{x} + \\epsilon \\mathbf{d}) - f(\\mathbf{x}) - \\mathtt{JacMV}(\\mathbf{x}, \\epsilon \\mathbf{d}) \\|$")
            plt.legend(loc="lower left")
            plt.yscale("log")
            plt.show()
            plt.clf()

        plot_jacobian_test("Weights", 0)
        plot_jacobian_test("Biases", 1)
        plot_jacobian_test("Inputs", 2)

    def plot_accuracy(self, training_results):



        results = training_results
        hp = results["hyperparams"]
        if hp["n_layers"] == 1:
            nn_info = "Softmax"
        else:
            nn_info = f"NN(hidden={hp['hidden']}, layers={hp['n_layers']})"

        train_accs = results["accuracy"]["train"]
        iterations = list(np.arange(1, len(train_accs)+1))
        # # print(train_accs)
        # # print(iterations)
        # plt.show(1,2)

        title = ("Training vs. Validation Accuracy\n"
                 f"{nn_info}\n"
                 f"Batch Size of {hp['batch_size']}")

        plt.title(title)

        plt.plot(iterations, train_accs)
        plt.show()
        # new_arr = list(zip(train_accs,iterations))
        # zip(*new_arr)
        #
        # plt.scatter(*zip(*new_arr))
        # plt.show()
        # print(new_arr)
        # df = pd.DataFrame(new_arr, columns=["x", "y"])
        #
        # # df["val"] = pd.Series([1, -1, 1]).apply(lambda x: "red" if x == 1 else "blue")
        #
        # sns.scatterplot(df["x"], df["y"]).plot()

        # print(train_accs)
        # fake = [[1,2], [2,4], [3,5]]
        # val_accs = results["accuracy"]["val"]
        #
        title = ("Training vs. Validation Accuracy\n"
                 f"{nn_info}\n"
                 f"Batch Size of {hp['batch_size']}")
        # # train_accs2 = sns.load_dataset(fake)
        #
        # ax1 = sns.boxplot(x="Epoch", data=val_accs)
        # ax1.set_title("Training Accuracy\n"
        #          f"{nn_info}\n"
        #          f"Batch Size of {hp['batch_size']}")
        #
        # ax1.set(xlim=(0, len(train_accs)))
        # ax1.set(ylim=(0, 21))
        #
        # val_accs2 = sns.load_dataset(val_accs)
        #
        # ax2 = sns.boxplot(x="Epoch", data=val_accs2)
        # ax2.set_title("Validation Accuracy\n"
        #          f"{nn_info}\n"
        #          f"Batch Size of {hp['batch_size']}")
        #
        # ax2.set(xlim=(0, len(val_accs)))
        # ax2.set(ylim=(0, 21))
        #

        # plt.title(title)
        # x = np.arange(len(train_accs))
        # plt.xticks(x, x + 1)
        # y = np.linspace(0, 1, 21)
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # tips = sns.load_dataset("tips")

        # # Create scatter plots
        # g = sns.FacetGrid(tips, col="sex", row="smoker", margin_titles=True)
        # g.map(sns.plt.scatter, "total_bill", "tip")
        # g = sns.FacetGrid(tips, col="sex", row="smoker", margin_titles=True)

        # sns.swarmplot(x="Epoch", y="Accuracy", data=train_accs, ax=ax1)
        # plt.show()
        #
        # sns.swarmplot(x="Epoch", y="Accuracy", data=val_accs, ax=ax2)
        # plt.show()
        #
        # plt.plot(train_accs, "-o", label="Training")
        # plt.plot(val_accs, "-o", label="Validation")
        # plt.legend()
        # plt.show()
        # plt.clf()

    def plot_loss(self, training_results):
        results = training_results
        hp = results["hyperparams"]
        if hp["n_layers"] == 1:
            nn_info = "Softmax"
        else:
            nn_info = f"NN(hidden={hp['hidden']}, layers={hp['n_layers']})"

        losses = results["losses"]

        title = ("Loss Graph\n"
                 f"{nn_info}\n"
                 f"Batch Size of {hp['batch_size']}")
        plt.title(title)
        x = np.arange(len(losses))
        plt.xticks(x, x + 1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(losses, "-o")

        plt.show()
        plt.clf()
