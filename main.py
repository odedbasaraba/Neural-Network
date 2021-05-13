# This is a sample Python script.
import numpy as np
from softmax import Activation_Softmax, SGD
from Neural_network import NeuralNetwork
from tests import MainTest
from plotter import ProjectPlotter
from train import Train
from matplotlib import pyplot as plt

import argparse
import os
import pathlib
from scipy.io import loadmat
from os.path import dirname, join as pjoin

def print_hi():
    # Use a breakpoint in the code line below to debug your script.
    print('Hi')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #dataset = loadmat("GMMData.mat")
    #dataset = loadmat("PeaksData.mat")
    dataset = loadmat("SwissRollData.mat")
    Yt = dataset["Yt"]
    Ct = dataset["Ct"]
    Yv = dataset["Yv"]
    Cv = dataset["Cv"]
    model = NeuralNetwork(Yt.shape[0], Ct.shape[0], 10, 5)
    # trainer = Train(model, SGD(0.1, model), 15, (Yt, Ct), (Yv, Cv), 32 )

    trainer = Train(model, Yt, Ct, Yv, Cv, 32, 15, SGD(0.1, model))
    results = trainer.training()
    # p = ProjectPlotter(5, 10, Yt, Ct )
    # plotter1 = p.plot_accuracy( results)
    # plotter2 = p.plot_loss(results)
    # plotter3 = p.gradient_test_plot()
    # plotter4 = p.plot_jacobian_tests()
    # plotter5 = p.gradient_test_plot2()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
