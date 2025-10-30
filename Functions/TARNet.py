import os
import logging
import fcntl
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

import seaborn as sns
import matplotlib.pyplot as plt

from geomloss import SamplesLoss


class CFR(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, classify=False):
        '''
        input_dim, output_dim:self-evident;
        hidden_dim: all neural networks in this model are using the same hidden layer dimension for simplicity
        '''
        super().__init__()

        self.classify = classify

        # func0 is used for predicting the outcome for control group, and func1 for predicting the outcome for treatment group.

        encoder = [nn.Linear(input_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2),
                   nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2)]
        self.encoder = nn.Sequential(*encoder)
        func0 = [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                 nn.Linear(hidden_dim, output_dim)]
        self.func0 = nn.Sequential(*func0)
        func1 = [nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                 nn.Linear(hidden_dim, output_dim)]
        self.func1 = nn.Sequential(*func1)

    def forward(self, X):
        # The input features (covariates) are first mapped in a hidden representation space
        # to measure the distance between Z0 and Z1
        Phi = self.encoder(X)

        # Pass the transformed features through treatments' predicting networks
        Y0 = self.func0(Phi)
        Y1 = self.func1(Phi)

        if self.classify:
            Y0 = torch.sigmoid(Y0)
            Y1 = torch.sigmoid(Y1)

        return Phi, Y0, Y1

class Wassertein_Loss(nn.Module):
    def __init__(self, p=2, blur=0.01):
        super(Wassertein_Loss, self).__init__()
        self.p = p
        self.blur = blur

    def forward(self, phi1, phi0):
        samples_loss = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur, backend="tensorized")
        imbalance_loss = samples_loss(phi1, phi0)
        return imbalance_loss
