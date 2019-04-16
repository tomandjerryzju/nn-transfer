# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from nn_transfer import transfer, util

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from keras.layers import Input, Dense
from keras.models import Model
import cPickle

class RelationNetwork_pytorch(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork_pytorch, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


def RelationNetwork_keras(input_size, hidden_size):
    input = Input(shape=(input_size,))
    x_model = Dense(hidden_size, activation='relu', name='fc1')(input)
    predictions = Dense(1, activation='sigmoid', name='fc2')(x_model)
    model = Model(inputs=input, outputs=predictions)

    return model


if __name__ == '__main__':
    relation_network_pytorch = RelationNetwork_pytorch(2048 * 2, 400)
    checkpoint_path = "/Users/hyc/workspace/LearningToCompare_FSL/imagenet_resnet2048/models/imagenet_resnet2048_relation_network_5way_10shot.pkl"
    relation_network_pytorch.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print("load relation network success")
    print(relation_network_pytorch)
    state_dict = relation_network_pytorch.state_dict()
    print(util.state_dict_layer_names(state_dict))

    relation_network_keras = RelationNetwork_keras(2048 * 2, 400)

    transfer.pytorch_to_keras(relation_network_pytorch, relation_network_keras)

    # cPickle.dump(fcl.get_weights(), open(output_model_file + '.pkl', 'wb'))

    data = torch.rand(6, 2048 * 2)
    data_keras = data.numpy()
    data_pytorch = Variable(data, requires_grad=False)
    keras_pred = relation_network_keras.predict(data_keras)
    pytorch_pred = relation_network_pytorch(data_pytorch).data.numpy()

    cPickle.dump(relation_network_keras.get_weights(), open('relation_network_keras.pkl', 'wb'))
    relation_network_keras.save('relation_network_keras.h5')

    # assert keras_pred.shape == pytorch_pred.shape
    #
    # plt.axis('Off')
    # plt.imshow(keras_pred)
    # plt.show()
    # plt.axis('Off')
    # plt.imshow(pytorch_pred)
    # plt.show()





