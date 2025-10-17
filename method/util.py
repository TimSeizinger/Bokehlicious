from typing import Literal
from torch import nn

from method.nn_util import SimpleGate, ChannelAttention, SimplifiedChannelAttention

Activation_Function = Literal['GELU', 'SG', 'Identity', 'ReLU', 'LReLU', 'Tanh', 'Sigmoid']
Attention_Mechanism = Literal['CA', 'SCA']

def get_activation(activation_type: Activation_Function = 'Identity'):
    """
    Get the activation function from the activation type.
    :param activation_type: Activation type
    :return: Activation function
    """
    if activation_type == 'GELU':
        return nn.GELU()
    elif activation_type == 'SG':
        return SimpleGate()
    elif activation_type == 'Identity':
        return nn.Identity()
    elif activation_type == 'ReLU':
        return nn.ReLU()
    elif activation_type == 'LReLU':
        return nn.LeakyReLU()
    elif activation_type == 'Tanh':
        return nn.Tanh()
    elif activation_type == 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f'Activation type {activation_type} is not implemented.')


def get_cnn_attention(attention_type: Attention_Mechanism = None):
    if attention_type == 'CA':
        return ChannelAttention
    elif attention_type == 'SCA':
        return SimplifiedChannelAttention
    else:
        raise NotImplementedError(f'Attention type {attention_type} is not implemented.')
