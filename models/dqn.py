"""Docstring """

import sonnet as snt
import tensorflow as tf

class DQN(snt.AbstractModule):
    """ Docstring for DQN """
    def __init__(self, hidden_size, output_size,
                 nonlinearity=tf.tanh, name="dqn"):
        """Docstring explaining __init__args"""
        super(DQN, self).__init__(name=name)
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._nonlinearity = nonlinearity

    def _build(self, inputs):
        """Compute output tensor from input teonsor"""
        lin_x_to_h = snt.Linear(output_size=self._hidden_size,
                                name="x_to_h")
        lin_h_to_o = snt.Linear(output_size=self._output_size,
                                name="h_to_o")
        return lin_h_to_o(self._nonlinearity(lin_x_to_h(inputs)))
