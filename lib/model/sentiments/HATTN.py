from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        weighted_input = x*K.expand_dims(weights,2)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
