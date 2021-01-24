from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow as tf


class AttentionBeforeConvolution(Layer):
    """
    Attention operation.
    Follows the work "Attention-based Convolutional Neural Networks for Sentence Classification"
    # Input shape
        3D tensor with shape: `(batches, steps, embedding_dimension)`.
    # Output shape
        3D tensor with shape: `(batches, steps, 2*embedding_dimension)`.
    How to use:
    Just put it on top of an INPUT/EMBEDDING Layer.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a TextCNN layer (for classification/regression) or whatever...
    """

    def __init__(self, max_len=100,
                 W_regularizer=None, u_regularizer=None, v_regularizer=None,
                 W_constraint=None, u_constraint=None, v_constraint=None,
                 bias=False,  attention_hidden_dim=100, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.v_regularizer = regularizers.get(v_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.v_constraint = constraints.get(v_constraint)

        self.attention_hidden_dim = attention_hidden_dim
        self.max_len = max_len

        self.bias = bias
        super(AttentionBeforeConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        the input shape is：(batches, seq_len, embedding_dims)
        :param input_shape:
        :return:
        '''
        assert len(input_shape) == 3

        self.attention_W = self.add_weight(shape=(int(input_shape[-1]), self.attention_hidden_dim,),
                                           initializer=self.init,
                                           name='{}_W'.format(self.name),
                                           regularizer=self.W_regularizer
                                           )

        self.attention_U = self.add_weight(shape=(int(input_shape[-1]), self.attention_hidden_dim,),
                                           initializer=self.init,
                                           name='{}_U'.format(self.name),
                                           regularizer=self.u_regularizer,
                                           )

        self.attention_V = self.add_weight(shape=(self.attention_hidden_dim, 1,),
                                           initializer=self.init,
                                           name='{}_V'.format(self.name),
                                           regularizer=self.v_regularizer,
                                           )

        # self.attention_W = self.add_weight(shape=tf.TensorShape((int(input_shape[-1]), self.attention_hidden_dim,)),
        #                                    initializer=self.init,
        #                                    name='{}_W'.format(self.name),
        #                                    regularizer=self.W_regularizer,
        #                                    constraint=self.W_constraint
        #                                    )
        #
        # self.attention_U = self.add_weight(shape=tf.TensorShape((int(input_shape[-1]), self.attention_hidden_dim,)),
        #                                    initializer=self.init,
        #                                    name='{}_U'.format(self.name),
        #                                    regularizer=self.u_regularizer,
        #                                    constraint=self.u_constraint
        #                                    )
        #
        # self.attention_V = self.add_weight(shape=tf.TensorShape((self.attention_hidden_dim, 1,)),
        #                                    initializer=self.init,
        #                                    name='{}_V'.format(self.name),
        #                                    regularizer=self.v_regularizer,
        #                                    constraint=self.v_constraint
        #                                    )


        if self.bias:
            self.b = self.add_weight(shape=tf.TensorShape((int(input_shape[-1]),)),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.v_regularizer,
                                     constraint=self.v_constraint)

        super(AttentionBeforeConvolution, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        assert x.shape.ndims == 3
        embed_dim = x.shape.as_list()[-1]

        input_att = tf.split(x, self.max_len, axis=1)
        output_att = list()

        for index, x_i in enumerate(input_att):
            x_i = tf.reshape(x_i, [-1, embed_dim])
            c_i = self.attention(x_i, input_att, index)
            inp = tf.concat([x_i, c_i], axis=1)
            output_att.append(inp)

        attention_out = tf.reshape(tf.concat(output_att, axis=1),
                                [-1, self.max_len, embed_dim*2],
                                name="attention_out")
        return attention_out

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        embed_dims = x_i.shape.as_list()[-1]

        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, embed_dims])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.max_len, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, embed_dims])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.max_len - 1, embed_dims])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.max_len, input_shape[-1]*2
