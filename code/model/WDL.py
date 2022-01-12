'''
Wide & Deep Learning for Recommender Systems
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input, Layer
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

import os

class DNN(Layer):
    def __init__(self, hidden_units, dnn_dropout=0, activation='relu'):
        """
        :param hidden_units:a neural network units
        :param dnn_dropout: dropout of dnn
        :param activation:a activation function of dnn
        """
        super(DNN, self).__init__()
        self.dnn_network=[Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """
        :param feature_length: the length of features
        :param w_reg:           The regularization coefficient of parameter w.
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name="w", 
                                 shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_reg), 
                                 trainable=True)
                            
    
    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)
        return result

class WideDeep(Model):
    def __init__(self,  feature_columns, hidden_units, dnn_dropout=0, activation='relu', embed_reg=1e-6, w_reg=1e-6):
        """
        :param feature_columns: a list sparse feature information
        :param hidden_units:    a neural network units
        :param dnn_dropout:     dropout of dnn
        :param activation:      a activation function of dnn
        :param embed_reg:       the regularizer of embedding
        :param w_reg:           the regularizer of linear
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layes = {
            'embed_'+str(i): Embedding(input_dim=feat['feat_num'], 
                                       input_length=1,
                                       output_dim=feat['embed_dim'],
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg)
                                       )  for i, feat in enumerate(self.sparse_feature_columns)
        }  
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.dnn_network = DNN(hidden_units, dnn_dropout, activation)
        self.linear = Linear(self.feature_length, w_reg)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_embed = tf.concat([self.embed_layes['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])], axis=-1)
        x = sparse_embed  #(batch_size, field * embed_dim)
        #wide sparse feature
        wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        wide_out = self.linear(wide_inputs)

        #deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)

        #out
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs
    
    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

