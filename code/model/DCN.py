'''
model: Deep & Cross Network for Ad Click Predictions
'''
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input

class CrossNetWork(Layer):
    def __init__(self, layer_num, cross_w_reg, cross_b_reg):
        """
        :param layer_num: a scaler.the depth of cross network
        :param cross_w_reg:a scaler. The regularizer of cross network.
        :param croww_b_reg:a scaler. The regularizer of cross network.
        """
        super(CrossNetWork, self).__init__()
        self.layer_num = layer_num
        self.cross_w_reg = cross_w_reg
        self.cross_b_reg = cross_b_reg

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [self.add_weight(name='w_'+str(i),
                                            shape=(dim, 1),
                                            initializer='random_normal',
                                            regularizer=l2(self.cross_w_reg),
                                            trainable=True
                                            )   for i in range(self.layer_num)]
        self.cross_bias = [self.add_weight(name='b_'+str(i),
                                            shape=(dim, 1),
                                            initializer='random_normal',
                                            regularizer=l2(self.cross_b_reg),
                                            trainable=True
                                            )  for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2) #(batch_size, dim, 1)
        x_l = x_0 #(None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0]) #(batch_size, dim, dim)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  #(batch_size, dim, 1)

        x_l = tf.squeeze(x_l, axis=2) #(batch_size, dim)
        return x_l


class DeepNetWork(Layer):
    def __init__(self, hidden_units, activation, dnn_dropout):
        """
        :param hidden_units: a list. dnn hidden units
        :param activation:a string. activation of dnn 
        :param dnn_dropout: a scaler. dropout of dnn
        """
        super(DeepNetWork, self).__init__()
        self.dnn_network = [Dense(units = unit, activation = activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu', dnn_dropout=0, embed_reg=1e-6, cross_w_reg=1e-6, cross_b_reg=1e-6):
        """
        :param feature_columns:a list. a sparse feature information
        :param hidden_units: a list. dnn hidden units
        :param activation:a string. activation of dnn 
        :param dnn_dropout: a scaler. dropout of dnn
        :param embed_reg:a scaler. The regularizer of embedding.
        :param cross_w_reg:a scaler. The regularizer of cross network.
        :param cross_b_reg:a scaler. The regularizer of cross network.
        """
        super(DCN, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layer = {
            'emb_'+str(i): Embedding(input_dim=feat['feat_num'],
                                    input_length=1,
                                    output_dim = feat['embed_dim'],
                                    embeddings_initializer='random_uniform',
                                    embeddings_regularizer=l2(embed_reg)
                                    ) for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetWork(self.layer_num, cross_w_reg, cross_b_reg)
        self.deep_network = DeepNetWork(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)


    def call(self, inputs, **kwargs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layer['emb_{}'.format(i)](sparse_inputs[:,i]) 
            for i in range(sparse_inputs.shape[1])], axis=1)
        x = sparse_embed

        #cross
        cross_x = self.cross_network(x)

        #deep
        dnn_x = self.deep_network(x)

        #concat
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))

        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()