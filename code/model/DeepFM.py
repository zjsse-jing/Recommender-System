import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Embedding, Dropout, Input
from tensorflow.keras.regularizers import l2

class FM(Layer):
    """
    wide part
    """
    def __init__(self, feature_length, w_reg=1e-6):
        """
        :param feature_length:a scalar. the length of features
        :param fm_w_reg:a scalar. the regularizer of w in fm
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg
    def build(self, input_shape):
        self.w = self.add_weight(name='w', 
                                shape=(self.feature_length,1),
                                initializer='random_normal',
                                regularizer=l2(self.w_reg),
                                trainable=True

        )
    def call(self, inputs, **kwargs):
        """
        :param inputs: a dict with shape (batch_size, {'sparse_inputs', 'embed_inputs'})
        sparse_inputs is 2D tensor with shape (batch_size, sum(field_num))
        embed_inputs  is 3D tensor with shape (batch_size, fields, embed_dim)
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        #first order: fm 构建一阶特征取消占用内存的tf.one_hot,改用tf.nn.embedding_lookup，通过映射实现
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1) #(batch_size, 1)
        #second_order 
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True)) #(batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True) #(batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2) #(batch_size, 1)
        return first_order + second_order


class DNN(Layer):
    """
    Deep part
    """
    def __init__(self, hidden_units=[200, 200, 200], activation='relu', dnn_dropout=0.):
        """
        :param hidden_units: a list. a list of dnn hidden units
        :param dnn_dropout: a scalar. dropout of dnn
        :param activation: a string. activation of dnn
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x



class DeepFM(Model):
    def __init__(self, feature_columns, dnn_dropout=0., hidden_units=[200,200,200], activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
        """
        :param feature_columns: a list. sparse column feature information 
        :param hidden_units: a list. a list of dnn hidden units
        :param dnn_dropout: a scalar. dropout of dnn
        :param activation: a string. activation of dnn
        :param fm_w_reg:a scalar. the regularizer of w in fm
        :param embed_reg:a scalar. the regularizer of embedding
        """
        super(DeepFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_'+str(i): Embedding(input_dim=feat['feat_num'],
                                        input_length=1,
                                        output_dim=feat['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']

        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.fm = FM(self.feature_length, fm_w_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)


    def call(self, inputs, **kwargs):
        """
        wide deep 两部分共享embedding
        """
        sparse_inputs = inputs
        #embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])], axis=-1) #(batch_size, embed_dim * fields)
        
        #wide
        sparse_inputs = sparse_inputs + tf.convert_to_tensor(self.index_mapping)
        wide_inputs = {'sparse_inputs':sparse_inputs, 'embed_inputs':tf.reshape(sparse_embed, shape=(-1, sparse_inputs.shape[1], self.embed_dim))}
        wide_outputs = self.fm(wide_inputs) #(batch_size, 1)

        #deep
        deep_outputs = self.dnn(sparse_embed)
        deep_outputs = self.dense(deep_outputs) #(batch_size, 1)

        #output
        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns), ), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()