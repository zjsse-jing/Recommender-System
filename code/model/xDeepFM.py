import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout, Embedding, Dense, Input
from tensorflow.keras.regularizers import l2

class Linear(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """
        :param feature_length: a scaler. the length of features
        :param w_reg: a scaler. the regularizer of linear
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                shape=(self.feature_length, 1),
                                regularizer=l2(self.w_reg),
                                trainable=True
                                )
    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1) #(batch_size, 1)
        return result

class DNN(Layer):
    def __init__(self, hidden_units, dnn_dropout=0, dnn_activation='relu'):
        """
        :param hidden_units: a list. a list of dnn hidden units
        :param dnn_dropout: a scaler. dropout of dnn
        :param dnn_activation: a string. activation function of dnn
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(unit, activation=dnn_activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class CIN(Layer):
    def __init__(self, cin_units, cin_reg=1e-4):
        """
        :param cin_units: a list.a list of the number of cin layers
        :param cin_reg: a scaler. the regularizer of cin
        """
        super(CIN, self).__init__()
        self.cin_size = cin_units
        self.l2_reg = cin_reg
    
    def build(self, input_shape):
        #get the number of embedding fields
        self.embedding_nums = input_shape[1]
        #a list of the number of cin
        self.field_nums = [self.embedding_nums] + self.cin_size
        #filters
        self.cin_w = {
            'CIN_W_'+str(i): self.add_weight(
                name='CIN_W_'+str(i),
                shape=(1, self.field_nums[0]*self.field_nums[i], self.field_nums[i+1]),
                initializer='uniform',
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums)-1)
        }
    
    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        #split dimension 2
        split_x_0 = tf.split(hidden_layers_results[0], dim, 2) # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_x_k = tf.split(hidden_layers_results[-1], dim, 2) # dim * (None, field_nums[i], 1)

            result_1 = tf.matmul(split_x_0, split_x_k, transpose_b=True) # (dim, None, field_nums[0], field_nums[i])

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums*self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1,0,2]) # (None, dim, field_nums[0]*field_nums[i])

            result_4 = tf.nn.conv1d(input = result_3, filters=self.cin_w['CIN_W_'+str(idx)], stride=1, padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0,1,2]) # (None, field_nums[i+1], dim)

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1) # (None, H_1+...+H_k, dim)
        result = tf.reduce_sum(result, axis=1) # (None, dim)
        
        return result


    
class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_units, dnn_dropout, dnn_activation='relu', embed_reg=1e-6, cin_reg=1e-6, w_reg=1e-6):
        """
        :param feature_columns: a list. sparse feature information
        :param hidden_units: a list. a list of dnn hidden units
        :param cin_units: a list.a list of the number of cin layers
        :param dnn_dropout: a scaler. dropout of dnn
        :param dnn_activation: a string. activation function of dnn
        :param embed_reg: a scaler. the regularizer of embedding
        :param cin_reg: a scaler. the regularizer of cin
        :param w_reg: a scaler. the regularizer of linear
        """
        super(xDeepFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_'+str(i): Embedding(input_dim = feat['feat_num'], 
                                        output_dim=self.embed_dim, 
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg)
                                        )
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']

        self.linear = Linear(self.feature_length, w_reg)
        self.cin = CIN(cin_units, cin_reg)
        self.dnn = DNN(hidden_units, dnn_dropout, dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        #linear
        linear_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        linear_output = self.linear(linear_inputs) # (batch_size, 1)

        #cin
        embed = [self.embed_layers['embed_{}'.format(i)](inputs[:,i]) for i in range(inputs.shape[1])]
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1,0,2])
        cin_out = self.cin(embed_matrix) # (batch_size, dim)
        cin_out = self.cin_dense(cin_out) # (batch_size, 1)

        #dnn
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1]*embed_matrix.shape[2]))
        dnn_output = self.dnn(embed_vector)
        dnn_output = self.dnn_dense(dnn_output)

        #output sigmoid
        output = tf.nn.sigmoid(linear_output + cin_out + dnn_output + self.bias)
        return output
    
    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs = sparse_inputs, outputs = self.call(sparse_inputs)).summary()
