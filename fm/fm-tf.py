'''
tensorflow实现fm的二分类
dataset: load_breast_cancer

'''
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
K = tf.keras.backend #Keras后端API

class FMLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=4, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FMLayer, self).__init__(**kwargs)
    

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                    shape=(self.input_dim, self.output_dim),
                                    initializer='glorot_uniform',
                                    trainable=True
                                    )
        super(FMLayer, self).build(input_shape)                            

    #FM的二阶交叉项的计算公式
    def call(self, x):
        a = K.pow(K.dot(x, self.kernel), 2)
        b = K.dot(K.pow(x,2), K.pow(self.kernel, 2))
        return K.sum(a-b, 1, keepdims=True) * 0.5

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    #线性回归
    linear = tf.keras.layers.Dense(units=1, 
                                    bias_regularizer=tf.keras.regularizers.l2(0.01),
                                    kernel_regularizer=tf.keras.regularizers.l1(0.02),)(inputs)
    #FM 二阶交叉项
    cross =  FMLayer(feature_dim)(inputs)
    #FM模型（线性回归+二阶交叉项）
    add = tf.keras.layers.Add()([linear, cross])
    predict = tf.keras.layers.Activation('sigmoid')(add)   

    model = tf.keras.Model(inputs=inputs, outputs=predict)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy])
    return model                              

if __name__ == '__main__':
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test  = train_test_split(data.data, data.target, test_size=0.2, random_state=11, stratify=data.target)
    fm = FM(30)
    fm.fit(x_train, y_train, epochs=5, batch_size=20, validation_data=(x_test, y_test))
    fm.summary
