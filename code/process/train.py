import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import os
from model import FM, DeepFM, WDL, DCN
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    # load data
    file = '/content/sample_data/criteo_sample.txt'
    test_size = 0.2
    feature_columns, train, test = create_criteo_dataset(file=file, test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test

    # build model
    k = 8
    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]
    cin_size = [128, 128]

    att_activation='sigmoid'
    ffn_activation='prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10


    #FM
    #model = FM(feature_columns=feature_columns, k=k)
    #model = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)    
    #model = DeepFM(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)  
    #model = DCN(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout) 
    model = xDeepFM(feature_columns, hidden_units=hidden_units, cin_units=cin_size, dnn_dropout=dnn_dropout, )
    model.summary()

    # compile
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

    # model checkpoint
    check_path = '/content/sample_data/save_fm_weights'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,verbose=1, period=5)

    model.fit(train_X, train_y, epochs=epochs, callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],         
        batch_size=batch_size,
        validation_split=0.1
        )
    auc =  model.evaluate(test_X, test_y, batch_size=batch_size)[1]
    print('test AUC: %f' % auc)

'''
数据量：100000

实验超参数：
    test_size: 0.2
    k： 隐因子 8
    学习率：0.001
    batch_size： 4096
    epochs：10
    embed_dim：8
    dnn_dropout：0.5
    hidden_units：[256, 128, 64]

不同模型试验结果：
fm          AUC: 0.750980
wide & deep AUC: 0.746426
deepFM      AUC: 0.749247
DCN         AUC: 0.739412
xDeepFM     AUC: 
'''
