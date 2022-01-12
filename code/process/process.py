import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split


def create_criteo_dataset(file, embed_dim=8, test_size=0.2):
    """
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    return: feature columns, train, test
    """
    df = pd.read_csv(file, sep='\t')

    sparse_features = ['C'+str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]

    features = sparse_features + dense_features

    df[sparse_features] = df[sparse_features].fillna('-1')
    df[dense_features] = df[dense_features].fillna(0)

    ## Bin continuous data into intervals.
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    df[dense_features] = est.fit_transform(df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    '''
    label	I1	I2	I3	I4	I5	I6	I7	I8	I9	I10	I11	I12	I13	C1	C2	C3	C4	C5	C6	C7	C8	C9	C10	C11	C12	C13	C14	C15	C16	C17	C18	C19	C20	C21	C22	C23	C24	C25	C26
    0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	16.0	1.0	0.0	0.0	229	250	43140	12137	23	5	6591	27	2	7231	2691	8957	424	4	2809	18686	9	2422	150	3	1163	0	2	9570	41	5604
    '''

    feature_columns = [{'feat':feat, 'feat_num':int(df[feat]).max+1, 'embed_dim':embed_dim}  for feat in features]

    train, test = train_test_split(df, test_size)
    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)

'''
数据处理部分：
采用部分criteo数据：10000条

缺失值填充
密集数据I1-I13离散化分桶（bins=100），稀疏数据C1-C26编码(LabelEncoder)
切分数据集
'''
