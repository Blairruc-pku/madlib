import os
import pdb
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
from DeepFM import DeepFM
from grad_avg.DataReader import FeatureDictionary, DataParser
from grad_avg import config
from grad_avg.metrics import accuracy


def DeepFM_DA(DeepFM):
    def __init__(self, **kwargs):
        DeepFM.__init__(self, **kwargs)




def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices



dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "gd",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": accuracy,
    "random_seed": config.RANDOM_SEED
}


# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()


# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                       numeric_cols=config.NUMERIC_COLS,
                       ignore_cols=config.IGNORE_COLS)
data_parser = DataParser(feat_dict=fd)
Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)

# data_all = np.hstack((np.array(Xi_train),np.array(Xv_train),np.array(y_train).reshape(-1,1)))
# df_all = pd.DataFrame(data_all)
# df_all.to_csv('data/driving_tf.csv',header=0)
# Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

dfm_params["feature_size"] = fd.feat_dim
dfm_params["field_size"] = len(Xi_train[0])

model_global = DeepFM(**dfm_params)
model_global.fit(Xi_train, Xv_train, y_train)

# _get = lambda x, l: [x[i] for i in l]
# for i, (train_idx, valid_idx) in enumerate(folds):
#     Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
#     Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
#     model_global.fit(Xi_train_, Xv_train_, y_train_)
#     pdb.set_trace()



