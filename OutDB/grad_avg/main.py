import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
import time
import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser

sys.path.append("..")
from DeepFM import DeepFM
from DeepFM1 import DeepFM1
from DeepFM2 import DeepFM2

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)
gradients_avg = []


def fit_with_grads(model, Xi, Xv, y, clear_grads):
    feed_dict = {model.feat_index: Xi,
                 model.feat_value: Xv,
                 model.label: y,
                 model.dropout_keep_fm: model.dropout_fm,
                 model.dropout_keep_deep: model.dropout_deep,
                 model.train_phase: True}
    global gradients_avg
    for i, placeholder in enumerate(model.grad_placeholders):
        if i < 2:
            feed_dict[placeholder] = np.stack([g[i].values for g in gradients_avg], axis=0).mean(axis=0)
            continue
        feed_dict[placeholder] = np.stack([g[i] for g in gradients_avg], axis=0).mean(axis=0)
    model.sess.run(model.train_op, feed_dict=feed_dict)
    if clear_grads:
        gradients_avg = []

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
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices

def gradients_compute(xi, xv, y, model):
    feed_dict = {model.feat_index: xi,
                 model.feat_value: xv,
                 model.label: y,
                 model.dropout_keep_fm: model.dropout_fm,
                 model.dropout_keep_deep: model.dropout_deep,
                 model.train_phase: True}

    if model.average_gradients == 1:
        loss, _ = model.sess.run([model.loss, model.train_op], feed_dict=feed_dict)
    else:
        loss, grads = model.sess.run([model.loss, model.grad_op], feed_dict=feed_dict)
        global gradients_avg
        gradients_avg.append(grads)
    return loss

def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, times):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    total_batch = int(len(y_train) / dfm_params["batch_size"])
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    Xi1_train_, Xv1_train_, y1_train_ = _get(Xi_train, folds[0][0]), _get(Xv_train, folds[0][0]), _get(y_train,
                                                                                                       folds[0][0])
    Xi2_train_, Xv2_train_, y2_train_ = _get(Xi_train, folds[1][0]), _get(Xv_train, folds[1][0]), _get(y_train,
                                                                                                       folds[1][0])
    Xi3_train_, Xv3_train_, y3_train_ = _get(Xi_train, folds[2][0]), _get(Xv_train, folds[2][0]), _get(y_train,
                                                                                                       folds[2][0])
    Xi1_valid_, Xv1_valid_, y1_valid_ = _get(Xi_train, folds[0][1]), _get(Xv_train, folds[0][1]), _get(y_train,
                                                                                                       folds[0][1])
    Xi2_valid_, Xv2_valid_, y2_valid_ = _get(Xi_train, folds[1][1]), _get(Xv_train, folds[1][1]), _get(y_train,
                                                                                                       folds[1][1])
    Xi3_valid_, Xv3_valid_, y3_valid_ = _get(Xi_train, folds[2][1]), _get(Xv_train, folds[2][1]), _get(y_train,
                                                                                                       folds[2][1])
    dfm1 = DeepFM2(**dfm_params)
    dfm2 = DeepFM2(**dfm_params)
    dfm3 = DeepFM2(**dfm_params)
    grad_count = 0
    for t in range(dfm_params["epoch"]):
        for i in range(total_batch):

            # compute gradients
            Xi1_batch, Xv1_batch, y1_batch = dfm1.get_batch(Xi1_train_, Xv1_train_, y1_train_, dfm_params["batch_size"], i)
            if len(y1_batch) != 1024 :
                break
            gradients_compute(Xi1_batch, Xv1_batch, y1_batch, dfm1)
            grad_count = grad_count + 1
            Xi2_batch, Xv2_batch, y2_batch = dfm2.get_batch(Xi2_train_, Xv2_train_, y2_train_, dfm_params["batch_size"], i)
            if len(y2_batch) != 1024:
                break
            gradients_compute(Xi2_batch, Xv2_batch, y2_batch, dfm2)
            grad_count = grad_count + 1
            Xi3_batch, Xv3_batch, y3_batch = dfm3.get_batch(Xi3_train_, Xv3_train_, y3_train_, dfm_params["batch_size"], i)
            if len(y3_batch) != 1024 :
                break
            gradients_compute(Xi1_batch, Xv1_batch, y1_batch, dfm3)
            grad_count = grad_count + 1
            if grad_count == dfm_params["average_gradients"] :
                grad_count = 0
                # fit_with grads_avg
                clear_grads = False
                fit_with_grads(dfm1, Xi1_batch, Xv1_batch, y1_batch, clear_grads)
                fit_with_grads(dfm2, Xi2_batch, Xv2_batch, y2_batch, clear_grads)
                clear_grads = True
                fit_with_grads(dfm3, Xi3_batch, Xv3_batch, y3_batch, clear_grads)

        # evaluate
        dfm1.fit_evaluate(Xi1_train_, Xv1_train_, y1_train_, Xi1_valid_, Xv1_valid_, y1_valid_, t)
        dfm2.fit_evaluate(Xi2_train_, Xv2_train_, y2_train_, Xi2_valid_, Xv2_valid_, y2_valid_, t)
        dfm3.fit_evaluate(Xi3_train_, Xv3_train_, y3_train_, Xi3_valid_, Xv3_valid_, y3_valid_, t)
    # plot1
    y_train_meta[folds[0][1], 0] = dfm1.predict(Xi1_valid_, Xv1_valid_)
    y_test_meta[:, 0] += dfm1.predict(Xi_test, Xv_test)
    gini_results_cv[0] = gini_norm(y1_valid_, y_train_meta[folds[0][1]])
    gini_results_epoch_train[0] = dfm1.train_result
    gini_results_epoch_valid[0] = dfm1.valid_result
    # plot1
    y_train_meta[folds[1][1], 0] = dfm2.predict(Xi2_valid_, Xv2_valid_)
    y_test_meta[:, 0] += dfm1.predict(Xi_test, Xv_test)
    gini_results_cv[1] = gini_norm(y1_valid_, y_train_meta[folds[1][1]])
    gini_results_epoch_train[1] = dfm2.train_result
    gini_results_epoch_valid[1] = dfm2.valid_result
    # plot1
    y_train_meta[folds[2][1], 0] = dfm3.predict(Xi3_valid_, Xv3_valid_)
    y_test_meta[:, 0] += dfm3.predict(Xi_test, Xv_test)
    gini_results_cv[2] = gini_norm(y1_valid_, y_train_meta[folds[2][1]])
    gini_results_epoch_train[2] = dfm3.train_result
    gini_results_epoch_valid[2] = dfm3.valid_result

    gini_results_train = gini_results_epoch_train.mean(axis=0)
    gini_results_valid = gini_results_epoch_valid.mean(axis=0)
    global gini_train_res
    global gini_valid_res
    gini_train_res[times] = gini_results_train
    gini_valid_res[times] = gini_results_valid
    y_test_meta /= float(len(folds))
    clf_str = "DeepFM"
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig_train(train_results, model_name):
    colors = ["red", "blue", "green", "yellow", "black"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(5):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        legends.append("train-avg-%d" % ((i + 1) * 3))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("./fig/DeepFM_grad_avg_train")
    plt.close()

def _plot_fig_valid(valid_results, model_name):
    colors = ["red", "blue", "green", "yellow", "black"]
    xs = np.arange(1, valid_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(5):
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("valid-avg-%d" % ((i + 1) * 3))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("./fig/DeepFM_grad_avg_valid")
    plt.close()

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

# ----------------- DeepFM Model ------------------
# params
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
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED,
    "average_gradients": 3
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 0)
'''epoch = dfm_params["epoch"]
gini_train_res = np.zeros((5, epoch), dtype=float)
gini_valid_res = np.zeros((5, epoch), dtype=float)
dfm_params["average_gradients"] = 3
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 0)
dfm_params["average_gradients"] = 6
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 1)
dfm_params["average_gradients"] = 9
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 2)
dfm_params["average_gradients"] = 12
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 3)
dfm_params["average_gradients"] = 15
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, 4)
print(gini_train_res.shape)
_plot_fig_train(gini_train_res, "DeepFM")
_plot_fig_valid(gini_valid_res, "DeepFM")'''
# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
# y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
# y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)


