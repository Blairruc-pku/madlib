import os
import pdb
import sys
import time
import Queue
import threading
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool

sys.path.append('..')
from OUTDB.DeepFM import DeepFM
from OUTDB.DataReader import FeatureDictionary, DataParser
from OUTDB import config


class DeepFM_Global(DeepFM):
    def __init__(self, **kwargs):
        DeepFM.__init__(self, **kwargs)


    #def save_weight(self):

    def get_embedding(self, ids):
        embeddings = self.sess.run(self.weights['feature_embeddings'])
        embeddings_bias = self.sess.run(self.weights["feature_bias"])
        return embeddings[ids], embeddings_bias[ids]

    def update_embedding(self, grad, feat_index):
        embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], feat_index)
        bias = tf.nn.embedding_lookup(self.weights["feature_bias"], feat_index)
        embedding_value, bias_value = self.sess.run(embeddings), self.sess.run(bias)
        embeddings_ = tf.Variable(embedding_value)
        bias_ = tf.Variable(bias_value)
        init_op = tf.variables_initializer([embeddings_, bias_])
        with tf.Session() as sess:
            sess.run(init_op)
            update = self.optimizer.apply_gradients(
                zip([tf.IndexedSlices(grad[0].values, grad[0].indices),
                     tf.IndexedSlices(grad[1].values, grad[1].indices)],
                    [embeddings_, bias_]))
            sess.run(update)
            embeddings_updated = sess.run(embeddings_)
            bias_updated = sess.run(bias_)

        return embeddings_updated, bias_updated

    def update_dense(self, grads):
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        variables_ = list()
        for v in variables_value:
            temp = tf.Variable(v)
            variables_.append(temp)
        init_op = tf.variables_initializer(variables_)
        with tf.Session() as sess:
            sess.run(init_op)
            update = self.optimizer.apply_gradients(zip(grads, variables_))
            sess.run(update)
            variables_updated = [sess.run(v) for v in variables_]

        return variables_updated


class DeepFM_Local(DeepFM):
    def __init__(self, **kwargs):
        self.emd_id_fetch = []
        self.feature_size = 10  # temp value
        self.embedding_size = kwargs['embedding_size']
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.embeddings = tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01)
            self.embeddings_bias = tf.random_uniform([self.feature_size, 1], 0.0, 1.0)
        DeepFM.__init__(self, **kwargs)

    def fetch_emb_id(self):
        self.emd_id_fetch = [1, 2, 3, 4, 5, 6]
        return self.emd_id_fetch

    def reset_graph(self, embeddings, embeddings_bias):
        self.embeddings = embeddings
        self.embeddings_bias = embeddings_bias
        self.graph = tf.Graph()
        self._init_graph()

    def gradients_compute(self, xi, xv, y):
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}

        loss, grads = self.sess.run([self.loss, self.grad_op], feed_dict=feed_dict)

        return grads

    def evaluate_per_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.train_phase: False}
        batch_out = self.sess.run(self.out, feed_dict=feed_dict)
        batch_out = np.reshape(batch_out,newshape=np.array(y).shape)
        y = np.array(y)
        correct_num = 0
        for i in range(len(batch_out)):
            if batch_out[i][0] > 0.5:
                batch_out[i][0] = 1
                if batch_out[i][0] == y[i][0]:
                    correct_num += 1
            else:
                batch_out[i][0] = 0
                if batch_out[i][0] == y[i][0]:
                    correct_num += 1
        return correct_num/len(y)

    def _init_graph(self):
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # None * F
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)  # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])  # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)  # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])  # dropout at each Deep layer

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grad_op = [x[0] for x in grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(grads_and_vars)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        weights = dict()
        # embeddings
        weights["feature_embeddings"] = tf.Variable(self.embeddings, name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(self.embeddings_bias, name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def get_embedding(self, ids):
        embeddings = self.sess.run(self.weights['feature_embeddings'])
        return embeddings[ids]




def worker_run(model, worker_id, data, dfm_params, model_global):
    Xi1_train_, Xv1_train_, y1_train_ = data[worker_id][0]
    total_batch = int(len(y1_train_) / dfm_params["batch_size"])
    print(total_batch)
    for t in range(dfm_params["epoch"]):
        for i in range(total_batch):
            # compute gradients
            Xi1_batch, Xv1_batch, y1_batch = model.get_batch(Xi1_train_, Xv1_train_, y1_train_, dfm_params["batch_size"], i)
            emd_id_unique = np.unique(np.array(Xi1_batch))
            emb_id_mapping = dict()
            for j, id in enumerate(emd_id_unique):
                emb_id_mapping[id] = j
            emd_id_local = np.vectorize(emb_id_mapping.get)(Xi1_batch)
            #print(emd_id_local)
            embeddings, embeddings_bias = model_global.get_embedding(emd_id_unique)
            model.reset_graph(embeddings,embeddings_bias)
            grads = model.gradients_compute(emd_id_local.tolist(), Xv1_batch, y1_batch)
            '''print([n.name for n in model.graph.as_graph_def().node if "Variable" in n.op])
            print(model.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))'''
            Q.put(item=(grads,emd_id_unique, worker_id, i))
            #model.update_dense(grads[2:])
            t1 = time.time()
            train_result = model.evaluate_per_batch(emd_id_local.tolist(), Xv1_batch, y1_batch)
            print("epoch[%d]-batch[%d] local worker-[%d] valid-result=%.4f [%.1f s]" % (t, i, worker_id, train_result, time.time() - t1))

    return emd_id_unique, grads

def _load_data():
    dfTrain = pd.read_csv('../OUTDB/data/train.csv')
    dfTest = pd.read_csv('../OUTDB/data/test.csv')

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


def check_for_update():
    while True:
        if not Q.empty():
            (grads, emd_id_unique, worker_id, batch_id) = Q.get()
            embedding_grads = grads[:2]
            dense_grads = grads[2:]
            print("Start apply gradient from worker {} batch {} ".format(worker_id, batch_id))
            model_global.update_embedding(embedding_grads, emd_id_unique)
            model_global.update_dense(dense_grads)
            print("Finish apply gradient from worker {} batch {} ".format(worker_id, batch_id))



dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 5000,
    "learning_rate": 0.001,
    "optimizer_type": "gd",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": accuracy_score,
    "random_seed": config.RANDOM_SEED,
}



Q = Queue.Queue()

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
Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)
dfm_params["feature_size"] = fd.feat_dim
dfm_params["field_size"] = len(Xi_train[0])

model_global = DeepFM_Global(**dfm_params)


_get = lambda x, l: [x[i] for i in l]

data_workers = []
for i, (train_idx, valid_idx) in enumerate(folds):
    Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
    Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
    data_workers.append(((Xi_train_, Xv_train_, y_train_), (Xi_valid_, Xv_valid_, y_valid_)))

dfm_params["feature_size"] = fd.feat_dim
dfm_params["field_size"] = len(Xi_train[0])


dfm1 = DeepFM_Local(**dfm_params)
dfm2 = DeepFM_Local(**dfm_params)

m1 = threading.Thread(target=check_for_update, name='master').start()
t1 = threading.Thread(target=worker_run, args=(dfm1, 0, data_workers, dfm_params, model_global), name='t1').start()
t2 = threading.Thread(target=worker_run, args=(dfm2, 1, data_workers, dfm_params, model_global), name='t2').start()
# emd_id_unique, grads = worker_run(dfm1, 0, data_workers, dfm_params, model_global)
# model_global.update_embedding(grads[:2], emd_id_unique)
# model_global.update_dense(grads[2:])
