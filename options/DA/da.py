import os
import pdb
import sys
import numpy as np
import psycopg2 as p2
import tensorflow as tf
import datetime
import time
sys.path.append('..')
from OUTDB.DeepFM import DeepFM
from OUTDB import config
from OUTDB.metrics import accuracy

logfile = os.path.join('../logs', 'da_' + str(int(time.time())) + '.res')

def log_record(content, ifprint = True):
    with open(logfile, 'a') as f:
        ct = str(datetime.datetime.now())
        content_out = '[' + ct + '] ' + '[DA] ' + content
        f.write(content_out + '\n')
        if ifprint:
            print(content_out)



class DeepFM_DA(DeepFM):
    def __init__(self, **kwargs):
        self.user = 'ruike.xy'
        self.host = 'localhost'
        self.dbname = 'driving'
        self.sample_tbl = 'driving'
        self.port = 5432
        self.total_sample = self._fetch_results("select count(*) from {self.sample_tbl}".format(**locals()))[0][0]
        self.Xi_train, self.Xv_train, self.y_train = list(), list(), list()
        columns = self._fetch_results("select column_name FROM information_schema.columns WHERE table_name  ='{self.sample_tbl}'".format(**locals()))
        self.xi_index = len(columns) / 2
        index_columns = columns[0:self.xi_index-1]
        feat_dim = 0
        for column in index_columns:
            column = column[0]
            num = self._fetch_results("select count(distinct {column}) from {self.sample_tbl}".format(**locals()))[0][0]
            feat_dim = feat_dim + num
        kwargs["field_size"] = self.xi_index - 1
        kwargs["feature_size"] = feat_dim
        log_record("Feat_dim:{}".format(feat_dim))

        DeepFM.__init__(self, **kwargs)
        log_record("Initialized model")


    def _connect_db(self):
        conn = p2.connect(host=self.host, user=self.user, dbname=self.dbname, port=self.port)
        return conn

    def _fetch_results(self, sql, json=False):
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            if cursor: cursor.close()
            if conn: conn.close()

            if json:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return results
        except:
            return  None


    def get_batch_da(self, batch_size, index):
        time_begin = time.time()
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < self.total_sample else self.total_sample
        select_query = "select * from {self.sample_tbl} where id >= {start} and id< {end}".format(**locals())
        data = self._fetch_results(select_query)
        xi = [item[1:self.xi_index] for item in data]
        xv = [item[self.xi_index: -1] for item in data]
        y = [[item[-1]] for item in data]
        self.Xi_train = self.Xi_train + xi
        self.Xv_train = self.Xv_train + xv
        self.y_train = self.y_train + y
        log_record("get batch {} takes {} sec".format(index, time.time() - time_begin), False)

        return xi, xv, y


    def predict(self, Xi=None, Xv=None):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        if not Xi is None:
            return DeepFM.predict(self, Xi, Xv)

        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch_da( self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            time_begin_infer = time.time()
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            log_record("Infer on batch {} takes {} sec".format(batch_index, time.time() - time_begin_infer), False)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch_da( self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        y = [[float(item[0])] for item in y]
        return self.eval_metric(y, y_pred)


    def fit(self, Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):

        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time.time()
            total_batch = int(self.total_sample / self.batch_size)
            log_record("Total batch:{}".format(total_batch))
            time_begin_epoch = time.time()
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch_da( self.batch_size, i)
                time_begin_batch = time.time()
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                if i % 50 == 0:
                    log_record("Fit on batch {} takes {} sec".format(i, time.time() - time_begin_batch))
                else:
                    log_record("Fit on batch {} takes {} sec".format(i, time.time() - time_begin_batch), False)
            log_record("Train on epoch {} takes {} sec".format(epoch, time.time() - time_begin_epoch))
            # evaluate training and validation datasets
            time_begin_evaluate = time.time()
            train_result = self.evaluate(self.Xi_train, self.Xv_train, self.y_train)
            self.train_result.append(train_result)
            log_record("Evaluate on epoch {} takes {} sec".format(epoch, time.time() - time_begin_evaluate))
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time.time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time.time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

            log_record("Total on epoch {} takes {} sec".format(epoch, time.time() - time_begin_epoch))


if __name__ == "__main__":
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

    log_record("Begin DA")
    model_global = DeepFM_DA(**dfm_params)
    model_global.fit()





