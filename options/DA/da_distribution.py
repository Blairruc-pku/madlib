import os
import pdb
import sys
import numpy as np
import psycopg2 as p2
import threading
import tensorflow as tf
import datetime
import time
from collections import  OrderedDict
from utils import serialize_nd_weights, deserialize_as_nd_weights, deserialize_gradient, serialize_gradient
from tensorflow.python.framework.ops import IndexedSlicesValue
sys.path.append('..')
from OUTDB.DeepFM import DeepFM
from OUTDB import config
from OUTDB.metrics import accuracy

logfile = os.path.join('../logs', 'da_' + str(int(time.time())) + '.res')

def log_record(content, ifprint = True):
    with open(logfile, 'a') as f:
        ct = str(datetime.datetime.now())
        content_out = '[' + ct + '] ' + '[DA] ' + str(content)
        f.write(str(content_out) + '\n')
        if ifprint:
            print(content_out)


class Schema:
    Model_Table = 'model'
    GRADIENT_TABLE = 'gradient'

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

    def _execute(self, sql):
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        if cursor: cursor.close()
        if conn: conn.close()

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
        except Exception, e:
            print(e)
            return  None

    def check_table_exists(self, name):
        sql_check_exists = "SELECT EXISTS( SELECT 1 FROM pg_class, pg_namespace WHERE relnamespace = pg_namespace.oid AND relname='{name}') as table_exists;".format(**locals())
        return self._fetch_results(sql_check_exists)[0][0]

    def clear(self):
        sql = "select pid, query from pg_stat_activity where datname='{self.sample_tbl}';".format(**locals())
        results = self._fetch_results(sql)
        for row in results:
            pid, query = row
            if not 'pg_stat_activity' in query:
                self._execute("select pg_terminate_backend({pid})".format(**locals()))


class DeepFM_Master( DeepFM_DA):
    def __init__(self, **kwargs):
        DeepFM_DA.__init__(self, **kwargs)
        self.clear()
        self.register_model()
        self.version = dict()


    def register_model(self, name='', description=''):
        model_table, gradient_table = Schema.Model_Table, Schema.GRADIENT_TABLE
        if not self.check_table_exists(model_table):
            colnames = ['model_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['SERIAL PRIMARY KEY', 'bytea', 'Text','TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {'model_table'} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(model_table))

        if not self.check_table_exists( gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape' , 'version', 'model_version','auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text' ,'int','int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format( gradient_table))

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        weight_serialized = serialize_nd_weights(variables_value)
        sql_insert = '''INSERT INTO {} VALUES(DEFAULT, %s, %s )'''.format(model_table)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert, (p2.Binary(weight_serialized), str(shapes)))
        conn.commit()

        self.model_id = self._fetch_results("SELECT max(model_id) from {model_table}".format(**locals()))[0][0]
        log_record("Register model {} in DB".format(self.model_id))


    def save_weight(self, weights):
        shapes = [v.shape for v in weights]
        weights_serialized = serialize_nd_weights(weights)
        sql_insert = '''UPDATE {} SET (model_id, weight, shape) = ({}, %s, %s) WHERE model_id = {}'''.format(Schema.Model_Table, self.model_id, self.model_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert, (p2.Binary(weights_serialized), str(shapes)))
        conn.commit()


    def update(self, grads):
        result = self._fetch_results("SELECT weight, shape FROM model WHERE model_id ={self.model_id}".format(**locals()))
        shapes_fetch = eval(result[0][1])
        weights = deserialize_as_nd_weights(result[0][0], shapes_fetch)
        variables_ = list()
        for v in weights:
            temp = tf.Variable(v)
            variables_.append(temp)
        init_op = tf.variables_initializer(variables_)
        for i, grad_ in enumerate(grads):
            if isinstance(grad_, IndexedSlicesValue):
                grads[i] = tf.IndexedSlices(grads[i].values, grads[i].indices.astype('int64'))

        with tf.Session() as sess:
            sess.run(init_op)
            update = self.optimizer.apply_gradients(zip(grads, variables_))
            sess.run(update)
            weights_updated = [sess.run(v) for v in variables_]

        self.save_weight(weights_updated)
        return weights_updated

    def apply_grads_loop(self):
        while True:
            self.apply_grads()

    def apply_grads(self):
        gradient_table = Schema.GRADIENT_TABLE
        query = '''SELECT worker_id, version FROM {gradient_table} WHERE model_id={self.model_id}'''.format(**locals())
        results = self._fetch_results(query)
        for row in results:
            worker_id, version = row
            if not worker_id in self.version.keys():
                self.version[worker_id] = 0
            if version == self.version[worker_id]:
                self.apply_grads_per_worker(worker_id)


    def apply_grads_per_worker(self, worker_id):
        gradient_table = Schema.GRADIENT_TABLE
        version = self.version[worker_id]
        query = '''SELECT gradient, shape, auxiliaries FROM {gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(**locals())
        results = self._fetch_results(query)
        log_record("[Master] [Worker{}] Receive gradients with version {}".format(worker_id, version))
        gradient_serialized, shape, auxiliaries = results[0]
        gradients = deserialize_gradient(gradient_serialized, eval(shape), eval(auxiliaries))
        self.update(gradients)
        self.version[worker_id] = self.version[worker_id] + 1

        query = "UPDATE {} SET model_version={} WHERE model_id={} AND worker_id={}".format(gradient_table, self.version[worker_id], self.model_id,worker_id)
        self._execute(query)
        log_record("[Master] [Worker{}] Save model with version {}".format(worker_id, self.version[worker_id]))


class DeepFM_Worker(DeepFM_DA):
    def __init__(self, worker_id, **kwargs):
        DeepFM_DA.__init__(self, **kwargs)
        self.worker_id = worker_id
        self.model_id = self._fetch_results("SELECT max(model_id) from model".format(**locals()))[0][0]
        self.version = 0
        self.total_sample_worker = self._fetch_results("SELECT count(*) FROM {self.sample_tbl} WHERE gp_segment_id={self.worker_id}".format(**locals()))[0][0]
        self.get_block_info()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.update_placehoders = list()
        self.update_ops = list()
        with self.graph.as_default():
            for variable_ in variables:
                placehoder_temp = tf.placeholder(variable_.dtype, variable_.shape)
                self.update_placehoders.append(placehoder_temp)
                self.update_ops.append(tf.assign(variable_, placehoder_temp))

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grad_op = [x[0] for x in grads_and_vars]

        self.init_weight = True

    def log_variables(self, i):
        variable = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]
        variables_value = self.sess.run(variable)
        log_record(variables_value)

    def pull_weights(self):
        sql_check_version = "SELECT model_version FROM {} WHERE worker_id={} AND model_id={} ".format(Schema.GRADIENT_TABLE, self.worker_id,self.model_id)
        result = self._fetch_results(sql_check_version)
        while not (self.init_weight or (result[0][0]==self.version)):
             result = self._fetch_results(sql_check_version)
        if not self.init_weight:
            log_record("[Worker{}] Pull weight with version {}".format(self.worker_id, self.version))

        if self.init_weight:
            self.init_weight = False
        result = self._fetch_results("SELECT weight, shape FROM model WHERE model_id =1")
        shapes_fetch = eval(result[0][1])
        weights = deserialize_as_nd_weights(result[0][0], shapes_fetch)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = weights[i]
        self.sess.run(self.update_ops, feed_dict=feed_dict)


    def get_block_info(self):
        query = "SELECT (ctid::text::point)[0]::bigint AS block_number, count(*) FROM {self.sample_tbl} where gp_segment_id={self.worker_id} group by block_number;".format(**locals())
        results = self._fetch_results(query)
        block_info = dict()
        for row in results:
            block_id, count = row
            block_info[int(block_id)] = int(count)

        self.block_info = OrderedDict(sorted(block_info.items()))


    def check_ctid(self, ctid):
        fid, sid = ctid
        count = 0
        for block_id in self.block_info.keys():
            if block_id < fid:
                count = count + self.block_info[block_id]
            else:
                count = count + sid - 1
                return count

    def get_ctid(self, record_id):
        record_cum = 0
        for block_id in list(self.block_info.keys()):
            if record_cum + self.block_info[block_id] >= record_id + 1:
                res = (block_id, record_id - record_cum + 1)
                assert  self.check_ctid(res) == record_id, "Wrong ctid generated"
                return res
            record_cum = record_cum + self.block_info[block_id]



    def get_batch_data_block(self, batch_size, index):
        time_begin = time.time()
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < self.total_sample else self.total_sample
        start_block = self.get_ctid(start)
        end_block = self.get_ctid(end)
        select_query = "select * from {self.sample_tbl} where ctid >= '{start_block}' and ctid< '{end_block}' AND gp_segment_id={self.worker_id}".format(**locals())
        data = self._fetch_results(select_query)
        xi = [item[1:self.xi_index] for item in data]
        xv = [item[self.xi_index: -1] for item in data]
        y = [[item[-1]] for item in data]
        assert len(y) == end - start, "Number of data selected ({}) doesn't match requirement ({})).".format(len(y), end-start)
        self.Xi_train = self.Xi_train + xi
        self.Xv_train = self.Xv_train + xv
        self.y_train = self.y_train + y
        log_record("[Worker{}] Get batch {} takes {} sec".format(self.worker_id, index, time.time() - time_begin), False)
        return xi, xv, y


    def gradients_compute(self, xi, xv, y):
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}

        loss, grads = self.sess.run([self.loss, self.grad_op], feed_dict=feed_dict)

        return grads


    def push_graident(self, grads):
        gradient_table = Schema.GRADIENT_TABLE
        grads_serialized, shapes, auxiliaries = serialize_gradient(grads)
        sql = "SELECT version FROM {gradient_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(**locals())
        result = self._fetch_results(sql)
        if result == []:
            sql_insert = '''INSERT INTO {gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {gradient_table} SET (gradient, shape, version, auxiliaries) = (%s, %s, {self.version} ,%s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        log_record("[Worker{self.worker_id}] Push gradient with version {self.version}".format(**locals()))
        self.version = self.version + 1

    def run_one_batch(self, batch_id):
        self.pull_weights()
        Xi_batch, Xv_batch, y_batch = self.get_batch_data_block( self.batch_size, batch_id)
        grads = self.gradients_compute( Xi_batch, Xv_batch, y_batch)
        self.push_graident(grads)

    def run(self):
        total_batch = int(self.total_sample_worker / self.batch_size)
        log_record("Total batch:{}".format(total_batch))
        for epoch in range(self.epoch):
            log_record("[Worker{self.worker_id}] Enter epoch {epoch}".format(**locals()))
            for i in range(total_batch):
                self.run_one_batch(i)






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
        "batch_norm": 0,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": accuracy,
        "random_seed": config.RANDOM_SEED
    }

    log_record("Begin DA")
    model_master = DeepFM_Master(**dfm_params)

    master = threading.Thread(target=model_master.apply_grads_loop, name='master').start()
    worker_num = 4

    def setup_worker(worker_id, dfm_params):
        model_worker = DeepFM_Worker(worker_id=worker_id, **dfm_params)
        model_worker.run()

    for i in range(worker_num):
        threading.Thread(target=setup_worker, args=(i, dfm_params), name='worker{}'.format(i)).start()




