dimport os
import pdb
import sys
import numpy as np
import psycopg2 as p2
import threading
import tensorflow as tf
import datetime
import time
from collections import OrderedDict
from utils import serialize_nd_weights, deserialize_as_nd_weights, deserialize_gradient, serialize_gradient
from tensorflow.python.framework.ops import IndexedSlicesValue
sys.path.append('..')
from OUTDB.DeepFM import DeepFM
from OUTDB import config
from OUTDB.metrics import accuracy

#logfile = os.path.join('../logs', 'da_' + str(int(time.time())) + '.res')
logfile = './log.res'

def log_record(content, ifprint = True):
    with open(logfile, 'a') as f:
        ct = str(datetime.datetime.now())
        content_out = '[' + ct + '] ' + '[DA] ' + str(content)
        f.write(str(content_out) + '\n')
        if ifprint:
            print(content_out)


class Schema:
    Embed_Model_Table = 'embed_model'
    Dense_Model_Table = 'dense_model'
    Embed_GRADIENT_TABLE = 'embed_gradient'
    Dense_GRADIENT_TABLE = 'dense_gradient'


class DeepFM_DA(DeepFM):
    def __init__(self, **kwargs):
        self.user = 'ruike.xy'
        self.host = '11.164.101.172'
        #self.user = 'gpadmin'
        #self.host = '192.168.1.2'
        self.dbname = 'driving'
        #self.database = 'gpdb'
        self.sample_tbl = 'driving'
        self.port = 5432
        self.total_sample = self._fetch_results("select count(*) from {self.sample_tbl}".format(**locals()))[0][0]
        self.Xi_train, self.Xv_train, self.y_train = list(), list(), list()
        columns = self._fetch_results("select column_name FROM information_schema.columns WHERE table_name ='{self.sample_tbl}'".format(**locals()))
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

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.update_placehoders = list()
        self.update_ops = list()
        with self.graph.as_default():
            for variable_ in variables:
                placehoder_temp = tf.placeholder(variable_.dtype, variable_.shape)
                self.update_placehoders.append(placehoder_temp)
                self.update_ops.append(tf.assign(variable_, placehoder_temp))

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
        except Exception as e:
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


class DeepFM_Master(DeepFM_DA):
    def __init__(self, **kwargs):
        DeepFM_DA.__init__(self, **kwargs)
        self.clear()
        self.register_model()
        self.version = dict()

    def register_model(self, name='', description=''):
        embed_model_table, dense_model_table, embed_gradient_table, dense_gradient_table =Schema.Embed_Model_Table, Schema.Dense_Model_Table, Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        if not self.check_table_exists(embed_model_table):
            colnames = ['model_id', 'embedding_weight', 'shape', 'embedding_bias', 'id', 'description']
            coltypes = ['SERIAL PRIMARY KEY', 'bytea', 'Text', 'int', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {'model_table'} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(embed_model_table))

        if not self.check_table_exists(dense_model_table):
            colnames = ['model_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['SERIAL PRIMARY KEY', 'bytea', 'Text','TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {'model_table'} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(dense_model_table))

        if not self.check_table_exists(embed_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'bias_gradient', 'shape', 'version', 'model_version', 'auxiliaries', 'id']
            coltypes = ['int', 'int', 'bytea', 'bytea','Text', 'int', 'int', 'Text', 'int']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(embed_gradient_table))

        if not self.check_table_exists(dense_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape', 'version', 'model_version','auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text', 'int', 'int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(dense_gradient_table))

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        weight_serialized = serialize_nd_weights(variables_value)
        sql_insert_dense = '''INSERT INTO {} VALUES(DEFAULT, %s, %s )'''.format(dense_model_table)

        embeddings = self.sess.run(self.weights['feature_embeddings'])
        embeddings_bias = self.sess.run(self.weights["feature_bias"])
        embed_weight_serialized = serialize_nd_weights(embeddings)
        embed_bias_weight_serialized = serialize_nd_weights(embeddings_bias)
        embedding_id = np.arange(len(embeddings)).tolist()
        sql_insert_embed = '''INSERT INTO {} VALUES(DEFAULT, %s, %s, %s, %s )'''.format(embed_model_table)

        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert_dense, (p2.Binary(weight_serialized), str(shapes)))
        cursor.execute(sql_insert_embed, (p2.Binary(embed_weight_serialized), str(shapes), p2.Binary(embed_bias_weight_serialized), embedding_id))
        conn.commit()

        self.model_id = self._fetch_results("SELECT max(model_id) from {model_table}".format(**locals()))[0][0]
        log_record("Register model {} in DB".format(self.model_id))

    def save_embedding(self, weights, embed_id):
        weights_serialized = serialize_nd_weights(weights[0])
        weights_serialized_bias = serialize_nd_weights(weights[1])
        sql_insert = '''UPDATE {} SET (model_id, embedding_weight, embedding_bias) = ({}, %s, %s) WHERE model_id = {} AND id IN {}'''.format(
            Schema.Embed_Model_Table, self.model_id, self.model_id, embed_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert, (p2.Binary(weights_serialized), p2.Binary(weights_serialized_bias)))
        conn.commit()

    def save_dense_weight(self, weights):
        shapes = [v.shape for v in weights]
        weights_serialized = serialize_nd_weights(weights)
        sql_insert = '''UPDATE {} SET (model_id, weight, shape) = ({}, %s, %s) WHERE model_id = {}'''.format(Schema.Dense_Model_Table, self.model_id, self.model_id)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert, (p2.Binary(weights_serialized), str(shapes)))
        conn.commit()

    def update(self, grads, embed_id):
        # 分别拉取dense并根据梯度用到的embedding_id拉取对应的embedding
        dense_result = self._fetch_results("SELECT weight, shape FROM dense_model WHERE model_id ={self.model_id}".format(**locals()))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        variables_ = list()
        embed_result = self._fetch_results("SELECT embedding_weight, shape, embedding_bias, id FROM embed_model WHERE model_id ={self.model_id} AND id IN {embed_id}".format(**locals()))
        shapes_fetch = eval(embed_result[0][1])
        embed_weights = deserialize_as_nd_weights(embed_result[0][0], shapes_fetch)
        embed_bias_weights = deserialize_as_nd_weights(embed_result[0][2], shapes_fetch)
        embedding = list()
        embedding_bias = list()
        cnt = 0
        for i in range(self.feature_size):
            if i in embed_id:
                embedding.append(embed_weights[cnt])
                embedding_bias.append(embed_bias_weights[cnt])
                cnt += 1
            else:
                # 为了参数feed_dict，未使用的embedding默认为0
                temp = np.zeros(shape=embed_weights[cnt].shape)
                embedding.append(temp)
                temp = np.zeros(shape=embed_bias_weights[cnt].shape)
                embedding_bias.append(temp)
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            print(i, placeholder)
            feed_dict[placeholder] = variables_[i]

        with tf.Session() as sess:
            self.sess.run(self.update_ops, feed_dict=feed_dict)
            update = self.optimizer.apply_gradients(zip(grads, variables_))
            sess.run(update)
            weights_updated = [sess.run(v) for v in variables_]
        self.save_embedding(weights_updated[0:2],embed_id)
        self.save_dense_weight(weights_updated[2:])
        return weights_updated

    def apply_grads_loop(self):
        while True:
            self.apply_grads()

    def apply_grads(self):
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        query = '''SELECT worker_id, version FROM {embed_gradient_table} WHERE model_id={self.model_id}'''.format(**locals())
        results = self._fetch_results(query)
        for row in results:
            worker_id, version = row
            if not worker_id in self.version.keys():
                self.version[worker_id] = 0
            if version == self.version[worker_id]:
                self.apply_grads_per_worker(worker_id)

    def apply_grads_per_worker(self, worker_id):
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        version = self.version[worker_id]
        query = '''SELECT gradient, shape, auxiliaries FROM {dense_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(**locals())
        results = self._fetch_results(query)
        log_record("[Master] [Worker{}] Receive gradients with version {}".format(worker_id, version))
        dense_gradient_serialized, shape, auxiliaries = results[0]
        dense_gradients = deserialize_gradient(dense_gradient_serialized, eval(shape), eval(auxiliaries))
        query = '''SELECT gradient, bias_gradient, shape, auxiliaries, embed_id FROM {embed_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(
            **locals())
        results = self._fetch_results(query)
        embed_gradient_serialized, embed_bias_gradient_serialized, shape, auxiliaries, embed_id = results[0]
        embed_gradients = deserialize_gradient(embed_gradient_serialized, eval(shape), eval(auxiliaries))
        embed_bias_gradients = deserialize_gradient(embed_gradient_serialized, eval(shape), eval(auxiliaries))
        grads = list()
        grads.append(tf.IndexedSlices(np.array(embed_gradients), np.range(self.feat_index)))
        grads.append(tf.IndexedSlices(np.array(embed_bias_gradients), np.range(self.feat_index)))
        grads.append(dense_gradient_serialized)
        self.update(grads,embed_id)
        self.version[worker_id] = self.version[worker_id] + 1

        query = "UPDATE {} SET model_version={} WHERE model_id={} AND worker_id={}".format(dense_gradient_table, self.version[worker_id], self.model_id,worker_id)
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

    def pull_weights(self,embed_id_unique):
        # 根据embedding_id进行拉取对应的embedding
        sql_check_version = "SELECT model_version FROM {} WHERE worker_id={} AND model_id={} ".format(Schema.Dense_Model_Table, self.worker_id,self.model_id)
        dense_result = self._fetch_results(sql_check_version)
        while not (self.init_weight or (dense_result[0][0]==self.version)):
             dense_result = self._fetch_results(sql_check_version)
        if not self.init_weight:
            log_record("[Worker{}] Pull weight with version {}".format(self.worker_id, self.version))

        if self.init_weight:
            self.init_weight = False
        variables_ = list()
        dense_result = self._fetch_results("SELECT weight, shape FROM dense_model WHERE model_id =1")
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        embed_result = self._fetch_results(
            "SELECT embedding_weight, shape, embedding_bias, id FROM embed_model WHERE model_id ={self.model_id} AND id IN {embed_id_unique}".format(
                **locals()))
        shapes_fetch = eval(embed_result[0][1])
        embed_weights = deserialize_as_nd_weights(embed_result[0][0], shapes_fetch)
        embed_bias_weights = deserialize_as_nd_weights(embed_result[0][2], shapes_fetch)
        embedding = list()
        embedding_bias = list()
        cnt = 0
        for i in range(self.feature_size):
            if i in embed_id_unique:
                embedding.append(embed_weights[cnt])
                embedding_bias.append(embed_bias_weights[cnt])
                cnt += 1
            else:
                temp = np.zeros(shape=embed_weights[cnt].shape)
                embedding.append(temp)
                temp = np.zeros(shape=embed_bias_weights[cnt].shape)
                embedding_bias.append(temp)
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            print(i, placeholder)
            feed_dict[placeholder] = variables_[i]
        self.sess.run(self.update_ops, feed_dict=feed_dict)
        return


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
        embed_grads = grads[0:2]
        dense_grads = grads[2:]
        embed_grads_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        # 分别更新dense表和embedding表
        grads_serialized, shapes, auxiliaries = serialize_gradient(dense_grads)
        sql = "SELECT version FROM {dense_gradient_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(**locals())
        result = self._fetch_results(sql)
        if result == []:
            sql_insert = '''INSERT INTO {dense_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {dense_gradient_table} SET (gradient, shape, version, auxiliaries) = (%s, %s, {self.version} ,%s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        embed_id_local = list()
        for i, grad_ in enumerate(embed_grads):
            if isinstance(grad_, IndexedSlicesValue):
                embed_grads[i] = tf.IndexedSlices(embed_grads[i].values, embed_grads[i].indices.astype('int64'))
                embed_id_local.append(embed_grads[i].indices.astype('int64'))
        embed_grad = list()
        embed_bias_grads = list()
        embed_id_unique = np.unique(np.array(embed_id_local))
        cnt = 0
        for i in range(self.feature_size):
            if i in embed_id_unique:
                embed_grad.append(embed_grads[0][cnt])
                embed_bias_grads.append(embed_grads[1][cnt])
                cnt+=1
            '''else:
                temp = np.zeros(shape=embed_grads[cnt].shape)
                embed_grad.append(temp)
                temp = np.zeros(shape=embed_bias_grads[cnt].shape)
                embed_bias_grads.append(temp)'''
        e_grads_serialized, shapes, auxiliaries = serialize_gradient(embed_grad)
        bias_grads_serialized,shapes, auxiliaries = serialize_gradient(embed_bias_grads)
        sql = "SELECT version FROM {embed_grads_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(**locals())
        result = self._fetch_results(sql)
        if result == []:
            sql_insert = '''INSERT INTO {embed_grads_table} (model_id, worker_id, gradient,bias_gradient, shape, version, auxiliaries, id) VALUES ({self.model_id}, {self.worker_id}, %s , %s, %s, {self.version}, %s, %s)'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_insert, (p2.Binary(e_grads_serialized),p2.Binary(bias_grads_serialized), str(shapes), str(auxiliaries), embed_id_unique))
            conn.commit()

        else:
            sql_update = '''UPDATE {embed_grads_table} SET (gradient,bias_gradient, shape, version, auxiliaries, id) = (%s, %s, %s, {self.version} ,%s, %s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(**locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(e_grads_serialized),p2.Binary(bias_grads_serialized), str(shapes), str(auxiliaries), embed_id_unique))
            conn.commit()

        log_record("[Worker{self.worker_id}] Push gradient with version {self.version}".format(**locals()))
        self.version = self.version + 1

    def run_one_batch(self, batch_id):
        Xi_batch, Xv_batch, y_batch = self.get_batch_data_block( self.batch_size, batch_id)
        emd_id_unique = np.unique(np.array(Xi_batch))
        self.pull_weights(emd_id_unique)
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

