# coding=utf-8
import os
import pdb
import sys
import numpy as np
import psycopg2 as p2
import threading
import tensorflow as tf
import datetime
import time
from collections import OrderedDict
from decimal import *

from sklearn.metrics import classification_report
from tensorflow import float32


from utils import serialize_nd_weights, deserialize_as_nd_weights, deserialize_gradient, serialize_gradient,serialize_embedding,deserialize_embedding
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
    Embed_Model_Table = 'embed_model'
    Dense_Model_Table = 'dense_model'
    Embed_GRADIENT_TABLE = 'embed_gradient_table'
    Dense_GRADIENT_TABLE = 'dense_gradient_table'


class DeepFM_DA(DeepFM):
    def __init__(self, **kwargs):
        self.user = 'ruike.xy'
        self.host = '11.164.101.172'
        self.dbname = 'driving'
        # self.user = 'gpadmin'
        # self.host = '192.168.1.2'
        # self.dbname = 'gpdb'
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

        kwargs["field_size"] = int(self.xi_index - 1)
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
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.update_placehoders = list()
        self.update_ops = list()
        self.gradient_placehoders = list()
        gradientL = list()
        with self.graph.as_default():
            for i, variable_ in enumerate(variables):
                placehoder_temp = tf.placeholder(variable_.dtype)
                self.update_placehoders.append(placehoder_temp)
                self.update_ops.append(tf.assign(variable_, placehoder_temp, validate_shape=False))
                if i < 2:
                    placehoder_value = tf.placeholder(variable_.dtype)
                    placehoder_indices = tf.placeholder('int64')
                    self.gradient_placehoders.append(placehoder_value)
                    self.gradient_placehoders.append(placehoder_indices)
                    gradientL.append(tf.IndexedSlices(placehoder_value,placehoder_indices))
                else:
                    self.gradient_placehoders.append(placehoder_temp)
                    gradientL.append(placehoder_temp)

            self.apply_grad_op = self.optimizer.apply_gradients(zip(gradientL, variables))

    def register_model(self, name='', description=''):
        embed_model_table, dense_model_table, embed_gradient_table, dense_gradient_table =Schema.Embed_Model_Table, Schema.Dense_Model_Table, Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        if not self.check_table_exists(embed_model_table):
            colnames = ['model_id', 'embedding_weight', 'shape', 'embedding_bias', 'id', 'description']
            coltypes = ['int', 'bytea', 'Text', 'bytea', 'int', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {embed_model_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(embed_model_table))

        if not self.check_table_exists(dense_model_table):
            colnames = ['model_id', 'weight', 'shape', 'name', 'description']
            coltypes = ['SERIAL PRIMARY KEY', 'bytea', 'Text', 'TEXT', 'TEXT']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {dense_model_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(dense_model_table))

        if not self.check_table_exists(embed_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape', 'version', 'model_version', 'auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text', 'int', 'int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {embed_gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(embed_gradient_table))

        if not self.check_table_exists(dense_gradient_table):
            colnames = ['model_id', 'worker_id', 'gradient', 'shape', 'version', 'model_version','auxiliaries']
            coltypes = ['int', 'int', 'bytea', 'Text', 'int', 'int', 'Text']
            col_defs = ','.join(map(' '.join, zip(colnames, coltypes)))
            sql = "CREATE TABLE {dense_gradient_table} ({col_defs})".format(**locals())
            self._execute(sql)
        else:
            log_record("Table {} exists".format(dense_gradient_table))

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2:]
        variables_value = [self.sess.run(v) for v in variables]
        shapes = [v.shape.as_list() for v in variables]
        weight_serialized = serialize_nd_weights(variables_value)
        weight = deserialize_as_nd_weights(weight_serialized, shapes)
        sql_insert_dense = '''INSERT INTO {} VALUES(DEFAULT, %s, %s )'''.format(dense_model_table)
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql_insert_dense, (p2.Binary(weight_serialized), str(shapes)))
        conn.commit()

        self.model_id = self._fetch_results("SELECT max(model_id) from {dense_model_table}".format(**locals()))[0][0]
        log_record("Register model {} in DB".format(self.model_id))

        embeddings = self.sess.run(self.weights['feature_embeddings'])
        embeddings_bias = self.sess.run(self.weights["feature_bias"])
        conn = self._connect_db()
        cursor = conn.cursor()
        for i in range(self.feature_size):
            embed_weight_serialized = serialize_embedding(embeddings[i])
            embed_bias_weight_serialized = serialize_embedding(embeddings_bias[i])
            shapes = [embeddings[i].shape,embeddings_bias[i].shape]
            sql_insert_embed = '''INSERT INTO {} VALUES({}, %s, %s, %s, %s )'''.format(embed_model_table, self.model_id)
            cursor.execute(sql_insert_embed, (p2.Binary(embed_weight_serialized), str(shapes), p2.Binary(embed_bias_weight_serialized), i))
        conn.commit()

    def save_embedding(self, weights, embed_id):
        conn = self._connect_db()
        cursor = conn.cursor()
        for i in range(len(embed_id)):
            weights_serialized = serialize_embedding(weights[0][i])
            weights_serialized_bias = serialize_embedding(weights[1][i])
            sql_insert = '''UPDATE {} SET ( embedding_weight, embedding_bias) = ( %s, %s) WHERE model_id = {} AND id = {}'''.format(
                Schema.Embed_Model_Table, self.model_id, embed_id[i])
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
        t1 = time.time()
        # 分别拉取dense并根据梯度用到的embedding_id拉取对应的embedding
        dense_result = self._fetch_results("SELECT weight, shape FROM {} WHERE model_id ={}".format(Schema.Dense_Model_Table, self.model_id))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        t2 = time.time()
        log_record("[Master] Pull dense [{} s]".format(round(t2-t1, 2)))
        embedding_result = self._fetch_results("SELECT embedding_weight, embedding_bias, id FROM {} WHERE id in {} and model_id={}".format(Schema.Embed_Model_Table, tuple(embed_id), self.model_id))
        emb_id_mapping = dict()
        embedding, embedding_bias = list(), list()
        embed_id_ = list()
        for i, row in enumerate(embedding_result):
            embedding.append(deserialize_embedding(row[0]))
            embedding_bias.append(deserialize_embedding(row[1]))
            emb_id_mapping[row[2]] = i
            embed_id_.append(row[2])
        t3 = time.time()
        log_record("[Master] Pull embedding [{} s]".format(round(t3-t2, 2)))
        variables_ = list()
        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)

        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)

        feed_dict = dict()
        placeholder_count = 0
        for i, grad_ in enumerate(grads):
            if isinstance(grad_, IndexedSlicesValue):
                indices = np.vectorize(emb_id_mapping.get)(grads[i].indices.astype('int64'))
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i].values
                placeholder_count = placeholder_count + 1
                feed_dict[self.gradient_placehoders[placeholder_count]] = indices
                placeholder_count = placeholder_count + 1
            else:
                feed_dict[self.gradient_placehoders[placeholder_count]] = grads[i]
                placeholder_count = placeholder_count + 1

        self.sess.run(self.apply_grad_op, feed_dict=feed_dict)

        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_updated = [self.sess.run(v) for v in variables]
        t4 = time.time()
        log_record("[Master] Update weight [{} s]".format(round(t4-t3, 2)))
        self.save_embedding(weights_updated[0:2], embed_id_)
        t5 = time.time()
        log_record("[Master] Save embedding  [{} s]".format(round(t5-t4, 2)))
        self.save_dense_weight(weights_updated[2:])
        t6 = time.time()
        log_record("[Master] Save dense  [{} s]".format(round(t6-t5, 2)))
        return weights_updated

    def apply_grads_loop(self):
        t1 = time.time()
        while True:
            embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
            embed_query = '''SELECT worker_id, version FROM {dense_gradient_table} WHERE model_id={self.model_id}'''.format(**locals())
            embed_results = self._fetch_results(embed_query)
            if embed_results:
                log_record("[Master] Wait for gradient [{} s]".format(round(time.time() - t1, 2)))
                for row in embed_results:
                    worker_id, version = row
                    if not worker_id in self.version.keys():
                        self.version[worker_id] = 0
                    if version == self.version[worker_id]:
                        self.apply_grads_per_worker(worker_id)
                t1 = time.time()

    def pull_grads(self, worker_id):
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        version = self.version[worker_id]
        query = '''SELECT gradient, shape, auxiliaries FROM {dense_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(**locals())
        results = self._fetch_results(query)
        dense_gradient_serialized, shape, auxiliaries = results[0]
        dense_gradients = deserialize_gradient(dense_gradient_serialized, eval(shape), eval(auxiliaries))
        log_record("[Master] [Worker{}] Receive dense gradients with version {}".format(worker_id, version))

        query = '''SELECT gradient, shape, auxiliaries FROM {embed_gradient_table} WHERE model_id={self.model_id} AND version={version} AND worker_id={worker_id}'''.format(
            **locals())
        results = self._fetch_results(query)
        embed_gradient_serialized, shape, auxiliaries = results[0]
        embed_gradients = deserialize_gradient(embed_gradient_serialized, eval(shape), eval(auxiliaries))
        log_record("[Master] [Worker{}] Receive embedding gradients with version {}".format(worker_id, version))
        grads = list()
        embed_id = list()
        for e in embed_gradients:
            grads.append(e)
            embed_id.append(e.indices.astype('int64'))
        for d in dense_gradients:
            grads.append(d)

        embed_id_unique = np.unique(np.array(embed_id)).tolist()

        return  grads, embed_id_unique

    def apply_grads_per_worker(self, worker_id):
        t1 = time.time()
        grads, embed_id_unique = self.pull_grads(worker_id)
        t2 = time.time()
        log_record("[Mater] Pull gradient  [{} s]".format(round(t2 - t1,2)))
        self.update(grads, embed_id_unique)
        t3 = time.time()
        log_record("[Mater] Update weight [{} s]".format(round(t3 - t2,2)))
        self.version[worker_id] = self.version[worker_id] + 1
        query = "UPDATE {} SET model_version={} WHERE model_id={} AND worker_id={}".format(Schema.Dense_GRADIENT_TABLE, self.version[worker_id], self.model_id,worker_id)
        self._execute(query)
        t4 = time.time()
        log_record("[Master] [Worker{}] Save model with version {}".format(worker_id, self.version[worker_id]))
        log_record("[Mater] Deal with worker {} takes {} sec ".format(worker_id, round(t4 - t1,2)))

class DeepFM_Worker(DeepFM_DA):
    def __init__(self, worker_id, **kwargs):
        DeepFM_DA.__init__(self, **kwargs)
        self.worker_id = worker_id
        self.model_id = self._fetch_results("SELECT max(model_id) from dense_model".format(**locals()))[0][0]
        self.version = 0
        self.total_sample_worker = self._fetch_results("SELECT count(*) FROM {self.sample_tbl} WHERE gp_segment_id={self.worker_id}".format(**locals()))[0][0]
        self.get_block_info()
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.update_placehoders = list()
        self.update_ops = list()
        with self.graph.as_default():
            for variable_ in variables:
                placehoder_temp = tf.placeholder(variable_.dtype, [None for i in  variable_.shape])
                self.update_placehoders.append(placehoder_temp)
                self.update_ops.append(tf.assign(variable_, placehoder_temp, validate_shape=False))

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grad_op = [x[0] for x in grads_and_vars]

        self.init_weight = True

    def log_variables(self, i):
        variable = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[i]
        variables_value = self.sess.run(variable)
        log_record(variables_value)

    def pull_weights(self, embed_id_unique):
        # 根据embedding_id进行拉取对应的embedding
        sql_check_version = "SELECT model_version FROM {} WHERE worker_id={} AND model_id={} ".format(Schema.Dense_GRADIENT_TABLE, self.worker_id,self.model_id)
        dense_result = self._fetch_results(sql_check_version)
        t1 = time.time()
        while not (self.init_weight or (dense_result[0][0]==self.version)):
             dense_result = self._fetch_results(sql_check_version)
        log_record("[Worker{}] Wait for master [{} s]".format(self.worker_id, round(time.time() - t1, 2)))
        if self.init_weight:
            self.init_weight = False
        variables_ = list()
        dense_result = self._fetch_results("SELECT weight, shape FROM dense_model WHERE model_id = {}".format(self.model_id))
        shapes_fetch = eval(dense_result[0][1])
        dense_weights = deserialize_as_nd_weights(dense_result[0][0], shapes_fetch)
        embedding = list()
        embedding_bias = list()
        embed_result = self._fetch_results( "SELECT embedding_weight, shape, embedding_bias, id FROM embed_model WHERE id in {} and model_id={}".format(tuple(embed_id_unique), self.model_id))
        emb_id_mapping = dict()
        for i, row in enumerate(embed_result):
            embedding.append(deserialize_embedding(row[0]))
            embedding_bias.append(deserialize_embedding(row[2]))
            emb_id_mapping[row[3]] = i

        variables_.append(np.array(embedding))
        variables_.append(np.array(embedding_bias))
        for v in dense_weights:
            variables_.append(v)
        feed_dict = dict()
        for i, placeholder in enumerate(self.update_placehoders):
            feed_dict[placeholder] = variables_[i]

        self.sess.run(self.update_ops, feed_dict=feed_dict)

        return emb_id_mapping


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
        xi = [ map(int, item[1:self.xi_index] )for item in data]
        xv = [map(float, item[self.xi_index: -1] )for item in data]
        y = [[float(item[-1])] for item in data]
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
        embed_gradient_table, dense_gradient_table = Schema.Embed_GRADIENT_TABLE, Schema.Dense_GRADIENT_TABLE
        # 更新embedding表
        grads_serialized, shapes, auxiliaries = serialize_gradient(embed_grads)
        sql = "SELECT version FROM {embed_gradient_table} WHERE model_id = {self.model_id} AND worker_id = {self.worker_id}".format(
            **locals())
        result = self._fetch_results(sql)
        if result == []:
            conn = self._connect_db()
            cursor = conn.cursor()
            sql_insert = '''INSERT INTO {embed_gradient_table} (model_id, worker_id, gradient, shape, version, auxiliaries) VALUES ({self.model_id}, {self.worker_id}, %s , %s, {self.version}, %s)'''.format(**locals())
            cursor.execute(sql_insert, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        else:
            sql_update = '''UPDATE {embed_gradient_table} SET (gradient, shape, version, auxiliaries) = (%s, %s, {self.version} ,%s) WHERE model_id = {self.model_id} AND worker_id={self.worker_id}'''.format(
                **locals())
            conn = self._connect_db()
            cursor = conn.cursor()
            cursor.execute(sql_update, (p2.Binary(grads_serialized), str(shapes), str(auxiliaries)))
            conn.commit()

        # 更新dense表
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

        self.version = self.version + 1


    def gradient_transform(self, grads, emb_id_mapping):
        def sum_by_group(values, groups):
            order = np.argsort(groups)
            groups = groups[order]
            values = values[order]
            values.cumsum(axis=0,out=values)
            index = np.ones(len(groups), 'bool')
            index[:-1] = groups[1:] != groups[:-1]
            values = values[index]
            groups = groups[index]
            values[1:] = values[1:] - values[:-1]
            return values, groups

        inv_map = {v: k for k, v in emb_id_mapping.iteritems()}
        for i in  range(len(grads)):
            grad = grads[i]
            if isinstance(grad, IndexedSlicesValue):
                indices = np.vectorize(inv_map.get)(grad.indices)
                values, indices = sum_by_group(grad.values, indices)
                grad = IndexedSlicesValue(values=values, indices=indices, dense_shape=grad.dense_shape)
                grads[i] = grad

        return  grads

    def run_one_batch(self, batch_id, epoch):
        t1 = time.time()
        Xi_batch, Xv_batch, y_batch = self.get_batch_data_block( self.batch_size, batch_id)
        t2 = time.time()
        log_record("[Worker{}] Get batch data  [{} s]".format(self.worker_id, round(t2-t1, 2)))
        emd_id_unique = np.unique(np.array(Xi_batch))
        emb_id_mapping = self.pull_weights(emd_id_unique)
        t3 = time.time()
        log_record("[Worker{}] Pull weight with version {} [{} s]".format(self.worker_id, self.version, round(t3-t2, 2)))
        Xi_batch_local = np.vectorize(emb_id_mapping.get)(Xi_batch)
        grads = self.gradients_compute( Xi_batch_local, Xv_batch, y_batch)
        grads = self.gradient_transform(grads, emb_id_mapping)
        t4 = time.time()
        log_record("[Worker{}] Compute gradient [{} s]".format(self.worker_id, round(t4-t3, 2)))
        self.push_graident(grads)
        t5 = time.time()
        log_record("[Worker{}] Push gradient with version {} [{} s]".format(self.worker_id, self.version-1, round(t5-t4, 2)  ))
        train_results = self.evaluate_per_batch(Xi_batch_local, Xv_batch, y_batch)
        t6 = time.time()
        log_record("[Worker%d] epoch%d batch%d train_results=%.4f [%.1f s]" % (self.worker_id, epoch,batch_id, float(float(train_results)/float(self.batch_size)), t6-t5))
        log_record("[Worker{}] Time for one batch [{} s]".format(self.worker_id, round(t6-t1, 2)  ))

    def run(self):
        total_batch = int(self.total_sample_worker / self.batch_size)
        log_record("Total batch:{}".format(total_batch))
        for epoch in range(self.epoch):
            log_record("[Worker{self.worker_id}] Enter epoch {epoch}".format(**locals()))
            for i in range(total_batch):
                self.run_one_batch(i, epoch)

    def evaluate_per_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.train_phase: False}
        batch_out,label_out = self.sess.run([self.out, self.label], feed_dict=feed_dict)
        correct_num = 0
        for i in range(len(batch_out)):
            if batch_out[i][0] > 0.5:
                batch_out[i][0] = 1.0
                if batch_out[i][0] == label_out[i][0]:
                    correct_num += 1
            else:
                batch_out[i][0] = 0.0
                if batch_out[i][0] == label_out[i][0]:
                    correct_num += 1
        return correct_num


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
