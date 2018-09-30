import tensorflow as tf
import numpy as np
from copy import deepcopy

from constants import RESULT_PATH, TF_SEED, NP_SEED
np.random.seed(NP_SEED)
tf.set_random_seed(TF_SEED)

import util
from visualizer import plot_metrics


def to_one_hot(y_sparse):
    # y_sparse : shape of (-1, 1)
    # return : shape of(-1, 3)
    if y_sparse.dtype != int:
        y_sparse = y_sparse.astype(int)
    return np.eye(3)[y_sparse.ravel()]

def to_sparse(y_one_hot):
    # y_sparse : shape of (-1, 3)
    # return : shape of(-1, 1)
    y_sparse = np.argmax(y_one_hot, axis=-1)
    return y_sparse

class NN:
    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):

        self.name = name
        self.n_class = n_class
        self.dropout_keep_prob = dropout_keep_prob
        self.n_lookback = n_lookback
        self.n_feature = n_feature
        self.random_feature_idx = None

        self.metrics = {'train': {'loss': []},
                        'val': {'loss': [], 'score_seq': []},
                        'test': {'loss': [], 'score_seq': []}}
        self.predicts = {'val':[], 'test':[], 'problem':[],
                         'val_softmax':[],'test_softmax':[],'problem_softmax':[]}

        self.sess = session

        with tf.name_scope(name):
            self.x = tf.placeholder(tf.float32, [None, n_lookback, n_feature], name='x')
            self.y = tf.placeholder(tf.float32, [None, n_class], name='y_onehot')
            self.w = tf.placeholder(tf.float32, [None, ], name='w')

            self.is_training = tf.placeholder(tf.bool)
            self.tf_dropout_keep_prob = tf.placeholder(tf.float32)
            global_step = tf.Variable(0, trainable=False)
            decaying_lr = tf.train.exponential_decay(lr, global_step, 100, lr_decay)

            x = self.x
            output = self.hidden_layers(x)  # (-1, 3)
            
            self.y_softmax = tf.nn.softmax(output)
            self.y_pred = tf.expand_dims(tf.argmax(output, axis=-1), 1)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=output))
            # self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=output, weights=self.w))

            self.optimizer = tf.contrib.opt.NadamOptimizer(decaying_lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            self.saver = tf.train.Saver()

    def hidden_layers(self, x):
        """
        x: shape=[-1, n_lookback, n_feature]
        :return: logit tensor (shape=[-1, n_class])
        """
        raise NotImplementedError("YOU MUST IMPLEMENT hidden_layers() FUNCTION")
        pass

    @staticmethod
    def swell_noise_layer(input_layer, std):
            """
            마지막 feature인 swell_t-1에 대해 가우시안 노이즈 추가.
            가우시안 노이즈 추가한 후 [0.0~2.0]으로 클리핑한 후 원래 input에 적용한다.
            :param input_layer: a tensor shape of [batch_size, n_lookback, n_feature]
            :param std: stddev of noise
            :return:
            """
            noise_shape = tf.shape(input_layer)[0], tf.shape(input_layer)[1], 1
            noise = tf.truncated_normal(shape=noise_shape, mean=0.0, stddev=std, dtype=tf.float32)
            noised_swell = input_layer[:, :, -1:] + noise
            noised_swell = tf.clip_by_value(noised_swell, 0.0, 2.0)
            input_layer = tf.concat([input_layer[:, :, :-1], noised_swell], axis=-1)
            return input_layer

    @staticmethod
    def add_noise_layer(input_layer, std):
        """
        마지막 feature(swell_t-1)를 제외한 나머지 feature 들에 대해
        가우시안 노이즈 추가한다.
        :param input_layer: a tensor shape of [batch_size, n_lookback, n_feature]
        :param std: stddev of noise
        :return:
        """
        noise_shape = tf.shape(input_layer)[0], tf.shape(input_layer)[1], tf.shape(input_layer)[2] - 1
        noise = tf.truncated_normal(shape=noise_shape, mean=0.0, stddev=std, dtype=tf.float32)
        noised_input = input_layer[:, :, :-1] + noise
        input_layer = tf.concat([noised_input, input_layer[:, :, -1:]], axis=-1)
        return input_layer

    def run_batch(self, x, y, w, batch_size, is_training):

        if batch_size == -1:
            batch_size = len(x)

        total_loss = 0
        y_preds = []

        total_steps = len(x) // batch_size
        last_batch_size = len(x) - total_steps * batch_size
        if last_batch_size != 0:
            print('last batch is size of {}'.format(last_batch_size))
            total_steps += 1

        if is_training:
            p = np.random.permutation(len(x))
            x, y, w = x[p], y[p], w[p]
            for j in range(total_steps):
                x_batch = x[j * batch_size: (j + 1) * batch_size]
                y_batch = y[j * batch_size: (j + 1) * batch_size]
                w_batch = w[j * batch_size: (j + 1) * batch_size]
                _, loss, y_pred = self.sess.run([self.train_op, self.loss, self.y_pred],
                                                   feed_dict={
                                                       self.x: x_batch,
                                                       self.y: y_batch,
                                                       self.w: w_batch,
                                                       self.tf_dropout_keep_prob: self.dropout_keep_prob,
                                                       self.is_training: True})
                total_loss += loss
                y_preds.append(y_pred)

        else:
            for j in range(total_steps):
                x_batch = x[j * batch_size: (j + 1) * batch_size]
                y_batch = y[j * batch_size: (j + 1) * batch_size]
                w_batch = w[j * batch_size: (j + 1) * batch_size]
                loss, y_pred = self.sess.run([self.loss, self.y_pred],
                                                     feed_dict={
                                                         self.x: x_batch,
                                                         self.y: y_batch,
                                                         self.w: w_batch,
                                                         self.tf_dropout_keep_prob: 1.,
                                                         self.is_training: False})
                total_loss += loss
                y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds)
        return total_loss/total_steps, y_preds

    def train(self, ds, BATCH_SIZE, EPOCH, feature_shuffle=False, train_all_data=False, verbose=True):


        # Feature Random Shuffling. 단, 마지막 feature는 swell_t-1으로 고정
        if feature_shuffle:
            ds = deepcopy(ds)
            p = np.random.permutation(ds['train']['x'].shape[2]-1)
            ds['train']['x'][:, :, :len(p)] = ds['train']['x'][:,:,p]
            ds['val']['x'][:, :, :len(p)] = ds['val']['x'][:,:,p]
            ds['test']['x'][:, :, :len(p)] = ds['test']['x'][:,:,p]
            ds['problem']['x'][:, :, :len(p)] = ds['problem']['x'][:,:,p]
            self.random_feature_idx = p

        # one hot
        ds['train']['y_onehot'] = to_one_hot(ds['train']['y'])
        ds['val']['y_onehot'] = to_one_hot(ds['val']['y'])
        ds['test']['y_onehot'] = to_one_hot(ds['test']['y'])

        # swell t-1과 맞춰야 하는 swell이 다른 경우에 weight를 2로 줌 -> 추후 loss 에서 사용 가능
        for d0 in ['train', 'val', 'test']:
            diff_samples = ds[d0]['x'][:, -1, -1] != ds[d0]['y'][:, -1]
            ds[d0]['w'] = np.array([1, 2])[diff_samples.astype(int)]

        for i in range(EPOCH):

            print('[NAME: {}, EPOCH: {}]'.format(self.name, i))
            # Train
            train_loss, _ = self.run_batch(ds['train']['x'], ds['train']['y_onehot'], ds['train']['w'], BATCH_SIZE, is_training=True)

            # Validation
            if verbose:
                print('predict ONE Validation')
            val_loss, val_pred_one = self.run_batch(ds['val']['x'], ds['val']['y_onehot'], ds['val']['w'], BATCH_SIZE, is_training=train_all_data)
            val_acc_one, val_score_one, val_max_score = util.calc_metric(ds['val']['y'].ravel(),
                                                          val_pred_one.round().astype(int).ravel(), self.n_class, verbose)
            
            if verbose:
                print('predict SEQ Validation')
            val_pred_seq, val_softmax_seq = self.predict_sequence(ds['val']['x'])
            val_pred_seq = val_pred_seq.round().astype(int).ravel()
            val_acc_seq, val_score_seq, val_max_score = util.calc_metric(ds['val']['y'].ravel(),
                                                          val_pred_seq, self.n_class, verbose)

            # Test
            if verbose:
                print('predict ONE Test')
            test_loss, test_pred_one = self.run_batch(ds['test']['x'], ds['test']['y_onehot'], ds['test']['w'], BATCH_SIZE, is_training=train_all_data)
            test_acc_one, test_score_one, test_max_score = util.calc_metric(ds['test']['y'].ravel(),
                                                            test_pred_one.ravel().round().astype(int), self.n_class, verbose)
            
            if verbose:
                print('predict SEQ Test')
            test_pred_seq, test_softmax_seq = self.predict_sequence(ds['test']['x'])
            test_pred_seq = test_pred_seq.round().astype(int).ravel()
            test_acc_seq, test_score_seq, test_max_score = util.calc_metric(ds['test']['y'].ravel(),
                                                            test_pred_seq, self.n_class, verbose)

            print("[SUMMARY]\n(Loss) train: {:.5} val: {:.5} test: {:.5}".format(train_loss, val_loss, test_loss))
            print("val_acc_seq : {:.5} val_score_seq : {:.5} (max: {:.5})".format(val_acc_seq, val_score_seq, val_max_score))
            print("test_acc_seq: {:.5} test_score_seq: {:.5} (max: {:.5})\n".format(test_acc_seq, test_score_seq, test_max_score))

            # append current epoch's metrics
            self.metrics['train']['loss'].append(train_loss)
            self.metrics['val']['loss'].append(val_loss)
            self.metrics['test']['loss'].append(test_loss)
            self.metrics['val']['score_seq'].append(val_score_seq)
            self.metrics['test']['score_seq'].append(test_score_seq)

            if verbose:
                plot_metrics(**self.metrics)

            # predict Problem
            problem_pred, problem_softmax = self.predict_sequence(ds['problem']['x'])
            problem_pred = problem_pred.astype(int).ravel()

            # append current epoch's predictions
            self.predicts['val'].append(val_pred_seq)
            self.predicts['test'].append(test_pred_seq)
            self.predicts['problem'].append(problem_pred)
            self.predicts['val_softmax'].append(val_softmax_seq)
            self.predicts['test_softmax'].append(test_softmax_seq)
            self.predicts['problem_softmax'].append(problem_softmax)
        # END for i in range(EPOCH):

        return self.predicts

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def predict_one(self, x):
        """
        :param x: shape=[N_LOOKBACK, N_FEATURE]
        :return: y
        """
        x = x[np.newaxis,]
        assert len(x[np.isnan(x)]) == 0
        y_pred = self.sess.run(self.y_pred, feed_dict={self.x: x,
                                                       self.tf_dropout_keep_prob: 1.,
                                                       self.is_training: False})
        return y_pred

    def predict_batch(self, x_list):
        y_pred, y_softmax = self.sess.run([self.y_pred, self.y_softmax], 
                                                    feed_dict={self.x: x_list,
                                                    self.tf_dropout_keep_prob: 1.,
                                                    self.is_training: False})
        return y_pred, y_softmax

    def predict_sequence(self, x_list):
        """
        test data와 같은 데이터로 예측할 때 사용하는 함수
        * 테스트 데이터같은 경우 연속된 24시간을 예측해야 하기 때문에, swell_t-1 데이터 중간에 nan이 들어있다.
        이를 NN을 이용해 한 시간 단위로 예측한 후 swell_t-1데이터를 채워나가며 예측을 진행한다.
        :param x_list: shape=[-1, N_LOOKBACK, N_FEATURE]
        :return:  list of y
        :param x_list:
        :return:
        """
        x_day_list = deepcopy(x_list.reshape(-1, 24, self.n_lookback, self.n_feature))

        y_preds = np.zeros((x_list.shape[0] // 24, 24))
        y_softmaxs = np.zeros((x_list.shape[0] // 24, 24, 3))

        for i in range(24):
            x_batch = x_day_list[:, i, :, :]
            y_pred_batch, y_softmax_batch = self.predict_batch(x_batch)
            y_preds[:, i:i + 1] = y_pred_batch
            y_softmaxs[:, i, :] = y_softmax_batch
            for j in range(1, min(self.n_lookback + 1, 24 - i)):
                x_day_list[:, i + j, -j, -1] = y_pred_batch.ravel()

        y_preds = y_preds.ravel()
        y_softmaxs = y_softmaxs.reshape(y_softmaxs.shape[0]*y_softmaxs.shape[1], 3)

        return y_preds, y_softmaxs


class Dense(NN):

    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)

    def hidden_layers(self, x):

        x = tf.cond(self.is_training, lambda: self.swell_noise_layer(x, 2.0), lambda: x)
        x = tf.cond(self.is_training, lambda: self.add_noise_layer(x, 0.1), lambda: x)
        X = tf.reshape(x, (-1, x.shape[1] * x.shape[2]))
        layer1 = tf.contrib.layers.fully_connected(X, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer1 = tf.layers.dropout(layer1, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer2 = tf.contrib.layers.fully_connected(layer1, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer2 = tf.layers.dropout(layer2, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer3 = tf.contrib.layers.fully_connected(layer2, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer3 = tf.layers.dropout(layer3, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer4 = tf.contrib.layers.fully_connected(layer3, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer4 = tf.layers.dropout(layer4, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        layer5 = tf.contrib.layers.fully_connected(layer4, 1024, activation_fn=tf.nn.relu,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        layer5 = tf.layers.dropout(layer5, rate=1 - self.dropout_keep_prob, training=self.is_training, seed=TF_SEED)

        output = tf.contrib.layers.fully_connected(layer5, 3, activation_fn=None,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(TF_SEED))
        return output


class RAND_CNN(NN):
    def __init__(self, session: tf.Session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name):
        super().__init__(session, n_class, n_lookback, n_feature, dropout_keep_prob, lr, lr_decay, name)
        
    def hidden_layers(self, x):
         #x: [None, n_lookback, n_feature]
        x = tf.cond(self.is_training, lambda: self.swell_noise_layer(x, 2.0), lambda: x)
        x = tf.cond(self.is_training, lambda: self.add_noise_layer(x, 0.1), lambda: x)

        X = tf.reshape(x, [-1, self.n_lookback, self.n_feature, 1])
        conv1 = tf.layers.conv2d(X, filters=64,kernel_size=[1,  3], strides=[1, 1],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu)
        conv1 = tf.layers.dropout(conv1, rate=0.5, training=self.is_training)

        conv2 = tf.layers.conv2d(conv1, filters=64,kernel_size=[conv1.shape[1], 4], strides=[1, 1],
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu)
        conv2 = tf.layers.dropout(conv2, rate=0.5, training=self.is_training)

        conv2 = tf.contrib.layers.flatten(conv2)
        output = tf.contrib.layers.fully_connected(conv2, 3, activation_fn=None)

        return output

