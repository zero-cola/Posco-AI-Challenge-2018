import os
import numpy as np
from copy import deepcopy

import tensorflow as tf
import util
from nn_clf import to_one_hot
from visualizer import plot_metrics
from constants import MODEL_PATH

class Ensembler:

    def __init__(self, sess, ds, batch_size, total_epoch, feature_shuffle, train_all_data, nn_list):

        self.sess = sess
        self.ds = ds
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.feature_shuffle = feature_shuffle
        self.train_all_data = train_all_data
        self.nn_list = nn_list

    def __save_model(self):
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        saver = tf.train.Saver()
        save_path = os.path.join(MODEL_PATH, 'ensemble')
        saver.save(self.sess, save_path, write_meta_graph=False)

    def __restore_model(self):
        restore_path = os.path.join(MODEL_PATH, 'ensemble')
        ckpt_path = restore_path
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_path)

    def train_per_nn(self, verbose=False):
        """
        nn_list에 있는 nn 각각을 total_epoch까지 학습한 뒤, 앙상블
        (전제조건: n_class == 3)
        verbose: False 이면 각각 nn에 대한 정보 적게 출력
        """
        random_shuffle_idx_list = []
        # nn 각각 total_epoch까지 학습
        for nn in self.nn_list:
            nn.train(self.ds, self.batch_size, self.total_epoch, 
                     self.feature_shuffle, # swell_t-1를 제외한 피쳐 셔플 여부
                     self.train_all_data, # val, test 데이터를 학습에 사용할 지 여부
                     verbose
                    )
            random_shuffle_idx_list.append(nn.random_feature_idx)
        self.__save_model()
        random_shuffle_idx_list = np.array(random_shuffle_idx_list)
        np.save(os.path.join(MODEL_PATH, 'random_shuffle_idx_list'), random_shuffle_idx_list)

        ensemble_metrics = {'val': {'score_seq': []},
                            'test': {'score_seq': []}}
        print("\n[Ensembled Model Val/Testing]")
        for i in range(self.total_epoch):
            print("[Ensemble EPOCH: {}]".format(i))
            val_softmax = np.zeros((len(self.ds['val']['x']),3))
            test_softmax = np.zeros((len(self.ds['test']['x']),3))
            problem_softmax = np.zeros((len(self.ds['problem']['x']),3))
            
            # 각각의 nn 모델의 softmax 결과값을 합산한뒤, argmax로 최종 예측값 도출
            for nn in self.nn_list:
                val_softmax += nn.predicts['val_softmax'][i]
                test_softmax += nn.predicts['test_softmax'][i]
                problem_softmax += nn.predicts['problem_softmax'][i]

            val_pred_seq = np.argmax(val_softmax, axis=1)
            test_pred_seq = np.argmax(test_softmax, axis=1)
            problem_pred_seq = np.argmax(problem_softmax, axis =1)
            
            val_acc_seq, val_score_seq, val_max_score = util.calc_metric(
                                            self.ds['val']['y'].ravel(),
                                            val_pred_seq, 3)
            test_acc_seq, test_score_seq, test_max_score = util.calc_metric(
                                            self.ds['test']['y'].ravel(),
                                            test_pred_seq, 3)
            
            print("[SUMMARY] val_acc_seq :{:.5}  val_score_seq :{:.5} (max:{:.5})"
                  .format(val_acc_seq, val_score_seq, val_max_score))
            print("test_acc_seq :{:.5}  test_score_seq :{:.5} (max:{:.5})"
                  .format(test_acc_seq, test_score_seq, test_max_score))
            
            ensemble_metrics['val']['score_seq'].append(val_score_seq)
            ensemble_metrics['test']['score_seq'].append(test_score_seq)
            plot_metrics(**ensemble_metrics)
            
            # epoch당 앙상블 모델의 예측 결과를 엑셀파일로 저장
            util.save_result_excel(problem_pred_seq,filename='result_'+str(i)+'ep.xlsx')

    def restore_and_testing(self):

        self.__restore_model()
        random_shuffle_idx_list = np.load(os.path.join(MODEL_PATH, 'random_shuffle_idx_list.npy'))

        problem_softmax = np.zeros((len(self.ds['problem']['x']), 3))

        for i, nn in enumerate(self.nn_list):
            nn.random_feature_idx = random_shuffle_idx_list[i]

            # Feature Random Shuffling. 단, 마지막 feature는 swell_t-1으로 고정
            if self.feature_shuffle:
                ds = deepcopy(self.ds)
                p = nn.random_feature_idx
                ds['problem']['x'][:, :, :len(p)] = ds['problem']['x'][:,:,p]

            # predict Problem
            problem_pred, _problem_softmax = nn.predict_sequence(ds['problem']['x'])

            # 각각의 nn 모델의 softmax 결과값을 합산한뒤, argmax로 최종 예측값 도출
            problem_softmax += _problem_softmax

        problem_pred_seq = np.argmax(problem_softmax, axis=1)

        # epoch당 앙상블 모델의 예측 결과를 엑셀파일로 저장
        util.save_result_excel(problem_pred_seq, filename='result_ensemble.xlsx')
