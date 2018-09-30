import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import shutil
from openpyxl import load_workbook

from constants import RAWDATA_PATH, RESULT_PATH

def save_result_excel(result, filename='result.xlsx', cvt_1_to_0=True):
    """
    최종 정답을 엑셀에 저장하는 함수
    :param result: 0, 1로 이루어진 ndarray. len=25*24
    """
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    shutil.copy(os.path.join(RAWDATA_PATH, 'result_sample.xlsx'),
                os.path.join(RESULT_PATH, filename))

    if cvt_1_to_0:
        result = np.array([0, 0, 1])[result.ravel()][:, np.newaxis]
    else:
        result = np.array([0, 1, 2])[result.ravel()][:, np.newaxis]

    book = load_workbook(os.path.join(RESULT_PATH, filename))
    writer = pd.ExcelWriter(os.path.join(RESULT_PATH, filename), engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    pd.DataFrame(result.reshape(-1, 24)).to_excel(writer, header=False, index=False, startrow=3, startcol=1)
    writer.save()

def calc_metric(y, y_pred, n_class=3, verbose=True):
    """
    TP : +2점
    FP : -1점
    TN : +1점
    FN : -2점
    :param y: sparse format e.g. [1, 0, 2, 0, 0 ..]
    :param y_pred: sparse format
    :return: accuracy, score
    """
    eps = 1e-6

    if n_class == 3:
        if verbose:
            print(confusion_matrix(y, y_pred))
        y = np.array([0, 0, 1])[y.ravel()]
        y_pred = np.array([0, 0, 1])[y_pred.ravel()]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    if verbose:
        print("  TN   FP   FN   TP\n {:^5}{:^5}{:^5}{:^5}".format(tn, fp, fn, tp), end=' ')

    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    if verbose:
        print("Recall: {:.5} Precision: {:.5}".format(recall, precision))

    mean_score = (tn - fp - (2*fn) + (2*tp)) / len(y)
    max_mean_score = (tn + fp + (2*fn) + (2*tp)) / len(y)
    acc = sum([tn, tp]) / len(y)
    
    return acc, mean_score, max_mean_score

def print_false_dates(y, y_pred, y_dt, epoch, cvt_1_to_0=True, name=''):

    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    if name != '':
        if not os.path.exists(os.path.join(RESULT_PATH, name)):
            os.mkdir(os.path.join(RESULT_PATH, name))

        save_path = os.path.join(RESULT_PATH, name, str(epoch) + '_epoch.csv')
    else:
        save_path = os.path.join(RESULT_PATH, str(epoch) + '_epoch.csv')

    if y_dt.ndim == 1:
        y_dt = y_dt[:, np.newaxis]

    if cvt_1_to_0:
        y = np.array([0, 0, 1])[y.ravel()][:, np.newaxis]
        y_pred = np.array([0, 0, 1])[y_pred.ravel()][:, np.newaxis]
    else:
        y = np.array([0, 1, 2])[y.ravel()][:, np.newaxis]
        y_pred = np.array([0, 1, 2])[y_pred.ravel()][:, np.newaxis]

    idx = np.array(y_pred != y)

    df = pd.DataFrame({'y': y[idx], 'y_pred': y_pred[idx]}, index=y_dt[idx])
    df = df.sort_index()
    df.to_csv(save_path, index=True, index_label='date')
    # print('날짜', '실제', '예측')
    # for dt, y_true, y_hat in zip(y_dt[idx], _y[idx], _y_pred[idx]):
    #     print(dt, y_true, y_hat)
