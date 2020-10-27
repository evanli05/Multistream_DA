from sklearn.metrics import f1_score



def cal_accuracy(y_true, y_predict):
    return f1_score(y_true, y_predict)

    