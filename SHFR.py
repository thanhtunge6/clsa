__author__ = 'ngot0008'

import numpy as np
from sklearn import linear_model

def SHFR(wt, ws, alpha):
    # wt (nc x Dt weight matrix in target language)
    # ws (nc x Ds weight matrix in source language)
    # alpha: regularization term
    # return transformation matrix G (wt_transpose = G x ws_transpose)
    h = wt.tolist()
    w = ws.tolist()
    clf = linear_model.Lasso(alpha=alpha,positive=True)
    clf.fit(w,h)
    return clf.coef_