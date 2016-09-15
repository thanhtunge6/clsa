__author__ = 'ngot0008'

import stackautoencoder as sda
from bow import vocabulary, disjoint_voc, load
import numpy as np
from sklearn import svm

def train():

    fname_s_train = '/home/tung/Desktop/cls-acl10-processed/st.txt'
    fname_s_unlabeled = '/home/tung/Desktop/cls-acl10-processed/su.txt'
    fname_t_unlabeled = '/home/tung/Desktop/cls-acl10-processed/tu.txt'
    # fname_s_train = 'cls-acl10-processed/en/books/train.processed'
    # fname_s_unlabeled = 'cls-acl10-processed/de/books/trans/en/books/test.processed'
    # fname_t_unlabeled = 'cls-acl10-processed/de/books/test.processed'
    max_unlabeled = 5000


    # Create vocabularies
    s_voc = vocabulary(fname_s_unlabeled, mindf=2, maxlines=max_unlabeled)
    t_voc = vocabulary(fname_t_unlabeled, mindf=2, maxlines=max_unlabeled)
    s_voc, t_voc, dim = disjoint_voc(s_voc, t_voc)
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))

    # Load labeled and unlabeled data
    s_train, labels, classes = load(fname_s_train, s_voc)
    print s_train.shape
    s_unlabeled, l1, c1 = load(fname_s_unlabeled, s_voc)
    print s_unlabeled.shape
    t_unlabeled, l2, c2 = load(fname_t_unlabeled, t_voc)
    print "start"
    [allhxs, Ws, Wt, G] = sda.mSDA(s_unlabeled.T, t_unlabeled.T, 0.8, 0.1, 1)
    print "end", allhxs.shape
    clf = svm.SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  probability=False, shrinking=True, tol=0.001, verbose=False)
    transformtrain = sda.transformsource(s_train.T, Ws, 1)
    print "learn svm"
    clf.fit(transformtrain.T, np.ravel(labels))
    print "done"
    transformhxt = sda.transformtarget(t_unlabeled.T, G, Wt, 1)
    y = clf.predict(transformhxt.T)
    print "Accuracy: ",np.sum(y == l2)/float(len(y))

train()
