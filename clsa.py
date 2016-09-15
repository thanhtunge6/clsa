__author__ = 'ngot0008'

import numpy as np
import stackautoencoder as sda
import optparse
from sklearn import svm
from bow import vocabulary, disjoint_voc, load
from compress import compressed_dump, compressed_load

class CLSAModel(object):
    def __init__(self, clf, Ws=None, Wt=None, G=None):
        self.clf = clf
        self.Ws = Ws
        self.Wt = Wt
        self.G = G
        self.s_voc = None
        self.t_voc =None
        self.layers = None


class CLSATrainer(object):
    def __init__(self, s_train, s_train_labels, s_unlabeled, t_unlabeled):
        self.s_train = s_train
        self.s_train_labels = s_train_labels
        self.s_unlabeled = s_unlabeled
        self.t_unlabeled = t_unlabeled

    def train(self, r, layers, noise):
        print "Stack auto encoder"
        [allhxs, Ws, Wt, G] = sda.mSDA(self.s_unlabeled.T, self.t_unlabeled.T, noise, r, layers)
        self.clf = svm.SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  probability=False, shrinking=True, tol=0.001, verbose=False)
        allhxtrain = sda.transformsource(self.s_train.T,Ws,layers)
        print "Train SVM"
        self.clf.fit(allhxtrain.T, np.ravel(self.s_train_labels))
        return CLSAModel(self.clf, Ws=Ws, Wt=Wt, G=G)


def train_args_parser():
    description = """Prefixes `s_` and `t_` refer to source and target language,\
     resp. Train and unlabeled files are expected to be in Bag-of-Words format.
        """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                         "s_lang t_lang s_train_file " \
                                         "s_unlabeled_file t_unlabeled_file " \
                                         "model_file",
                                   description=description)

    parser.add_option("-r",
                      dest="lamda",
                      help="regularization parameter lambda",
                      default=0.1,
                      metavar="float",
                      type="float")

    parser.add_option("--layers",
                      dest="layers",
                      help="number of layers",
                      default=2,
                      metavar="int",
                      type="int")

    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                           "-1 for unlimited.",
                      default=-1,
                      metavar="int",
                      type="int")

    parser.add_option("-n",
                      dest="noise",
                      help="corruption rate",
                      default=0.5,
                      metavar="float",
                      type="float")
    return parser

def train():
    """Training script for CLSCL.

    TODO: different translators.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 6:
        parser.error("incorrect number of arguments (use `--help` for help).")

    slang = argv[0]
    tlang = argv[1]

    fname_s_train = argv[2]
    fname_s_unlabeled = argv[3]
    fname_t_unlabeled = argv[4]

    max_unlabeled = 50000

    # Create vocabularies
    s_voc = vocabulary(fname_s_unlabeled, mindf=2, maxlines=max_unlabeled)
    t_voc = vocabulary(fname_t_unlabeled, mindf=2, maxlines=max_unlabeled)
    s_voc, t_voc, dim = disjoint_voc(s_voc, t_voc)
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    # print("|V| = %d" % dim)

    # Load labeled and unlabeled data
    s_train, labels, classes = load(fname_s_train, s_voc, dim)
    print("classes = {%s}" % ",".join(classes))
    print("|s_train| = %d" % s_train.shape[0])

    s_unlabeled, l1, c1 = load(fname_s_unlabeled, s_voc, dim)
    t_unlabeled, l2, c2 = load(fname_t_unlabeled, t_voc, dim)
    clsa_trainer = CLSATrainer(s_train,labels, s_unlabeled,
                                 t_unlabeled)

    model = clsa_trainer.train(options.lamda, options.layers, options.noise)
    model.s_voc = s_voc
    model.t_voc = t_voc
    model.dim = dim
    model.layers = options.layers
    compressed_dump(argv[5], model)


def predict_args_parser():
    """Create argument and option parser for the
    prediction script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "model_file " \
                                   "t_test_file",
                                   description=description)

    return parser


def predict():
    """Prediction script for CLSA.  """
    parser = predict_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 2:
        parser.error("incorrect number of arguments (use `--help` for help).")

    fname_model = argv[0]
    fname_t_test = argv[1]

    print "Load model"
    clsa_model = compressed_load(fname_model)

    clf = clsa_model.clf
    Ws = clsa_model.clf
    Wt = clsa_model.Wt
    G = clsa_model.G
    t_voc = clsa_model.t_voc
    dim = clsa_model.dim
    layers = clsa_model.layers

    t_test, labels, classes = load(fname_t_test, t_voc, dim)
    transformhxt = sda.transformtarget(t_test.T, G, Wt, layers)
    y = clf.predict(transformhxt.T)

    print "Accuracy: ",np.sum(y == labels)/float(len(y))