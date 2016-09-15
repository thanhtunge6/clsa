Cross-Lingual Sentiment Analysis
================================

TOC
---

  * Requirements_
  * Installation_
  * Documentation_
     - [Hybrid  Heterogeneous  Transfer  Learning]_
     - NER_
  * References_

.. _Requirements:

Requirements
------------

To install nut you need:

   * Python 2.7
   * Numpy (>= 1.11)
   * Sklearn (>= 0.17)

.. _Installation:

Installation
------------

To clone the repository run, 

   git clone git://github.com/thanhtunge6/clsa.git

.. _Documentation:

Documentation
-------------

.. _Hybrid  Heterogeneous  Transfer  Learning:

HHTL
~~~~

An implementation of Hybrid  Heterogeneous  Transfer  Learning through Deep Learning
See [Joey2014]_ for a detailed description.

The data for cross-language sentiment classification that has been used in the above
study can be found here [#f1]_.

clscl_train
???????????

Training script for CLSCL. See `./clscl_train --help` for further details. 

Usage::

    $ ./clscl_train en de cls-acl10-processed/en/books/train.processed cls-acl10-processed/en/books/unlabeled.processed cls-acl10-processed/de/books/unlabeled.processed cls-acl10-processed/dict/en_de_dict.txt model.bz2 --phi 30 --max-unlabeled=50000 -k 100 -m 450 --strategy=parallel

    |V_S| = 64682
    |V_T| = 106024
    |V| = 170706
    |s_train| = 2000
    |s_unlabeled| = 50000
    |t_unlabeled| = 50000
    debug: DictTranslator contains 5012 translations.
    mutualinformation took 5.624 sec
    select_pivots took 7.197 sec
    |pivots| = 450
    create_inverted_index took 59.353 sec
    Run joblib.Parallel
    [Parallel(n_jobs=-1)]: Done   1 out of 450 |elapsed:    9.1s remaining: 67.8min
    [Parallel(n_jobs=-1)]: Done   5 out of 450 |elapsed:   15.2s remaining: 22.6min
    [..]
    [Parallel(n_jobs=-1)]: Done 449 out of 450 |elapsed: 14.5min remaining:    1.9s
    train_aux_classifiers took 881.803 sec
    density: 0.1154
    Ut.shape = (100,170706)
    learn took 903.588 sec
    project took 175.483 sec

.. note:: If you have access to a hadoop cluster, you can use `--strategy=hadoop` to train the pivot classifiers even faster, however, make sure that the hadoop nodes have Bolt (feature-mask branch) [#f3]_ installed. 

clscl_predict
?????????????

Prediction script for CLSCL.

Usage::

    $ ./clscl_predict cls-acl10-processed/en/books/train.processed model.bz2 cls-acl10-processed/de/books/test.processed 0.01
    |V_S| = 64682
    |V_T| = 106024
    |V| = 170706
    load took 0.681 sec
    load took 0.659 sec
    classes = {negative,positive}
    project took 2.498 sec
    project took 2.716 sec
    project took 2.275 sec
    project took 2.492 sec
    ACC: 83.05
    
.. _References:
References
----------

.. [#f1] http://www.uni-weimar.de/en/media/chairs/webis/corpora/corpus-webis-cls-10/

.. [Joey2014] Zhou, P. T., Pan, S. J., Tsang I. W. and Yan Y. `Hybrid Heterogeneous Transfer Learning through Deep Learning <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8181/8869>`_. In Proceedings of AAAI 2014.
