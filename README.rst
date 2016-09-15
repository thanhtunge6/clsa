Cross-Lingual Sentiment Analysis
================================

TOC
---

  * Requirements_
  * Installation_
  * Documentation_
     - HHTL_
  * References_

.. _Requirements:

Requirements
------------

To install clsa you need:

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

.. _HHTL:

HHTL
~~~~

An implementation of Hybrid  Heterogeneous  Transfer  Learning through Deep Learning.
See [Joey2014]_ for a detailed description.

The data for cross-language sentiment classification that has been used in the above
study can be found here [#f1]_.

clsa_train
??????????

Training script for CLSA. See `./clsa_train --help` for further details. 

Usage::

    $ python ./clsa_train en de cls-acl10-processed/en/books/train.processed cls-acl10-processed/de/books/trans/en/books/test.processed cls-acl10-processed/de/books/test.processed model.bz2 -r 0.1 --layer 1 -n 0.8


    vocabulary took 0.065 sec
    vocabulary took 0.072 sec
    |V_S| = 5271
    |V_T| = 5936
    load took 0.373 sec
    classes = {negative,positive}
    |s_train| = 1000
    load took 0.261 sec
    load took 0.265 sec
    Stack auto encoder
    Stacking hidden layers...
    layer 0
    Learn mapping
    layer  1
    Compute hidden layer source
    Compute hidden layer target
    Learn maping
    Layer  1  took  20.8635218143  seconds
    Train SVM
    Save model


clsa_predict
????????????

Prediction script for CLSA.

Usage::

    $ python ./clsa_test model.bz2 cls-acl10-processed/de/books/test.processed

    Load model
    load took 0.274 sec
    Accuracy:  0.75125

.. _References:
References
----------

.. [#f1] http://www.uni-weimar.de/en/media/chairs/webis/corpora/corpus-webis-cls-10/

.. [Joey2014] Zhou, P. T., Pan, S. J., Tsang I. W. and Yan Y. `Hybrid Heterogeneous Transfer Learning through Deep Learning <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8181/8869>`_. In Proceedings of AAAI 2014.
