__author__ = 'ngot0008'
import numpy as np
import time


def mDA(xx, noise, Lambda):
    # xx : dxn input
    # noise: corruption level
    # Lambda: regularization
    # hx: dxn hidden representation
    # W: dx(d+1) mapping
    [d, n] = xx.shape
    xxb = np.concatenate((xx, np.matrix(np.ones(n))), axis=0)

    # scatter matrix S
    S = xxb*xxb.T

    # corrupting
    q = np.ones((d+1,1))*(1-noise)
    q[d,0]=1
    qm = q*q.T

    # Q: (d+1)x(d+1)
    Q = np.multiply(S, qm)
    for i in range(0, d):
        Q[i,i]=S[i,i]*q[i]


    # P:dx(d+1)
    P = S[0:d,:]*(1-noise)
    P[:,d]=S[0:d,d]

    # final W = P*Q^-1, dx(d+1);
    reg = Lambda*np.identity(d+1)
    reg[d,d]=0
    W = np.linalg.solve((Q+reg).T, P.T).T
    hx = W*xxb;

    hx = np.tanh(hx)
    return [hx, W]


def mSDA(xxs, xxt, noise,Lambda,layers):
    # xxs : s input
    # xxt : target input
    # noise: corruption level
    # layers: number of layers to stack

    # allhx: (layers*d)xn stacked hidden representations
    print "Stacking hidden layers..."
    prevhxs = xxs
    allhxs = xxs
    prevhxt = xxt
    allhxt = xxs
    G = np.empty(layers+1, dtype=object)
    print "layer 0"
    print "Learn mapping"
    G[0]=mapping(xxs,xxt,0.1)
    Ws = np.empty(layers, dtype=object)
    Wt = np.empty(layers, dtype=object)
    for layer in range(0,layers):
        print "layer ",layer+1
        start = time.time()
        print "Compute hidden layer source"
        [newhxs, W1] = mDA(prevhxs,noise,Lambda)
        Ws[layer] = W1
        print "Compute hidden layer target"
        [newhxt, W2] = mDA(prevhxt,noise,Lambda)
        Wt[layer] = W2
        print "Learn maping"
        G[layer+1]=mapping(newhxs,newhxt,0.1)
        print 'Layer ',layer+1,' took ',time.time()-start," sec"
        allhxs = np.concatenate((allhxs, newhxs), axis=0)
        allhxt = np.concatenate((allhxt, newhxt), axis=0)
        prevhxs = newhxs
        prevhxt = newhxt
    return [allhxs, Ws, Wt, G]


def mapping(source,target,Lambda):
    [ds,n] = source.shape
    [dt, nt] = target.shape
    # add bias term
    sourceb = np.concatenate((source, np.matrix(np.ones(n))), axis=0)
    targetb = np.concatenate((target, np.matrix(np.ones(n))), axis=0)
    # calculate transform matrix G
    reg = Lambda*np.identity(dt+1)
    H = (targetb*targetb.T+reg).T
    K = targetb*sourceb.T
    G = np.linalg.solve(H, K).T
    return G


def transformsource(source,Ws,layers):
    [d, n] = source.shape
    prevhx = source
    allhx = source
    for layer in range(0, layers):
        xxb = np.concatenate((prevhx, np.matrix(np.ones(n))), axis=0)
        newhx = (Ws[layer] * xxb)[0:d,:]
        allhx = np.concatenate((allhx, newhx), axis=0)
        prevhx = newhx
    return allhx


def transformtarget(target,G,Wt,layers):
    [d, n] = target.shape
    sourcedim = G[0].shape[0]-1
    prevhx = target
    allhx = (G[0]*np.concatenate((target, np.matrix(np.ones(n))), axis=0))[0:sourcedim,:]
    for layer in range (0,layers):
        xxb = np.concatenate((prevhx, np.matrix(np.ones(n))), axis=0)
        newhx = Wt[layer]*xxb
        transformhx = (G[layer+1]*np.concatenate((newhx, np.matrix(np.ones(n))), axis=0))[0:sourcedim,:]
        allhx = np.concatenate((allhx, transformhx), axis=0)
        prevhx = newhx
    return allhx

