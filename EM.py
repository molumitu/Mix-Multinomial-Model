from matplotlib.pyplot import axis
import numpy as np
from scipy.special import logsumexp
from data_loader import cal_coeff
from torch.utils.tensorboard import SummaryWriter


def EM_Circle(rng, X, K, gamma=None, eps=1e-5, maxiter=1000):
    writer = SummaryWriter()
    # X (D, W) ((18271, 83735))
    D = X.shape[0]
    Nd = np.sum(X, axis=1) #(D,)
    Coeff = cal_coeff(X,Nd)
    if gamma is None:
        gamma = rng.dirichlet(np.ones(K), D)  # responsibility (D,K)
    gamma_delta = 1
    L_list = []
    i = 0
    while i < maxiter and gamma_delta > eps:
        pi = np.sum(gamma, axis=0) / D # the weight of mixture models (K,)
        mu = (X.T.dot(gamma) / (np.sum(gamma.T * Nd, axis=1))).T  #(K, W)
        L = np.log(pi) + X.dot(np.log(mu.T+eps)) - Coeff#(D, K)
        gamma_new = np.exp((L.T - (logsumexp(L, axis=1))).T)
        gamma_delta = np.max(np.abs(gamma_new-gamma))
        gamma = gamma_new
        i += 1
        likelihood = np.sum(logsumexp(L, axis=1))
        L_list.append(likelihood)
        writer.add_scalar('Train/Log-Likelihood', likelihood, i)
    return pi, mu, gamma, L_list

