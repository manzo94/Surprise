"""
the :mod:`factorized_neighborhood` module includes some algorithms combining
the weighted KNN and the matrix factorization
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
cimport cython
import numpy as np
from six.moves import range
import heapq
import sys
from tqdm import tqdm

import time

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class FactorizedNeighborhood(AlgoBase):
    """ The *FactorizedNeighborhood* algorithm is an extension of :class:`SVDpp` combining
        matrix factorization and KNN models. 
    """
    # TODO: check if the algorithm doesn't work because of some error in the code
    # or becuase it is very difficult to find the right parameters.

    def __init__(self, n_factors=20, n_epochs=20, k=20, bsl_options={}, sim_options={},
                 init_mean=0, init_std_dev=.1, init_knn='zeros', clip=False, lr_all=.007, reg_all=.02, lr_bsl=0.007, lr_mf=0.007, lr_yj=0.007,
                 lr_knn=0.001, lr_cij=0.001, reg_bsl=0.005, reg_mf=0.015, reg_yj=0.015, reg_knn=0.015, reg_cij=0.015, shuffle_sgd=True, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.k = k
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.init_knn = init_knn
        self.clip = clip
        self.lr_bsl = lr_bsl if lr_bsl is not None else lr_all
        self.lr_mf = lr_mf if lr_mf is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.lr_knn = lr_knn if lr_knn is not None else lr_all
        self.lr_cij = lr_cij if lr_cij is not None else lr_all
        self.reg_bsl = reg_bsl if reg_bsl is not None else reg_all
        self.reg_mf = reg_mf if reg_mf is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.reg_knn = reg_knn if reg_knn is not None else reg_all
        self.reg_cij = reg_cij if reg_cij is not None else reg_all
        self.shuffle_sgd = shuffle_sgd
        self.verbose = verbose

        AlgoBase.__init__(self, bsl_options=bsl_options, sim_options=sim_options)

    def train(self, trainset):
    
        if self.verbose:
            print('Starting train with parameters:')
            print('n_factors: {}, k: {}, init_mean: {}, init_std: {}'.format(self.n_factors, self.k, self.init_mean, self.init_std_dev))
            print('init_knn: {}, clip: {}'.format(self.init_knn, self.clip))
            print('Learning rates: {}, {}, {}, {}, {}'.format(self.lr_bsl, self.lr_mf, self.lr_yj, self.lr_knn, self.lr_cij))
            print('Regulizers: {}, {}, {}, {}, {}'.format(self.reg_bsl, self.reg_mf, self.reg_yj, self.reg_knn, self.reg_cij))
            print('Shuffled: ', self.shuffle_sgd)

        AlgoBase.train(self, trainset)
        
        self.sqrt_Iu = np.zeros(trainset.n_users, np.double)
        self.near_items = np.empty(trainset.n_items, dtype=list)
        
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        self.bu_base = self.bu
        self.bi_base = self.bi
        self.sgd(trainset)

        
    # Directives to speed up the code. Comment out while developing
    @cython.wraparound(False)  # turn off negative index wrapping
    @cython.boundscheck(False)  # turn off bounds check
    def sgd(self, trainset):
    
        # KNN ratings
        cdef np.ndarray[np.double_t, ndim=2] ruj
        # KNN list of nearest items
        cdef np.ndarray[list] near_items
        cdef list rated_j
        # KNN weights
        cdef np.ndarray[np.double_t, ndim=2] wij
        # Implicit ratings in KNN
        cdef np.ndarray[np.double_t, ndim=2] cij
        # Baseline biases
        cdef np.ndarray[np.double_t] bu_base = self.bu_base
        cdef np.ndarray[np.double_t] bj_base = self.bi_base
        cdef double b_glob_u, buj
        # KNN params
        cdef double lr_knn = self.lr_knn
        cdef double reg_knn = self.reg_knn
        
        # KNN estimate
        cdef double knn, eff_neigh
        cdef np.ndarray[np.double_t] ratbuj
        
        # Development utils
        cdef double knn_err
        cdef double mf_err
        cdef double global_err
        
        cdef list all_ratings#np.ndarray[tuple] all_ratings
        
        # Fixed parameters for each users, no need to update at each iteration over ratings
        cdef np.ndarray[list] Iu
        cdef np.ndarray[np.double_t] sqrt_Iu
        
        # Init KNN weights
        if self.init_knn=='sim':
            wij = self.sim
            cij = np.zeros((trainset.n_items, trainset.n_items))
        elif self.init_knn=='zeros':
            wij = np.zeros((trainset.n_items, trainset.n_items))
            cij = np.zeros((trainset.n_items, trainset.n_items))
        else:
            wij = np.random.normal(self.init_mean, self.init_std_dev,
                               (trainset.n_items, trainset.n_items))
            cij = np.random.normal(self.init_mean, self.init_std_dev,
                               (trainset.n_items, trainset.n_items))
                               
        ruj = np.zeros((trainset.n_users, self.k), np.double)
        ratbuj = np.zeros(trainset.n_items, np.double)
        
        sqrt_Iu = np.zeros(trainset.n_users, np.double)
        Iu = np.empty(trainset.n_users, dtype=list)
        
        near_items = np.empty(trainset.n_items, dtype=list)
        all_ratings = []#np.empty(trainset.n_ratings, dtype=tuple)
        rated_j = []
        
        
        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi
        # item implicit factors
        cdef np.ndarray[np.double_t, ndim=2] yj

        cdef int u, i, j, f, n
        cdef double r, err, dot, puf, qif, _
        cdef double global_mean = self.trainset.global_mean
        cdef np.ndarray[np.double_t] u_impl_fdb

        cdef double lr_bsl = self.lr_bsl
        cdef double lr_mf = self.lr_mf
        cdef double lr_yj = self.lr_yj
        cdef double lr_cij = self.lr_cij
        
        cdef double reg_bsl = self.reg_bsl
        cdef double reg_mf = self.reg_mf
        cdef double reg_yj = self.reg_yj
        cdef double reg_cij = self.reg_cij

        bu = self.bu
        bi = self.bi

        pu = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_users, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))
        yj = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors, np.double)
        

        # Find fixed user info
        if self.verbose:
                print(" Building user fixed info...")
        for u in trainset.all_users():
            # number of items rated by users
            Iu[u] = [j for (j, _) in trainset.ur[u]]
            sqrt_Iu[u] = np.sqrt(len(Iu[u]))
            self.sqrt_Iu[u] = sqrt_Iu[u]
        
        # Find nearest neighbors fo each items 
        if self.verbose:
            print(" Building items neighborhood...")
            mean = 0
        for i in trainset.all_items():
            # find best k neghbours of items
            near_items[i] = self.get_neighbors(i, self.k)
            self.near_items[i] = near_items[i]
            if self.verbose:
                mean += len(near_items[i])
        if self.verbose:
            print('Average number of valid neghbours: ', mean/trainset.n_items)
        
        # Iteration of the SGD 
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
                start = time.time()
                knn_err = 0
                mf_err = 0
                global_err = 0
            # Iterate over all ratings
            all_ratings = list(trainset.all_ratings())
            if self.shuffle_sgd:
                np.random.shuffle(all_ratings)
            for n, (u, i, r) in enumerate(tqdm(all_ratings, mininterval=10)):

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for f in range(self.n_factors):
                    for j in Iu[u]:
                        u_impl_fdb[f] += yj[j, f]
                    u_impl_fdb[f] /= sqrt_Iu[u]

                # Reset estimations
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                knn = 0
                
                # Matrix factorization estimation
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])
                
                # KNN estimation
                b_glob_u = global_mean + bu_base[u]
                # Find items rated that are in the neghborhood
                rated_j = [(j, rat) for (j, rat) in trainset.ur[u] if j in near_items[i]]
                # Add only items which are neighbors
                if rated_j:
                    for (j, rat) in rated_j:
                        buj = b_glob_u + bj_base[j]
                        # explicit and explicit KNN estimation
                        ratbuj[j] = (rat - buj)
                        knn += ratbuj[j] * wij[i,j] + cij[i,j]
                    eff_neigh = np.sqrt(len(rated_j))
                    knn /= eff_neigh
                
                # estimate rating
                err = global_mean + bu[u] + bi[i]
                if self.verbose:
                    knn_err += np.square(r - np.clip(err + dot, 1, 5))
                    mf_err += np.square(r - np.clip(err + knn, 1, 5))
                    global_err += np.square(r - np.clip(err + dot + knn, 1, 5))
                    
                err += dot + knn    
                
                # Clip estimation
                if self.clip:
                    if err>5:
                        err = 5
                    elif err<1:
                        err = 1
                
                # compute error
                err = r - err

                # BSL factors
                bu[u] += lr_bsl * (err - reg_bsl * bu[u])
                bi[i] += lr_bsl * (err - reg_bsl * bi[i])

                # update factors
                for f in range(self.n_factors):
                    # MF factors
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_mf * (err * qif - reg_mf * puf)
                    qi[i, f] += lr_mf * (err * (puf + u_impl_fdb[f]) -
                                         reg_mf * qif)
                    
                    for j in Iu[u]:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu[u] -
                                             reg_yj * yj[j, f])
                    
                # Knn factors
                if len(rated_j):
                    err /= eff_neigh
                    for (j, _) in rated_j:
                        wij[i, j] += lr_knn * (err * ratbuj[j] -
                                               reg_knn * wij[i, j])
                        cij[i, j] += lr_cij * (err - reg_cij * cij[i,j])
                                             
            # Scale learning rate at each iteration
            lr_bsl *= 0.9
            lr_mf *= 0.9
            lr_yj  *=0.9
            lr_knn *= 0.9
            lr_cij   *= 0.9
            
            # Print time info
            if self.verbose:
                end = time.time()
                knn_err = np.sqrt(knn_err / trainset.n_ratings)
                mf_err = np.sqrt(mf_err / trainset.n_ratings)
                global_err = np.sqrt(global_err / trainset.n_ratings)
                print('KNN err: ', knn_err)
                print('MF err: ', mf_err)
                print('Global err: ', global_err)
                print('Time of iteration: ', end - start)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj
        self.wij = wij
        self.cij = cij

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # MF estimation
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / self.sqrt_Iu[u])
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)
            
            # KNN estimation
            rated_j = [(j, rat) for (j, rat) in self.trainset.ur[u] if j in self.near_items[i]]
            knn = 0
            if len(rated_j):
                for (j, rat) in rated_j:
                    buj = self.trainset.global_mean + self.bu_base[u] + self.bi_base[i]
                    # explicit and explicit KNN estimation
                    knn += (rat - buj) * self.wij[i,j] + self.cij[i,j]
                knn /= np.sqrt(len(rated_j))
            est += knn
            
        return est
        

class WeightedNeighborhood(AlgoBase):
    """ The *WeightedNeighborhood* algorithm is a KNN based algorithm in which
        the weights are not computed with a similarity measure but are directly
        learnt by the algorithm with a SGD.
    """

    # TODO: check if the algorithm doesn't work because of some error in the code
    # or becuase it is very difficult to find the right parameters.

    def __init__(self, n_epochs=20, k=20, bsl_options={}, sim_options={},
                 init_mean=0, init_std_dev=.1, init_knn='zeros', clip=False, lr_all=.007, reg_all=.02, lr_bsl=0.007,
                 lr_knn=0.001, lr_cij=0.001, reg_bsl=0.005, reg_knn=0.015, reg_cij=0.015, shuffle_sgd=True, verbose=False):
        
        self.n_epochs = n_epochs
        self.k = k
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.init_knn = init_knn
        self.clip = clip
        self.lr_bsl = lr_bsl if lr_bsl is not None else lr_all
        self.lr_knn = lr_knn if lr_knn is not None else lr_all
        self.lr_cij = lr_cij if lr_cij is not None else lr_all
        self.reg_bsl = reg_bsl if reg_bsl is not None else reg_all
        self.reg_knn = reg_knn if reg_knn is not None else reg_all
        self.reg_cij = reg_cij if reg_cij is not None else reg_all
        self.shuffle_sgd = shuffle_sgd
        self.verbose = verbose

        AlgoBase.__init__(self, bsl_options=bsl_options, sim_options=sim_options)

    def train(self, trainset):
    
        if self.verbose:
            print('Starting train with parameters:')
            print('k: {}, init_mean: {}, init_std: {}'.format(self.k, self.init_mean, self.init_std_dev))
            print('init_knn: {}, clip: {}'.format(self.init_knn, self.clip))
            print('Learning rates: {}, {}, {}'.format(self.lr_bsl, self.lr_knn, self.lr_cij))
            print('Regulizers: {}, {}, {}'.format(self.reg_bsl, self.reg_knn, self.reg_cij))
            print('Shuffled: ', self.shuffle_sgd)

        AlgoBase.train(self, trainset)
        
        self.near_items = np.empty(trainset.n_items, dtype=list)
        
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        self.bu_base = self.bu
        self.bi_base = self.bi
        self.sgd(trainset)

    def sgd(self, trainset):
    
        # KNN ratings
        cdef np.ndarray[np.double_t, ndim=2] ruj
        # KNN list of nearest items
        cdef np.ndarray[list] near_items
        cdef list rated_j
        # KNN weights
        cdef np.ndarray[np.double_t, ndim=2] wij
        # Implicit ratings in KNN
        cdef np.ndarray[np.double_t, ndim=2] cij
        # Baseline biases
        cdef np.ndarray[np.double_t] bu_base = self.bu_base
        cdef np.ndarray[np.double_t] bj_base = self.bi_base
        cdef double b_glob_u, buj
        
        # KNN estimate
        cdef double knn, eff_neigh
        cdef np.ndarray[np.double_t] ratbuj
        
        # Development utils
        cdef double global_err
        
        cdef list all_ratings
        
        # Init KNN weights
        if self.init_knn=='sim':
            wij = self.sim
            cij = np.zeros((trainset.n_items, trainset.n_items))
        elif self.init_knn=='zeros':
            wij = np.zeros((trainset.n_items, trainset.n_items))
            cij = np.zeros((trainset.n_items, trainset.n_items))
        else:
            wij = np.random.normal(self.init_mean, self.init_std_dev,
                               (trainset.n_items, trainset.n_items))
            cij = np.random.normal(self.init_mean, self.init_std_dev,
                               (trainset.n_items, trainset.n_items))
                               
        ruj = np.zeros((trainset.n_users, self.k), np.double)
        ratbuj = np.zeros(trainset.n_items, np.double)
        
        near_items = np.empty(trainset.n_items, dtype=list)
        all_ratings = []
        rated_j = []
        
        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi

        cdef int u, i, j, f
        cdef double r, err, dot, puf, qif, _
        cdef double global_mean = self.trainset.global_mean
        cdef np.ndarray[np.double_t] u_impl_fdb

        # params
        cdef double lr_bsl = self.lr_bsl
        cdef double reg_bsl = self.reg_bsl
        cdef double lr_knn = self.lr_knn
        cdef double lr_cij = self.lr_cij
        cdef double reg_knn = self.reg_knn
        cdef double reg_cij = self.reg_cij

        bu = self.bu
        bi = self.bi

        # Find nearest neighbors for each items 
        if self.verbose:
            print(" Building items neighborhood...")
            mean = 0
        for i in trainset.all_items():
            # find best k neghbours of items
            near_items[i] = self.get_neighbors(i, self.k)
            self.near_items[i] = near_items[i]
            if self.verbose:
                mean += len(near_items[i])
        if self.verbose:
            print('Average number of valid neghbours: ', mean/trainset.n_items)
        
        # Iteration of the SGD 
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
                start = time.time()
                global_err = 0
            # Iterate over all ratings
            all_ratings = list(trainset.all_ratings())
            if self.shuffle_sgd:
                np.random.shuffle(all_ratings)
            for u, i, r in tqdm(all_ratings, mininterval=10):

                # Reset estimation
                knn = 0
                # KNN estimation
                b_glob_u = global_mean + bu_base[u]
                # Find items rated that are in the neghborhood
                rated_j = [(j, rat) for (j, rat) in trainset.ur[u] if j in near_items[i]]
                # Add only items which are neighbors
                if rated_j:
                    for (j, rat) in rated_j:
                        buj = b_glob_u + bj_base[j]
                        # explicit and explicit KNN estimation
                        ratbuj[j] = (rat - buj)
                        knn += ratbuj[j] * wij[i,j] + cij[i,j]
                    eff_neigh = np.sqrt(len(rated_j))
                    knn /= eff_neigh
                    
                # estimate rating
                err = global_mean + bu[u] + bi[i] + knn
                if self.verbose:
                    global_err += np.square(r - np.clip(err, 1, 5))
                
                # Clip estimation
                if self.clip:
                    if err>5:
                        err = 5
                    elif err<1:
                        err = 1
                
                # compute error
                err = r - err

                # update biases
                bu[u] += lr_bsl * (err - reg_bsl * bu[u])
                bi[i] += lr_bsl * (err - reg_bsl * bi[i])
                    
                # Knn factors
                if rated_j:
                    err /= eff_neigh
                    for (j, _) in rated_j:
                        wij[i, j] += lr_knn * (err * ratbuj[j] -
                                               reg_knn * wij[i, j])
                        cij[i, j] += lr_cij * (err - reg_cij * cij[i,j])
                                             
            # Scale learning rate at each iteration
            lr_bsl *= 0.9
            lr_knn *= 0.9
            lr_cij *= 0.9
            
            # Print time info
            if self.verbose:
                end = time.time()
                global_err = np.sqrt(global_err / trainset.n_ratings)
                print('Global err: ', global_err)
                print('Time of iteration: ', end - start)

        self.bu = bu
        self.bi = bi
        self.wij = wij
        self.cij = cij

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # KNN estimation
            rated_j = [(j, rat) for (j, rat) in self.trainset.ur[u] if j in self.near_items[i]]
            knn = 0
            if len(rated_j):
                for (j, rat) in rated_j:
                    buj = self.trainset.global_mean + self.bu_base[u] + self.bi_base[i]
                    # explicit and explicit KNN estimation
                    knn += (rat - buj) * self.wij[i,j] + self.cij[i,j]
                knn /= np.sqrt(len(rated_j))
            est += knn
            
        return est