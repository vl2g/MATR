import numpy as np
import torch
import math
from torch import log, exp
import torch.nn.functional as F

from model.model_utils import unique_softmax, cosine_sim
from dp.dp_utils import VarTable, minGamma, minProb


device = "cuda" if torch.cuda.is_available() else "cpu"


def softDTW(step_features, frame_features, labels, dist_type='inner', softning='prob',
            gamma_min=0.1, gamma_xz=0.1, step_normalize=True):
    """ function to obtain a soft (differentiable) version of DTW
            step_features: torch.tensor[K, d], step language embeddings, i.e. sequence1,
            frame_featurues: torch.tensor[N, d], embedding of video frames, i.e. seuqence2
            labels: array[K], labels of steps
    """
    # defining the function
    _min_fn = minProb if softning == 'prob' else minGamma
    min_fn = lambda x: _min_fn(x, gamma=gamma_min)

    # first get a pairwise distance matrix
    if dist_type == 'inner':
        dist = step_features @ frame_features.T
    else:
        dist = cosine_sim(step_features, frame_features)
    if step_normalize:
        if labels is not None:
            norm_dist = unique_softmax(dist, labels, gamma_xz)
        else:
            norm_dist = torch.softmax(dist / gamma_xz, 0)
        dist = -log(norm_dist)
    
    # initialize soft-DTW table
    nrows, ncols = dist.shape
    # sdtw = torch.zeros((nrows+1,ncols+1)).to(torch.float).to(device)
    sdtw = VarTable((nrows + 1, ncols + 1))
    for i in range(1, nrows + 1):
        sdtw[i, 0] = 9999999999
    for j in range(1, ncols + 1):
        sdtw[0, j] = 9999999999

    # obtain dtw table using min_gamma or softMin relaxation
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            neighbors = torch.stack([sdtw[i, j - 1], sdtw[i - 1, j - 1], sdtw[i - 1, j]])
            di, dj = i - 1, j - 1    # in the distance matrix indices are shifted by one
            new_val = dist[di, dj] + min_fn(neighbors) 
            sdtw[i, j] = torch.squeeze(new_val, 0)
    sdtw_loss = sdtw[nrows, ncols] / step_features.shape[0]
    return sdtw_loss, sdtw, dist


def dropDTW(step_features, frame_features, labels, softning='prob',
            gamma_min=1, gamma_xz=1, step_normalize=True, eps=1e-5):
    """ function to obtain a soft (differentiable) version of DTW
            step_features: torch.tensor[K, d], step language embeddings, i.e. sequence1,
            frame_featurues: torch.tensor[N, d], embedding of video frames, i.e. seuqence2
            labels: array[K], labels of steps
    """
    # defining the function
    _min_fn = minProb if softning == 'prob' else minGamma
    min_fn = lambda x: _min_fn(x, gamma=gamma_min)

    # first get a pairwise distance matrix and drop costs
    dist = step_features @ frame_features.T
    inf = torch.tensor([9999999999], device=dist.device, dtype=dist.dtype)
    if step_normalize:
        norm_dist = unique_softmax(dist, labels, gamma_xz)
        drop_costs = 1 - norm_dist.max(dim=0).values  # assuming dist is in [0, 1]
        zx_costs = -log(norm_dist)
        drop_costs = -log(drop_costs + eps)
    else:
        zx_costs = 1 - dist
        drop_costs = 1 - zx_costs.max(dim=0).values  # assuming dist is in [0, 1]

    cum_drop_costs = torch.cumsum(drop_costs, dim=0)

    # initialize soft-DTW table
    K, N = dist.shape

    D = VarTable((K + 1, N + 1, 3))    # 3-dim DP table, instead of 3 1-dim tables above
    for i in range(1, K + 1):
        D[i, 0] = torch.zeros_like(D[i, 0]) + inf
    for j in range(1, N + 1):
        filling = torch.zeros_like(D[0, j]) + inf
        filling[[0, 1]] = cum_drop_costs[j-1]
        D[0, j] = filling

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1    # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            min_prev_cost = torch.zeros_like(d_diag) + min_fn([d_diag, d_left])

            # positive transition, i.e. matching x_i to z_j
            Dp = min_prev_cost + zx_costs[z_cost_ind, x_cost_ind]
            # negative transition, i.e. dropping xi
            Dm = d_left + drop_costs[x_cost_ind]

            # update final solution matrix
            D_final = torch.zeros_like(Dm) + min_fn([Dm, Dp])
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)
    min_cost = D[K, N][0]

    return min_cost, D, dist


def batch_dropDTW(zx_costs_list, drop_costs_list, softning='prob',
                  exclusive=True, contiguous=True, drop_mode='DropDTW', gamma_min=1):
    """ A batch version of soft Drop-DTW. Operates on a list of pairwise similarities and drop costs.
            zx_costs_list: a list of pairwise similarity matrices [tensor[N_1, K_1], ..., tensor[N_B, K_B]]
            drop_costs_list: a list contatining drop costs for rows in each DP table [tensor[N_1], ..., tensor[N_B]]
    """
    # defining the min function
    min_fn = minProb if softning == 'prob' else minGamma
    inf = 9999999999

    # pre-processing
    B = len(zx_costs_list)
    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if drop_mode == 'OTAM':
            # add zero row in order to skip in the end
            zero_row = torch.zeros_like(zx_costs_list[i][-1]) 
            zx_costs_list[i] = torch.cat([zx_costs_list[i], zero_row[None, :]], dim=0)
            Ki += 1

        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # preparing padded tables
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)

        # padding everything to the size of the largest N and K
        row_pad = torch.zeros([N - Ns[i]]).to(zx_costs.device)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        multirow_pad = torch.stack([row_pad + inf] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + inf
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)
    
    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 3, B))    # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + inf
    for xi in range(1, N + 1):
        if drop_mode == 'DropDTW':
            D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1):xi]
        elif drop_mode == 'OTAM':
            D[0, xi] = torch.zeros_like(D[0, xi])
        else:  # drop_mode == 'DTW'
            D[0, xi] = torch.zeros_like(D[0, xi]) + inf

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            z_cost_ind, x_cost_ind = zi - 1, xi - 1    # indexind in costs is shifted by 1

            d_diag, d_left = D[zi - 1, xi - 1][0:1], D[zi, xi - 1][0:1]
            dp_left, dp_up = D[zi, xi - 1][2:3], D[zi - 1, xi][2:3]
            
            if drop_mode == 'DropDTW':
                # positive transition, i.e. matching x_i to z_j
                if contiguous:
                    pos_neighbors = [d_diag, dp_left]
                else:
                    pos_neighbors = [d_diag, d_left]
                if not exclusive:
                    pos_neighbors.append(dp_up)

                Dp = min_fn(pos_neighbors, gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]

                # negative transition, i.e. dropping xi
                Dm = d_left + all_drop_costs[x_cost_ind]

                # update final solution matrix
                D_final = min_fn([Dm, Dp], gamma=gamma_min)
            else:
                d_right = D[zi - 1, xi][0:1]
                D_final = Dm = Dp = min_fn([d_diag, d_left, d_right], gamma=gamma_min) + all_zx_costs[z_cost_ind, x_cost_ind]
            D[zi, xi] = torch.cat([D_final, Dm, Dp], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][0, i]
        min_costs.append(min_cost_i / Ni)

    return min_costs, D


def batch_NW(zx_costs_list, drop_costs_list, softning='prob', gamma_min=1):
    """ function to obtain a soft (differentiable version of DTW) 
            embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
            and D: dimensionality of of the embedding vector)
    """
    # defining the function
    min_fn = minProb if softning == 'prob' else minGamma

    # pre-processing
    B = len(zx_costs_list)
    Ns, Ks = [], []
    for i in range(B):
        Ki, Ni = zx_costs_list[i].shape
        if Ki >= Ni:
            # in case the number of steps is greater than the number of frames,
            # duplicate every frame and let the drops do the job.
            mult = math.ceil(Ki / Ni)
            zx_costs_list[i] = torch.stack([zx_costs_list[i]] * mult, dim=-1).reshape([Ki, -1])
            drop_costs_list[i] = torch.stack([drop_costs_list[i]] * mult, dim=-1).reshape([-1])
            Ni *= mult
        Ns.append(Ni)
        Ks.append(Ki)
    N, K = max(Ns), max(Ks)

    # preparing padded tables
    padded_cum_drop_costs, padded_drop_costs, padded_zx_costs = [], [], []
    for i in range(B):
        zx_costs = zx_costs_list[i]
        drop_costs = drop_costs_list[i]
        cum_drop_costs = torch.cumsum(drop_costs, dim=0)

        # padding everything to the size of the largest N and K
        row_pad = torch.zeros([N - Ns[i]]).to(zx_costs.device)
        padded_cum_drop_costs.append(torch.cat([cum_drop_costs, row_pad]))
        padded_drop_costs.append(torch.cat([drop_costs, row_pad]))
        multirow_pad = torch.stack([row_pad + 9999999999] * Ks[i], dim=0)
        padded_table = torch.cat([zx_costs, multirow_pad], dim=1)
        rest_pad = torch.zeros([K - Ks[i], N]).to(zx_costs.device) + 9999999999
        padded_table = torch.cat([padded_table, rest_pad], dim=0)
        padded_zx_costs.append(padded_table)

    all_zx_costs = torch.stack(padded_zx_costs, dim=-1)
    all_cum_drop_costs = torch.stack(padded_cum_drop_costs, dim=-1)
    all_drop_costs = torch.stack(padded_drop_costs, dim=-1)
    
    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, B))    # This corresponds to B 3-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + 9999999999
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[xi - 1]

    # obtain dtw table using min_gamma or softMin relaxation
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            match_cost = all_zx_costs[zi - 1, xi - 1]
            drop_cost = all_drop_costs[xi - 1]
            transition_costs = [D[zi - 1, xi - 1] + drop_cost, D[zi - 1, xi] + match_cost, D[zi, xi - 1] + match_cost]
            D[zi, xi] = min_fn(transition_costs, gamma=gamma_min, keepdim=False)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        Ni, Ki = Ns[i], Ks[i]
        min_cost_i = D[Ki, Ni][i]
        min_costs.append(min_cost_i / Ni)

    return min_costs, D


def batch_double_dropDTW(zx_costs_list, drop_costs_list,
                         exclusive=True, contiguous=True, gamma_min=1):
    """ function to obtain a soft (differentiable version of DTW) 
            embs1, embs2: embedding of size N*D and M*D (N and M : number of video frames
            and D: dimensionality of of the embedding vector)
    """
    min_fn = lambda x: minProb(x, gamma=gamma_min)

    # assuming sequences are the same length
    B = len(zx_costs_list)
    N, K = zx_costs_list[0].shape
    cum_drop_costs_list = [torch.cumsum(drop_costs_list[i], dim=0) for i in range(B)]

    all_zx_costs = torch.stack(zx_costs_list, dim=-1)
    all_cum_drop_costs = torch.stack(cum_drop_costs_list, dim=-1)
    all_drop_costs = torch.stack(drop_costs_list, dim=-1)

    # Creating and initializing DP tables
    D = VarTable((K + 1, N + 1, 4, B))    # This corresponds to B 4-dim DP tables
    for zi in range(1, K + 1):
        D[zi, 0] = torch.zeros_like(D[zi, 0]) + all_cum_drop_costs[(zi - 1):zi]
    for xi in range(1, N + 1):
        D[0, xi] = torch.zeros_like(D[0, xi]) + all_cum_drop_costs[(xi - 1):xi]

    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_costs = [D[zi - 1, xi - 1][s] for s in diag_neigh_states]

            left_neigh_states = [0, 1]  # zx and z-
            left_neigh_costs = [D[zi, xi - 1][s] for s in left_neigh_states]

            upper_neigh_states = [0, 2]  # zx and -x
            upper_neigh_costs = [D[zi - 1, xi][s] for s in upper_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1    # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_costs_zx = diag_neigh_costs + upper_neigh_costs + left_neigh_costs
            D0 = min_fn(neigh_costs_zx) + all_zx_costs[z_cost_ind, x_cost_ind]

            # DP 1: coming to z-
            neigh_costs_z_ = left_neigh_costs
            D1 = min_fn(neigh_costs_z_) + all_drop_costs[x_cost_ind]

            # DP 2: coming to -x
            neigh_costs__x = upper_neigh_costs
            D2 = min_fn(neigh_costs__x) + all_drop_costs[z_cost_ind]

            # DP 3: coming to --
            costs___ = ([d + all_drop_costs[z_cost_ind] * 2 for d in diag_neigh_costs] +
                        [D[zi, xi - 1][3] + all_drop_costs[x_cost_ind], D[zi - 1, xi][3] + all_drop_costs[z_cost_ind]])
            D3 = min_fn(costs___)

            D[zi, xi] = torch.cat([D0, D1, D2, D3], dim=0)

    # Computing the final min cost for the whole batch
    min_costs = []
    for i in range(B):
        min_cost_i = min_fn(D[K, N][:, i])
        min_costs.append(min_cost_i / N)
    return min_costs, D
