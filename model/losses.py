import torch
import torch.nn.functional as F
from torch import log, exp
import numpy as np

from dp.soft_dp import batch_dropDTW, batch_NW
from dp.dp_utils import compute_all_costs
from model.model_utils import unique_softmax


def mil_nce(features_1, features_2, correspondance_mat, eps=1e-8, gamma=1, hard_ratio=1):
    corresp = correspondance_mat.to(torch.float32)
    prod = features_1 @ features_2.T / gamma
    # logsumexp trick happens here
    prod_exp = exp(prod - prod.max(dim=1, keepdim=True).values)
    nominator = (prod_exp * corresp).sum(dim=1)
    denominator = prod_exp.sum(dim=1)
    nll = -log(nominator / (denominator + eps))
    if hard_ratio < 1:
        n_hard_examples = int(nll.shape[0] * hard_ratio)
        hard_indices = nll.sort().indices[-n_hard_examples:]
        nll = nll[hard_indices]
    return nll.mean()


def compute_clust_loss(samples, distractors, l2_normalize=False, frame_gamma=10,
                      xz_gamma=10, xz_hard_ratio=0.3, all_classes_distinct=False,
                      bg_scope='global'):
    # aggregating videos with attentino for their steps, i.e. done per each step
    all_pooled_frames, pooled_frames_labels = [], []
    all_step_features, all_step_labels = [], []
    global_step_id_count = 0
    for i, sample in enumerate(samples):
        step_features, frame_features = sample['step_features'], sample['frame_features']

        # Used for YouCook2, where text descriptions are unique for each step
        if all_classes_distinct:
            n_samples = sample['step_ids'].shape[0]
            step_ids = torch.arange(global_step_id_count,
                                    global_step_id_count + n_samples)
            global_step_id_count += n_samples
        else:
            step_ids = sample['step_ids']

        if distractors is not None:
            bg_step_id = torch.tensor([99999]).to(step_ids.dtype).to(step_ids.device)
            if bg_scope == 'class':
                bg_step_id = bg_step_id + sample['cls']
            if bg_scope == 'video':
                global_step_id_count += 1
                bg_step_id = bg_step_id + global_step_id_count
                
            step_ids = torch.cat([step_ids, bg_step_id])
            step_features = torch.cat([step_features, distractors[i][None, :]])

        if l2_normalize:
            step_features = F.normalize(step_features, p=2, dim=1)
            frame_features = F.normalize(frame_features, p=2, dim=1)

        unique_step_labels, unique_idxs = [
            torch.from_numpy(t) for t in np.unique(step_ids.detach().cpu().numpy(), return_index=True)]
        unique_step_features = step_features[unique_idxs]  # size [K, d]
        sim = unique_step_features @ frame_features.T
        frame_weights = F.softmax(sim / frame_gamma, dim=1)  # size [K, N]
        step_pooled_frames = frame_weights @ frame_features  # size [K, d]
        all_pooled_frames.append(step_pooled_frames)
        pooled_frames_labels.append(unique_step_labels)
        all_step_features.append(step_features)
        all_step_labels.append(step_ids)
    all_pooled_frames = torch.cat(all_pooled_frames, dim=0)
    pooled_frames_labels = torch.cat(pooled_frames_labels, dim=0)
    all_step_features = torch.cat(all_step_features, dim=0)
    all_step_labels = torch.cat(all_step_labels, dim=0)
    assert pooled_frames_labels.shape[0] == all_pooled_frames.shape[0], "Shape mismatch occured"

    unique_labels, unique_idxs = [
        torch.from_numpy(t) for t in np.unique(all_step_labels.detach().cpu().numpy(), return_index=True)]
    unique_step_features = all_step_features[unique_idxs]
    N_steps = all_pooled_frames.shape[0]

    # creating the matrix of targets for the MIL-NCE contrastive objective
    xz_label_mat = torch.zeros([N_steps, unique_labels.shape[0]]).to(all_pooled_frames.device)
    for i in range(all_pooled_frames.shape[0]):
        for j in range(unique_labels.shape[0]):
            xz_label_mat[i, j] = pooled_frames_labels[i] == unique_labels[j]

    # reinforcing existing alignment with step descriptors
    xz_loss = mil_nce(all_pooled_frames, unique_step_features, xz_label_mat,
                      gamma=xz_gamma, hard_ratio=xz_hard_ratio)
    return xz_loss


def compute_alignment_loss(samples, distractors, l2_normalize=False, drop_cost_type='max',
                           dp_algo='DropDTW', keep_percentile=1, contiguous=True, softning='prob',
                           gamma_xz=10, gamma_min=1, aggregate_loss=True):
    gamma_xz = 0.1 if l2_normalize else gamma_xz
    gamma_min = 0.1 if l2_normalize else gamma_min

    # do pre-processing
    zx_costs_list = []
    drop_costs_list = []
    for i, sample in enumerate(samples):
        distractor = None if distractors is None else distractors[i]
        zx_costs, drop_costs, _ = compute_all_costs(sample, distractor, gamma_xz, drop_cost_type,
                                                    keep_percentile, l2_normalize)
        zx_costs_list.append(zx_costs)
        drop_costs_list.append(drop_costs)

    if dp_algo == 'NW':
        min_costs, _ = batch_NW(zx_costs_list, drop_costs_list, gamma_min=gamma_min, softning=softning)
    else:
        min_costs, A = batch_dropDTW(zx_costs_list, drop_costs_list, gamma_min=gamma_min,
                                     drop_mode=dp_algo, contiguous=contiguous, softning=softning)
    dtw_losses = [c / len(samples) for c in min_costs]
    if aggregate_loss:
        return sum(dtw_losses), A
    else:
        return dtw_losses, A
