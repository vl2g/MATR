import os
import pdb
import h5py
import nncore
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from main.config_hl import TVSUM_SPLITS, YOUTUBE_SPLITS
from utils.basic_utils import load_jsonl, load_pickle, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from utils.span_utils import span_xx_to_cxw
from random import shuffle
import io

logger = logging.getLogger(__name__)



class DatasetMR(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """
    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, v_feat_dim, q_feat_dim,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 use_cache=-1, fix_len=-1, add_easy_negative=1, easy_negative_only=-1):
        self.dset_name = dset_name
        self.data_path = data_path[0] if isinstance(data_path, list) else data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.v_feat_dim = v_feat_dim
        self.q_feat_dim = q_feat_dim
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.fix_len = fix_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_cache = use_cache
        self.add_easy_negative = add_easy_negative
        self.easy_negative_only = easy_negative_only
        
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        self.v_feat_types = [feat_dir.split('/')[-1] for feat_dir in self.v_feat_dirs]
        t_feat_type = q_feat_dir.split('/')[-1]

        if self.use_cache > 0:
            print('Loading the off-line features...')
            dset_dir = os.path.join('data', self.dset_name)
            vid_keys = [meta['vid'] for meta in self.data]
            qid_keys = [meta['qid'] for meta in self.data]

            self.vid_cache = {}
            for v_feat_type in self.v_feat_types:
                assert 'vid' in v_feat_type
                with h5py.File(os.path.join(dset_dir, 'h5py', v_feat_type + '.hdf5'), 'r') as f:
                    self.vid_cache[v_feat_type] = {key: f[str(key)][:] for key in tqdm(vid_keys)}

            assert 'txt' in t_feat_type
            self.txt_cache = {}
            with h5py.File(os.path.join(dset_dir, 'h5py', t_feat_type + '.hdf5'), 'r') as f:
                for key in tqdm(qid_keys):
                    try:
                        self.txt_cache[key] = f[str(key)][:]
                    except:
                        logger.info(f"text {key} is not in the cache.")

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.dset_name in ['actnet','sportsmr']:
            for i, window_i in enumerate(meta["relevant_windows"]):
                if window_i[1] - window_i[0] < self.clip_len:
                    center = (window_i[1] + window_i[0]) / 2 
                    window_i[0] = max(0, center - 0.5 * self.clip_len)
                    window_i[1] = min(float(meta['duration']), center + 0.5 * self.clip_len)
                    window_i[1] = max(self.clip_len, window_i[1])

        model_inputs["timestamp"] = ( (torch.arange(0, ctx_l) + self.clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

        if 'test' in self.data_path and 'qvhighlights' in self.dset_name:
            meta["relevant_windows"] = [[0, 150]]
        relevant_windows = torch.Tensor(meta["relevant_windows"])

        # assign the nearest window for each timestamp i.e., qvhighlights.
        num_vid_seq = model_inputs["timestamp"].shape[0]
        num_windows = relevant_windows.shape[0]

        relevant_windows_ts = relevant_windows / (ctx_l * self.clip_len)
        relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(num_vid_seq, 1, 1)
        model_inputs_ts = model_inputs["timestamp"].unsqueeze(1).repeat(1, num_windows, 1)

        if meta['qid'] is not None:
            nn_window_ts = torch.zeros_like(model_inputs["timestamp"])
            diff_left = model_inputs_ts[..., 0]  - relevant_windows_ts[..., 0]
            diff_right = relevant_windows_ts[..., 1] - model_inputs_ts[..., 1]
            assign_idx = torch.where((diff_left >= 0) * (diff_right >= 0))
            if min(assign_idx[0].shape) == 0:   # not assigned, happened in activitynet.
                nn_window_ts = relevant_windows_ts.squeeze(1)
            else:
                nn_window_ts[assign_idx[0]] = relevant_windows_ts[assign_idx[0], assign_idx[1]]

        model_inputs["span_labels_nn"] = nn_window_ts
        model_inputs["timestamp_window"] = 1 * (model_inputs["timestamp"][:,0] >= nn_window_ts[:,0])  & (model_inputs["timestamp"][:,1] <= nn_window_ts[:,1])

        # for activitynet.
        if model_inputs["timestamp_window"].sum() < 1:
            idx = int(meta['relevant_windows'][0][0] / self.clip_len)
            idx = max(0, min(idx, ctx_l-1))
            model_inputs["timestamp_window"][idx] = 1

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
            if 'saliency_scores' in meta.keys():
                model_inputs["saliency_scores"] = torch.zeros(ctx_l).double()
                limit = meta["relevant_clip_ids"].index(ctx_l) if (np.array(meta["relevant_clip_ids"]) >= ctx_l).any() else None
                model_inputs["saliency_scores"][meta["relevant_clip_ids"][:limit]] = torch.tensor(np.mean(np.array(meta["saliency_scores"][:limit]), -1))
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
            else:
                model_inputs["saliency_scores"] = model_inputs["timestamp_window"]
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"] = \
                    self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
                model_inputs["saliency_pos_labels"] = [ random.choice(torch.where(model_inputs['saliency_scores'])[0].tolist()) ]

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=1):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_st = min(gt_st, ctx_l-1)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_ed = gt_st

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st] * max_n #[gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))

        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # Check if scores and rel_clip_ids are not empty
        if not scores or not rel_clip_ids:
            return [], []

        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        
        # Ensure scores is at least 2D
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=1)
        
        agg_scores = np.sum(scores, axis=1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # Check if sort_indices is not empty
        if len(sort_indices) == 0:
            return [], []

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]

        if agg_scores[sort_indices[-1]] == agg_scores[sort_indices[0]]:
            hard_neg_clip_indices = hard_pos_clip_indices

        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        # pdb.set_trace()
        if self.add_easy_negative > 0:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        if self.easy_negative_only > 0:
            return easy_pos_clip_indices, easy_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices


    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.use_cache > 0:
            try:
                q_feat = self.txt_cache[qid]
            except:
                q_feat = np.zeros((10, self.q_feat_dim)).astype(np.float32)
            return  torch.from_numpy(q_feat)               
                q_feat_path = join("data/query_featurs", f"qid{qid}.npz")
                q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)

            except:
                q_feat = np.zeros((10, 512)).astype(np.float32)

                logger.info(f"Something wrong when loading the query feature {q_feat_path}.")
         else:
            q_feat = np.zeros((10, 512)).astype(np.float32)

        if self.q_feat_type == "last_hidden_state":
            # q_feat = q_feat[:self.max_q_l]
            q_feat = q_feat
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.txt_drop_ratio > 0:
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        v_feat_list = []
        for feat_type, _feat_dir in zip(self.v_feat_types, self.v_feat_dirs):
            if self.use_cache > 0:
                _feat = self.vid_cache[feat_type][vid]
            else:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"].astype(np.float32)
                # _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
            v_feat_list.append(_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)

