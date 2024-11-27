import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dp.exact_dp import drop_dtw
from models.losses import compute_all_costs

color_code = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'lime']
shape_code = ["o", "s", "P", "*", "h", ">", 'X', 'd', 'D', 'v', '<']
color_code, shape_code = color_code * 3, shape_code * 3  # to protect against long step sequences
color_code = color_code + ['black']
shape_code = shape_code + ['p']


def visualize_drop_dtw_matching(samples, distractor=None, gamma_f=10, drop_cost='logit', keep_percentile=0.3, shape=(10, 2)):
    gamma_f = [gamma_f] * len(samples) if not isinstance(gamma_f, (list, tuple)) else gamma_f
    plt.rcParams["figure.figsize"] = (shape[0], shape[1] * len(samples))
    for i, (sample_name, sample) in enumerate(samples.items()):
        ax = plt.subplot(len(samples), 1, i + 1)
        ax.set_title(f"{sample_name}: drop-dtw matching, gamma {gamma_f[i]}")
        frame_features = sample['frame_features']
        step_features = sample['step_features']

        zx_costs, drop_costs, _ = compute_all_costs(
            sample, distractor, gamma_f[i], drop_cost_type=drop_cost, keep_percentile=keep_percentile)
        zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]

        min_cost, path, frames_dropped = drop_dtw(zx_costs, drop_costs)
        frame_labels = np.zeros_like(drop_costs) - 1
        for label, frame_id in path:
            if frame_id * label > 0 and frame_id not in frames_dropped:
                frame_labels[frame_id - 1] = sample['step_ids'][label - 1].item()
            
        gt_labels = np.zeros_like(frame_labels) - 1
        for i in range(gt_labels.shape[0]):
            for sample_id, start, end in zip(sample['step_ids'], sample['step_starts'], sample['step_ends']):
                if (i >= start.item()) and (i <= end.item()):
                    gt_labels[i] = sample_id.item()
        unique_labels = np.unique(sample['step_ids'].numpy())
        step_colors = dict(zip(unique_labels, color_code))
        step_shapes = dict(zip(unique_labels, shape_code))

        tick_freq = 20 if len(frame_labels) > 100 else 10
        plt.xticks(np.arange(0, len(frame_labels) * 3.2, tick_freq))
        plt.xlim(0, len(frame_labels) * 3.2)
        plt.tick_params(bottom=True, top=False, left=True, right=True, labelright=True)
        plt.grid()

        added_step_ids = []
        for si, step_id in enumerate(unique_labels):
            gt_x = np.arange(len(gt_labels))[gt_labels == step_id]
            pred_x = np.arange(len(frame_labels))[frame_labels == step_id]
            step_color, step_shape = step_colors[step_id], step_shapes[step_id]
            plt.plot(gt_x * 3.2, [2] * len(gt_x), step_shape, color=step_color)
            plt.plot(pred_x * 3.2, [1] * len(pred_x), step_shape, color=step_color)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB'))
    return img


def visualize_step_strength(samples, distractor=None, gamma_f=10, drop_cost='logit', keep_percentile=0.3, shape=(10, 2)):
    gamma_f = [gamma_f] * len(samples) if not isinstance(gamma_f, (list, tuple)) else gamma_f
    step_ids = list(samples.values())[0]['step_ids']
    unique_step_mask = torch.zeros_like(step_ids).to(torch.bool)
    unique_step_ids = []
    for si, step_id in enumerate(step_ids):
        if step_id.item() not in unique_step_ids:
            unique_step_ids.append(step_id.item())
            unique_step_mask[si] = True
    step_colors = dict(zip(np.sort(unique_step_ids), color_code))

    plt.rcParams["figure.figsize"] = (shape[0], shape[1] * len(samples))
    for i, (sample_name, sample) in enumerate(samples.items()):
        ax = plt.subplot(len(samples), 1, i + 1)
        ax.set_title(f"{sample_name}: frame-step product, gamma {gamma_f[i]}")
        step_ids = sample['step_ids']
        frame_features = sample['frame_features']
        step_features = sample['step_features'][unique_step_mask]
        descr_clip_similarity = (step_features @ frame_features.T / gamma_f[i]).detach().cpu().numpy()
        N_frames = frame_features.shape[0]

        tick_freq = 20 if N_frames > 100 else 10
        plt.xticks(np.arange(0, N_frames * 3.2, tick_freq))
        plt.tick_params(bottom=True, top=False, left=True, right=True, labelright=True)
        plt.grid()

        added_step_ids = []
        for si, step_id in enumerate(step_ids):
            step_color = step_colors[step_id.item()]
            plt.plot([sample['step_starts_sec'][si], sample['step_ends_sec'][si]],
                     [descr_clip_similarity.max() + 0.1] * 2, color=step_color)
            
            if step_id not in added_step_ids:
                added_step_ids.append(step_id)
                step_id_scores = descr_clip_similarity[np.array(unique_step_ids) == step_id.item()][0]
                plt.plot(np.arange(N_frames) * 3.2, step_id_scores, color=step_color)

        if distractor is not None and i == 0:
            distractor_activations = (frame_features @ distractor / gamma_f[i]).detach().cpu().numpy()
            plt.plot(np.arange(N_frames) * 3.2, distractor_activations, color=color_code[-1])
        if drop_cost == 'logit' and i == 0:
            sim_vec = descr_clip_similarity.reshape([-1])
            k = max([1, int(sim_vec.shape[0] * keep_percentile)])
            baseline_logit = np.sort(sim_vec)[-k]

            drop_threshold = np.ones(N_frames) * baseline_logit
            plt.plot(np.arange(N_frames) * 3.2, drop_threshold, color=color_code[-1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB'))
    return img
