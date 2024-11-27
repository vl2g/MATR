import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from run_on_video import clip, vid2clip, txt2clip

def parse_args():
    parser = argparse.ArgumentParser(description='Process a video and text query.')
    parser.add_argument('--csv_file', type=str, default='/data/test_data.csv')
    parser.add_argument('--video_folder', type=str, default='/data/vid_clip')
    parser.add_argument('--query_folder', type=str, default='/data/test_feats')
    parser.add_argument('--save_dir', type=str, default='./tmp', help='Directory to save intermediate files.')
    parser.add_argument('--resume', type=str, default='/data/ckpt/model_best.ckpt', help='Path to model checkpoint.')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU ID to use.')
    parser.add_argument("--output", type=str, default='results.csv', help='Path to save the output CSV with predictions.')
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Constants
model_version = "ViT-B/32"
output_feat_size = 512
clip_len = 2
overwrite = True
num_decoding_thread = 4
half_precision = False

clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir,q_path,v_path):
    vid = np.load(v_path)['features'].astype(np.float32)
    txt = np.load(q_path)['last_hidden_state'].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 2
    ctx_l = vid.shape[0]

    timestamp = ((torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, query,q_path, v_path):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir,q_path,v_path)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
    
    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits
    
    top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()
    k = min(5, len(pred_confidence.flatten()))
    top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=k)
    top5_windows = pred_windows[top5_indices].tolist()
    top5_confidences = top5_values.tolist()
    
    result = {
        "top1_interval": top1_window,
        "top5_intervals": top5_windows,
        "top5_confidences": top5_confidences
    }

    return result

def extract_vid(vid_path, save_dir):
    # vid_features = vid2clip(clip_model, vid_path, save_dir)
    # read feaute from the vid_path
    vid_features = np.load(vid_path)

    return "Video features extracted."

def extract_txt(query, save_dir):
    # txt_features = txt2clip(clip_model, query, save_dir)
    txt_features = np.load(query)

    return "Text features extracted."

if __name__ == '__main__':
    os.makedirs(args.save_dir, exist_ok=True)
    data = pd.read_csv(args.csv_file)

    top1_intervals = []
    top5_intervals = []
    top5_confidences = []

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing videos and queries"):
        video_id = row['reference_id']
        query = row['query_id']
        video_path = os.path.join(args.video_folder, f"{video_id.split('.')[0]}.npz")
        query_path = os.path.join(args.query_folder, f"qid{query.split('.')[0]}.npz")


        if not os.path.exists(query_path):
            logger.error(f"Video file {video_path} does not exist.")
            top1_intervals.append(None)
            top5_intervals.append(None)
            top5_confidences.append(None)
            continue

        extract_vid(video_path, args.save_dir)
        extract_txt(query_path, args.save_dir)
        result = forward(vtg_model, args.save_dir, query,query_path,video_path)

        top1_intervals.append(result['top1_interval'])
        top5_intervals.append(result['top5_intervals'])
        top5_confidences.append(result['top5_confidences'])

    data['top1_interval'] = top1_intervals
    data['top5_intervals'] = top5_intervals
    data['top5_confidences'] = top5_confidences

    data.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
