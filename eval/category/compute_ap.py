import json
import pickle
import numpy as np
import sys
import cv2
import torch
from pathlib import Path
from pycocotools import mask as maskUtils
import pickle
import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from util import compute_degree_cm_mAP


def VOCap(rec, prec):
    ap = np.sum(rec[1:] * prec)
    return ap


def cal_auc(add_dis, max_dis=1.):
    D = np.array(add_dis)
    D[np.where(np.isnan(D) == True)[0]] = np.inf
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.ones((n-1,)) / (n - 1)
    aps = VOCap(D, acc)
    return aps * 100


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    root = Path(__file__).parent.parent.parent
    
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--ap_outpath', type=str, default=None)
    
    args = parser.parse_args()
    
    method = args.method
    split = args.split

    preds = pickle.load(open(root / f'prediction/category/{method}_pred.pkl', 'rb'))
    gts = pickle.load(open(root / f'eval/category/catpose_gts_{split}.pkl', 'rb'))
    categories = open(root / 'eval/category/category_names.txt').read().splitlines()

    results = []
    valid_cats = []
    for image_path in tqdm(preds):
        image_path_gt = image_path
        gt = gts[image_path_gt]
        
        if len(gt['gt_RTs']) == 0:
            continue
        
        pred = preds[image_path]
        pred.update(**gt)
        pred['gt_handle_visibility'] = np.ones((len(pred['gt_RTs']),))
        pred['pred_bboxes'] = np.zeros((len(pred['pred_RTs']), 4))
        results.append(pred)
        valid_cats.extend([categories[cls_id - 1] for cls_id in gt['gt_class_ids']])
    
    valid_cats = list(set(valid_cats))
    # print(valid_cats)
    cat_names = [cat.replace('/', '-') for cat in categories]

    iou_3d_aps, pose_aps = compute_degree_cm_mAP(results[:], 
            ['BG'] + cat_names,
            'tmp',
        degree_thresholds=np.linspace(0, 1, 61) * 60, 
        shift_thresholds=np.linspace(0, 1, 61) * 15,
        iou_3d_thresholds=np.linspace(0, 1, 101),
        iou_pose_thres=0.1,
        use_matches_for_pose=True)

    if args.ap_outpath is not None:
        pickle.dump({
            'iou_3d_aps': iou_3d_aps,
            'pose_aps': pose_aps,
        }, open(args.ap_outpath, 'wb'))
    
    # next compute the APs
    artic_idxs = np.array([i for i, cat in enumerate(categories) if '/' in cat]) + 1
    rigid_idxs = np.array([i for i, cat in enumerate(categories) if '/' not in cat]) + 1

    categories_test = open(root / 'eval/category/category_names_test.txt').read().splitlines()
    artic_idxs_test = np.array([idx for idx in artic_idxs if categories[idx - 1] in categories_test])
    rigid_idxs_test = np.array([idx for idx in rigid_idxs if categories[idx - 1] in categories_test])
    # rigid_idxs_test = np.array([idx for idx in rigid_idxs if categories[idx - 1] in valid_cats])
    
    for idxs, name in zip([rigid_idxs_test, artic_idxs_test], ['rigid', 'artic']):
        print(name)
        pose_ap = np.mean(pose_aps[idxs], axis=0)
        iou_3d_ap = np.mean(iou_3d_aps[idxs], axis=0)
        print('iou25 {:.1f}'.format(iou_3d_ap[25] * 100))  # iou25
        print('iou50 {:.1f}'.format(iou_3d_ap[50] * 100))  # iou50
        print('20deg {:.1f}'.format(np.mean(pose_ap[:21, -1]) * 100.))  # degree auc
        print('60deg {:.1f}'.format(np.mean(pose_ap[:61, -1]) * 100.))  # degree auc
        print('5cm {:.1f}'.format(np.mean(pose_ap[-1, :21]) * 100.))  # translation auc
        print('15cm {:.1f}'.format(np.mean(pose_ap[-1, :61]) * 100.))  # translation auc
        print('20deg 5cm {:.1f}'.format(np.mean(np.diag(pose_ap[:21, :21])) * 100.))  # degree and translation auc
        print('60deg 15cm {:.1f}'.format(np.mean(np.diag(pose_ap[:61, :61])) * 100.))  # degree and translation auc

        print()
    print()