import json
import pickle
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from pycocotools.coco import COCO


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: lists of integers [x_min, y_min, width, height]

    Returns:
    float: IoU of box1 and box2
    """
    
    # Extract the coordinates and dimensions of each box
    x1_min, y1_min, width1, height1 = box1
    x2_min, y2_min, width2, height2 = box2

    # Calculate the (x, y) coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_min + width1, x2_min + width2)
    y_inter_max = min(y1_min + height1, y2_min + height2)

    # Calculate the area of intersection rectangle
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = width1 * height1
    box2_area = width2 * height2

    # Calculate the union area by using the formula: union_area = box1_area + box2_area - inter_area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU by dividing the intersection area by the union area
    iou = inter_area / union_area

    return iou


if __name__ == '__main__':
    root = Path(__file__).parent.parent.parent
    
    parser = ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    
    args = parser.parse_args()
    split = args.split
    
    # load coco annotations
    gt_coco = COCO(str(root / f'dataset/pace/{split}_all_coco_det_modal_partcat.json'))
    image_ids = gt_coco.getImgIds()
    
    # load object info
    models_info = json.load(open(root / 'dataset/pace/models/models_info.json'))
    artic_info = json.load(open(root / 'dataset/pace/models/artic_info.json'))
    artic_insts = list(artic_info.keys())
    objid2cats = {}
    objid2insts = {}
    all_cats = open(root / 'eval/category/category_names.txt').read().splitlines()
    for i, obj_id in enumerate(models_info.keys()):
        identifier = models_info[str(obj_id)]['identifier']
        objid2cats[int(obj_id)] = identifier.split('/')[0]
        if len(identifier.split('/')) > 2:
            objid2cats[int(obj_id)] += '/{}'.format(identifier.split('/')[-1])
        assert(objid2cats[int(obj_id)] in all_cats)
        objid2insts[int(obj_id)] = '/'.join(identifier.split('/')[:2])

    cats2id = {}
    for i, cat in enumerate(all_cats):
        cats2id[cat] = i
    
    pkls = {}
    for img_id in tqdm(image_ids[:len(image_ids)]):
        # Load the image and its annotations
        image_data = gt_coco.loadImgs(img_id)[0]
        annotations = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_id),)
        out = {
            'gt_bboxes': [],
            'gt_segs': [],
            'gt_RTs': [],
            'gt_scales': [],
            'gt_class_ids': [],
            'gt_obj_ids': [],
            'image_path': image_data['file_name']
        }
        scene_gt = json.load(open(f'/orion/u/yangyou/COLSPA/data/bop/colspa/{split}/' + image_data['file_name'].split('/')[1] + '/scene_gt.json'))
        scene_gt_info = json.load(open(f'/orion/u/yangyou/COLSPA/data/bop/colspa/{split}/' + image_data['file_name'].split('/')[1] + '/scene_gt_info.json'))
        img_id = int(image_data['file_name'].split('/')[-1].split('.')[0])
        gts = scene_gt[str(img_id)]
        gt_infos = scene_gt_info[str(img_id)]
        obj_ids = [a['obj_id'] for a in gts]
        cat_ids = [cats2id[objid2cats[obj_id]] for obj_id in obj_ids]
            
        for anno in annotations:
            if cat_ids.count(anno['category_id']) >= 2:
                cand_idxs = [i for i, cat_id in enumerate(cat_ids) if cat_id == anno['category_id']]
                best_iou = -1
                gt_idx = None
                for idx in cand_idxs:
                    gt_info = gt_infos[idx]
                    iou = compute_iou(gt_info['bbox_visib'], anno['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        gt_idx = idx
                if best_iou < 0.5:  # in very rare cases, the iou is less than 0.5 because the object has only a few pixels visible on the image boundary
                    continue
            else:
                gt_idx = cat_ids.index(anno['category_id'])
            
            gt = gts[gt_idx]
            out['gt_bboxes'].append(anno['bbox'])
            out['gt_segs'].append(anno['segmentation'])
            out['gt_class_ids'].append(anno['category_id'] + 1)
            model_info = models_info[str(obj_ids[gt_idx])]
            scale = np.array([model_info['size_x'], model_info['size_y'], model_info['size_z']]) / 1000.
            rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
            trans = np.array(gt['cam_t_m2c']) / 1000.
            rt = np.eye(4)
            rt[:3, :3] = rot
            rt[:3, 3] = trans
            out['gt_obj_ids'].append(obj_ids[gt_idx])
            out['gt_RTs'].append(rt)
            out['gt_scales'].append(scale)
        for key in out:
            if key not in ['gt_segs', 'image_path']:
                if len(out[key]) > 0:
                    out[key] = np.stack(out[key])
            
            assert len(out['gt_bboxes']) == len(out['gt_RTs']) == len(out['gt_scales']) == len(out['gt_class_ids'])
        pkls[image_data['file_name']] = out
    pickle.dump(pkls, open(root / f'eval/category/catpose_gts_{split}.pkl', 'wb'))