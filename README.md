

 <h1 align="center">
PACE: Pose Annotations in Cluttered Environments
</h1>

<p align='center'>
<img align="center" src='images/teaser.jpg' width='70%'> </img>
</p>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, <a href="https://xiongkai.netlify.app/">Kai Xiong</a>, Zhening Yang, <a href="https://github.com/huangzhengxiang">Zhengxiang Huang</a>, <a href="https://github.com/Zhou-jw">Junwei Zhou</a>, <a href="https://rshi.top/">Ruoxi Shi</a>, Zhou Fang, <a href="https://adamharley.com/">Adam W Harley</a>, <a href="https://geometry.stanford.edu/member/guibas/">Leonidas Guibas</a>, <a href="https://www.mvig.org/">Cewu Lu</a>
<br>
<br>
<a href='https://arxiv.org/pdf/2312.15130.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/projects/pace'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
<!-- <a href='https://youtu.be/IyGzkdR5MLU'>
<img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Video'/>
</a> -->
<br>
</h3>
</div>
 
We introduce PACE (Pose Annotations in Cluttered Environments), a large-scale benchmark designed to advance the development and evaluation of pose estimation methods in 
cluttered scenarios. PACE encompasses 54,945 frames with 257,673 annotations across 300 videos, covering 576 objects from 44 categories and featuring a mix of rigid and articulated items in cluttered scenes.

# Why a new dataset?
- Our objective is to rigorously assess the generalization capabilities of current state-of-the-art methods in a broad and large-scale testing environment. This will enable us to explore and quantify the 'simulation-to-reality' gap, providing deeper insights into the effectiveness of these methods in practical applications.


# Contents
- [Dataset Download](#dataset-download)
- [Dataset Format](#dataset-format)
- [Dataset Visualization](#dataset-visualization)
- [Annotation Tools](#annotation-tools)
- [Citation](#citation)

# Dataset Download
Our dataset can be downloaded on [OneDrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/qq456cvb_sjtu_edu_cn/Ei3YV1Iz5U1Ai2fkgD7wMO0BlnAjzgRSahLu3YwD8W-dZQ) or [BaiduPan](https://pan.baidu.com/s/1EJ5deSOJWk4vqQMc81bLeg?pwd=mf7d).
  
# Dataset Format
Our dataset mainly follows [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) with the following structure:
```
camera[_pbr|_real].json
models[_simplified|_eval]
├─ models_info.json
├─ [artic_info.json]
├─ obj_OBJ_ID.ply
train_pbr|val|test
├─ SCENE_ID
│  ├─ scene_camera.json
│  ├─ scene_gt.json
│  ├─ scene_gt_info.json
│  ├─ scene_gt_coco_det_modal[_partcat|_inst].json
│  ├─ depth
│  ├─ mask
│  ├─ mask_visib
│  ├─ rgb
|  ├─ [rgb_nocs]
```

- `models[_simplified|_eval]` - 3D object models. `models_simplified` is a simplified version of original 3D meshes (e.g., can be used for rendering or annotation); `models_eval` contains uniformly sampled point cloud from the original meshes, and can be used for evaluation (e.g., computing chamfer distance).
- `models_info.json` - Meta information about the meshes, including diameters (the largest distance between any two points), bounds and scales, all in mm. The scales represent the size of axis-aligned bounding boxes. This file also contains the mapping from `obj_id` (int) to object `identifier` (string). **Notice**: Articulated objects can contain multiple `obj_id` with information stored in `artic_info.json`.
- `artic_info.json` - contain the part information of articulated objects, with `identifier` as the key.
- `scene_camera.json` - Camera parameters.
- `scene_gt.json` - Ground-truth annotations. See [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#ground-truth-annotations) for details.
- `scene_gt_info.json` - Meta information about ground-truth poses. See [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#meta-information-about-the-ground-truth-poses) for details.
- `scene_gt_coco_det_modal[_partcat|_inst].json` - 2D bounding box and instance segmentation labels in COCO format. `scene_gt_coco_det_modal_partcat.json` treats individual parts of articulated objects as different categories, which is useful when evaluating articulate-agnostic category-level pose estimation methods. `scene_gt_coco_det_modal_inst.json` treats each object **instance** as a separate category, which is useful when evaluating instance-level pose estimation methods. **Notice**: there are slightly more categories than those reported in the paper since some objects only appear in the synthetic dataset but not in the real one.
- `rgb` - Color images.
- `rgb_nocs` - Normalized coordinates of objects encoded as RGB colors (mapped from `[-1, 1]` to `[0, 1]`).
- `depth` - Depth images (saved as 16-bit unsigned short). To convert depth into actual meters, divide by 10000 for `train_pbr` and 1000 for `val|test`.
- `mask` - Masks of objects.
- `mask_visib` - Masks of visible parts of objects.


# Dataset Visualization
We provide a visualization script to visualize the ground-truth pose annotations together wich their rendered 3D models. You can run `visualizer.ipynb` and get the following rgb/rendering/pose/mask visualizations:

<p align='center'>
<img align="left" src='images/rgb_000000.png' width='45%'> </img> <img align="center" src='images/render_000000.png' width='45%'> </img>
</p>
<p align='center'>
<img align="left" src='images/pose_000000.png' width='45%'> </img> <img align="center" src='images/mask_000000.png' width='45%'> </img>
</p>

# Annotation Tools
We also provide the source code of our annotation tools, organized as follows:
```
annotation_tool
├─ inpainting
├─ obj_align
├─ obj_sym
├─ pose_annotate
├─ postprocessing
├─ TFT_vs_Fund
```

- `inpainting` - Code for inpainting the markers so that the result image is more realistic.
- `obj_align` - Code for aligning objects into a consistent orientation within the same category.
- `obj_sym` - Code for annotating object symmetry information.
- `pose_annotate` - The main program of pose annotation.
- `postprocessing` - Code for various post processing steps, e.g., remove the markers, automatically refine the extrinsics, and manually align the extrinsics.
- `TFT_vs_Fund` - Used in refining the extrinsics of the 3-cameras.

More detailed documentation for the annotation software is coming soon.

# Citation
```
@misc{you2023pace,
    title={PACE: Pose Annotations in Cluttered Environments},
    author={Yang You and Kai Xiong and Zhening Yang and Zhengxiang Huang and Junwei Zhou and Ruoxi Shi and Zhou Fang and Adam W. Harley and Cewu Lu},
    year={2023},
    eprint={2312.15130},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
