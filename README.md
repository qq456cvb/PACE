# <div align="center">PACE: Pose Annotations in Cluttered Environments<br>(ECCV 2024)</div>

<p align="center">
  <img src="images/teaser.jpg" width="100%" alt="PACE Teaser"/>
</p>

<div align="center">
  <h3>
    <a href="https://qq456cvb.github.io">Yang You</a>, <a href="https://xiongkai.netlify.app/">Kai Xiong</a>, Zhening Yang, <a href="https://github.com/huangzhengxiang">Zhengxiang Huang</a>, <a href="https://github.com/Zhou-jw">Junwei Zhou</a>, <a href="https://rshi.top/">Ruoxi Shi</a>, Zhou Fang, <a href="https://adamharley.com/">Adam W Harley</a>, <a href="https://geometry.stanford.edu/member/guibas/">Leonidas Guibas</a>, <a href="https://www.mvig.org/">Cewu Lu</a>
    <br><br>
    <a href="https://arxiv.org/pdf/2312.15130.pdf">
      <img src="https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange" alt="Paper PDF">
    </a>
    <a href="https://qq456cvb.github.io/files/pace_supp.pdf">
      <img src="https://img.shields.io/badge/Supp-PDF-blue?style=flat&logo=arxiv&logoColor=green" alt="Supplementary">
    </a>
    <a href="https://qq456cvb.github.io/projects/pace">
      <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green" alt="Project Page">
    </a>
    <!--
    <a href="https://youtu.be/IyGzkdR5MLU">
      <img src="https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red" alt="Video"/>
    </a>
    -->
    <br>
  </h3>
</div>

---

**PACE** (Pose Annotations in Cluttered Environments) is a large-scale benchmark designed to advance pose estimation in challenging, cluttered scenarios. PACE provides comprehensive real-world and simulated datasets for both instance-level and category-level tasks, featuring:

- **55K frames** with **258K annotations** across **300 videos**
- **238 objects** from **43 categories** (rigid and articulated)
- An innovative annotation system using a calibrated 3-camera setup
- **PACESim**: 100K photo-realistic simulated frames with 2.4M annotations across 931 objects

We evaluate state-of-the-art algorithms on PACE for both pose estimation and object pose tracking, highlighting the benchmark's challenges and research opportunities.

<p align="center">
  <img src="images/video.gif?raw=true" width="50%"/>
</p>

---

## Why a New Dataset?
- PACE rigorously tests the generalization of state-of-the-art methods in complex, real-world environments, enabling exploration and quantification of the 'simulation-to-reality' gap for practical applications.

## 🔥News
- Try our latest pose estimator **[CPPF++](https://github.com/qq456cvb/CPPF2)** (*TPAMI*), which achieves state-of-the-art performance on PACE.

## Update Log
- **2024/07/22**: PACE v1.1 uploaded to [HuggingFace](https://huggingface.co/datasets/qq456cvb/PACE/tree/main). Benchmark evaluation code released.
- **2024/03/01**: PACE v1.0 released.

## Table of Contents
- [Dataset Download](#dataset-download)
- [Dataset Format](#dataset-format)
- [Dataset Visualization](#dataset-visualization)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Annotation Tools](#annotation-tools)
- [License](#license)
- [Citation](#citation)

---

## Dataset Download
Download the dataset from [HuggingFace](https://huggingface.co/datasets/qq456cvb/PACE/tree/main). Unzip all `tar.gz` files and place them under `dataset/pace` for evaluation. Large files are split into chunks; merge them with, e.g., `cat test_chunk_* > test.tar.gz`.

---

## Dataset Format
PACE follows the [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) with the following structure (regex syntax):

```
camera_pbr.json
models(_eval|_nocs)?
├─ models_info.json
├─ (artic_info.json)?
├─ obj_${OBJ_ID}.ply
model_splits
├─ category
|  ├─ ${category}_(train|val|test).txt
|  ├─ (train|val|test).txt
├─ instance
|  ├─ (train|val|test).txt
(train(_pbr_cat|_pbr_inst)|val(_inst|_pbr_cat)|test)
├─ ${SCENE_ID}
│  ├─ scene_camera.json
│  ├─ scene_gt.json
│  ├─ scene_gt_info.json
│  ├─ scene_gt_coco_det_modal(_partcat|_inst)?.json
│  ├─ depth
│  ├─ mask
│  ├─ mask_visib
│  ├─ rgb
|  ├─ (rgb_nocs)?
```

**Key components:**
- `camera_pbr.json`: Camera parameters for PBR rendering; real camera parameters are in each scene's `scene_camera.json`.
- `models(_eval|_nocs)?`: 3D object models. `models` contains original scanned meshes; `models_eval` has uniformly sampled point clouds for evaluation (e.g., Chamfer distance); all models (except articulated parts, ID 545–692) are recentered and normalized to a unit bounding box. `models_nocs` recolors vertices by NOCS coordinates.
  - `models_info.json`: Mesh metadata (diameter, bounds, scales in mm), and mapping from `obj_id` to object `identifier`. Articulated objects have multiple parts, each with a unique `obj_id`; associations are in `artic_info.json`.
  - `artic_info.json`: Part information for articulated objects, keyed by `identifier`.
  - `obj_${OBJ_ID}.ply`: Mesh file for object `${OBJ_ID}`.
- `model_splits`: Model IDs for train/val/test splits. Instance-level splits share IDs; category-level splits differ per category.
- `train(_pbr_cat|_pbr_inst)|val(_inst|_pbr_cat)|test`: Synthetic and real data for category/instance-level training and validation; real-world test data for both.
  - `${SCENE_ID}`: Each scene in a separate folder (e.g., `000011`).
    - `scene_camera.json`: Camera parameters.
    - `scene_gt.json`: Ground-truth annotations ([BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#ground-truth-annotations)).
    - `scene_gt_info.json`: Meta info about ground-truth poses ([BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md#meta-information-about-the-ground-truth-poses)).
    - `scene_gt_coco_det_modal(_partcat|_inst)?.json`: 2D bounding box and instance segmentation in COCO format.
      - `scene_gt_coco_det_modal_partcat.json`: Treats articulated parts as separate categories (for category-level evaluation).
      - `scene_gt_coco_det_modal_inst.json`: Treats each object instance as a separate category (for instance-level evaluation). Note: There may be more categories than reported in the paper, as some objects appear only in synthetic data.
    - `rgb`: Color images.
    - `rgb_nocs`: Normalized object coordinates as RGB (mapped from `[-1, 1]` to `[0, 1]`), normalized w.r.t. object bounding box. Example normalization:
      ```python
      mesh = trimesh.load_mesh(ply_fn)
      bbox = mesh.bounds
      center = (bbox[0] + bbox[1]) / 2
      mesh.apply_translation(-center)
      extent = bbox[1] - bbox[0]
      colors = np.array(mesh.vertices) / extent.max()
      colors = np.clip(colors + 0.5, 0, 1.)
      ```
      See [this paper](https://arxiv.org/abs/1908.07640) for disambiguation method.
    - `depth`: 16-bit depth images. Convert to meters by dividing by 10,000 (PBR) or 1,000 (real).
    - `mask`: Object masks.
    - `mask_visib`: Visible part masks.

---

## Dataset Visualization
A visualization script is provided to display ground-truth pose annotations and rendered 3D models. Run `visualizer.ipynb` to generate visualizations like the following:

<p align="center">
  <img src="images/rgb_000000.png" width="45%"/> <img src="images/render_000000.png" width="45%"/>
</p>
<p align="center">
  <img src="images/pose_000000.png" width="45%"/> <img src="images/mask_000000.png" width="45%"/>
</p>
<p align="center">
  <img src="images/depth_000000.png" width="45%"/> <img src="images/nocs_000000.png" width="45%"/>
</p>

---

## Benchmark Evaluation
Unzip all `tar.gz` files from [HuggingFace](https://huggingface.co/datasets/qq456cvb/PACE/tree/main) and place them under `dataset/pace` for evaluation.

### Instance-Level Pose Estimation
- Ensure the `bop_toolkit` submodule is cloned: after `git clone`, run `git submodule update --init`, or use `git clone --recurse-submodules git@github.com:qq456cvb/PACE.git`.
- Place prediction results at `prediction/instance/${METHOD_NAME}_pace-test.csv` (baseline results available [here](https://drive.google.com/drive/folders/1_MfVn815u0oWzGG4H9bcRIy42rVLzOr0?usp=sharing)).
- Run:
  ```sh
  cd eval/instance
  sh eval.sh ${METHOD_NAME}
  ```

### Category-Level Pose Estimation
- Place prediction results at `prediction/category/${METHOD_NAME}_pred.pkl` (baseline results available [here](https://drive.google.com/drive/folders/1_Z22KjGJ55yimboSuN2nVp6M0_Yz1-Dr?usp=sharing)).
- Download ground-truth labels in compatible `pkl` format from [here](https://drive.google.com/file/d/1a_Ld_8COxQAXL2dJI4L2qvrwbpX6qsa2/view?usp=sharing) and place at `eval/category/catpose_gts_test.pkl`.
- Run:
  ```sh
  cd eval/category
  sh eval.sh ${METHOD_NAME}
  ```

**Note:** There are more categories (55) in `category_names.txt` than reported in the paper, as some categories lack real-world test images. The actual evaluation categories (47) are in `category_names_test.txt` (parts are counted separately). Ground-truth class IDs in `catpose_gts_test.pkl` use indices 1–55, matching `category_names.txt`.

---

## Annotation Tools
The source code for our annotation tools is organized as follows:

```
annotation_tool/
├─ inpainting
├─ obj_align
├─ obj_sym
├─ pose_annotate
├─ postprocessing
├─ TFT_vs_Fund
├─ utils
```

- `inpainting`: Inpaints markers for more realistic images.
- `obj_align`: Aligns objects to a consistent orientation within categories.
- `obj_sym`: Annotates object symmetry information.
- `pose_annotate`: Main pose annotation program.
- `postprocessing`: Post-processing steps (e.g., marker removal, extrinsics refinement/alignment).
- `TFT_vs_Fund`: Refines 3-camera extrinsics.
- `utils`: Miscellaneous helper functions.

*Detailed documentation is coming soon. We are working to make the annotation tools as user-friendly as possible for accurate 3D pose annotation.*

---

## License
[MIT](https://opensource.org/license/mit) license for all contents **except**:

- Models with IDs 693–1260 are from [SketchFab](https://sketchfab.com/) under [CC BY](https://creativecommons.org/licenses/by/4.0/). Original posts: `https://sketchfab.com/3d-models/${OBJ_IDENTIFIER}` (find the identifier in `models_info.json`).
- Models 1165 and 1166 are from [GrabCAD](https://grabcad.com/library/squeegee-2) (identical geometry, different colors). See [GrabCAD license](https://help.grabcad.com/article/246-how-can-models-be-used-and-shared?locale=en).

---

## Citation
```bibtex
@misc{you2023pace,
    title={PACE: Pose Annotations in Cluttered Environments},
    author={You, Yang and Xiong, Kai and Yang, Zhening and Huang, Zhengxiang and Zhou, Junwei and Shi, Ruoxi and Fang, Zhou and Harley, Adam W. and Guibas, Leonidas and Lu, Cewu},
    booktitle={European Conference on Computer Vision},
    year={2024},
    organization={Springer}
}
``` 
