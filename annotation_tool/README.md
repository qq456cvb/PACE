# PACE Annotation Tools

This directory contains the full annotation pipeline used to create PACE: a multi-camera 3D pose annotation system built around a calibrated 3-camera RGB-D rig and an ArUco marker board. The tools cover everything from camera calibration and video capture to object model preparation, interactive pose annotation, and mask generation.

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Requirements](#requirements)
- [Data Layout](#data-layout)
- [Step 1 — Camera Calibration](#step-1--camera-calibration)
- [Step 2 — Video Capture](#step-2--video-capture)
- [Step 3 — Extrinsic Refinement](#step-3--extrinsic-refinement)
- [Step 4 — Marker Removal](#step-4--marker-removal)
- [Step 5 — Object Model Preparation](#step-5--object-model-preparation)
- [Step 6 — Pose Annotation](#step-6--pose-annotation)
- [Step 7 — Mask Generation and Inspection](#step-7--mask-generation-and-inspection)
- [Annotation File Format](#annotation-file-format)
- [Configuration](#configuration)
- [Notes](#notes)

## Pipeline Overview

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│ 1. Calibrate    │ → │ 2. Capture      │ → │ 3. Refine           │
│ utils/          │   │ inpainting/     │   │ postprocessing/     │
│ calc_extrin.py  │   │ inpaint.py      │   │ refine_extrinsic.py │
└─────────────────┘   └─────────────────┘   └─────────────────────┘
                                                       ↓
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│ 5. Prepare      │ → │ 6. Annotate     │ ← │ 4. Remove marker    │
│ obj_align/      │   │ pose_annotate/  │   │ postprocessing/     │
│ obj_sym/        │   │ mainwindow.py   │   │ remove_marker.py    │
└─────────────────┘   └─────────────────┘   └─────────────────────┘
                               ↓
                      ┌─────────────────┐
                      │ 7. Masks/export │
                      │ postprocessing/ │
                      │ generate_seg.py │
                      └─────────────────┘
```

All tools resolve data paths relative to the working directory, so run them from the `annotation_tool/` root, e.g.:

```sh
cd annotation_tool
python pose_annotate/mainwindow.py
```

## Requirements

**Hardware (for capture only).** Three Intel RealSense RGB-D cameras (1280x720) rigidly mounted around the capture area, plus a printed 100 mm ArUco marker (`DICT_4X4_50`, ID 0) attached to a movable board on which objects are placed. Annotation of already-captured data only needs a CUDA GPU.

**Software.** Python 3.8+ with:

- GUI / visualization: `PyQt5`, `pyqtgraph`, `open3d`, `seaborn`
- Geometry / rendering: `torch` (CUDA), `nvdiffrast`, `trimesh`, `pymeshlab`, `scipy`, `kornia`
- I/O and misc: `opencv-contrib-python` (ArUco), `xmltodict`, `tqdm`, `Pillow`
- Capture only: `pyrealsense2`, `scikit-video`

Some optional features of the pose annotation GUI call external projects which need to be set up separately if you want to use them:

- [XMem](https://github.com/hkchengrex/XMem) — video object segmentation, used by the *Bundle track* propagation
- [BundleTrack](https://github.com/wenbowen123/BundleTrack) — 6-DoF object tracking from masks
- BCOT / [RBOT](https://github.com/henningtjaden/RBOT) region-based tracker — used by the *Tracking selection* action
- [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) and a MATLAB installation with the bundled `TFT_vs_Fund` toolbox — used by the automatic extrinsic refinement scripts

The core manual annotation workflow (PnP initialization + keyboard pose refinement + rendered mask generation) works without these.

The GUI tools embed an Open3D window inside Qt and were developed on Windows (`win32gui`); on Linux the pose annotation tool falls back to `xwininfo` (X11 required).

## Data Layout

Captured scenes are organized as:

```
data/videos/scene_X/video_Y/
├─ intrinsics.npy            # (3, 3, 3) intrinsics of cam0/1/2
├─ extrinsics.npy            # (3, 4, 4) initial extrinsics from marker calibration
├─ extrinsics_refined.npy    # (3, 4, 4) refined extrinsics (Step 3)
├─ cam{0,1,2}/
│  ├─ rgb/rgb0000.png        # raw color frames
│  ├─ rgb_marker/rgb0000.png # marker-inpainted color frames (Step 4)
│  ├─ depth/depth0000.png    # 16-bit depth
│  ├─ pose/0000.json         # pose annotations (Step 6 output)
│  └─ mask/0000.png          # instance masks (Step 7 output)
└─ aux1/
   ├─ bg0000.png             # background reference clip (for marker removal)
   └─ rel0000.png            # relative-pose reference clip
```

Object models are organized as:

```
data/models_aligned_lowres/<category>/<object>.obj        # simplified meshes used by the GUI
data/models_aligned_highres/<category>/<object>.obj       # full-resolution meshes
data/models_aligned_highres/<category>/<object>.sym.npy   # symmetry annotations (Step 5)
```

Articulated objects are directories containing a `.urdf` plus one mesh per link. Mesh units are millimeters (converted to meters at load time).

## Step 1 — Camera Calibration

```sh
python utils/calc_extrin.py
```

Opens all three RealSense streams and detects the ArUco marker. Press `S` to start collecting marker observations, and `S` again to stop. The script averages the per-frame marker poses and writes:

- `data/intrinsics.npy` — per-camera intrinsics
- `data/extrinsics.npy` — per-camera extrinsics in the marker (board) coordinate frame

These two files are picked up by the capture tool and copied into every recorded video.

## Step 2 — Video Capture

```sh
python inpainting/inpaint.py
```

A capture GUI for recording annotation videos. Despite living in `inpainting/`, this is the main recording tool — it also previews live marker inpainting so you can check the auxiliary clips are good. Before recording a scene, capture two short reference clips:

1. **Background** — the empty board with the marker visible (used to inpaint the marker region later)
2. **Relative Pose** — move the board around so the marker is seen from several angles

Then record the actual sequence with objects on the board.

| Button | Key | Action |
|---|---|---|
| Open/Close Cam | `D` | start/stop the RealSense streams |
| Background | `Z` | record the background reference clip |
| Relative Pose | `X` | record the relative-pose reference clip |
| Record/Stop | `C` | record the main RGB-D sequence |
| OK | `A` | accept the current recording |
| Cancel | `S` | discard the current recording |

Each accepted recording is saved as a new `data/videos/video_N/` folder containing `cam{0,1,2}/rgb`, `cam{0,1,2}/depth`, `aux1/`, and copies of `intrinsics.npy` / `extrinsics.npy`. The RealSense serial numbers at the top of `inpaint.py` must be edited to match your devices.

## Step 3 — Extrinsic Refinement

Marker-based extrinsics are a good initialization but not pixel-perfect. Three refinement options are available (edit the `root = Path('data/videos/...')` glob at the bottom of each script to select your scene):

- **Automatic** — `python postprocessing/refine_extrinsic.py`
  Extracts SuperPoint/SuperGlue correspondences across the three views, estimates relative poses with the trifocal-tensor toolbox (`TFT_vs_Fund`, via the MATLAB engine), and runs bundle adjustment. Writes `extrinsics_refined.npy`.

- **Semi-automatic** — `python postprocessing/refine_extrinsic_manual.py`
  Same optimization, but lets you inspect and edit the correspondences in OpenCV windows first: left-click to add a point, right-click near a point to delete it across views, `Q` to finish editing.

- **Manual point-cloud alignment** — `python postprocessing/manual_align.py`
  Visualizes the fused point clouds from all cameras and lets you nudge a camera's extrinsics with the same translation/rotation keys as the pose annotation tool (see below). Press `Esc` to save and exit.

## Step 4 — Marker Removal

```sh
python postprocessing/remove_marker.py
```

Inpaints the ArUco marker region in every frame using the background and relative-pose reference clips (`aux1/`), warping the clean background via homography and blending with seamless cloning. This produces the `rgb_marker/` frames consumed by the pose annotation GUI, so the marker does not leak into the dataset imagery. Edit the hardcoded `root` path to point at your scene.

## Step 5 — Object Model Preparation

Each scanned object must be aligned to its category-canonical frame and annotated with its symmetries before pose annotation.

### 5a. Canonical alignment — `python obj_align/aligner.py`

Open a raw scanned `.obj` (File → Open, defaults to `data/models`), orient it into the canonical category frame, and save (File → Save). The viewer shows the mesh together with its three axis-plane projections (XY / XZ / YZ) so you can verify the alignment at a glance.

- **Auto Align** — initial alignment from PCA of the mesh
- **Swap X-Y / X-Z / Y-Z**, **Flip X / Y / Z** — quick axis fixes
- Keyboard — `Q/E`, `W/S`, `A/D` translate and `Z/X`, `C/V`, `B/N` rotate for fine adjustment
- **Simplify** — decimate the mesh with pymeshlab; saving a simplified mesh defaults to `models_aligned_lowres/`, otherwise to `models_aligned_highres/`

Save both a high-res and a simplified low-res copy of every object: the GUI annotates with low-res meshes for speed, while the high-res meshes go into the released dataset.

### 5b. Symmetry annotation — `python obj_sym/annotator.py`

Iterates over `data/models_aligned_highres/*/*` and shows the first object without a `.sym.npy` file. Click the button matching the object's symmetry and the tool writes the corresponding stack of rotation matrices and advances to the next object:

| Button | Meaning |
|---|---|
| `none` | no symmetry (identity only) |
| `x180` / `y180` / `z180` | 2-fold (180°) symmetry around the axis |
| `x90` / `y90` / `z90` | 4-fold (90°) symmetry around the axis |
| `xinf` / `yinf` / `zinf` | continuous rotational symmetry around the axis (sampled every 10°) |
| `allinf` | spherical symmetry |

## Step 6 — Pose Annotation

```sh
python pose_annotate/mainwindow.py
```

The main annotation GUI. Open a video folder with **File → Open scene...** (`Ctrl+O`) — it expects the layout from [Data Layout](#data-layout) with `extrinsics_refined.npy` and `rgb_marker/` present. The window shows three tabs (**2D** image view, **3D** point-cloud view, **Segmentation**), the **Current Annotations** tree, and the **Object Database** tree listing everything under `data/models_aligned_lowres`.

### Adding an object

1. Select the object in the **Object Database** and press **Add**.
2. Click at least 4 distinctive points on the object in the 2D image.
3. In the Open3D window that opens, pick the same points on the mesh in the same order.
4. Press **Add** again (now acting as Confirm) — the pose is solved by PnP and the rendered object is overlaid on the image. `Ctrl+Z` undoes the last picked point; **Cancel** aborts.

### Refining a pose

Hover over a rendered object to highlight it and click to select it (or press its ID digit `0`–`9`). Then refine with the keyboard; the rendered overlay updates live in all views:

| Keys | Action |
|---|---|
| `Q`/`E`, `W`/`S`, `A`/`D` | translate along the object's local X / Z / Y axis (1 mm) |
| `Z`/`X`, `C`/`V`, `B`/`N` | rotate around the object's local X / Y / Z axis (0.5°) |
| hold `Alt` | 10x step size |
| `R` | toggle between 2D and 3D tabs |
| `F` | store the current 3D viewpoint |
| `Delete` | delete the selected annotation |
| `Ctrl+S` | **save** the current frame's poses and masks |

Navigate frames with **Prev10 / Prev / Next / Next10** or jump with the **Frame ID → Go** box. For articulated objects, individual links can be selected and posed separately.

### Propagating annotations (AutoLabel menu)

Annotating every frame by hand is unnecessary — annotate keyframes and propagate:

- **Extrinsic extrapolation** — since objects sit still on the moving marker board, this transfers the annotated poses to other frames using the per-frame marker pose. Select target frames in the thumbnail dialog (click to toggle, `Shift+click` for ranges).
- **Tracking selection** — propagates poses with the BCOT/RBOT region-based tracker.
- **Bundle track** — segments the object with XMem and tracks it with BundleTrack; useful when objects are moved by hand.
- **Segmentation generation** — batch-renders instance masks for every frame that has a pose JSON.

### Segmentation tab

Masks rendered from poses can be touched up manually: mouse wheel changes the brush radius, left-click picks an instance ID, `Ctrl+left-drag` paints with the selected ID, `Ctrl+right-drag` erases. Press **Generate from Pose** to re-render the mask from the current poses.

## Step 7 — Mask Generation and Inspection

- `python postprocessing/generate_seg.py` — batch-renders instance masks for all three cameras from the saved pose JSONs (edit the `root` path; output goes to `cam{0,1,2}/mask/`).
- `python postprocessing/view_pose.py` — quick sanity check that re-projects an annotated object's vertices into all three views.
- `python postprocessing/calibrate_depth.py` — visualizes depth/RGB alignment as a fused heatmap.
- `python postprocessing/mesh2eval.py` — simplifies released meshes into the `models_eval` point-cloud models used for evaluation.

## Annotation File Format

One JSON per frame at `cam0/pose/<frame>.json`, keyed by object ID:

```json
{
  "1": {
    "name": "bottle_001",
    "path": "data/models_aligned_lowres/bottle/001.obj",
    "articulated": false,
    "m2c_R": [[...], [...], [...]],
    "m2c_t": [tx, ty, tz]
  },
  "2,4": {
    "name": "laptop_003",
    "path": "data/models_aligned_lowres/laptop/003",
    "articulated": true,
    "m2c_R": [[...], [...], [...]],
    "m2c_t": [tx, ty, tz],
    "links": {
      "2": {"name": "base", "m2c_R": [[...]], "m2c_t": [...]},
      "3": {"name": "screen", "m2c_R": [[...]], "m2c_t": [...]}
    }
  }
}
```

`m2c_R` / `m2c_t` are the model-to-camera rotation (3x3) and translation (meters) in the reference camera (`cam0`). Rigid objects use a single ID; articulated objects use an ID range `"first,last"` with per-link poses under `links`. Use `utils.io.load_anno` / `save_anno` to read and write this format. Poses for the other cameras follow from the extrinsics: `pose_cam_i = extrinsics[i] @ pose_cam_0`.

## Configuration

Global constants live in `utils/config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `FRAME_WIDTH`, `FRAME_HEIGHT` | 1280, 720 | capture resolution |
| `DEPTH_SCALE` | 1e-4 | depth-unit-to-meter scale used by the GUI |
| `MARKER_SIZE` | 100 | ArUco marker side length (mm) |
| `CAMS_TO_ANNO` | `[0]` | cameras shown in the annotation GUI |
| `MODEL_ROOT` | `data/models_aligned_lowres` | object database root |

## Notes

- The standalone postprocessing scripts use hardcoded `root = Path('data/videos/scene_X/video_Y')` paths near the top of their `__main__` blocks — edit these to point at the scene you are processing.
- `TFT_vs_Fund/` is a third-party MATLAB toolbox ([Julià & Monasse, *A Critical Review of the Trifocal Tensor Decomposition*](https://github.com/LauraFJulia/TFT_vs_Fund)) bundled for the extrinsic refinement step; see `TFT_vs_Fund/LICENSE.txt` for its license.
- Please open an issue if you hit problems setting up the pipeline — we are happy to help.
