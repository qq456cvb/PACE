import os
import pymeshlab
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def process_model(i, in_dir='data/models', out_dir='data/models_eval'):
    model_in_path = f'{in_dir}/obj_{i:06d}.ply'
    model_out_path = f'{out_dir}/obj_{i:06d}.ply'
    
    print(model_out_path)
    # Check if the model has already been processed
    # if os.path.exists(model_out_path):
    #     return

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_in_path)
    ms.generate_resampled_uniform_mesh(cellsize=pymeshlab.Percentage(.25), offset=pymeshlab.AbsoluteValue(0.), mergeclosevert=True, 
                               discretize=False, multisample=False, absdist=False)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=0, targetperc=0.025, qualitythr=0.5,
                                                       preserveboundary=True, boundaryweight=1,
                                                       preservenormal=True, preservetopology=False,
                                                       optimalplacement=True, planarquadric=True,
                                                       qualityweight=False, autoclean=True, selected=False)
    ms.save_current_mesh(model_out_path)
    

def simplify(i):
    
    model_in_path = f'data/models/obj_{i:06d}.ply'
    model_out_path = f'data/models_simplified/obj_{i:06d}.ply'
    
    # Check if the model has already been processed
    # if os.path.exists(model_out_path):
    #     return
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_in_path)
    # ms.generate_resampled_uniform_mesh(cellsize=pymeshlab.Percentage(0.25), offset=pymeshlab.Percentage(0), mergeclosevert=True, 
    #                                    discretize=False, multisample=True, absdist=False)
    # ms.compute_texcoord_transfer_vertex_to_wedge()
    for _ in range(1):  # simplify three times
      # ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=0, targetperc=0.025, qualitythr=0.5,
      #                                             preserveboundary=True, boundaryweight=1,
      #                                             preservenormal=True, 
      #                                             # preservetopology=False,
      #                                             optimalplacement=True, planarquadric=True,
      #                                             # qualityweight=False, 
      #                                             # autoclean=True, 
      #                                             selected=False)
    #   ms.simplification_quadric_edge_collapse_decimation_with_texture()
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetperc=0.1, preserveboundary=True, preservenormal=True)
    
    # ms.compute_texcoord_transfer_wedge_to_vertex()
    ms.save_current_mesh(model_out_path)
    
    
def copy_mesh(i):
    
    model_in_path = f'data/models/obj_{i:06d}.ply'
    model_out_path = f'data/models_simplified/obj_{i:06d}.ply'
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(model_in_path)
    ms.save_current_mesh(model_out_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Resample and simplify meshes into evaluation models.')
    parser.add_argument('--in-dir', default='data/models', help='input mesh directory containing obj_XXXXXX.ply files')
    parser.add_argument('--out-dir', default='data/models_eval', help='output directory for evaluation meshes')
    parser.add_argument('--start', type=int, default=148, help='first object id (inclusive)')
    parser.add_argument('--end', type=int, default=693, help='last object id (exclusive)')
    parser.add_argument('--workers', type=int, default=cpu_count() // 2, help='number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    indices = list(range(args.start, args.end))
    
    worker = partial(process_model, in_dir=args.in_dir, out_dir=args.out_dir)
    with Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap(worker, indices), total=len(indices)))

if __name__ == "__main__":
    main()
