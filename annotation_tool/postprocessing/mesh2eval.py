import os
import pymeshlab
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_model(i):
    model_in_path = f'data/models/obj_{i:06d}.ply'
    model_out_path = f'data/models_eval/obj_{i:06d}.ply'
    
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
    # indices = [100, 142, 222, 345, 480, 500, 528, 530, 531, 545, 546, 557, 558, 559, 560, 575, 576, 577, 579, 582, 583, 588, 593, 595] + list(range(610, 693))
    # indices = [480]
    # indices = list(range(545, 693))  # do not simplify articulated objects
    indices = list(range(148, 693))
    
    # Using all available CPUs, but you can adjust the number if necessary
    with Pool(processes=cpu_count() // 2) as pool:
        list(tqdm(pool.imap(process_model, indices), total=len(indices)))

if __name__ == "__main__":
    main()
