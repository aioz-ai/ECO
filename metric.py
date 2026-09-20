import os
from pathlib import Path
import numpy as np
import argparse
import open3d as o3d
from gen_human_meshes import gen_human_meshes
import json
from tqdm import tqdm
import torch 
import torch.nn.functional as F
from mesh_to_sdf import mesh_to_voxels, mesh_to_sdf,  get_surface_point_cloud
from utils import create_o3d_mesh_from_vertices_faces, trimesh_from_o3d
import trimesh


   
# calculate the non-collision score mentioned in the paper https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Generating_3D_People_in_Scenes_Without_People_CVPR_2020_paper.pdf
# code is reproduced from https://github.com/yz-cnsdqz/PSI-release/blob/master/utils/utils_eval_collision_habitat.py and https://github.com/yz-cnsdqz/PSI-release/blob/master/demo.ipynb
# def cal_non_collision(human_verts, sdf, cam_ext, sdf_data):
#     cam_ext = torch.tensor(cam_ext, dtype=torch.float32, device=device)

#     human_verts = torch.tensor(human_verts, dtype=torch.float32, device=device).unsqueeze(0) # [b, 655, 3]
#     bs, n_verts, _ = human_verts.shape 
#     cam_ext_batch = cam_ext.repeat(bs,1,1)
#     s_grid_min_batch = torch.tensor(sdf_data['min'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
#     s_grid_max_batch = torch.tensor(sdf_data['max'], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

#     human_verts = GeometryTransformer.verts_transform(human_verts, cam_ext_batch)
#     norm_verts_batch = (human_verts - s_grid_min_batch) / (s_grid_max_batch - s_grid_min_batch) *2 -1

#     sdf = sdf.repeat(bs,1,1,1)
#     body_sdf_batch = F.grid_sample(sdf.unsqueeze(1), 
#                                             norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
#                                             padding_mode='border')
#     if body_sdf_batch.lt(0).sum().item() < 1: # if the number of negative sdf entries is less than one, score is maximum 
#         score = torch.tensor(1, dtype=torch.float32, device=device)
#     else:
#         score = (body_sdf_batch > 0).sum()
#         score = score*1.0 / (n_verts * bs)

    # return score
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fitting_results_path", type=str, help="Path to the fitting results of some motion sequence")
    parser.add_argument("--vertices_path", type=str, help="Path to human vertices of some motion sequence")

    args = parser.parse_args()
    input_dir = Path(args.fitting_results_path)
    vertices_path = Path(args.vertices_path)
    seq_name = input_dir.stem

    # load best obj to form scene
    res_dir = input_dir / 'fit_best_obj'
    obj_mesh_list = []
    obj_mesh_dir = []
    for obj_class_dir in res_dir.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            best_obj_id = best_obj_json['best_obj_id']
            best_obj_path = obj_dir / best_obj_id / 'opt_best.obj'
            print(best_obj_path)
            obj_mesh_dir.append(best_obj_path)
            obj_mesh = o3d.io.read_triangle_mesh(str(best_obj_path))
            obj_mesh.compute_vertex_normals()
            obj_mesh_list.append(obj_mesh)

    human_vertices = np.load(open(vertices_path, "rb"))

    seq_score = 0


    # calculate non-collision score between human motion and each obj, then avarage all the scores
    for obj_idx in range(len(obj_mesh_list)): 
        each_obj = obj_mesh_list[obj_idx]
        # score for each object is average from all human frames
        non_coll = 0
        mesh = trimesh_from_o3d(each_obj)
        n_verts = human_vertices.shape[1] #655 

        # this code computes the signed distance values for human vertices as query points 
        # code from https://github.com/marian42/mesh_to_sdf/blob/master/mesh_to_sdf/__init__.py
        point_cloud = get_surface_point_cloud(mesh)
        for i in range(human_vertices.shape[0]):
            query_points = human_vertices[i,:,:]
            # calculate sdfs
            sdf  = point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)
            # keep positive sdf values
            pos_sdf = (sdf>0).sum()
            non_coll +=  pos_sdf/n_verts
            print(f"frame {i} - {pos_sdf/n_verts}")
        non_collision_score = non_coll*1.0/human_vertices.shape[0]
        seq_score+= non_collision_score
        print(f"Non collision score for obj {obj_mesh_dir[obj_idx]}: {non_collision_score}")
    print(f"Non collision for sequence {seq_name}: {seq_score/len(obj_mesh_list)}")
