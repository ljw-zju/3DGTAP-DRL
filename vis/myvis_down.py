


import numpy as np
import os
import json
import copy
import argparse
from pyvirtualdisplay import Display
import open3d as o3d
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm
from PIL import Image
import shutil


ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] \
    + [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]



base_meshes = []

def getTransform(json_path):
    with open(json_path) as fh:
        data = json.load(fh)
        pos_data = data["positions"]
        rot_data = data["rotations"]
        remove_idx = data.get("remove_idx", [])
        step_num = len(pos_data)
        return pos_data, rot_data, remove_idx, step_num

def getCoord(path):
    coords = []
    for id_val in ids:
        model_path = f'{path}/{id_val}._Root.stl'
        if not os.path.exists(model_path):
            coords.append(None)
            continue
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()
        coords.append(mesh)
    return coords

def render_pass_optimized(output_dir, meshes_to_render, pos_data, rot_data, total_steps, background_color, mesh_color, removed_idx_set, is_removed_pass=False):
  
    pass_type = "removed" if is_removed_pass else "opaque"
    print(f"Executing Optimized Pass: Rendering {pass_type} teeth...")
    
    os.makedirs(output_dir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.mesh_show_back_face = True
    

    for mesh in meshes_to_render.values():
        vis.add_geometry(mesh)
        
    view_rotation = R_scipy.from_euler('xyz', (np.pi, 0, np.pi)).as_matrix()

    for g_idx in tqdm(range(total_steps), desc=f"Rendering {pass_type} pass"):
        for i, mesh_to_update in meshes_to_render.items():
            original_mesh = base_meshes[i] 
            mesh_copy = copy.deepcopy(original_mesh)
            
        
            step_index = g_idx
            # ---
            
            q_wxyz = np.array(rot_data[step_index][i], dtype=np.float64)
            pos = pos_data[step_index][i]
            q_xyzw = q_wxyz[[1, 2, 3, 0]]
            
            mesh_copy.rotate(R_scipy.from_quat(q_xyzw).as_matrix(), center=(0,0,0)).translate(pos)
            mesh_copy.rotate(view_rotation, center=(0,0,0)).translate([0,0,20])

            if is_removed_pass:
                if i in removed_idx_set:
                 
                    mesh_to_update.paint_uniform_color([0.0176, 0.0726*2, 0.0334*2]) 
                    pass
             
                else:
               
                    mesh_to_update.paint_uniform_color(mesh_color)

          
            mesh_to_update.vertices = mesh_copy.vertices
            mesh_to_update.vertex_normals = mesh_copy.vertex_normals
            vis.update_geometry(mesh_to_update)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_dir, f"step_{g_idx+1:04d}.png"), do_render=True)
        
    vis.destroy_window()

def composite_images_numpy(pass1_dir, pass2_dir, final_dir, alpha):

    print("Compositing images with NumPy for high performance...")
    os.makedirs(final_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(pass1_dir) if f.endswith('.png')])

    for f in tqdm(files, desc="Compositing"):
        base_img_np = np.array(Image.open(os.path.join(pass1_dir, f)))
        overlay_img_np = np.array(Image.open(os.path.join(pass2_dir, f)))
        
        mask = np.any(overlay_img_np > [10, 10, 10], axis=-1)
        
        composite_img_np = np.where(
            mask[..., np.newaxis],
            (base_img_np * (1 - alpha) + overlay_img_np * alpha).astype(np.uint8),
            base_img_np
        )
        
        Image.fromarray(composite_img_np).save(os.path.join(final_dir, f))

def render_headless(root_data_dir, vis_output_dir, json_path, mesh_color):

    global base_meshes
    
    view_name = "down"
    
    final_output_path = os.path.join(vis_output_dir, view_name)
    pass1_opaque_dir = os.path.join(vis_output_dir, "temp_pass_opaque")
    pass2_removed_dir = os.path.join(vis_output_dir, "temp_pass_removed")
    
    base_meshes = getCoord(root_data_dir)
    pos_data, rot_data, remove_idx, total_steps = getTransform(json_path)
    removed_idx_set = set(remove_idx)
    
    view_rotation = R_scipy.from_euler('xyz', (np.pi, 0, np.pi)).as_matrix()

    
    opaque_meshes = {}
  
    all_lower_meshes = {}

    for i, mesh in enumerate(base_meshes):
        if mesh is None or i < 14: continue
        
        mesh_for_scene = copy.deepcopy(mesh)
        
        q_wxyz = np.array(rot_data[0][i], dtype=np.float64)
        pos = pos_data[0][i]
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        mesh_for_scene.rotate(R_scipy.from_quat(q_xyzw).as_matrix(), center=(0,0,0)).translate(pos)
        mesh_for_scene.rotate(view_rotation, center=(0,0,0)).translate([0,0,20])
        
        all_lower_meshes[i] = mesh_for_scene
        if i not in removed_idx_set :
            mesh_for_scene_opaque = copy.deepcopy(mesh_for_scene)
            mesh_for_scene_opaque.paint_uniform_color(mesh_color)
            opaque_meshes[i] = mesh_for_scene_opaque


    if opaque_meshes:
        render_pass_optimized(
            output_dir=pass1_opaque_dir, 
            meshes_to_render=opaque_meshes,
            pos_data=pos_data, rot_data=rot_data, total_steps=total_steps,
            background_color=[1.0, 1.0, 1.0],
            mesh_color=mesh_color,
            removed_idx_set=removed_idx_set,
            is_removed_pass=False
        )

  
    if removed_idx_set:
        print("Removed teeth detected. Proceeding to second pass and composition.")
        render_pass_optimized(
            output_dir=pass2_removed_dir,
            meshes_to_render=all_lower_meshes,
            pos_data=pos_data, rot_data=rot_data, total_steps=total_steps,
            background_color=[0.0, 0.0, 0.0],
            mesh_color=mesh_color,
            removed_idx_set=removed_idx_set,
            is_removed_pass=True
        )
        
        composite_images_numpy(pass1_opaque_dir, pass2_removed_dir, final_output_path, alpha=0.15)
        
        print("Cleaning up temporary directories...")
       
    else:
        print("Info: No removed teeth detected. Finalizing.")
        if os.path.exists(final_output_path): shutil.rmtree(final_output_path)
        shutil.copytree(pass1_opaque_dir, final_output_path)

    
    print(f"🎉 All done! Final (buggy) images saved to: {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless rendering for the 'down' view with optimized performance.")
    parser.add_argument("--target_sample_name", type=str, required=True)
    parser.add_argument("--json_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_gt", type=lambda x: (str(x).lower() == 'true'), required=True)
    args = parser.parse_args()  

    with Display(visible=0, size=(1920, 1080)):
        client = args.target_sample_name
        color = [0.6, 0.6, 0.6] if args.is_gt else [0.65, 0.4, 0.5]
      
        mesh_root_list=["3D_mesh"]
        rootdir = None
        for mesh_root in mesh_root_list:
            path_to_check = os.path.join(mesh_root, client, "models")
            if os.path.isdir(path_to_check):
                rootdir = path_to_check
                break
        
        if rootdir is None: 
            raise FileNotFoundError(f"Could not find model directory for sample {client}")

        if args.is_gt:
            output_dir_base = os.path.join('render_gt', client, 'gt')
        else:
            output_dir_base = os.path.join('result_render', client, args.model_name)
        json_file = os.path.join(args.json_root, f"{client}.json")

        if not os.path.exists(json_file): 
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        render_headless(
            root_data_dir=rootdir,
            vis_output_dir=output_dir_base,
            json_path=json_file,
            mesh_color=color
        )


