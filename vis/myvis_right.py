# vis/myvis_right.py

import numpy as np
import os
import json
import copy
import argparse
from pyvirtualdisplay import Display
import open3d as o3d

# Define the standard order of tooth IDs
ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] \
    + [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
oid = {id: i for i, id in enumerate(ids)}

def getTransform(json_path):
    with open(json_path) as fh:
        data = json.load(fh)
        pos_data = data["positions"]
        rot_data = data["rotations"]
        remove_idx = data.get("remove_idx", [])
        step_num = len(pos_data)
        return pos_data, rot_data, remove_idx, step_num

def getCoord(path):
    """
    Loads tooth mesh models from STL files.
    """
    coords = []
    print(f"Loading mesh models from: {path}/")
    for id in ids:
        model_path = f'{path}/{id}._Root.stl'
        if not os.path.exists(model_path):
            coords.append(None)
            continue
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()
        coords.append(mesh)
    return coords

def render_headless(root_data_dir, vis_output_dir, json_path, mesh_color, view_rot, view_name, exclude_indices):
    """
    Renders tooth animation in a headless environment and saves frames as images.

    Args:
        root_data_dir (str): Root directory containing STL models.
        vis_output_dir (str): Output directory to save rendered images.
        json_path (str): Path to the JSON file with position and rotation data.
        mesh_color (list): Color for the tooth models.
        view_rot (tuple): Fixed rotation (rx, ry, rz) to set the camera view.
        view_name (str): Name of the view for the output subdirectory.
        exclude_indices (list): List of tooth indices to exclude from rendering.
    """
    print(f"Starting headless rendering for '{view_name}' view...")
    
    # --- 1. Create Output Directory ---
    output_path = os.path.join(vis_output_dir, view_name)
    os.makedirs(output_path, exist_ok=True)
    
    # --- 2. Load Data ---
    base_meshes = getCoord(root_data_dir)
    pos_data, rot_data,remove_idx ,total_steps = getTransform(json_path)

    # --- 3. Initialize Open3D Headless Renderer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) 
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0]) # White background
    opt.mesh_show_back_face = True

    # --- 4. Initialize Scene (Frame 0) ---
    scene_meshes = []
    for i, base_mesh in enumerate(base_meshes):
        if base_mesh is None or i in exclude_indices or i in remove_idx:
            scene_meshes.append(None)
            continue
            
        mesh_copy = copy.deepcopy(base_mesh)
        q = np.array(rot_data[0][i], dtype=np.float64)
        
        # Using quaternion order (x, y, z, w) as requested
        q_o3d = np.array([q[0], q[1], q[2], q[3]])

        R = mesh_copy.get_rotation_matrix_from_quaternion(q_o3d)
        mesh_copy.rotate(R, center=(0, 0, 0))
        mesh_copy.translate(pos_data[0][i])
        
        R_view = mesh_copy.get_rotation_matrix_from_xyz(view_rot)
        mesh_copy.rotate(R_view, center=(0, 0, 0))
        
        mesh_copy.paint_uniform_color(mesh_color)
        
        vis.add_geometry(mesh_copy)
        scene_meshes.append(mesh_copy)
    
    # --- 5. Render Frame by Frame ---
    print(f"Rendering {total_steps} frames...")
    for g_idx in range(total_steps):
        print(f"  - Frame {g_idx+1}/{total_steps}")
        
        for i, base_mesh in enumerate(base_meshes):
            if base_mesh is None or i in exclude_indices or i in remove_idx:
                continue

            mesh_to_update = scene_meshes[i]
            mesh_copy = copy.deepcopy(base_mesh)
            
            q = np.array(rot_data[g_idx][i], dtype=np.float64)
            pos = pos_data[g_idx][i]
            q_o3d = np.array([q[0], q[1], q[2], q[3]])

            R = mesh_copy.get_rotation_matrix_from_quaternion(q_o3d)
            mesh_copy.rotate(R, center=(0, 0, 0))
            mesh_copy.translate(pos)

            R_view = mesh_copy.get_rotation_matrix_from_xyz(view_rot)
            mesh_copy.rotate(R_view, center=(0, 0, 0))
            
            mesh_to_update.vertices = mesh_copy.vertices
            mesh_to_update.compute_vertex_normals()
            vis.update_geometry(mesh_to_update)
            
        vis.poll_events()
        vis.update_renderer()
        
        image_name = f"step_{g_idx+1:04d}.png"
        image_path = os.path.join(output_path, image_name)
        vis.capture_screen_image(image_path, do_render=True)

    # --- 6. Clean Up ---
    vis.destroy_window()
    print(f"Rendering complete. Images saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless rendering for the 'right' view.")
    parser.add_argument("--target_sample_name", type=str, required=True)
    parser.add_argument("--json_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_gt", type=lambda x: (str(x).lower() == 'true'), required=True)
    args = parser.parse_args()  

    with Display(visible=0, size=(1920, 1080)):
        client = args.target_sample_name
        
        color=[0.6, 0.6, 0.6] if args.is_gt else [0.65, 0.4, 0.5]
     
        
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
            output_dir_base = os.path.join('render_gt', client, args.model_name if not args.is_gt else 'gt')
        else:
            output_dir_base = os.path.join('result_render', client, args.model_name if not args.is_gt else 'gt')
        json_file = os.path.join(args.json_root, f"{client}.json")

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
   
        # Define the rotation for the RIGHT view
        right_view_rotation = (np.pi/2, 0, -np.pi/2)
        
        # Define the indices of teeth to be EXCLUDED from rendering
        # This is the mirror of the left view's exclusion
        exclude_range = [i for i in range(0, 5)] + [i for i in range(14, 19)]
        
        render_headless(
            root_data_dir=rootdir,
            vis_output_dir=output_dir_base,
            json_path=json_file,
            mesh_color=color,
            view_rot=right_view_rotation,
            view_name="right",
            exclude_indices=exclude_range
        )