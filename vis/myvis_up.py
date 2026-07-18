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


ids = [i for i in range(17, 10, -1)] + [i for i in range(21, 28)] + [i for i in range(47, 40, -1)] + [i for i in range(31, 38)]
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

def render_pass_optimized(output_dir, meshes_to_render, pos_data, rot_data, total_steps,
                          background_color, mesh_color,
                          view_rotation, view_translation,
                          is_removed_pass=False, transparency_config=None,
                          camera_parameters=None):

    pass_type = "transparent_overlay" if is_removed_pass else "opaque"
    print(f"Executing Optimized Pass: Rendering {pass_type} teeth...")
    
    os.makedirs(output_dir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)
    opt.mesh_show_back_face = True
    
    for mesh in meshes_to_render.values():
        vis.add_geometry(mesh)

    if camera_parameters:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)
        
    for g_idx in tqdm(range(total_steps), desc=f"Rendering {pass_type} pass"):
        for i, mesh_to_update in meshes_to_render.items():
            original_mesh = base_meshes[i] 
            mesh_copy = copy.deepcopy(original_mesh)
            
            step_index = g_idx
            q_wxyz = np.array(rot_data[step_index][i], dtype=np.float64)
            pos = pos_data[step_index][i]
            q_xyzw = q_wxyz[[1, 2, 3, 0]]
            
            mesh_copy.rotate(R_scipy.from_quat(q_xyzw).as_matrix(), center=(0,0,0)).translate(pos)
            mesh_copy.rotate(view_rotation, center=(0,0,0)).translate(view_translation)

            if is_removed_pass:
                if transparency_config and i in transparency_config['id_to_color_map']:
                    id_color = transparency_config['id_to_color_map'][i]
                    mesh_to_update.paint_uniform_color(id_color)
                else:
                    mesh_to_update.paint_uniform_color([0,0,0])
            
            mesh_to_update.vertices = mesh_copy.vertices
            mesh_to_update.vertex_normals = mesh_copy.vertex_normals
            vis.update_geometry(mesh_to_update)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(output_dir, f"step_{g_idx+1:04d}.png"), do_render=True)
        
    vis.destroy_window()

def composite_images_numpy_multi_alpha(pass1_dir, pass2_dir, final_dir, transparency_config):
    print("Compositing images with NumPy using multi-alpha strategy...")
    os.makedirs(final_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(pass1_dir) if f.endswith('.png')])
    if not files:
        print(f"Error: No base images found in {pass1_dir} to composite.")
        return

    final_transparent_color_np = (np.array(transparency_config['final_color']) * 255).astype(np.uint8)

    color_tuple_to_alpha = {
        tuple((np.array(color) * 255).astype(int)): alpha
        for color, alpha in transparency_config['color_to_alpha_map'].items()
    }

    for f in tqdm(files, desc="Compositing"):
        base_img_np = np.array(Image.open(os.path.join(pass1_dir, f)))
        overlay_path = os.path.join(pass2_dir, f)
        if not os.path.exists(overlay_path):
            continue
            
        overlay_img_np = np.array(Image.open(overlay_path))
        
        composite_img_np = base_img_np.copy()
        
        for color_tuple, alpha in color_tuple_to_alpha.items():
            # mask = np.all(overlay_img_np == color_tuple, axis=-1)
            mask = np.any(overlay_img_np > [10, 10, 10], axis=-1)
            
            if np.any(mask):
                new_pixels = (
                    base_img_np[mask] * (1 - alpha) + 
                    final_transparent_color_np * alpha
                ).astype(np.uint8)
                composite_img_np[mask] = new_pixels

        Image.fromarray(composite_img_np).convert("RGB").save(os.path.join(final_dir, f))

def render_headless_up(root_data_dir, vis_output_dir, json_path, mesh_color, transparent_teeth_spec):
    global base_meshes
    
    view_name = "up"
    final_output_path = os.path.join(vis_output_dir, view_name)
    pass1_opaque_dir = os.path.join(vis_output_dir, f"temp_pass_opaque_{view_name}")
    pass2_removed_dir = os.path.join(vis_output_dir, f"temp_pass_removed_{view_name}")
    
    base_meshes = getCoord(root_data_dir)
    pos_data, rot_data, remove_idx, total_steps = getTransform(json_path)
    
    DEFAULT_REMOVED_ALPHA = 0.15
    all_transparent_teeth = {}
    
    for tooth_id in remove_idx:
        all_transparent_teeth[tooth_id] = DEFAULT_REMOVED_ALPHA
        
    for tooth_id, alpha in transparent_teeth_spec.items():
        all_transparent_teeth[tooth_id] = alpha
    
    all_transparent_ids_set = set(all_transparent_teeth.keys())
    
    transparency_config = None
    if all_transparent_ids_set:
        print(f"'Up' View - Transparency enabled for teeth: {all_transparent_teeth}")
   
        transparency_config = {
            'id_to_color_map': {},
            'color_to_alpha_map': {},
            'final_color': [0.0176, 0.0726 * 2, 0.0334 * 2]
   
        }
        
        unique_alphas = sorted(list(set(all_transparent_teeth.values())))
        base_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        alpha_to_id_color = {}
        for i, alpha in enumerate(unique_alphas):
            alpha_to_id_color[alpha] = base_colors[i % len(base_colors)]

        for tooth_id, alpha in all_transparent_teeth.items():
            id_color = alpha_to_id_color[alpha]
            transparency_config['id_to_color_map'][tooth_id] =  [0.0176, 0.0726 * 2, 0.0334 * 2]
            transparency_config['color_to_alpha_map'][tuple(id_color)] = alpha

    
    r1 = R_scipy.from_euler('xyz', (np.pi, 0, np.pi)) 
    

    angle_x_tilt = np.deg2rad(0)  
    angle_y_tilt = np.deg2rad(0)  
    
    r2 = R_scipy.from_euler('xyz', (np.pi - angle_x_tilt, angle_y_tilt, 0))
    
    view_rotation = (r2 * r1).as_matrix()
    view_translation = [0, 0, 20]

    opaque_meshes = {}
    all_upper_meshes_for_view = {}
    all_upper_meshes_for_camera = []
    
    print("Preparing initial meshes for 'up' view rendering...")
    for i, mesh in enumerate(base_meshes):
        if mesh is None or i >= 14: continue
        
        mesh_for_scene = copy.deepcopy(mesh)
        q_wxyz = np.array(rot_data[0][i], dtype=np.float64)
        pos = pos_data[0][i]
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        
        mesh_for_scene.rotate(R_scipy.from_quat(q_xyzw).as_matrix(), center=(0,0,0)).translate(pos)
        mesh_for_scene.rotate(view_rotation, center=(0,0,0)).translate(view_translation)
        
        all_upper_meshes_for_view[i] = mesh_for_scene
        all_upper_meshes_for_camera.append(mesh_for_scene)
        
        if i not in all_transparent_ids_set:
            mesh_for_scene_opaque = copy.deepcopy(mesh_for_scene)
            mesh_for_scene_opaque.paint_uniform_color(mesh_color)
            opaque_meshes[i] = mesh_for_scene_opaque

    camera_parameters = None
    if all_upper_meshes_for_camera:
        print("Capturing a consistent camera view...")
        cam_setup_vis = o3d.visualization.Visualizer()
        cam_setup_vis.create_window(visible=False)
        for mesh in all_upper_meshes_for_camera:
            cam_setup_vis.add_geometry(mesh)
        cam_setup_vis.poll_events()
        cam_setup_vis.update_renderer()
        camera_parameters = cam_setup_vis.get_view_control().convert_to_pinhole_camera_parameters()
        cam_setup_vis.destroy_window()
        del cam_setup_vis
        print("Consistent camera view captured.")
    else:
        print("Warning: No upper teeth found. Aborting.")
        return

    print("\n--- Pass 1: Generating Base Images ---")
    if opaque_meshes:
        render_pass_optimized(
            output_dir=pass1_opaque_dir, 
            meshes_to_render=opaque_meshes,
            pos_data=pos_data, rot_data=rot_data, total_steps=total_steps,
            background_color=[1.0, 1.0, 1.0], mesh_color=mesh_color,
            view_rotation=view_rotation, view_translation=view_translation,
            camera_parameters=camera_parameters
        )
    else:
        print("All teeth are transparent. Creating blank white background images for Pass 1.")
        os.makedirs(pass1_opaque_dir, exist_ok=True)
        blank_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        for g_idx in tqdm(range(total_steps), desc="Creating blank images"):
            Image.fromarray(blank_img).save(os.path.join(pass1_opaque_dir, f"step_{g_idx+1:04d}.png"))
    
    if all_transparent_ids_set:
        print("\n--- Pass 2 & Compositing: Handling Transparent Teeth ---")
        render_pass_optimized(
            output_dir=pass2_removed_dir,
            meshes_to_render=all_upper_meshes_for_view,
            pos_data=pos_data, rot_data=rot_data, total_steps=total_steps,
            background_color=[0.0, 0.0, 0.0], mesh_color=mesh_color,
            view_rotation=view_rotation, view_translation=view_translation,
            is_removed_pass=True,
            transparency_config=transparency_config,
            camera_parameters=camera_parameters
        )
        composite_images_numpy_multi_alpha(pass1_opaque_dir, pass2_removed_dir, final_output_path, transparency_config)
    else:
        print("\n--- Finalizing: No transparent teeth ---")
        if os.path.exists(final_output_path): shutil.rmtree(final_output_path)
        os.makedirs(final_output_path, exist_ok=True)
        for f in os.listdir(pass1_opaque_dir):
            if f.endswith('.png'):
                src_path = os.path.join(pass1_opaque_dir, f)
                dst_path = os.path.join(final_output_path, f)
                Image.open(src_path).convert('RGB').save(dst_path)
    
    print("\nCleaning up temporary directories...")
    if os.path.exists(pass1_opaque_dir): shutil.rmtree(pass1_opaque_dir)
    if os.path.exists(pass2_removed_dir): shutil.rmtree(pass2_removed_dir)
    
    print(f"🎉 All done! 'Up' view images saved to: {final_output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Advanced headless rendering for the 'up' view.")
    parser.add_argument("--target_sample_name", type=str, required=True)
    parser.add_argument("--json_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_gt", type=lambda x: (str(x).lower() == 'true'), required=True)
    args = parser.parse_args()  

    transparent_teeth_spec_up = {
        # 2: 0.1,
        # 3:0.5,
        # 4: 0.15,
        # 5: 0.15,
        # 6: 0.15,
        # 7: 0.15,
        # 8: 0.15,
        # 9: 0.15,
        # 10: 0.15,
        # 11:0.55,
        # 3: 0.8,
    }

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
            output_dir_base = os.path.join('render_gt', client, 'gt')
        else:
            output_dir_base = os.path.join('result_render', client, args.model_name)
        json_file = os.path.join(args.json_root, f"{client}.json")

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        render_headless_up(
            root_data_dir=rootdir,
            vis_output_dir=output_dir_base,
            json_path=json_file,
            mesh_color=color,
            transparent_teeth_spec=transparent_teeth_spec_up
        )


