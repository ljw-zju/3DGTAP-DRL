import os
import numpy as np
import json
import cv2
# print("设置 OPEN3D_CPU_RENDERING=true")
# os.environ['OPEN3D_CPU_RENDERING'] = 'true'
import copy
import glob
from scipy.spatial.transform import Rotation as R
from utils.utils_np import quat_to_matrix9D
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# 
from utils.prioritized_memory_copy import Memory
from utils.running_mean_std_test import Scale,RunningMeanStdState
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw, ImageFont

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser()
parser.add_argument("--target_sample_name", type=str, required=True)
parser.add_argument("--model_name",type=str,required=True)
parser.add_argument("--temperature",type=float,default=0.5)
parser.add_argument("--model_path",type=str,required=True)
parser.add_argument("--diff_path",type=str,required=True)
parser.add_argument("--relative_pos", type=lambda x: (str(x).lower() == 'true'), required=True)
parser.add_argument("--is_eval",type=lambda x: (str(x).lower() == 'true'), required=True)
args = parser.parse_args()

error_cases=['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002776709.json', 'C01002778059.json', 'C01002778116.json', 'C01002781299.json', 'C01002782256.json', 'C01002782469.json', 'C01002785002.json', 'C01002787969.json', 'C01002788634.json', 'C01002791605.json', 'C01002791650.json', 'C01002792909.json', 'C01002793517.json', 'C01002795801.json', 'C01002796969.json', 'C01002799164.json', 'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 'C01002747516.json', 'C01002774628.json', 'C01002785507.json', 'C01002796914.json', 'C01002803328.json', 'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 'C01002763390.json', 'C01002775708.json', 'C01002789185.json', 'C01002801630.json', 'C01002814870.json', 'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C01002780423.json', 'C01002801146.json', 'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json']

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)]
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
all_ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(all_ids)}
#####################

diff=np.load(f"{args.diff_path}").astype(np.float32)
teeth_ids=all_ids
teeth_ids_pos=[oid[id] for id in teeth_ids]
max_diff=diff[teeth_ids_pos,:]
max_diff_tensor=torch.from_numpy(max_diff)
shape_dir="data/feature_pointr_108d"
hull_dir="data/hull_512"
teeth_mean=np.load("data/mean.npy")
shape_mean=np.load("data/shape_mean.npy")
def weight_initialization(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for weights
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  # Initialize biases to zero

n_teeth = len(all_ids)
up_pos = {id: i for i, id in enumerate(up_ids)}
down_pos = {id: i for i, id in enumerate(down_ids)}
d_matrix = np.zeros((n_teeth, n_teeth), dtype=np.float32)
for i in range(n_teeth):
    for j in range(n_teeth):
        id_i = all_ids[i]
        id_j = all_ids[j]
        if (id_i in up_ids and id_j in up_ids):
            dist = abs(up_pos[id_i] - up_pos[id_j]) 
            d_matrix[i, j] = max(0, dist)
        elif (id_i in down_ids and id_j in down_ids):
            dist = abs(down_pos[id_i] - down_pos[id_j])
            d_matrix[i, j] = max(0, dist)
        else:
            d_matrix[i, j] = 99 

rho_matrix = -d_matrix
rho_matrix_tensor = torch.from_numpy(rho_matrix)

class RelativePositionBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.mu = nn.Parameter(torch.Tensor(n_heads))
        with torch.no_grad():
            self.mu.normal_(0.0, 0.02).abs_()

    def forward(self, rho_matrix):
        mu_positive = F.softplus(self.mu)
        bias_per_head = mu_positive.view(-1, 1, 1) * rho_matrix.unsqueeze(0)
        return bias_per_head



class Actor(nn.Module):
    def __init__(self, n_teeth=28, state_dim=135, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1, action_scale=1.0):
        super(Actor, self).__init__()
        self.n_teeth = n_teeth
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        input_dim = state_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.shape_embedding = nn.Linear(shape_dim, shape_dim)
        self.shape_ln = nn.LayerNorm(shape_dim)
        self.relu=nn.ReLU()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout),
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  
        )

        self.weight_out=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,1),
  
        )
        self.sigmod=nn.Sigmoid()

        self.pos_embedding = nn.Embedding(n_teeth, hidden_dim)


    def forward(self, state):
        """
        state: (batch_size, n_teeth, state_dim)
        """
        shape = self.shape_embedding(state[:, :, 36:])
        shape = self.shape_ln(shape)
        shape=self.relu(shape)

        state_features = state[:, :, :36]

        x = torch.cat([state_features, shape], dim=-1) 
        # x=state_features
        x = self.embedding(x)

        pos_embed= self.pos_embedding(torch.tensor(list(range(self.n_teeth)),device=state.device))
        x = x + pos_embed.unsqueeze(0)


        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        action = self.fc_out(x)
      
        mask=self.weight_out(x)
        mask=self.sigmod(mask/temperature)
        mask_action=action*mask

        return mask_action,mask



class Actor_pos(nn.Module):
    def __init__(self, n_teeth=28, state_dim=135, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1, action_scale=1.0):
        super(Actor_pos, self).__init__()
        self.n_teeth = n_teeth
        self.n_heads = n_heads
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        input_dim = state_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.shape_embedding = nn.Linear(shape_dim, shape_dim)
        self.shape_ln = nn.LayerNorm(shape_dim)
        self.relu = nn.ReLU()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout),
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.weight_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.sigmod = nn.Sigmoid()

        self.pos_embedding = nn.Embedding(n_teeth, hidden_dim)
        
        self.relative_bias_generator = RelativePositionBias(n_heads)
        self.register_buffer('rho_matrix', rho_matrix_tensor)
        

    def forward(self, state):
        batch_size = state.shape[0]
        shape = self.shape_embedding(state[:, :, 36:])
        shape = self.shape_ln(shape)
        shape = self.relu(shape)
        
        state_features = state[:, :, :36]
        
        x = torch.cat([state_features, shape], dim=-1)
        x = self.embedding(x)
        
        pos_embed = self.pos_embedding(torch.arange(self.n_teeth, device=state.device))      
        x = x + pos_embed.unsqueeze(0)

        relative_pos_bias = self.relative_bias_generator(self.rho_matrix.to(x.device))
        expanded_bias = relative_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        final_bias = expanded_bias.reshape(batch_size * self.n_heads, self.n_teeth, self.n_teeth)
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x, mask=final_bias)
        x = x.permute(1, 0, 2)
        
        action = self.fc_out(x) 
        mask = self.weight_out(x)
        mask = self.sigmod(mask/temperature) 
        mask_action = action * mask
        
        return mask_action, mask


import json
from utils import utils_np
from env.ortho_env_test import OrthoEnv
import open3d as o3d
def create_env(sampleFileName=None,teeth_ids=None,shape_dir=None,hull_dir=None,args=None,remove_idx=None):
    step_paths=[f for f in os.listdir(sampleFileName) if f.startswith("step")]
    seq_state=[]
    teeth_is_null=[]
    for step in range(1,len(step_paths)+1):
        json_path=os.path.join(sampleFileName,f'step{step}.json')
        teeth28=[]
        with open(json_path,'r') as file:
            data=json.load(file)
            for id in teeth_ids:
                if f'{id}' in data.keys() and (oid[id] not in remove_idx):
                    x,y,z,qx,qy,qz,qw=data[f'{id}']
                    teeth28.append([x,y,z,qw,qx,qy,qz])
                else:
                    teeth28.append([0]*7)
                    if step==1:
                        teeth_is_null.append(len(teeth28)-1)
                        if f'{id}' in data.keys():
                            x,y,z,qx,qy,qz,qw=data[f'{id}']
                            remove_tran[oid[id]]=[x,y,z]
                            remove_rot[oid[id]]=[qw,qx,qy,qz]
                
        seq_state.append(teeth28)

    seq_state=np.array(seq_state)
    xyz=seq_state[:,:,:3]
    rotation=seq_state[:,:,3:]
    rotation=utils_np.quat_to_matrix9D(rotation.reshape(-1,4))
    rotation=utils_np.matrix9D_to_6D(rotation).reshape(-1,len(teeth_ids),6)
    seq_state=np.concatenate((xyz,rotation),axis=2)
    seq_state[:,teeth_is_null,:]=teeth_mean[teeth_is_null,:]
    convex_hull=[]
    for id in teeth_ids:
        hull_path=os.path.join(hull_dir,f"{sampleFileName.split('/')[-1]}_{id}.ply")
        if os.path.exists(hull_path) and (oid[id] not in remove_idx):
            hull = o3d.io.read_triangle_mesh(hull_path)
            convex_hull.append(hull)
        else:
            convex_hull.append(None)
    teeth_shape=[]
    for id in teeth_ids:
        shape_path=os.path.join(shape_dir,f"{sampleFileName.split('/')[-1]}-{id}.npy")
        if os.path.exists(shape_path) and (oid[id] not in remove_idx):
            shape=np.load(shape_path)
            # print(shape.shape)
            teeth_shape.append(shape)
        else:
            # print(shape_mean[oid[id]].shape)
            # teeth_shape.append(np.array([[0]*108]))
            teeth_shape.append(shape_mean[oid[id]].reshape(1,-1))
    # print(teeth_shape)
    teeth_shape=np.concatenate(teeth_shape)
    first_step=np.concatenate((seq_state[0],seq_state[seq_state.shape[0]-1],teeth_shape),axis=1)
    env=OrthoEnv(first_step.astype(np.float32),seq_state.astype(np.float32),convex_hull=convex_hull,teeth_ids=teeth_ids,collision_punishment=1,remove_idx=remove_idx)
    print("env,",remove_idx)
    return env



def get_pos(obs):
    teeth_tran=obs[:,:3]
    teeth_rot_mat=utils_np.matrix6D_to_9D(obs[:,3:9])
    quat=utils_np.matrix9D_to_quat(teeth_rot_mat)
    return teeth_tran.tolist(),quat.tolist()

@torch.no_grad()
def test(env):
    tran_list,rot_list=[],[]
    obs=env.reset()
    tran,rot=get_pos(obs.copy())
    tran_list.append(tran)
    rot_list.append(rot)
    for step in range(200):
        actions,mask=actor(state_normalizer.normalize(torch.Tensor(obs).to(device)))
        actions = actions.cpu().numpy()
        mask=mask.cpu().numpy()
        # print(print(mask.shape),mask[0][remove_idx])
        next_obs,reward,done,_,reward_dict,collision_record,tran_errors,rot_errors=env.step(actions*max_diff)
        obs=next_obs
        tran,rot=get_pos(obs.copy())
        tran_list.append(tran)
        rot_list.append(rot)
        if done:
            break
    return tran_list,rot_list

if __name__ == "__main__":
    center,extent,bounds=None,None,None
    target_sample_name=args.target_sample_name
    phase=args.model_name
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.backends.cudnn.deterministic =True

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    
    if args.relative_pos:
        actor=Actor_pos(n_teeth=28, state_dim=144, action_dim=9,
                n_heads=4, n_layers=2,
                hidden_dim=256, dropout=0.1).to(device).to(torch.float32)
        print("*"*50)
    else:
        actor=Actor(n_teeth=28, state_dim=144, action_dim=9,
                n_heads=4, n_layers=2,
                hidden_dim=256, dropout=0.1).to(device).to(torch.float32)
    cal_diff=True
    temperature = args.temperature
    actor_state,_=torch.load(f"{args.model_path}",
                             map_location=device)
    actor.load_state_dict(actor_state)
    if args.is_eval:
        actor.eval()
    state_normalizer=RunningMeanStdState(shape=(28,108+9),teeth_pos=teeth_ids_pos,device=device)

    with open("remove_idx_summary.json",'r') as f:
        remove_idx_dict=json.load(f)
    remove_idx=remove_idx_dict[target_sample_name]
    print("create_test_json",remove_idx)
    remove_rot,remove_tran={},{}
    env=create_env(os.path.join("data/test_data",target_sample_name),teeth_ids,shape_dir=shape_dir,hull_dir=hull_dir,args=None,remove_idx=remove_idx)

    tran_list,rot_list=test(env)
    print(remove_rot)
    for key,value in remove_tran.items():
        for i in range(len(tran_list)):
            tran_list[i][key][0]=value[0]
            tran_list[i][key][1]=value[1]
            tran_list[i][key][2]=value[2]
    for key,value in remove_rot.items():
        for i in range(len(rot_list)):
            rot_list[i][key][0]=value[0]
            rot_list[i][key][1]=value[1]
            rot_list[i][key][2]=value[2]
            rot_list[i][key][3]=value[3]
    data_to_write={
            "positions":tran_list,
            "rotations":rot_list,
            "remove_idx":remove_idx
        }

    filename=target_sample_name
    output_dir=os.path.join("result_json",f"{args.model_name}")
    os.makedirs(output_dir,exist_ok=True)
    file_path=os.path.join(output_dir,f"{filename.split('/')[-1]}.json")
    with open(file_path,'w',encoding='utf-8') as json_file:
        json.dump(data_to_write,json_file,indent=4,ensure_ascii=False)    

    