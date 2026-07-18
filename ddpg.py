import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import json
from utils import utils_np
import open3d as o3d
from env.ortho_env import OrthoEnv
from tqdm import tqdm
from utils.prioritized_memory import Memory
from utils.running_mean_std import Scale,RunningMeanStdState
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2024
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    """the learning rate of the optimizer"""
    buffer_size: int = int(1500000)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    policy_frequency: int = 50
    plot_frequency: int = 5000
    """the frequency of training policy (delayed)"""
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    train_dir='data/train_data'
    test_dir='data/test_data'
    n_head=4
    n_layers=2
    dropout=0.1
    hidden_dim=256
    state_dim=144
    q_learning_rate=1e-4
    final_q_learning_rate = 1e-6
    actor_learning_rate=1e-5
    n_step=3
    q_alpha=0.5
    collision_punishment=10
    alpha_trans=100
    alpha_rotation=100
    q_weight_decay=1e-2
    actor_weight_decay=1e-4
    thred_completion=0.3
    save_path=None
    lambda_mask=100
    alpha_L2=100
    temperature = 0.5
    anneal_lr=True
    expert_weight=5
    weight_sum=4
    alpha_smooth=200
    collision_tolerance=0.20
    grad_clip_norm: float = 1.0 
    is_stage2=True



up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}  


diff=np.load("data/max_movement.npy").astype(np.float32)
teeth_ids=ids
teeth_ids_pos=[oid[id] for id in teeth_ids]
max_diff=diff[teeth_ids_pos,:]
max_diff_tensor=torch.from_numpy(max_diff)
shape_dir="data/feature_pointr_108d"#shape 
hull_dir="data/hull_512" 
teeth_mean=np.load("data/mean.npy")
shape_mean=np.load("data/shape_mean.npy")



n_teeth = len(ids)
up_pos = {id: i for i, id in enumerate(up_ids)}
down_pos = {id: i for i, id in enumerate(down_ids)}
d_matrix = np.zeros((n_teeth, n_teeth), dtype=np.float32)
for i in range(n_teeth):
    for j in range(n_teeth):
        id_i = ids[i]
        id_j = ids[j]
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




import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Linear):
    
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)  
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight) 
        init.zeros_(m.bias)  
    elif isinstance(m, nn.Embedding):
        init.xavier_normal_(m.weight)  


class RelativePositionBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.mu = nn.Parameter(torch.Tensor(n_heads))
        with torch.no_grad():
            self.mu.normal_(0.0, 0.02).abs_()

    def forward(self, rho_matrix):
        mu_positive = F.softplus(self.mu)
        # (n_heads, 1, 1) * (1, n_teeth, n_teeth) -> (n_heads, n_teeth, n_teeth)
        bias_per_head = mu_positive.view(-1, 1, 1) * rho_matrix.unsqueeze(0)
        return bias_per_head

class QNetwork(nn.Module):
    def __init__(self, n_teeth=28, state_dim=18, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1):
        super(QNetwork, self).__init__()
        self.n_teeth = n_teeth
        self.n_heads = n_heads
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + action_dim
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
            nn.Linear(hidden_dim, 1)
        )
        self.pos_embedding = nn.Embedding(n_teeth, hidden_dim)  
        self.fc2 = nn.Linear(self.n_teeth, 1) 

        self.relative_bias_generator = RelativePositionBias(n_heads)
        self.register_buffer('rho_matrix', rho_matrix_tensor)

        self.apply(weights_init)

    def forward(self, state, action):
        batch_size = state.shape[0]
        
        shape = self.shape_embedding(state[:, :, 36:])
        shape = self.shape_ln(shape)
        shape = self.relu(shape)
        
        state_features = state[:, :, :36]
        
        x = torch.cat([state_features, shape, action], dim=-1)
        x = self.embedding(x)
        
        pos_embed = self.pos_embedding(torch.arange(self.n_teeth, device=x.device))
        x = x + pos_embed.unsqueeze(0)
        
        relative_pos_bias = self.relative_bias_generator(self.rho_matrix.to(x.device))
        expanded_bias = relative_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        final_bias = expanded_bias.reshape(batch_size * self.n_heads, self.n_teeth, self.n_teeth)
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x, mask=final_bias)
        x = x.permute(1, 0, 2)
        
        q_per_tooth = self.fc_out(x).squeeze(-1)
        q_value = self.fc2(q_per_tooth)
        
        return q_value

class Actor(nn.Module):
    def __init__(self, n_teeth=28, state_dim=135, action_dim=9, shape_dim=108,
                 n_heads=4, n_layers=2, hidden_dim=64, dropout=0.1, action_scale=1.0):
        super(Actor, self).__init__()
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
        
        self.apply(weights_init)

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
        mask = self.sigmod(mask/args.temperature) 
        mask_action = action * mask
        
        return mask_action, mask


def create_env(sampleFileName=None,teeth_ids=None,shape_dir=None,hull_dir=None,args=None,shape_id=None,remove_idx=None):
    step_paths=[f for f in os.listdir(sampleFileName) if f.startswith("step")]
    seq_state=[]
    teeth_is_null=[]
    for step in range(1,len(step_paths)+1):
        json_path=os.path.join(sampleFileName,f'step{step}.json')
        teeth28=[]
        with open(json_path,'r') as file:
            data=json.load(file)
            for id in teeth_ids:
                if f'{id}' in data.keys() and (oid[id] not in remove_idx) :
                    x,y,z,qx,qy,qz,qw=data[f'{id}']
                    teeth28.append([x,y,z,qw,qx,qy,qz])
                else:
                    teeth28.append([0]*7)
                    if step==1:
                        teeth_is_null.append(len(teeth28)-1)
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
            teeth_shape.append(shape)
        else:
            teeth_shape.append(shape_mean[oid[id]].reshape(1,-1))
    teeth_shape=np.concatenate(teeth_shape)
    teeth_shape_tensor=torch.from_numpy(teeth_shape).to(device=device).to(dtype=torch.float32)
    if shape_id is None:
        teeth_shape_list.append(teeth_shape_tensor)
        sampletoid[sampleFileName]=len(teeth_shape_list)-1
        shape_id=len(teeth_shape_list)-1
    first_step=np.concatenate((seq_state[0],seq_state[seq_state.shape[0]-1]),axis=1)
    env=OrthoEnv(first_step.astype(np.float32),seq_state.astype(np.float32),convex_hull=convex_hull,teeth_ids=teeth_ids,
                collision_punishment=args.collision_punishment,alpha_trans=args.alpha_trans,alpha_angle=args.alpha_rotation,
                shape_id=shape_id,alpha_smooth=args.alpha_smooth,collision_tolerance=args.collision_tolerance,stage2=args.is_stage2)
    return env



def make_env_list(filename_list,args=None,shape_id=None):
    env_list=[]
    if len(filename_list)==1:
        remove_idx_path=os.path.join(filename_list[0],"remove_idx.json")
        if os.path.exists(remove_idx_path):
            with open(remove_idx_path, 'r') as f:
                remove_idx = json.load(f)
                print(remove_idx_path)
        else:
            remove_idx=[]
        env=create_env(filename_list[0],teeth_ids,shape_dir=shape_dir,hull_dir=hull_dir,args=args,shape_id=shape_id,remove_idx=remove_idx)
        env.reset()
        env_list.append(env)
        return env_list
    
    for filename in tqdm(filename_list):
        remove_idx_path=os.path.join(filename,"remove_idx.json")
        if os.path.exists(remove_idx_path):
            with open(remove_idx_path, 'r') as f:
                remove_idx = json.load(f)
                print(remove_idx_path)
        else:
            remove_idx=[]
        env=create_env(filename,teeth_ids,shape_dir=shape_dir,hull_dir=hull_dir,args=args,remove_idx=remove_idx)
        env.reset()
        env_list.append(env)
        
    return env_list

def load_expert_data(env_list):
    for env in tqdm(env_list[:],desc="main"):
        obs=env.reset()
        done=False
        n_done=False
        data_list=[]
        while(not done and not n_done):
            n_reward,n_obs,n_done,n_discount,action=env.default_action()
            next_obs,reward,done,_,reward_dict,collision_record,_=env.step(action,is_load_expert=True)
          
            action=(action/max_diff).clip(-1.0,1.0)
            data=(obs,action,next_obs,reward,done,n_obs,n_reward,n_done,n_discount,env.shape_id)
            obs=next_obs
            rb.add(data,expert_data=True)
            data_list.append(data)
      
        obs_np=np.array([data[0] for data in data_list])
        state_normalizer.update(obs_np)

        reward_np=np.array([data[3] for data in data_list]+[data[6] for data in data_list])
        reward_scaler.update(reward_np)

        env.bvh_objects.clear()
        env.close()
        env.reset()
        del env

@torch.no_grad()
def create_agent_data(env,total_reward):
    obs=np.concatenate((env.state.copy(),env.last_action),axis=1)
    init_obs=obs.copy()
    n_reward=0
    data=[]
    for step in range(args.n_step):
        actions,mask=actor(state_normalizer.normalize(torch.from_numpy(obs).to(device),[env.shape_id]))
        actions += torch.normal(0, torch.ones_like(max_diff_tensor)*get_noise(global_step)).to(device)
        actions = actions.cpu().numpy().clip(-1,1)
        next_obs,reward,done,_,reward_dict,_,truncated=env.step(actions*max_diff,global_step=global_step)

        obs=next_obs
        if args.gamma>0:
            n_reward+=(args.gamma**step)*reward
        else:
            n_reward+=reward
        if len(data) == 0:
            data.extend([init_obs,actions,next_obs,reward,done])
        total_reward+=reward
        
    data.extend([next_obs,n_reward,done,args.gamma**(step+1),env.shape_id])
    normal_state=state_normalizer.normalize(torch.from_numpy(data[2]).to(device),[env.shape_id])
    next_state_actions,_=target_actor(normal_state)
    qf1_next_target = qf1_target(normal_state, next_state_actions)
    next_q_value = torch.Tensor(reward_scaler.normalize(data[3])).to(device).flatten() + (1 - data[4]) * args.gamma * (qf1_next_target).view(-1)
    n_normal_state=state_normalizer.normalize(torch.Tensor(data[5]).to(device),[env.shape_id])
    next_n_actions,_=target_actor(n_normal_state)
    qf1_n_target=qf1_target(n_normal_state,next_n_actions)
    next_n_q_value=torch.Tensor(reward_scaler.normalize(data[6])).to(device).flatten()+(1-data[7])*data[8]*(qf1_n_target).view(-1)
    next_q_value=args.q_alpha*next_q_value+(1-args.q_alpha)*next_n_q_value



    qf1_a_values = qf1(state_normalizer.normalize(torch.from_numpy(data[0]).to(device),[env.shape_id]), torch.Tensor(data[1]).to(device)).view(-1)
    td_error = torch.abs(qf1_a_values-next_q_value).cpu().numpy()

    ####action
    data[1]=data[1].squeeze(0)
    rb.add(tuple(data),error=td_error.item())

    length=env.T
    if done or truncated:
        env.reset()
    return done or truncated,total_reward,length,reward_dict


def plot(x,y,label,xlable,ylabel,save_path):
    plt.figure(figsize=(10,8))
    plt.plot(x,y,label=label)
    plt.xlabel(xlabel=xlable)
    plt.ylabel(ylabel=ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(save_path)
    plt.close()

test_reward_list=[[]]
test_len_list=[[]]
@torch.no_grad()
def test():
    for i in range(len(test_env_list)):
        env=test_env_list[i]
        obs=env.reset()
        done=False
        test_total_reward=0
        discount=1
        truncated=False
        while (not done) and (not truncated):
            actions,_=actor(state_normalizer.normalize(torch.Tensor(obs).to(device),[env.shape_id]))
            actions = actions.cpu().numpy()
            next_obs,reward,done,_,reward_dict,_,truncated=env.step(actions*max_diff)
            obs=next_obs
            test_total_reward+=reward
        test_reward_list[i].append(max(-1000,test_total_reward))
        test_len_list[i].append(env.T)
    test_x=[i*args.plot_frequency for i in range(len(test_reward_list[i]))]
    plt.figure(figsize=(10,8))
    for i in range(len(test_reward_list)):
        plt.plot(test_x,test_reward_list[i],label=f"sample_{i}")
    plt.xlabel("step")
    plt.ylabel("eposide reward")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_root,f"{global_step}_reward_episode.png"))
    plt.close()



def get_lambda_mask(cur_step):
    return args.lambda_mask

def get_noise(cur_step):
    return max(0.001,args.exploration_noise-cur_step/1e7)

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}_beta"
    save_root=f"result/{len(teeth_ids)}_collpun_{args.collision_punishment}_\
lr_{args.actor_learning_rate}_gamma_{args.gamma}_noise_{args.exploration_noise}_{args.hidden_dim}_\
tran_{args.alpha_trans}_{args.alpha_rotation}_smooth_{args.alpha_smooth}_{args.anneal_lr}_{args.collision_tolerance}"
    os.makedirs(save_root,exist_ok=True)
   
    sampletoid={}
    teeth_shape_list=[]


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    filename_list=[os.path.join(args.train_dir,f) for f in os.listdir(args.train_dir) if f.startswith("C")]
    env_list=make_env_list(filename_list,args)
    filename_list_test=[os.path.join(args.test_dir,f) for f in os.listdir(args.test_dir) if f.startswith("C")]
    test_env_list=make_env_list(filename_list_test,args)

    state_normalizer=RunningMeanStdState(shape=(28,108+9),teeth_pos=teeth_ids_pos,teeth_shape_list=teeth_shape_list,device=device)
    reward_scaler=Scale(shape=(1,))
    rb=Memory(args.buffer_size,device=device,state_normalizer=state_normalizer,reward_scaler=reward_scaler,weight_expert=args.expert_weight)
    
    load_expert_data(env_list)

    del(env_list)

    actor=Actor(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    
    target_actor=Actor(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                    n_heads=args.n_head,n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    
    qf1=QNetwork(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    qf1_target=QNetwork(n_teeth=len(teeth_ids), state_dim=args.state_dim, action_dim=9,
                n_heads=args.n_head, n_layers=args.n_layers,
                hidden_dim=args.hidden_dim, dropout=args.dropout).to(device).to(torch.float32)
    if args.save_path is not None:
        actor_state,q_state=torch.load(args.save_path,map_location=device)
        actor.load_state_dict(actor_state,strict=False)
        qf1.load_state_dict(q_state,strict=False)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.AdamW(qf1.parameters(), lr=args.q_learning_rate, 
                              weight_decay=args.q_weight_decay,betas=(0.9,0.999))
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_learning_rate, 
                                weight_decay=args.actor_weight_decay,betas=(0.9,0.999))

    done,n_done=False,False
    env_i=random.randint(0,len(filename_list)-1)
    env_cur=make_env_list([filename_list[env_i]],args,shape_id=sampletoid[filename_list[env_i]])[0]
    env_cur.reset()
    total_reward,total_trans_reward,total_rotation_reward,total_collision_punishment,total_smooth_punishment=0,0,0,0,0
    total_reward_list,trans_list,rotation_list,q_loss_list,actor_loss_list,reward_list,collision_punishment_list,smooth_punishment_list=[],[],[],[],[],[],[],[]
    num_punishment_list,total_num_punishment=[],0
    L1_loss_list,L2_loss_list=[],[]
    for global_step in tqdm(range(args.total_timesteps)):
        if args.anneal_lr:
            q_lrnow = args.final_q_learning_rate + 0.5 * (args.q_learning_rate - args.final_q_learning_rate) * \
              (1 + math.cos(math.pi * global_step / args.total_timesteps))
            actor_lrnow=q_lrnow*0.1
            q_optimizer.param_groups[0]["lr"] = q_lrnow
            actor_optimizer.param_groups[0]["lr"]=actor_lrnow
        done,total_reward,length,reward_dict=create_agent_data(env_cur,total_reward)
        if done:
            env_i=random.randint(0,len(filename_list)-1)
            env_cur=make_env_list([filename_list[env_i]],args,shape_id=sampletoid[filename_list[env_i]])[0]
            env_cur.reset()
            total_reward_list.append(total_reward)
            total_reward=0

        batch,idxs,is_weight=rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions,_=target_actor(batch.next_obs)
            qf1_next_target=qf1_target(batch.next_obs,next_state_actions)
            next_q_value = batch.reward.flatten() + (1 - batch.done.flatten()) * args.gamma * (qf1_next_target).view(-1)

            n_state_actions,_=target_actor(batch.n_obs)
            qf1_n_target=qf1_target(batch.n_obs,n_state_actions)
            n_q_value=batch.n_reward.flatten()+(1-batch.n_done.flatten())*batch.n_discount*(qf1_n_target).view(-1)

            q_value=args.q_alpha*next_q_value+(1-args.q_alpha)*n_q_value

        qf1_a_values=qf1(batch.obs,batch.action).view(-1)
        td_error=torch.abs(qf1_a_values-q_value).detach().cpu().numpy().flatten()
        qf1_loss=(F.mse_loss(qf1_a_values,q_value,reduction='none')*torch.from_numpy(is_weight).to(device)).mean()
        q_optimizer.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_norm_(qf1.parameters(), args.grad_clip_norm)
        q_optimizer.step()

        for i in range(len(idxs)):
            rb.update(idxs[i],td_error[i].item())


        
        if global_step % args.policy_frequency == 0:
            action,mask=actor(batch.obs)
         
           
            upper = F.relu((mask[:, :14, :].reshape(-1, 14).sum(dim=-1) - args.weight_sum)).mean()
            lower = F.relu((mask[:, 14:, :].reshape(-1, 14).sum(dim=-1) - args.weight_sum)).mean()
            L1_loss = get_lambda_mask(global_step) * (upper + lower)
            
            L2_loss=args.alpha_L2*((mask*(1-mask)).mean())
            
            actor_loss = -qf1(batch.obs, action).mean()+L1_loss+L2_loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        total_rotation_reward+=reward_dict["rotation_reward"]
        total_trans_reward+=reward_dict["trans_reward"]
        total_collision_punishment+=reward_dict["collision_punishment"]
        total_smooth_punishment+=reward_dict["smooth_punishment"]
        total_num_punishment+=reward_dict["num_punishment"]
        if (global_step)%args.plot_frequency==0:
            print(reward_dict)
            trans_list.append(total_trans_reward/(global_step+1))
            rotation_list.append(total_rotation_reward/(global_step+1))
            collision_punishment_list.append(total_collision_punishment/(global_step+1))
            smooth_punishment_list.append(total_smooth_punishment/(global_step+1))
            # num_punishment_list.append(total_num_punishment/(global_step+1))
            q_loss_list.append(qf1_loss.item())
            actor_loss_list.append(actor_loss.item())
            reward_list.append(batch.reward.mean().item())
            L1_loss_list.append(L1_loss.item())
            L2_loss_list.append(L2_loss.item())

            x=[i*args.plot_frequency for i in range(len(trans_list))]
            plot(x,trans_list,label="tran",xlable="step",ylabel="tran_reward",save_path=os.path.join(save_root,f"{global_step:04d}_trans.png"))
            plot(x,rotation_list,label="angle",xlable="step",ylabel="rotation_reward",save_path=os.path.join(save_root,f"{global_step:04d}_rotation.png"))
            plot(x,collision_punishment_list,label="collision_punishment",xlable="step",ylabel="collision_punishmet",save_path=os.path.join(save_root,f"{global_step:04d}_collision.png"))
            plot(x,q_loss_list,label="q_loss",xlable="step",ylabel="q_loss",save_path=os.path.join(save_root,f"{global_step:04d}_q_loss.png"))
            plot(x,actor_loss_list,label="actor_loss",xlable="step",ylabel="actor_loss",save_path=os.path.join(save_root,f"{global_step:04d}_actor_loss.png"))
            plot(x,reward_list,label="reward_mean",xlable="step",ylabel="step_reward",save_path=os.path.join(save_root,f"{global_step:04d}_reward_mean.png"))
            plot(x,smooth_punishment_list,label="smooth",xlable="step",ylabel="smooth",save_path=os.path.join(save_root,f"{global_step:04d}_smooth.png"))
            plot(x,L1_loss_list,label="L1_loss",xlable="step",ylabel="L1_loss",save_path=os.path.join(save_root,f"{global_step:04d}_L1_loss.png"))
            plot(x,L2_loss_list,label="L2_loss",xlable="step",ylabel="L2_loss",save_path=os.path.join(save_root,f"{global_step:04d}_L2_loss.png"))
            test()

        if args.save_model and global_step%99999==0:
            model_path = f"{save_root}/{global_step}.cleanrl_model"
            torch.save((actor.state_dict(), qf1.state_dict()), model_path)
       



   

 

    
    
