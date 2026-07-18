import numpy as np
import math
import gym
from gym import spaces
from utils import utils_np
import open3d as o3d
import fcl




up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}  



transform_matrix_original_to_fcl = np.array([[-1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, -1]], dtype=np.float32)

class OrthoEnv(gym.Env):
    def __init__(self,first_step,seq_state,convex_hull=None,alpha_angle=100,alpha_trans=100,
                 collision_tolerance=0.30,completion_prize=100,gamma=0.9,
                 teeth_ids=None,collision_punishment=1,n_step=3,max_step=200,thre_completion=0.3,remove_idx=None):

        """
        first_step:(NUM_TEEth,126)
        seq_state:(seq_len,NUM_TEETH,126)
        convex_hull(list):(NUM_TEETH,)
        """
        assert len(teeth_ids)==len(convex_hull)
        assert len(teeth_ids)==first_step.shape[0]
        assert len(teeth_ids)==seq_state.shape[1]

        self.first_step=first_step
        self.state=first_step.copy()
        self.last_state=None
        self.seq_state=seq_state
        self.last_action=None
        self.seq_len=self.seq_state.shape[0]
        self.convex_hull=convex_hull
        self.alpha_angle=alpha_angle
        self.alpha_trans=alpha_trans
        self.collision_tolerance=collision_tolerance
        self.completion_prize=completion_prize
        self.gamma=gamma
        self.teeth_ids=teeth_ids
        self.num_teeth=len(teeth_ids)
        self.collision_punishment=collision_punishment
        self.n_step=n_step
        self.max_step=max_step
        self.thre_completion=thre_completion
        self.collision_record=[]
        self.bvh_objects=[]
        self.T=0
        self.collision_dict={}
        self.remove_idx=remove_idx

       
        teeth_ids_pos=[oid[id] for id in self.teeth_ids]

        

        max_diff=np.load('data/max_movement.npy')
        self.max_diff=max_diff[teeth_ids_pos,:]
        self.action_space=spaces.Box(low=-1*self.max_diff,high=self.max_diff,shape=(self.num_teeth,9),dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=self.first_step.shape, dtype=np.float32)
        

    def reset(self,):
        self.state=self.first_step.copy()
        self.last_state=None
        self.last_action=np.zeros((self.num_teeth,9)).astype(np.float32)
        self.T=0
        return np.concatenate((self.state.copy(),self.last_action),axis=1)
    
    def step(self,action,is_load_expert=False,global_step=0):
        action=action.reshape(self.num_teeth,9).astype(np.float32)
        if len(self.remove_idx):
            action[self.remove_idx]=0
        self.collision_record=[]
        done=False
        reward=0
        self.T+=1
        self.last_state=self.state.copy()
        self.state[:,:9]+=action
       
        current_positions=self.state[:,:3]
        last_positions=self.last_state[:,:3]
        trans_reward=self.calculate_translation_reward(current_positions,last_positions)

        current_rotations=self.state[:,3:9]
        last_rotations=self.last_state[:,3:9]
        angle_reward=self.calculate_rotation_reward(current_rotations,last_rotations)

        num_punishment=self.calculate_num_punishment(action)

        if self.T==1:
            smooth_punishment=np.abs(0)
        else:
            smooth_punishment=self.calculate_smooth_puhishmnet(action.copy(),self.last_action)
        self.last_action=action


        if self.collision_punishment == 0:
            collision_punishment=0
        elif is_load_expert:
            collision_punishment=self.collision_dict[self.T]
        else:
            collision_punishment=self.calculate_collision_penalty(self.state[:,:9])

    

        reward=trans_reward+angle_reward+collision_punishment+smooth_punishment+num_punishment

    
        done=self.check_done()

        if done:
            reward=reward+self.completion_prize
        if self.T>self.max_step:
            done=True
        reward_dict={
            "trans_reward":trans_reward.item(),
            "rotation_reward":angle_reward.item(),
            "collision_punishment":collision_punishment,
            "smooth_punishment":smooth_punishment.item(),
            "total_reward":reward.item(),
            "num_punishment":num_punishment
        }
        return np.concatenate((self.state,self.last_action),axis=1),reward,done,False,reward_dict,self.collision_record,self.trans_error,self.angle_errors


    def calculate_num_punishment(self,action):
        return 0

    def calculate_smooth_puhishmnet(self,action,last_action):
        tran_punishmet=np.sum(np.abs(action[:,:3]-last_action[:,:3]))
        rotation_punishment=np.sum(np.abs(action[:,3:9]-last_action[:,3:9]))

        return -(tran_punishmet*1.2)/(self.num_teeth)



    def calculate_translation_reward(self,current_positions,last_positions,k=16.0):
        target_positions = self.first_step[:, 9:12]
        last_distance=np.square(np.linalg.norm(last_positions-target_positions,axis=1))
        current_distance = np.square(np.linalg.norm(current_positions - target_positions, axis=1))
        reward = -np.log((1+k*current_distance)/(1+k*last_distance))
        reward = np.mean(reward)
        return reward*self.alpha_trans
    
    def calculate_rotation_reward(self,current_rotations,last_rotations,k=16.0):
    
        target_rotations=self.first_step[:,12:18]
        last_angle_error=np.square(self.get_angle_error_np(last_rotations,target_rotations))
        current_angle_error=np.square(self.get_angle_error_np(current_rotations,target_rotations))
        reward=-np.log((1+k*current_angle_error)/(1+k*last_angle_error))
        reward=np.mean(reward)
        return reward*self.alpha_angle


    def get_angle_error_np(self,a, b):
        n = a.shape[0]
        res = np.zeros(n)

        zero_rows_mask = np.all(a == 0, axis=1)
        res[zero_rows_mask] = 0.0


        non_zero_rows_mask = ~zero_rows_mask

        a_non_zero = a[non_zero_rows_mask]
        b_non_zero = b[non_zero_rows_mask]

        if a_non_zero.size > 0:
            a_9d = utils_np.matrix6D_to_9D(a_non_zero)
            b_9d = utils_np.matrix6D_to_9D(b_non_zero)

            rm = np.matmul(np.swapaxes(a_9d, -2, -1), b_9d)
            tr = np.trace(rm, axis1=-2, axis2=-1)
            angle_error = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))
            res[non_zero_rows_mask] = angle_error

        return res
    
    def get_bvh_objects(self):
        assert self.num_teeth == len(self.convex_hull)
        for i in range(len(self.convex_hull)):
            obj=self.convex_hull[i]
            if obj is None:
                self.bvh_objects.append(None)
            else:
             
                hull=obj
                hull_vertices = np.asarray(hull.vertices)
                hull_triangles = np.asarray(hull.triangles)
                bvh = fcl.BVHModel()
                bvh.beginModel()

                for tri in hull_triangles:
                    p1 = hull_vertices[tri[0]]
                    p2 = hull_vertices[tri[1]]
                    p3 = hull_vertices[tri[2]]
                    bvh.addTriangle(
                        np.asarray(p1, dtype=np.float32),
                        np.asarray(p2, dtype=np.float32),
                        np.asarray(p3, dtype=np.float32)
                    )
                bvh.endModel()
                self.bvh_objects.append(bvh)    

    def calculate_collision_penalty(self,state):
        reward=0
        all_trans=state[:,:3]
        all_rot=utils_np.matrix6D_to_9D(state[:,3:9])
        if len(self.bvh_objects)==0:
            self.get_bvh_objects()
        for i in range(1,len(self.bvh_objects)):
            bvh1=self.bvh_objects[i-1]
            bvh2=self.bvh_objects[i]
            if bvh1 is None or bvh2 is None:
                continue
            trans1=transform_matrix_original_to_fcl@all_trans[i-1]
            trans2=transform_matrix_original_to_fcl@all_trans[i]
            rot1=transform_matrix_original_to_fcl@all_rot[i-1]
            rot2=transform_matrix_original_to_fcl@all_rot[i]
            transform1=fcl.Transform(rot1,trans1)
            transform2=fcl.Transform(rot2,trans2)
            obj1=fcl.CollisionObject(bvh1,transform1)
            obj2=fcl.CollisionObject(bvh2,transform2)
            request=fcl.CollisionRequest(num_max_contacts=1,enable_contact=True)
            result=fcl.CollisionResult()
            fcl.collide(obj1,obj2,request,result)
            if result.is_collision:
                if result.contacts:
                    depth=[contact.penetration_depth for contact in result.contacts]
                    penetration_depth=sum(depth)/(len(depth))
                    if penetration_depth>0.3:
                        reward-=self.collision_punishment
        return reward

    def check_done(self):
        self.trans_error=np.linalg.norm(self.state[:, :3] - self.state[:, 9:12], axis=1).max()
        self.angle_errors=self.get_angle_error_np(self.state[:, 3:9], self.state[:, 12:18]).max()*180
        if np.linalg.norm(self.state[:, :3] - self.state[:, 9:12], axis=1).max() < 0.3 and \
            self.get_angle_error_np(self.state[:, 3:9], self.state[:, 12:18]).max()*90 < np.pi*1.5:
            return True
        return False

    def default_action(self,):
        n_reward=0
        T=self.T
        reward_list=[]
        n_done=False
        for step in range(self.n_step):
            assert T+step+1<self.seq_len
            last_positions=self.seq_state[T+step,:,:3]
            last_rotations=self.seq_state[T+step,:,3:9]
            current_positions=self.seq_state[T+step+1,:,:3]
            current_rotations=self.seq_state[T+step+1,:,3:9]
            reward_trans=self.calculate_translation_reward(current_positions,last_positions)
            reward_angle=self.calculate_rotation_reward(current_rotations,last_rotations)
            if T+step+1 in self.collision_dict:
                collision_punishment=self.collision_dict[T+step+1]
            else:
               
                collision_punishment=0
                self.collision_dict[T+step+1]=collision_punishment

            cur_action=self.seq_state[T+step+1,:,:9]-self.seq_state[T+step,:,:9]
            if T+step>0:
                last_action=self.seq_state[T+step,:,:9]-self.seq_state[T+step-1,:,:9]
                smooth_punishment=self.calculate_smooth_puhishmnet(cur_action,last_action)
            else:
                smooth_punishment=0
            cur_reward=reward_trans+reward_angle+collision_punishment+smooth_punishment
            reward_list.append(cur_reward)
            if self.seq_len-1==T+step+1:
                reward_list[-1]=self.completion_prize+reward_list[-1]
                n_done=True
                break
        
        ####n_obs###
        n_obs=np.concatenate((self.state.copy(),cur_action),axis=1)
        n_obs[:,:9]=self.seq_state[min(self.seq_len-1,T+self.n_step)]

        ###n_reward###
        for i in range(len(reward_list)):
            if self.gamma>0:
                n_reward+=reward_list[i]*(self.gamma**i)
            else:
                n_reward+=reward_list[i]
        
        ######n_discount#######
        n_discount=self.gamma**(len(reward_list))

        ####action  (numteeth,9)####
        action=self.seq_state[T+1]-self.seq_state[T]

        return n_reward,n_obs,n_done,n_discount,action

