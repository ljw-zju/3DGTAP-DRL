import numpy as np
import torch
class Scale:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m2 = self.var * self.count
        m2_batch = batch_var * batch_count
        new_var = (m2 + m2_batch + np.square(delta) * self.count * batch_count / total_count) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return x.reshape(-1).astype(np.float32)
        

    def denormalize(self, x):
        std = np.sqrt(self.var)
        return x * (std + 1e-8) + self.mean

    def state_dict(self):
     
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict):
       
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class RunningMeanStdState:
    def __init__(self, epsilon=1e-4, shape=(),teeth_pos=[],teeth_shape_list=None,device=None):
        
        self.obs_mean=torch.from_numpy(np.load("data/mean.npy").astype(np.float32)).to(device)
        self.obs_std=torch.from_numpy(np.load("data/std.npy").astype(np.float32)).to(device)
        self.shape_mean=torch.from_numpy(np.load("data/shape_mean.npy").astype(np.float32)).to(device)
        self.shape_std=torch.from_numpy(np.load("data/shape_std.npy").astype(np.float32)).to(device)
        self.obs_mean=self.obs_mean[teeth_pos,:]
        self.obs_std=self.obs_std[teeth_pos,:]
        self.shape_mean=self.shape_mean[teeth_pos,:]
        self.shape_std=self.shape_std[teeth_pos,:]
        self.teeth_pos=teeth_pos
        if teeth_shape_list is not None:
            teeth_shape_Tensor=torch.stack(teeth_shape_list)
            self.teeth_shape_tensor=teeth_shape_Tensor[:,teeth_pos,:]
        self.count = epsilon

    def init_shape(self,teeth_shape_list):
        teeth_shape_Tensor=torch.stack(teeth_shape_list)
        self.teeth_shape_tensor=teeth_shape_Tensor[:,self.teeth_pos,:]

    def update(self, x):
        pass


    def normalize(self, x,shape_id_list):
        """
        Standardize the data using the current running mean and standard deviation.

        Args:
            x (ndarray): The data to be standardized.

        Returns:
            ndarray: The standardized data.
        """ 
        if len(x.shape)==2:
            x=x.unsqueeze(0)
        cur_state=x[:,:,:9]
        tar_state=x[:,:,9:18]
        diff=tar_state-cur_state
        shape=self.teeth_shape_tensor[shape_id_list,:,:].clone()
        cur_state=(cur_state-self.obs_mean)/(self.obs_std+1e-8)
        diff=(diff)/(self.obs_std+1e-8)
        tar_state=(tar_state-self.obs_mean)/(self.obs_std+1e-8)
        shape=(shape-self.shape_mean)/(self.shape_std+1e-8)
        res=torch.cat((cur_state,diff,tar_state,x[:,:,-9:],shape),axis=-1)
      
        return res  

  


