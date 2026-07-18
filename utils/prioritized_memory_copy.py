import random
import numpy as np
from utils.SumTree import SumTree
import torch

class Data:
    def __init__(self,obs,action,next_obs,reward,done,n_obs,n_reward,n_done,n_discount):
        self.obs=obs
        self.action=action
        self.next_obs=next_obs
        self.reward=reward
        self.done=done
        self.n_obs=n_obs
        self.n_reward=n_reward
        self.n_done=n_done
        self.n_discount=n_discount



class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.0001

    def __init__(self, capacity,device=None,state_normalizer=None,reward_scaler=None,weight_expert=1):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.state_normalizer=state_normalizer
        self.reward_scaler=reward_scaler
        self.device=device
        self.weight_expert=weight_expert


    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, sample,error=1.0,expert_data=False,is_no_pority=False):
        if expert_data:
            self.tree.add(np.abs(self.weight_expert),sample,expert_data=expert_data)
        else:
            if is_no_pority:
                self.tree.add(np.abs(self.weight_expert), sample,expert_data=expert_data)
            else:
                p = self._get_priority(error)
                self.tree.add(p, sample,expert_data=expert_data)
            # self.tree.add(p, sample,expert_data=expert_data)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta).astype(np.float32)
        is_weight /= is_weight.max()


        ####batch -> tensor
        obs,action,next_obs,reward,done,n_obs,n_reward,n_done,n_discount,shape_id_list=[],[],[],[],[],[],[],[],[],[]
        for data in batch:
            obs.append(data[0])
            action.append(data[1])
            next_obs.append(data[2])
            reward.append(data[3])
            done.append(1.0 if data[4] else 0.0)
            n_obs.append(data[5])
            n_reward.append(data[6])
            n_done.append(1.0 if data[7] else 0.0)
            n_discount.append(data[8])
            shape_id_list.append(data[9])
        obs=self.state_normalizer.normalize(torch.from_numpy(np.array(obs)).to(self.device),shape_id_list)
        action=torch.from_numpy(np.array(action)).to(self.device)
        next_obs=self.state_normalizer.normalize(torch.from_numpy(np.array(next_obs)).to(self.device),shape_id_list)
        reward=torch.from_numpy(self.reward_scaler.normalize(np.array(reward))).to(self.device,dtype=torch.float32)
        done=torch.from_numpy(np.array(done)).to(self.device,torch.float32)
        n_obs=self.state_normalizer.normalize(torch.from_numpy(np.array(n_obs)).to(self.device),shape_id_list)
        n_reward=torch.from_numpy(self.reward_scaler.normalize(np.array(n_reward))).to(self.device,dtype=torch.float32)
        n_done=torch.from_numpy(np.array(n_done)).to(self.device,dtype=torch.float32)
        n_discount=torch.from_numpy(np.array(n_discount)).to(self.device,dtype=torch.float32)

        batch=Data(obs,action,next_obs,reward,done,n_obs,n_reward,n_done,n_discount)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)