
import os
import json
import numpy as np
import math
from utils import utils_np
import open3d as o3d
import fcl
from tqdm import tqdm



import numpy as np

def batch_euler_to_quat_wxyz(euler_angles_rad, order='zyx'):
   

    # (N, 3)
    roll, pitch, yaw = euler_angles_rad[:, 0], euler_angles_rad[:, 1], euler_angles_rad[:, 2]


    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)


    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

 
    quats_wxyz = np.stack([w, x, y, z], axis=1)
    
    return quats_wxyz

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids
oid = {id: i for i, id in enumerate(ids)}  

transform_matrix_original_to_fcl = np.array([[-1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, -1]], dtype=np.float32)
error_cases=['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002776709.json', 'C01002778059.json', 'C01002778116.json', 'C01002781299.json', 'C01002782256.json', 'C01002782469.json', 'C01002785002.json', 'C01002787969.json', 'C01002788634.json', 'C01002791605.json', 'C01002791650.json', 'C01002792909.json', 'C01002793517.json', 'C01002795801.json', 'C01002796969.json', 'C01002799164.json', 'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 'C01002747516.json', 'C01002774628.json', 'C01002785507.json', 'C01002796914.json', 'C01002803328.json', 'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 'C01002763390.json', 'C01002775708.json', 'C01002789185.json', 'C01002801630.json', 'C01002814870.json', 'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C01002780423.json', 'C01002801146.json', 'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json']

class Evaluator:
    def __init__(self,gt_path,test_path,sample_name,collision_tolerance,remove_idx,num_max_contacts,is_transformer,args) -> None:
        self.is_wolf=args.is_wolf
        self.is_ddim=args.is_ddim
        self.sample_name=sample_name
        self.init_data(gt_path,test_path)
        self.get_mesh()
        
        self.COLLISION_TOLERANCE=collision_tolerance
        self.remove_idx=remove_idx
        self.num_max_contacts=num_max_contacts
        self.is_transformer=is_transformer
        self.trans_exit=args.trans_exit
        self.rot_exit=args.rot_exit
        
    def generate_aligned_indices(self,len_a: int, len_b: int):
        if len_a <= len_b:
            len_short = len_a
            len_long = len_b
            is_a_shorter = True
        else:
            len_short = len_b
            len_long = len_a
            is_a_shorter = False

   
        indices_short = list(range(len_short))
    
  
        indices_long_sampled = []


        if len_short == 1:
            indices_long_sampled = [0]
        else:
            for i in range(len_short):
                progress = i / (len_short - 1)
                index = int(round(progress * (len_long - 1)))
                indices_long_sampled.append(index)
        if is_a_shorter:
            return indices_short, indices_long_sampled
        else:
            return indices_long_sampled, indices_short


    def init_data(self,gt_path,test_path):
        print(gt_path,test_path)
        assert os.path.exists(gt_path) and os.path.exists(test_path) 
        
        with open(gt_path, 'r') as f:
            data = json.load(f)
            gt_tran=data['positions']
            gt_rot=data['rotations']
            self.gt_tran=np.array(gt_tran)
            self.gt_rot=np.array(gt_rot)
        with open(test_path,'r') as f:
            data=json.load(f)
            test_tran=data['positions']
            test_rot=data['rotations']
            self.test_tran=np.array(test_tran)
            self.test_rot=np.array(test_rot)  
            if self.is_ddim:
                self.test_rot=(-1)*self.test_rot  
            if self.is_wolf:
                self.test_rot=batch_euler_to_quat_wxyz(self.test_rot.reshape(-1,3)).reshape(-1,28,4)

    def get_mesh(self,):
        self.bvh_objects={}
        for id in ids:
            filename=os.path.join('data/hull_512',f'{self.sample_name}_{id}.ply')
            if os.path.exists(filename):
                hull=o3d.io.read_triangle_mesh(filename)
                hull_vertices=np.asarray(hull.vertices)
                hull_triangles=np.asarray(hull.triangles)
                bvh=fcl.BVHModel()
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
                self.bvh_objects[id]=bvh

    def get_length_diff(self,):
        gt_length=self.gt_tran.shape[0]
        if self.is_transformer:
            for i in range(self.test_tran.shape[0]):
                tran_error=self.calculate_distance(self.test_tran[i],self.gt_tran[gt_length-1])
                angle_error=self.get_angle_error_np(self.test_rot[i],self.gt_rot[gt_length-1])
                if tran_error.max()<self.trans_exit and angle_error.max()*180/np.pi<self.rot_exit:
                    return abs(i+1-gt_length),i+1
                # if tran_error.max()<self.trans_exit:
                #     return abs(i+1-gt_length),i+1
        return abs(self.gt_tran.shape[0]-self.test_tran.shape[0]),self.test_tran.shape[0]
    
    def calculate_distance(self,a,b):
        distance=np.linalg.norm(a-b,axis=1)
        distance[self.remove_idx]=0
        return distance
    
    def get_angle_error_np(self,a, b):
        n = a.shape[0]
        res = np.zeros(n)


        tmp=a
        zero_rows_mask = np.all(tmp == 0, axis=1)
        res[zero_rows_mask] = 0.0


        non_zero_rows_mask = ~zero_rows_mask

        a_non_zero = a[non_zero_rows_mask]
        b_non_zero = b[non_zero_rows_mask]

        if a_non_zero.size > 0:
            a_9d = utils_np.quat_to_matrix9D(a_non_zero)
            b_9d = utils_np.quat_to_matrix9D(b_non_zero)

            rm = np.matmul(np.swapaxes(a_9d, -2, -1), b_9d)
            tr = np.trace(rm, axis1=-2, axis2=-1) 
            angle_error = np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))
            res[non_zero_rows_mask] = angle_error
        # print(non_zero_rows_mask)
        
        res[self.remove_idx]=0
     
        return res

    def get_total_length(self,length):
        total_length=0
        total_angle=0
        for i in range(1,length):
            cur_distance=self.calculate_distance(self.test_tran[i],self.test_tran[i-1])
            mask=cur_distance<0.01
            total_length+=np.sum(cur_distance)
            angle_error=self.get_angle_error_np(self.test_rot[i],self.test_rot[i-1])
            angle_error1=self.get_angle_error_np(self.test_rot[i],(-1)*self.test_rot[i-1])
            angle_error=np.minimum(angle_error,angle_error1)
     
            print(angle_error.max(),angle_error.sum())
  
            total_angle+=np.sum(np.abs(angle_error))
          
        
        return total_length,total_angle

    def get_error(self,length):
        angle_error=self.get_angle_error_np(self.test_rot[length-1],self.gt_rot[-1])
        tran_error=self.calculate_distance(self.test_tran[length-1],self.gt_tran[-1])

        return tran_error.max(),angle_error.max()*180/(np.pi)
        
    def calculate_collision_step(self,step):
        collision_count1,collision_count3,collision_count5=0,0,0
        for i in range(1,len(ids)):
            j=i-1
            if (i in self.remove_idx) or (j in self.remove_idx):
                continue
            if (ids[i] not in self.bvh_objects) or (ids[j] not in self.bvh_objects):
                continue
            trans_a=transform_matrix_original_to_fcl@self.test_tran[step][i]
            trans_b=transform_matrix_original_to_fcl@self.test_tran[step][j]
            rot_a=transform_matrix_original_to_fcl@utils_np.quat_to_matrix9D(self.test_rot[step][i])
            rot_b=transform_matrix_original_to_fcl@utils_np.quat_to_matrix9D(self.test_rot[step][j])
            transform_a=fcl.Transform(rot_a,trans_a)
            transform_b=fcl.Transform(rot_b,trans_b)
            obj1=fcl.CollisionObject(self.bvh_objects[ids[i]],transform_a)
            obj2=fcl.CollisionObject(self.bvh_objects[ids[j]],transform_b)
            request=fcl.CollisionRequest(num_max_contacts=self.num_max_contacts,enable_contact=True)
            result=fcl.CollisionResult()
            fcl.collide(obj1,obj2,request,result)
            if result.is_collision:
                if result.contacts:
                    contacts=[contact.penetration_depth for contact in result.contacts]
                    penetration_depth=sum(contacts)/len(contacts)
                    if penetration_depth>0.5:
                        collision_count5+=1
                    if penetration_depth>0.3:
                        collision_count3+=1
                    if penetration_depth>0.1:
                        collision_count1+=1


        return collision_count5,collision_count3,collision_count1

    def calculate_collision_count(self,length):
        collision_count5,collision_count3,collision_count1=0,0,0
        # length=self.gt_tran.shape[0]
        for step in range(1,length):
            tmp1,tmp2,tmp3=self.calculate_collision_step(step)
            collision_count5+=tmp1
            collision_count3+=tmp2
            collision_count1+=tmp3
        
        return collision_count5,collision_count3,collision_count1
    
    def calculate_smooth(self,length):
        total_discontinuity_count=0
        for i in range(1,length):
            move_distance=(self.calculate_distance(self.test_tran[i],self.test_tran[i-1])) >0.5
            # if self.is_ddim:
            #     thre=10
            # else:
            #     thre=3
            mask=move_distance<0.01
            move_angle=(self.get_angle_error_np(self.test_rot[i],self.test_rot[i-1])*180/np.pi)
            if self.is_ddim:
                move_angle[mask]=0
            move_angle=move_angle>3
            discontinuity_count=np.logical_or(move_distance,move_angle)
            discontinuity_count[self.remove_idx]=False
            total_discontinuity_count+=np.sum(discontinuity_count)
        return total_discontinuity_count
    
    def calculate_acceleration_smoothness(self, length):
       
        if length < 3:
            return 0.0


        absolute_path = self.test_tran[:length, :, :]

        velocities = absolute_path[1:, ...] - absolute_path[:-1, ...]
        

        accelerations = velocities[1:, ...] - velocities[:-1, ...]
        

        accel_magnitudes = np.linalg.norm(accelerations, axis=-1)

        accel_magnitudes[:, self.remove_idx] = 0.0

        total_accel_norm = np.sum(accel_magnitudes)
        
 
        num_valid_accel_points = (28 - len(self.remove_idx)) * (length - 2)

        if num_valid_accel_points == 0:
            return 0.0


        return total_accel_norm / num_valid_accel_points

    def calculate_direct_distance(self):
        tran=self.calculate_distance(self.gt_tran[0],self.gt_tran[-1])
        angle=self.get_angle_error_np(self.gt_rot[0],self.gt_rot[-1])
        
        return np.sum(tran),np.sum(angle)

    def calculate_error(self,gt_list,test_list):
        total_rot,total_tran=0,0
        for i in range(len(gt_list)):
            tran_error=self.calculate_distance(self.gt_tran[gt_list[i]],self.test_tran[test_list[i]])
            rot_error=self.get_angle_error_np(self.gt_rot[gt_list[i]],self.test_rot[test_list[i]])
            rot_error1=self.get_angle_error_np(self.gt_rot[gt_list[i]],(-1)*self.test_rot[test_list[i]])
            rot_error=np.minimum(rot_error,rot_error1)
            total_rot+=rot_error.sum()/(28-len(self.remove_idx))
            total_tran+=tran_error.sum()/(28-len(self.remove_idx))
        return total_rot/len(gt_list),total_tran/len(gt_list)

    def eval(self,):
        sequence_length_diff,length=self.get_length_diff()
        # length=length-1
        total_length,total_angle=self.get_total_length(length)
        tran_error,angle_error=self.get_error(length)
        collision_count5,collision_count3,collision_count1=self.calculate_collision_count(length)
        discontinuity_count=self.calculate_smooth(length)
        fail_case=0
        direct_tran,direct_angle=self.calculate_direct_distance()
        gt_list,test_list=self.generate_aligned_indices(self.gt_rot.shape[0],length)
        r_error,t_error=self.calculate_error(gt_list,test_list)
        if length>=149:
            fail_case=1
        

        acceleration_smoothnes=self.calculate_acceleration_smoothness(length)
        return {"sequence_length_diff":sequence_length_diff,
                "path_length":total_length,
                "path_angle":total_angle,
                "tran_error":tran_error,
                "angle_error":angle_error,
                "collision_count5":collision_count5,
                "collision_count3":collision_count3,
                "collision_count1":collision_count1,
                "discontinuity_count":discontinuity_count,
                "sequence_length":length,
                "fail_case":fail_case,
                "direct_tran":direct_tran,
                "direct_angle":direct_angle,
                "r_error":r_error,
                "t_error":t_error,
                "smooth":acceleration_smoothnes}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EVAL")
    parser.add_argument("--test_dir", type=str, default='result_json/no_pos')
    parser.add_argument("--fail",type=bool,default=False)
    parser.add_argument("--is_transformer",type=bool,default=False)
    parser.add_argument("--is_ddim",type=bool,default=False)
    parser.add_argument("--is_wolf",type=bool,default=False)
    parser.add_argument("--collision_tolerance",type=float,default=0.3)
    parser.add_argument("--num_max_contacts",type=int,default=1)
    parser.add_argument("--trans_exit",type=float,default=0.3)
    parser.add_argument("--rot_exit",type=float,default=3)

    args = parser.parse_args()  

    with open("remove_idx_summary.json",'r') as f:
        remove_idx_dict=json.load(f)

    tmp_dir='json_transformer_ddim/transformer'
    filenamelist=[f for f in os.listdir(tmp_dir) if f.endswith('.json')]
    fail_case_list=[]
    collision3_list=[]
    total_sequence_length_diff=0
    total_path_length=0
    total_path_angle=0
    total_tran_error=0
    total_angle_error=0
    total_collision_count5,total_collision_count3,total_collision_count1=0,0,0
    total_discontinuity_count=0
    total_sequence_length=0
    fail_case=0
    sample_count=0
    direct_distance,direct_angle=0,0
    total_r_error,total_t_error=0,0
    total_count=0
    total_smooth=0
    for filename in tqdm(filenamelist):
      
        if  (filename in error_cases):
            continue
        gt_path=os.path.join("data/gt_json",filename)
        test_path=os.path.join(args.test_dir,filename)
        remove_idx=remove_idx_dict[filename.split('.')[0]]
        evaluator=Evaluator(gt_path=gt_path,test_path=test_path,sample_name=filename.split('.')[0],
                            collision_tolerance=args.collision_tolerance,
                            remove_idx=remove_idx,
                            num_max_contacts=args.num_max_contacts,
                            is_transformer=args.is_transformer,
                            args=args)
        
        result=evaluator.eval()

        if result["fail_case"]==1 and args.fail:
            fail_case_list.append((filename.split('.')[0],result["angle_error"],result["tran_error"]))
            continue
      
        if result["collision_count1"]!=0:
            collision3_list.append(filename)
        sample_count+=1
        total_sequence_length_diff+=result["sequence_length_diff"]
        total_path_length+=result["path_length"]
        total_path_angle+=result["path_angle"]
        total_tran_error+=result["tran_error"]
        total_angle_error+=result["angle_error"]
        total_collision_count5+=result["collision_count5"]
        total_collision_count3+=result["collision_count3"]
        total_collision_count1+=result["collision_count1"]
        total_discontinuity_count+=result["discontinuity_count"]
        total_sequence_length+=result["sequence_length"]
        fail_case+=result["fail_case"]
        direct_distance+=result["direct_tran"]
        direct_angle+=result["direct_angle"]
        total_r_error+=result["r_error"]
        total_t_error+=result["t_error"]
        total_count+=(28-len(remove_idx))*result["sequence_length"]
        total_smooth+=result["smooth"]

    result={
        "model":args.test_dir,
        "sample_count":sample_count,
        "total_delta_N":total_sequence_length_diff,
        "mean_delta_N":total_sequence_length_diff/sample_count,
        "mean_path_length":total_path_length/sample_count,
        "mean_path_angle":total_path_angle/sample_count,
        "direct_distance":direct_distance/sample_count,
        "direct_angle":direct_angle/sample_count,
        "tran_error":total_tran_error/sample_count,
        "angle_error":total_angle_error/sample_count,
        "total_discontinuity_count":total_discontinuity_count,
        "smooth":total_smooth/sample_count,
        "total_sequence_length":total_sequence_length,
        "total_collision_count5":total_collision_count5,
        "collision_frequence5":total_collision_count5/(sample_count*28),
        "total_collision_count3":total_collision_count3,
        "collision_frequence3":total_collision_count3/(sample_count*28),
        "total_collision_count1":total_collision_count1,
        "collision_frequence1":total_collision_count1/(sample_count*28),
        "fail_case":fail_case,
        "fail_frequence":fail_case/sample_count,
        "fail_case_list":fail_case_list,
        "r_error":total_r_error/sample_count,
        "t_error":total_t_error/sample_count,
        "total_count":total_count,
        "success":fail_case/sample_count
        # "collision_list":collision3_list
    }
    output_path=os.path.join("evaluate/result_hull",f"{args.test_dir.split('/')[-1]}_{args.fail}.json")
    with open(output_path,'w',encoding='utf-8') as json_file:
       
        json.dump(result,json_file,indent=4,ensure_ascii=False, cls=NumpyEncoder)



