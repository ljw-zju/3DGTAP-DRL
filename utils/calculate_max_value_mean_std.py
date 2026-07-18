import os
import json
import numpy as np
from utils_np import quat_to_matrix9D,matrix9D_to_6D
from tqdm import tqdm 
up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)] 
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids+down_ids

error_cases=['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002776709.json', 'C01002778059.json', 'C01002778116.json', 'C01002781299.json', 'C01002782256.json', 'C01002782469.json', 'C01002785002.json', 'C01002787969.json', 'C01002788634.json', 'C01002791605.json', 'C01002791650.json', 'C01002792909.json', 'C01002793517.json', 'C01002795801.json', 'C01002796969.json', 'C01002799164.json', 'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 'C01002747516.json', 'C01002774628.json', 'C01002785507.json', 'C01002796914.json', 'C01002803328.json', 'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 'C01002763390.json', 'C01002775708.json', 'C01002789185.json', 'C01002801630.json', 'C01002814870.json', 'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C01002780423.json', 'C01002801146.json', 'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json']


def make_data_list(root=None):
    sample_paths=[os.path.join(root,f) for f in os.listdir(root) if os.path.isdir(os.path.join(root,f)) and f.startswith("C")]
    all_data=[]
    for sample in tqdm(sample_paths):
        if f'{sample.split("/")[-1]}.json' in error_cases:
            print("error_case",sample)
            continue
        json_paths=[os.path.join(sample,f) for f in os.listdir(sample) if f.startswith("step")]
        sample_data=[]
        for json_path_step in json_paths:
            teeth28=[]
            with open(json_path_step,'r') as file:
                data=json.load(file)
                for teeth_id in ids:
                    if f'{teeth_id}' in data.keys():
                        x,y,z,qx,qy,qz,qw=data[f'{teeth_id}']
                        teeth28.append([x,y,z,qw,qx,qy,qz])
                    else:
                        teeth28.append([0]*7)
            sample_data.append(teeth28)
        all_data.append(sample_data)
    return all_data

if __name__ == "__main__":
    data=make_data_list('data/train_data')
    result_list=[]
    for sublist_3d in data:
        numpy_array_3d=np.array(sublist_3d)
        xyz=numpy_array_3d[:,:,:3]
        rotation=quat_to_matrix9D(numpy_array_3d.reshape(-1,7)[:,3:])
        rotation=matrix9D_to_6D(rotation).reshape(-1,28,6)
        numpy_array_3d=np.concatenate((xyz,rotation),axis=2)
        result_list.append(numpy_array_3d)
    data_np=np.concatenate(result_list,axis=0)
    print("data_np.shape: ",data_np.shape)
    mean_array = np.mean(data_np, axis=0, keepdims=True)
    std_array = np.std(data_np, axis=0, keepdims=True)
    abs_max_array = np.max(np.abs(data_np), axis=0, keepdims=True)
    print(mean_array.shape,std_array.shape,abs_max_array.shape)
    mean_array = np.squeeze(mean_array, axis=0)
    std_array = np.squeeze(std_array, axis=0)
    abs_max_array = np.squeeze(abs_max_array, axis=0)

    print("mean_array:",mean_array)
    print("*"*50)
    print("std_array:",std_array)
    print("*"*50)
    print("abs_max_array",abs_max_array)


    output_dir = "RL-Staging/data"

 
    np.save(os.path.join(output_dir, "mean_copy.npy"),mean_array)
    np.save(os.path.join(output_dir, "std_copy.npy"),std_array)
    np.save(os.path.join(output_dir, "max_value_copy.npy"),abs_max_array)

    