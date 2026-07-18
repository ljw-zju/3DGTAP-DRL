import os
import json
import numpy as np
from utils_np import quat_to_matrix9D, matrix9D_to_6D
from tqdm import tqdm
import matplotlib.pyplot as plt

up_ids = [i for i in range(17, 10, -1)] \
    + [i for i in range(21, 28)]
down_ids = [i for i in range(47, 40, -1)] \
    + [i for i in range(31, 38)]
ids = up_ids + down_ids

error_cases=['C01002722632.json', 'C01002722812.json', 'C01002724937.json', 'C01002726883.json', 'C01002728672.json', 'C01002737908.json', 'C01002739797.json', 'C01002739809.json', 'C01002740294.json', 'C01002742285.json', 'C01002742814.json', 'C01002743376.json', 'C01002748270.json', 'C01002752736.json', 'C01002753894.json', 'C01002757078.json', 'C01002760218.json', 'C01002760285.json', 'C01002762513.json', 'C01002764234.json', 'C01002770466.json', 'C01002772985.json', 'C01002774123.json', 'C01002774594.json', 'C01002775269.json', 'C01002784742.json', 'C01002791706.json', 'C01002792886.json', 'C01002796891.json', 'C01002800505.json', 'C01002807805.json', 'C01002809896.json', 'C01002810292.json', 'C01002811406.json', 'C01002811855.json', 'C01002812430.json', 'C01002817413.json', 'C01002818931.json', 'C01002828437.json', 'C01002828482.json', 'C01002837246.json', 'C01002838124.json', 'C01002838337.json', 'C01002840587.json', 'C01002844772.json', 'C01002849621.json', 'C01002722823.json', 'C01002725118.json', 'C01002725736.json', 'C01002727154.json', 'C01002727817.json', 'C01002728762.json', 'C01002735973.json', 'C01002736749.json', 'C01002736806.json', 'C01002737627.json', 'C01002738954.json', 'C01002742982.json', 'C01002744298.json', 'C01002744513.json', 'C01002744715.json', 'C01002745255.json', 'C01002746492.json', 'C01002746762.json', 'C01002746784.json', 'C01002747392.json', 'C01002748258.json', 'C01002750688.json', 'C01002751746.json', 'C01002752343.json', 'C01002752398.json', 'C01002756167.json', 'C01002761703.json', 'C01002763288.json', 'C01002763514.json', 'C01002764458.json', 'C01002764650.json', 'C01002767170.json', 'C01002767967.json', 'C01002770411.json', 'C01002772389.json', 'C01002772402.json', 'C01002772660.json', 'C01002775270.json', 'C01002776709.json', 'C01002778059.json', 'C01002778116.json', 'C01002781299.json', 'C01002782256.json', 'C01002782469.json', 'C01002785002.json', 'C01002787969.json', 'C01002788634.json', 'C01002791605.json', 'C01002791650.json', 'C01002792909.json', 'C01002793517.json', 'C01002795801.json', 'C01002796969.json', 'C01002799164.json', 'C01002800279.json', 'C01002801236.json', 'C01002805533.json', 'C01002808367.json', 'C01002811237.json', 'C01002811934.json', 'C01002821159.json', 'C01002823780.json', 'C01002824747.json', 'C01002830711.json', 'C01002831149.json', 'C01002834276.json', 'C01002835435.json', 'C01002836043.json', 'C01002836706.json', 'C01002840767.json', 'C01002844996.json', 'C01002846987.json', 'C01002847045.json', 'C01002722788.json', 'C01002747516.json', 'C01002774628.json', 'C01002785507.json', 'C01002796914.json', 'C01002803328.json', 'C01002815310.json', 'C01002735210.json', 'C01002737403.json', 'C01002756831.json', 'C01002763198.json', 'C01002763390.json', 'C01002775708.json', 'C01002789185.json', 'C01002801630.json', 'C01002814870.json', 'C01002826165.json', 'C01002725466.json', 'C01002726265.json', 'C01002745749.json', 'C01002757180.json', 'C01002766258.json', 'C01002767675.json', 'C01002771849.json', 'C01002780423.json', 'C01002801146.json', 'C01002847483.json', 'C01002744748.json', 'C01002776833.json', 'C01002790266.json', 'C01002796879.json', 'C01002826705.json', 'C01002827975.json', 'C01002838720.json', 'C01002845920.json']

def make_data_list(root=None):
    global error_count
    sample_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and f.startswith("C")]
    all_data = []
    for sample in tqdm(sample_paths):
        if f'{sample.split("/")[-1]}.json' in error_cases:
            error_count+=1
            print("error_case",sample,error_count)
            continue
        if os.path.isdir(os.path.join("result_vis/gt_train",sample.split('/')[-1])):
            print(f"❌ error_case2  {sample},error data")
            continue

        json_paths = [os.path.join(sample, f) for f in os.listdir(sample) if f.startswith("step")]
        sample_data = []
        for step in range(1,len(json_paths)+1):  # Ensure steps are in order
            json_path_step=os.path.join(sample,f'step{step}.json')
            teeth28 = []
            with open(json_path_step, 'r') as file:
                data = json.load(file)
                for teeth_id in ids:
                    if f'{teeth_id}' in data.keys():
                        x,y,z,qx,qy,qz,qw=data[f'{teeth_id}']
                        teeth28.append([x,y,z,qw,qx,qy,qz])
                    else:
                        teeth28.append([0] * 7)
            sample_data.append(teeth28)
        all_data.append(sample_data)
    return all_data


def remove_outliers_std(data_np, n_std=3):
    mean = np.mean(data_np, axis=0, keepdims=True)
    std = np.std(data_np, axis=0, keepdims=True)
    upper_bound = mean + n_std * std
    lower_bound = mean - n_std * std
    cleaned_data = np.clip(data_np, lower_bound, upper_bound)
    return cleaned_data



def plot_feature_distribution_with_extremes(data, teeth_id, feature_index, title, save_path):
    values = data[:, teeth_id, feature_index]
    if values.size == 0:
        return  # Skip plotting if no data

    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(values, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")


    max_val = np.max(values)
    bin_max_index = np.digitize(max_val, bins) - 1
    if 0 <= bin_max_index < len(patches):
        max_count = n[bin_max_index]
        bin_center_max = (bins[bin_max_index] + bins[bin_max_index + 1]) / 2
        plt.text(bin_center_max, max_count * 1.05, f'Max: {max_val:.2f}', ha='center', va='bottom', color='red', fontsize=8)
        plt.scatter(bin_center_max, max_count, color='red', s=20)


    min_val = np.min(values)
    bin_min_index = np.digitize(min_val, bins) - 1
    if 0 <= bin_min_index < len(patches):
        min_count = n[bin_min_index]
        bin_center_min = (bins[bin_min_index] + bins[bin_min_index + 1]) / 2
        plt.text(bin_center_min, min_count * 1.05, f'Min: {min_val:.2f}', ha='center', va='bottom', color='green', fontsize=8)
        plt.scatter(bin_center_min, min_count, color='green', s=20)

    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    root_path = 'data/train_data'
    output_dir = 'data/feature_distributions_extremes'
    os.makedirs(output_dir, exist_ok=True)
    error_count=0
    data = make_data_list(root_path)
    result_list = []
    for sublist_3d in data:
        numpy_array_3d = np.array(sublist_3d)
        if numpy_array_3d.shape[0] > 1:  # Ensure at least two steps for diff
            xyz = numpy_array_3d[:, :, :3]
            rotation = quat_to_matrix9D(numpy_array_3d.reshape(-1, 7)[:, 3:])
            rotation = matrix9D_to_6D(rotation).reshape(-1, 28, 6)
            combined = np.concatenate((xyz, rotation), axis=2)
            diff = np.diff(combined, axis=0)
            result_list.append(diff)
        else:
            print(f"Warning: Sample has {numpy_array_3d.shape[0]} steps, skipping diff calculation.")

    if result_list:
        data_np = np.concatenate(result_list, axis=0)
        print("Shape of data_np after concatenation:", data_np.shape)

        num_teeth = 28
        num_features = 9

     
        output_dir_before = os.path.join(output_dir, 'before_outlier')
        os.makedirs(output_dir_before, exist_ok=True)
        for i in range(num_teeth):
            for j in range(num_features):
                save_path = os.path.join(output_dir_before, f"teeth_{ids[i]}_feature_{j}.png")
                plot_feature_distribution_with_extremes(data_np, i, j, f"Before Outlier Removal - Teeth {ids[i]}, Feature {j}", save_path)
        print(f"Saved {num_teeth * num_features} images (before outlier removal) with extremes to {output_dir_before}")

   
        cleaned_data_np = remove_outliers_std(data_np, n_std=3.5)
        print("Shape of cleaned_data_np:", cleaned_data_np.shape)

        output_dir_after = os.path.join(output_dir, 'after_outlier')
        os.makedirs(output_dir_after, exist_ok=True)
        for i in range(num_teeth):
            for j in range(num_features):
                save_path = os.path.join(output_dir_after, f"teeth_{ids[i]}_feature_{j}.png")
                plot_feature_distribution_with_extremes(cleaned_data_np, i, j, f"After Outlier Removal (3 STD) - Teeth {ids[i]}, Feature {j}", save_path)
        print(f"Saved {num_teeth * num_features} images (after outlier removal) with extremes to {output_dir_after}")

     
        abs_max_cleaned = np.max(np.abs(cleaned_data_np), axis=0)
        print("Shape of abs_max_cleaned:", abs_max_cleaned.shape)
        print("abs_max_cleaned:\n", abs_max_cleaned)

        output_path = os.path.join(output_dir.replace('tmp', 'data'), "max_diff_cleaned_with_extremes.npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            print(f"Warning: File '{output_path}' already exists and will be overwritten.")
        np.save('data/diff.npy', abs_max_cleaned)

    else:
        print("No valid data for diff calculation.")