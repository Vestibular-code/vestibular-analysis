import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial"]

def preprocessing(base_folder, landmarks_folder, data_folder):
    """
    Pre-processing function: convert CSV files in landmarks folder to CSV files and save to specified directory
    
    Parameters:
        base_folder: base folder path
        landmarks_folder: folder containing landmarks CSV files
        data_folder: target folder path for data saving
    """
    # Read target sample folder names from target_list.txt
    target_list_file = os.path.join(landmarks_folder, 'target_list.txt')
    with open(target_list_file, 'r') as f:
        target_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # Get all CSV files starting with 'landmarks_' in landmarks folder
    csv_files = [f for f in os.listdir(landmarks_folder) 
                if f.startswith('landmarks_') and f.endswith('.csv')]
    
    # Iterate through each CSV file
    for i, csv_filename in enumerate(csv_files, 1):
        # Get full path of current CSV file
        csv_file_path = os.path.join(landmarks_folder, csv_filename)
        
        # Read CSV file
        data = pd.read_csv(csv_file_path, header=None)
        
        # Extract folder name from filename (remove "landmarks_" prefix)
        folder_name = os.path.splitext(csv_filename)[0]
        folder_name = folder_name[10:]  # Remove "landmarks_" prefix
        
        # Construct points.csv file save path
        points_file_path = os.path.join(data_folder, folder_name, 'six_point')
        
        # Ensure target folder exists
        os.makedirs(points_file_path, exist_ok=True)
        
        # Process data: extract coordinates and convert to arrays
        # Assume columns 3-6 in CSV file correspond to Var3-Var6 in original MATLAB code
        fl_tuj1_coordinates = data.iloc[:, 2:4].to_numpy()  # Columns 3-4 (indices 2-3)
        target_coordinates = data.iloc[:, 4:6].to_numpy()   # Columns 5-6 (indices 4-5)
        
        # Save flTuj1Coordinates to points.csv
        pd.DataFrame(fl_tuj1_coordinates).to_csv(
            os.path.join(points_file_path, 'points.csv'),
            index=False,  # Don't save index
            header=False  # Don't save column names, consistent with original MAT file structure
        )
        
        # Process target data from first file
        if i == 1:
            for target_name in target_list:
                target_points_folder = os.path.join(data_folder, target_name, 'six_point')
                os.makedirs(target_points_folder, exist_ok=True)
                
                points_file_name = os.path.join(target_points_folder, 'points.csv')
                # Save targetCoordinates data
                pd.DataFrame(target_coordinates).to_csv(
                    points_file_name,
                    index=False,
                    header=False
                )
    
    print('All coordinate data successfully saved to /six_point/points.csv files in each folder.')
    
def calculate_intersection(a, b, c, d):
    """
    Calculate intersection of two lines ab and cd.
    a, b, c, d: all length-2 arrays or lists, representing points (x, y)
    Return intersection point (ox, oy)
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    # Calculate line equation coefficients
    m1 = (y2 - y1) / (x2 - x1)
    c1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3)
    c2 = y3 - m2 * x3
    ox = (c2 - c1) / (m1 - m2)
    oy = (m1 * c2 - c1 * m2) / (m1 - m2)
    return np.array([ox, oy]) 

def norm_rot_point(data_folder, summary_folder, sub_folders):
    """
    Calculate standardized coordinates for each sample and save results.
    """
    all_points_list = []
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        points_file = os.path.join(current_folder, 'six_point', 'points.csv')
        points = np.genfromtxt(points_file, delimiter=',')
        all_points_list.append(points)
    all_pj_points = []
    all_vg_size = []
    for i, folder in enumerate(sub_folders):
        tiff_file_name = folder
        points = all_points_list[i]
        big_sharp = points[0]
        big_flat = points[1]
        mid_sharp = points[2]
        mid_flat = points[3]
        small_sharp = points[4]
        small_flat = points[5]
        small2mid = points[6]
        big2mid = points[7]
        big_mid = (big_sharp + big_flat) / 2
        small_mid = (small_sharp + small_flat) / 2
        origin = calculate_intersection(big_mid, small_mid, mid_sharp, mid_flat)
        x_axis = small_mid - big_mid
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = mid_sharp - mid_flat
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_proj = np.dot(points - origin, x_axis)
        y_proj = np.dot(points - origin, y_axis)
        x_min, x_max = np.min(x_proj), np.max(x_proj)
        y_min, y_max = np.min(y_proj), np.max(y_proj)
        proj_x = x_proj / (x_max - x_min)
        proj_y = y_proj / (y_max - y_min)
        pj_points = np.stack([proj_x, proj_y], axis=1)
        all_vg_size.append([x_max - x_min, y_max - y_min])
        all_pj_points.append(pj_points.flatten())
        # Plot
        plt.figure()
        plt.plot(pj_points[:, 0], pj_points[:, 1], 'ro')
        for idx, label in enumerate(['big sharp', 'big flat', 'mid sharp', 'mid flat', 'small sharp', 'small flat', 'small2mid', 'big2mid']):
            plt.text(pj_points[idx, 0], pj_points[idx, 1], label, va='bottom', ha='right')
        plt.xlabel('Normalized X')
        plt.ylabel('Normalized Y')
        plt.title(f'Standardized Coordinates - {tiff_file_name}')
        plt.axis('equal')
        plt.grid(True)
        save_path = os.path.join(data_folder, tiff_file_name, 'six_point', 'normalized.png')
        plt.savefig(save_path)
        plt.close()
        # Save standardized points
        np.savetxt(os.path.join(data_folder, tiff_file_name, 'six_point', 'norm_points.csv'), pj_points, delimiter=',')
    os.makedirs(summary_folder, exist_ok=True)
    size_file = os.path.join(summary_folder, 'vg_size.csv')
    pd.DataFrame(all_vg_size).to_csv(size_file, header=False, index=False)

def registration_to_frame(data_folder, summary_folder, sub_folders, point_label_list, order_list):
    all_pj_points = []
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        points_file = os.path.join(current_folder, 'six_point', 'norm_points.csv')
        pj_points = np.genfromtxt(points_file, delimiter=',')
        all_pj_points.append(pj_points)
    num_of_points = len(point_label_list)
    total_points = []
    for pj_points in all_pj_points:
        if pj_points.shape[0] != num_of_points:
            raise ValueError('All samples must have the same number of points')
        total_points.append(pj_points.flatten())
    total_points = np.stack(total_points, axis=1)
    average_points = np.mean(total_points, axis=1)
    average_points = average_points.reshape((num_of_points, 2))
    os.makedirs(summary_folder, exist_ok=True)
    average_vg_table = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    pd.DataFrame(average_points).to_csv(average_vg_table, header=False, index=False)
    print(f'Average VG points saved to {average_vg_table}')
    order = order_list  # Python indices start from 0
    A_ordered = average_points[order, :]

    x = A_ordered[:, 0]
    y = A_ordered[:, 1]
    cs = CubicSpline(np.arange(x.size), np.c_[x, y], axis=0, bc_type='periodic')

    t_fine = np.linspace(0, x.size-1, 300)
    x_fine, y_fine = cs(t_fine).T
    
    plt.figure()
    plt.plot(x_fine, y_fine, '-')        # smooth curve
    plt.scatter(x, y, color='k', zorder=5)  # original points, confirm curve passes through them
    for idx, label in enumerate(point_label_list):
        plt.text(average_points[idx, 0], average_points[idx, 1], label, va='bottom', ha='right')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.title('Average VG Points')
    plt.grid(False)
    plt.savefig(os.path.join(summary_folder, 'average_vg.png'))
    plt.close() 

def mapping_landmarks(data_folder, summary_folder, sub_folders):
    results_file_path = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    results_data = pd.read_csv(results_file_path, header=None)
    n_point = len(results_data)
    vg_size_path = os.path.join(summary_folder, 'vg_size.csv')
    vg_size = pd.read_csv(vg_size_path, header=None)
    avg_vg_size = vg_size.mean(axis=0).values
    BX = results_data.iloc[:, 0].values * avg_vg_size[0]
    BY = results_data.iloc[:, 1].values * avg_vg_size[1]
    final_data = []
    pt_labels = [f'Pt-{i}' for i in range(n_point + 1)]
    for i in range(n_point):
        final_data.append([pt_labels[i], 'TRUE', None, None, BX[i], BY[i]])
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        points_file_path = os.path.join(current_folder, 'six_point', 'points.csv')
        if os.path.exists(points_file_path):
            points = np.genfromtxt(points_file_path, delimiter=',')
            for i in range(n_point):
                final_data[i][2] = points[i, 0]
                final_data[i][3] = points[i, 1]
        else:
            print(f"{points_file_path} does not exist")
        final_file_path = os.path.join(current_folder, 'six_point', 'final_mapping_landmarks.csv')
        pd.DataFrame(final_data).to_csv(final_file_path, header=False, index=False)
        print(f'Saved to {final_file_path}')
        usethis_file_path = os.path.join(current_folder, 'six_point', 'usethis_mapping_landmarks.csv')
        pd.DataFrame(final_data).to_csv(usethis_file_path, header=False, index=False, quoting=1)
        print(f'Saved to {usethis_file_path}') 

def true_mapping_roi(marker_list, data_folder, sub_folders):
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        csv_file = os.path.join(current_folder, "raw_data", f'{folder}_data.csv')
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
        else:
            print(f'File not found: {csv_file}')
        for marker in marker_list:
            positive_cols = [f'{marker}_isPositive']
            interested_cols = ['barcode', 'actual_cell_type', 'relative_row', 'relative_col']
            select_cols = interested_cols + positive_cols
            selected_data = data[select_cols].copy()
            # 2. Add Label column: folder string + barcode string (key step)
            # Convert barcode column to string first (avoid errors when concatenating numeric types)
            selected_data['barcode_str'] = selected_data['barcode'].astype(str)
            # Concatenate: folder (e.g. "wt1") + barcode_str (e.g. "10") → result like "wt1_10" (separator can be customised)
            selected_data['Label'] = folder + '_' + selected_data['barcode_str']  # use "_" separator for clarity
            # (Optional) remove temporary barcode_str column to keep data clean
            selected_data = selected_data.drop(columns=['barcode_str'])

            selected_data['pos_x'] = selected_data['relative_col']
            selected_data['pos_y'] = selected_data['relative_row']
            selected_data['isPositive'] = selected_data[f'{marker}_isPositive']
            selected_data = selected_data[['Label', f'{marker}_isPositive', 'pos_x', 'pos_y', 'isPositive']]
            output_csv_file = os.path.join(current_folder, 'six_point', f'roi_data_{marker}.csv')
            os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
            selected_data.to_csv(output_csv_file, index=False)

            # Extract columns 8 and 9
            if os.path.exists(csv_file):
                data_to_save = selected_data[['pos_x', 'pos_y']].values
                roi_file = os.path.join(current_folder, 'six_point', f'test_{marker}.csv')
                np.savetxt(roi_file, data_to_save, delimiter=',')
                print(f'Processed folder: {current_folder}, results saved to: {output_csv_file}')
