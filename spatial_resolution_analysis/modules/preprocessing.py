import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial"]

def preprocessing(base_folder, landmarks_folder, data_folder):
    """
    预处理函数，将landmarks文件夹中的CSV文件转换为CSV文件并保存到指定目录
    
    参数:
        base_folder: 基础文件夹路径
        landmarks_folder: 包含landmarks CSV文件的文件夹路径
        data_folder: 数据保存目标文件夹路径
    """
    # 读取target_list.txt中的目标样本文件夹名
    target_list_file = os.path.join(landmarks_folder, 'target_list.txt')
    with open(target_list_file, 'r') as f:
        target_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # 获取landmarks文件夹中所有以'landmarks_'开头的csv文件
    csv_files = [f for f in os.listdir(landmarks_folder) 
                if f.startswith('landmarks_') and f.endswith('.csv')]
    
    # 遍历每个csv文件
    for i, csv_filename in enumerate(csv_files, 1):
        # 获取当前csv文件的完整路径
        csv_file_path = os.path.join(landmarks_folder, csv_filename)
        
        # 读取csv文件
        data = pd.read_csv(csv_file_path, header=None)
        
        # 从文件名中提取文件夹名（去掉"landmarks_"前缀）
        folder_name = os.path.splitext(csv_filename)[0]
        folder_name = folder_name[10:]  # 去掉"landmarks_"前缀
        
        # 构造points.csv文件保存路径
        points_file_path = os.path.join(data_folder, folder_name, 'six_point')
        
        # 确保目标文件夹存在
        os.makedirs(points_file_path, exist_ok=True)
        
        # 处理数据：提取坐标并转换为数组
        # 假设CSV文件中第3-6列对应原始MATLAB代码中的Var3-Var6
        fl_tuj1_coordinates = data.iloc[:, 2:4].to_numpy()  # 第3-4列（索引2-3）
        target_coordinates = data.iloc[:, 4:6].to_numpy()   # 第5-6列（索引4-5）
        
        # 保存flTuj1Coordinates到points.csv
        pd.DataFrame(fl_tuj1_coordinates).to_csv(
            os.path.join(points_file_path, 'points.csv'),
            index=False,  # 不保存索引
            header=False  # 不保存列名，与原MAT文件数据结构保持一致
        )
        
        # 处理第一个文件的目标数据
        if i == 1:
            for target_name in target_list:
                target_points_folder = os.path.join(data_folder, target_name, 'six_point')
                os.makedirs(target_points_folder, exist_ok=True)
                
                points_file_name = os.path.join(target_points_folder, 'points.csv')
                # 保存targetCoordinates数据
                pd.DataFrame(target_coordinates).to_csv(
                    points_file_name,
                    index=False,
                    header=False
                )
    
    print('所有坐标数据已成功保存到各个文件夹下的/six_point/points.csv文件中。')
    
def calculate_intersection(a, b, c, d):
    """
    计算两条直线 ab 和 cd 的交点。
    a, b, c, d: 均为长度为2的数组或列表，表示点的 (x, y)
    返回交点 (ox, oy)
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    # 计算直线方程的系数
    m1 = (y2 - y1) / (x2 - x1)
    c1 = y1 - m1 * x1
    m2 = (y4 - y3) / (x4 - x3)
    c2 = y3 - m2 * x3
    ox = (c2 - c1) / (m1 - m2)
    oy = (m1 * c2 - c1 * m2) / (m1 - m2)
    return np.array([ox, oy]) 

def norm_rot_point(data_folder, summary_folder, sub_folders):
    """
    计算每个样品的标准化坐标，并保存结果。
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
        # 绘图
        plt.figure()
        plt.plot(pj_points[:, 0], pj_points[:, 1], 'ro')
        for idx, label in enumerate(['big sharp', 'big flat', 'mid sharp', 'mid flat', 'small sharp', 'small flat', 'small2mid', 'big2mid']):
            plt.text(pj_points[idx, 0], pj_points[idx, 1], label, va='bottom', ha='right')
        plt.xlabel('Normalized X')
        plt.ylabel('Normalized Y')
        plt.title(f'标准化坐标 - {tiff_file_name}')
        plt.axis('equal')
        plt.grid(True)
        save_path = os.path.join(data_folder, tiff_file_name, 'six_point', 'normalized.png')
        plt.savefig(save_path)
        plt.close()
        # 保存标准化点
        np.savetxt(os.path.join(data_folder, tiff_file_name, 'six_point', 'norm_points.csv'), pj_points, delimiter=',')
    os.makedirs(summary_folder, exist_ok=True)
    size_file = os.path.join(summary_folder, 'vg_size.csv')
    pd.DataFrame(all_vg_size).to_csv(size_file, header=False, index=False)

def registration_to_frame(data_folder, summary_folder, sub_folders, point_label_list, order_list):
    all_pj_points = []
    for folder in sub_folders:
        tiff_file_name = folder
        points_file = os.path.join(data_folder, tiff_file_name, 'six_point', 'norm_points.csv')
        pj_points = np.genfromtxt(points_file, delimiter=',')
        all_pj_points.append(pj_points)
    num_of_points = len(point_label_list)
    total_points = []
    for pj_points in all_pj_points:
        if pj_points.shape[0] != num_of_points:
            raise ValueError('所有样品的点数必须一致')
        total_points.append(pj_points.flatten())
    total_points = np.stack(total_points, axis=1)
    average_points = np.mean(total_points, axis=1)
    average_points = average_points.reshape((num_of_points, 2))
    os.makedirs(summary_folder, exist_ok=True)
    average_vg_table = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    pd.DataFrame(average_points).to_csv(average_vg_table, header=False, index=False)
    print(f'平均VG点已保存到 {average_vg_table}')
    order = order_list  # Python索引从0开始
    A_ordered = average_points[order, :]

    x = A_ordered[:, 0]
    y = A_ordered[:, 1]
    cs = CubicSpline(np.arange(x.size), np.c_[x, y], axis=0, bc_type='periodic')

    t_fine = np.linspace(0, x.size-1, 300)
    x_fine, y_fine = cs(t_fine).T
    
    plt.figure()
    plt.plot(x_fine, y_fine, '-')        # 平滑曲线
    plt.scatter(x, y, color='k', zorder=5)  # 原始点，确认曲线确实穿过它们
    for idx, label in enumerate(point_label_list):
        plt.text(average_points[idx, 0], average_points[idx, 1], label, va='bottom', ha='right')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.title('平均VG点')
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
            print(f"{points_file_path} 不存在")
        final_file_path = os.path.join(current_folder, 'six_point', 'final_mapping_landmarks.csv')
        pd.DataFrame(final_data).to_csv(final_file_path, header=False, index=False)
        print(f'已保存到 {final_file_path}')
        usethis_file_path = os.path.join(current_folder, 'six_point', 'usethis_mapping_landmarks.csv')
        pd.DataFrame(final_data).to_csv(usethis_file_path, header=False, index=False, quoting=1)
        print(f'已保存到 {usethis_file_path}') 

def true_mapping_roi(marker_list, data_folder, sub_folders):
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        csv_file = os.path.join(current_folder, "raw_data", f'{folder}_data.csv')
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
        else:
            print(f'未找到文件: {csv_file}')
        for marker in marker_list:
            positive_cols = [f'{marker}_isPositive']
            interested_cols = ['barcode', 'actual_cell_type', 'relative_row', 'relative_col']
            select_cols = interested_cols + positive_cols
            selected_data = data[select_cols].copy()
            # 2. 新增Label列：folder字符串 + barcode字符串（关键步骤）
            # 先将barcode列转为字符串（避免数字类型拼接时出现错误）
            selected_data['barcode_str'] = selected_data['barcode'].astype(str)
            # 拼接：folder（如"wt1"） + barcode_str（如"10"） → 结果如"wt1_10"（可自定义分隔符）
            selected_data['Label'] = folder + '_' + selected_data['barcode_str']  # 用"_"分隔，清晰区分
            # （可选）删除临时的barcode_str列，保持数据整洁
            selected_data = selected_data.drop(columns=['barcode_str'])

            selected_data['pos_x'] = selected_data['relative_col']
            selected_data['pos_y'] = selected_data['relative_row']
            selected_data['isPositive'] = selected_data[f'{marker}_isPositive']
            selected_data = selected_data[['Label', f'{marker}_isPositive', 'pos_x', 'pos_y', 'isPositive']]
            output_csv_file = os.path.join(current_folder, 'six_point', f'roi_data_{marker}.csv')
            os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
            selected_data.to_csv(output_csv_file, index=False)

            # 提取第8、9列
            if os.path.exists(csv_file):
                data_to_save = selected_data[['pos_x', 'pos_y']].values
                roi_file = os.path.join(current_folder, 'six_point', f'test_{marker}.csv')
                np.savetxt(roi_file, data_to_save, delimiter=',')
                print(f'已处理文件夹: {current_folder}，保存结果到: {output_csv_file}') 