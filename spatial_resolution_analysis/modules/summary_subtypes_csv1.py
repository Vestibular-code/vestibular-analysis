import os
import pandas as pd
import re

def summary_subtypes_csv1(summary_folder, data_folder):
    sample_info_path = os.path.join(summary_folder, 'sample_info.csv')
    sample_info = pd.read_csv(sample_info_path)
    individual_sum_folder = os.path.join(summary_folder, 'individual_sum')
    os.makedirs(individual_sum_folder, exist_ok=True)
    unique_sam_names = sample_info['sam_name'].unique()
    for sam_name in unique_sam_names:
        sam_folders = sample_info.loc[sample_info['sam_name'] == sam_name, 'sam_folder']
        all_summary = []
        for sam_folder in sam_folders:
            current_folder = os.path.join(data_folder, sam_folder)
            roi_data_files = [f for f in os.listdir(os.path.join(current_folder, 'six_point')) if f.startswith('roi_data_') and f.endswith('.csv')]
            warped_roi_path = os.path.join(current_folder, 'six_point', 'testoutput.csv')
            warped_roi = pd.read_csv(warped_roi_path)
            pos_x = warped_roi.iloc[:, 0].values
            pos_y = warped_roi.iloc[:, 1].values
            for roi_data_file in roi_data_files:
                marker = roi_data_file[9:-4]
                roi_info_path = os.path.join(current_folder, 'six_point', roi_data_file)
                roi_info = pd.read_csv(roi_info_path)
                is_positive = roi_info['isPositive']
                modified_labels = []
                for label in roi_info['Label']:
                    original_label = f'{sam_folder}_{label}'
                    m = re.match(r'^([^_]*_)(.*?\.tif:.*)$', original_label)
                    if m:
                        idx = m.group(2).find(':')
                        modified_label = m.group(1) + 'tif:' + m.group(2)[idx+1:]
                    else:
                        modified_label = original_label
                    modified_labels.append(modified_label)
                current_summary = pd.DataFrame({
                    'Label': modified_labels,
                    'Mean': roi_info['Mean'],
                    'Major': roi_info['Major'],
                    'pos_X': roi_info['pos_x'],
                    'pos_Y': roi_info['pos_y'],
                    'warpedROIvar1': pos_x,
                    'warpedROIvar2': pos_y,
                    'isPositive': is_positive,
                    'sam_name': [marker]*len(roi_info)
                })
                all_summary.append(current_summary)
        if not all_summary:
            continue
        all_summary_df = pd.concat(all_summary, ignore_index=True)
        unique_sample_names = all_summary_df['sam_name'].unique()
        for sample_name in unique_sample_names:
            sample_data = all_summary_df[all_summary_df['sam_name'] == sample_name]
            summary_file_path = os.path.join(summary_folder, f'{sam_name}_summary_{sample_name}.csv')
            sample_data.to_csv(summary_file_path, index=False)
            # individual_sum 分割
            unique_numbers = []
            for label in sample_data['Label']:
                s_index = label.find('s')
                if s_index > 0:
                    prefix = label[:s_index]
                    numbers = re.findall(r'\d+', prefix)
                    if numbers:
                        unique_numbers.append(numbers[0])
                    else:
                        unique_numbers.append(prefix)
                else:
                    unique_numbers.append(label)
            unique_numbers = list(set(unique_numbers))
            for number_part in unique_numbers:
                number_data = sample_data[sample_data['Label'].str.contains(f'^{number_part}')]  # 简化筛选
                if not number_data.empty:
                    individual_file_path = os.path.join(individual_sum_folder, f'{sam_name}_{number_part}_summary_{sample_name}.csv')
                    number_data.to_csv(individual_file_path, index=False) 