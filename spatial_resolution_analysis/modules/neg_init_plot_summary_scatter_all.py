import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python.getColorCombinationd import get_color_combination_d
from python.flxdensity2C import flxdensity2C

def neg_init_plot_summary_scatter_all(summary_folder, marker_names, bin_num):
    sample_info_file_path = os.path.join(summary_folder, 'sample_info.csv')
    sample_info_data = pd.read_csv(sample_info_file_path)
    unique_sam_names = sample_info_data['sam_name'].unique()
    results_file_path = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    results_data = pd.read_csv(results_file_path, header=None)
    vg_size_path = os.path.join(summary_folder, 'vg_size.csv')
    vg_size = pd.read_csv(vg_size_path, header=None)
    avg_vg_size = vg_size.mean(axis=0).values
    BX = results_data.iloc[:, 0].values
    BY = results_data.iloc[:, 1].values
    for sam_name in unique_sam_names:
        for i in range(1):
            sample_name = marker_names[i]
            file_name = f'{sam_name}_summary_{sample_name}.csv'
            min_color, zero_color, max_color, hist_color = get_color_combination_d(marker_names, sample_name)
            summary_file_path = os.path.join(summary_folder, file_name)
            if not os.path.exists(summary_file_path):
                continue
            summary_data = pd.read_csv(summary_file_path)
            standardized_x = summary_data.iloc[:, 5] / avg_vg_size[0]
            standardized_y = summary_data.iloc[:, 6] / avg_vg_size[1]
            pos_x = standardized_x.values
            pos_y = standardized_y.values
            xedges = np.linspace(-0.7, 0.7, 50)
            yedges = np.linspace(-0.6, 0.6, 50)
            CData, _, colorMap = flxdensity2C(pos_x, pos_y, xedges, yedges, min_color, max_color, zero_color)
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
            sc = ax1.scatter(pos_x, pos_y, c=CData, s=30, alpha=0.7)
            ax1.set_xlim([-0.7, 0.8])
            ax1.set_ylim([-0.6, 0.6])
            order = [0, 7, 2, 6, 4, 5, 3, 1]
            BX_ordered = BX[order]
            BY_ordered = BY[order]
            BX_ordered = np.append(BX_ordered, BX_ordered[0])
            BY_ordered = np.append(BY_ordered, BY_ordered[0])
            ax1.plot(BX_ordered, BY_ordered, color=1-np.array(zero_color)*0.3+0.5, linewidth=2)
            ax1.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)
            ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
            bin_edges = np.linspace(BX_ordered.min(), BX_ordered.max(), bin_num+1)
            ax2.hist(pos_x, bins=bin_edges, color=hist_color, alpha=0.5, density=True)
            ax2.set_xlim(ax1.get_xlim())
            ax3 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
            bin_edges = np.linspace(BY_ordered.min(), BY_ordered.max(), bin_num+1)
            ax3.hist(pos_y, bins=bin_edges, color=hist_color, alpha=0.5, density=True, orientation='horizontal')
            ax3.set_ylim(ax1.get_ylim())
            plt.tight_layout()
            density_save_file_path = os.path.join(summary_folder, f'{sam_name}_neginit_scatter_density_{sample_name}.png')
            plt.savefig(density_save_file_path)
            plt.close() 