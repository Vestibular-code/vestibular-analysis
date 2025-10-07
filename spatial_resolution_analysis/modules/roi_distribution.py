import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline
from matplotlib.colors import ListedColormap  # for custom colour mapping
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial"]

def plot_individually(summary_folder, sub_folders, data_folder):
    results_file_path = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    results_data = pd.read_csv(results_file_path, header=None)
    vg_size_path = os.path.join(summary_folder, 'vg_size.csv')
    vg_size = pd.read_csv(vg_size_path, header=None)
    avg_vg_size = vg_size.mean(axis=0).values
    BX = results_data.iloc[:, 0].values * avg_vg_size[0]
    BY = results_data.iloc[:, 1].values * avg_vg_size[1]
    for folder in sub_folders:
        current_folder = os.path.join(data_folder, folder)
        warped_roi_path = os.path.join(current_folder, 'six_point', 'testoutput.csv')
        warped_roi = pd.read_csv(warped_roi_path, header=None)
        roi_data_files = [f for f in os.listdir(os.path.join(current_folder, 'six_point')) if f.startswith('roi_data_') and f.endswith('.csv')]
        for roi_data_file in roi_data_files:
            marker = roi_data_file[9:-4]
            roi_info_path = os.path.join(current_folder, 'six_point', roi_data_file)
            roi_info = pd.read_csv(roi_info_path, header=0)
            pos_x = warped_roi.iloc[:, 0].values
            pos_y = warped_roi.iloc[:, 1].values
            # Ensure data lengths are consistent
            if len(roi_info['isPositive']) != len(pos_x) or len(roi_info['isPositive']) != len(pos_y):
                raise ValueError("Data lengths don't match, please check data source")

            # Plot directly using Boolean indexing, no new variables needed
            plt.figure()
            plt.grid(True)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'{folder} - {marker} Sample ROI Registered Coordinates and Sample Landmark Points')

            # Complete in one line when plotting using conditional filtering
            plt.plot(pos_x[roi_info['isPositive'] == 1], pos_y[roi_info['isPositive'] == 1], 
                     'ro', markersize=8, label='pos-ROI', alpha=0.5)
            plt.plot(pos_x[roi_info['isPositive'] == 0], pos_y[roi_info['isPositive'] == 0], 
                     'bo', markersize=8, label='neg-ROI', alpha=0.5)

            plt.plot(BX, BY, 'kx', markersize=10, label='mapping anchors')
            plt.legend(loc='best')
            # Save image
            save_path = os.path.join(current_folder, 'six_point', f'warped_roi_{marker}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
            plt.close()

def summary_subtypes_csv(summary_folder, data_folder):
    sample_info_path = os.path.join(summary_folder, 'sample_info.csv')
    sample_info = pd.read_csv(sample_info_path, header=0)
    unique_sam_names = sample_info['sam_name'].unique()
    for sam_name in unique_sam_names:
        sam_folders = sample_info.loc[sample_info['sam_name'] == sam_name, 'sam_folder']
        all_summary = []
        for sam_folder in sam_folders:
            current_folder = os.path.join(data_folder, sam_folder)
            roi_data_files = [f for f in os.listdir(os.path.join(current_folder, 'six_point')) if f.startswith('roi_data_') and f.endswith('.csv')]
            warped_roi_path = os.path.join(current_folder, 'six_point', 'testoutput.csv')
            warped_roi = pd.read_csv(warped_roi_path, header=None)
            pos_x = warped_roi.iloc[:, 0].values
            pos_y = warped_roi.iloc[:, 1].values
            for roi_data_file in roi_data_files:
                marker = roi_data_file[9:-4]
                roi_info_path = os.path.join(current_folder, 'six_point', roi_data_file)
                roi_info = pd.read_csv(roi_info_path, header=0)
                is_positive = roi_info['isPositive']
                modified_labels = []
                for label in roi_info['Label']:
                    original_label = f'{sam_folder}_{label}'
                    import re
                    m = re.match(r'^([^_]*_)(.*?\.tif:.*)$', original_label)
                    if m:
                        idx = m.group(2).find(':')
                        modified_label = m.group(1) + 'tif:' + m.group(2)[idx+1:]
                    else:
                        modified_label = original_label
                    modified_labels.append(modified_label)
                current_summary = pd.DataFrame({
                    'Label': modified_labels,
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
            sample_data = sample_data.drop_duplicates()
            summary_file_path = os.path.join(summary_folder, f'{sam_name}_summary_{sample_name}.csv')
            sample_data.to_csv(summary_file_path, index=False) 

def get_color_combination_d(marker_names, sample_name):
    """
    Get colour combination (ensure all colour values are 0-1 floats)
    """
    first_group_color = {
        # Correct: RGB each channel is 0-1 float
        'maxColor': [0.08, 0.08, 1.0],  # blue
        'minColor': [1.0, 0.95, 0.1],  # yellow
        'histColor': [0.5647, 0.7490, 0.9765],  # Correct: already in 0-1 range
        'zeroColor': [0.5, 0.5, 0.5]  # Correct: medium grey (0-1 range)
    }
    min_color = first_group_color['minColor']
    zero_color = first_group_color['zeroColor']
    max_color = first_group_color['maxColor']
    hist_color = first_group_color['histColor']
    return min_color, zero_color, max_color, hist_color 

def color_func_factory(color_list):
    """
    Generate colour interpolation function (fix: use linear interpolation instead of cubic to avoid boundary derivative errors)
    """
    # Colour node x-coordinates: from 0 to 1, same number as colour_list
    x = np.linspace(0, 1, len(color_list))
    # Extract RGB three channel values
    y1 = [c[0] for c in color_list]  # R channel
    y2 = [c[1] for c in color_list]  # G channel
    y3 = [c[2] for c in color_list]  # B channel
    # -------------------------- Core fix: kind='linear' --------------------------
    # Linear interpolation (linear): only needs 2 or more data points, no boundary derivative requirements, stable and reliable
    interp_r = interp1d(x, y1, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_g = interp1d(x, y2, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_b = interp1d(x, y3, kind='linear', bounds_error=False, fill_value="extrapolate")
    def color_func(X):
        # Stack RGB channels to generate final colour data (shape: [n_points, 3])
        return np.stack([interp_r(X), interp_g(X), interp_b(X)], axis=-1)
    return color_func

def flxdensity2C(X, Y, XList, YList, minColor, maxColor, zeroColor):
    """
    2D kernel density estimation and generate colour matrix
    """    
    # 1. Generate grid and kernel density estimation
    XMesh, YMesh = np.meshgrid(XList, YList)
    XYi = np.vstack([XMesh.ravel(), YMesh.ravel()]).T  # grid point coordinates
    values = np.vstack([X, Y])  # input data (for KDE)
    
    # -------------------------- New: KDE calculation exception capture --------------------------
    try:
        kde = gaussian_kde(values)  # initialise kernel density estimator
        F = kde(XYi.T)  # compute density value at each grid point
    except Exception as e:
        print(f"Warning: kernel density estimation failed -> {str(e)}")
        return np.array([]), np.array([]), np.array([])
    
    # 2. Build density grid (ensure same size as XMesh/YMesh)
    ZMesh = np.zeros_like(XMesh)
    # Avoid F length exceeding ZMesh (extreme case where grid point count is abnormal)
    fill_length = min(len(F), ZMesh.size)
    ZMesh.ravel()[:fill_length] = F
    
    # 3. Interpolate to get density value for each original data point (replace interp2)
    from scipy.interpolate import griddata
    h = griddata(XYi, F, (X, Y), method='linear', fill_value=0)  # linear interpolation more stable
    
    # -------------------------- Fix: density normalisation (avoid division by zero error) --------------------------
    h_min = np.min(h)
    # If all density values are same (e.g. only 1 data point), avoid dividing by 0, directly set to 0
    if np.max(h) - h_min < 1e-8:
        h_norm = np.zeros_like(h)
    else:
        # Original logic: divide by 4 (can be adjusted according to needs, if normalising to 0-1 range can use (h - h_min)/(np.max(h)-h_min))
        h_norm = (h - h_min) / 4    
    # 4. Generate colour data (core fix: use linear interpolation instead of cubic interpolation)
    colour_list = [minColor, zeroColor, maxColor]
    # Call modified colour_func_factory (using linear interpolation)
    colour_func = colour_func_factory(colour_list)
    CData = colour_func(h_norm)
    
    # 5. Generate colour mapping matrix (for possible colorbar later, keep linear interpolation)
    # -------------------------- Keep colorMap (for building custom_cmap) --------------------------
    num_colors = 64
    x_colour = np.linspace(0, 1, len([minColor, zeroColor, maxColor]))
    y1 = [c[0] for c in [minColor, zeroColor, maxColor]]
    y2 = [c[1] for c in [minColor, zeroColor, maxColor]]
    y3 = [c[2] for c in [minColor, zeroColor, maxColor]]
    colorMap = np.stack([
        np.interp(np.linspace(0, 1, num_colors), x_colour, y1),
        np.interp(np.linspace(0, 1, num_colors), x_colour, y2),
        np.interp(np.linspace(0, 1, num_colors), x_colour, y3)
    ], axis=-1)
    
    # Return h_norm (for mapping colours), h (original density), colorMap (custom cmap)
    return h_norm, h, colorMap  # Core: CData → replaced with h_norm

def plot_summary_scatter(summary_folder, marker_names, order_list, num_xbins, num_ybins):
    sample_info_file_path = os.path.join(summary_folder, 'sample_info.csv')
    sample_info_data = pd.read_csv(sample_info_file_path)
    unique_sam_names = sample_info_data['sam_name'].unique()
    results_file_path = os.path.join(summary_folder, 'vg_averageing_labels.csv')
    results_data = pd.read_csv(results_file_path, header=None)
    vg_size_path = os.path.join(summary_folder, 'vg_size.csv')
    vg_size = pd.read_csv(vg_size_path, header=None)
    avg_vg_size = vg_size.mean(axis=0).values
    BX = results_data.iloc[:, 0].values * avg_vg_size[0]
    BY = results_data.iloc[:, 1].values * avg_vg_size[1]
    for sam_name in unique_sam_names:
        for sample_name in marker_names:
            file_name = f'{sam_name}_summary_{sample_name}.csv'
            min_color, zero_color, max_color, hist_color = get_color_combination_d(marker_names, sample_name)
            summary_file_path = os.path.join(summary_folder, file_name)
            summary_data = pd.read_csv(summary_file_path, header=0)
            summary_data = summary_data.drop_duplicates()
            # Directly use conditional filtering where needed, no separate variables required
            standardized_x = summary_data['warpedROIvar1'] / avg_vg_size[0]
            standardized_y = summary_data['warpedROIvar2'] / avg_vg_size[1]

            # Directly obtain pos_x and pos_y through conditions, omit intermediate variables
            pos_x = standardized_x[summary_data['isPositive'] == 1].values
            pos_y = standardized_y[summary_data['isPositive'] == 1].values

            xedges = np.linspace(-0.6, 0.6, num_xbins+1)
            yedges = np.linspace(-0.6, 0.6, num_ybins+1)
            h_norm, _, colorMap = flxdensity2C(pos_x, pos_y, xedges, yedges, min_color, max_color, zero_color)
            custom_cmap = ListedColormap(colorMap)  # convert to matplotlib recognisable colour map
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot2grid((8, 8), (2, 0), colspan=6, rowspan=6)
            sc = ax1.scatter(
                pos_x, pos_y,
                c=h_norm,          # pass values (density), not direct RGB
                cmap=custom_cmap,  # bind custom mapping (blue-grey-yellow)
                s=30, 
                alpha=0.7,
                vmin=0, vmax=1
            )
            ax1.set_xlim([-0.6, 0.6])
            ax1.set_ylim([-0.6, 0.6])

            order = order_list
            BX_ordered = BX[order]
            BY_ordered = BY[order]
            cs = CubicSpline(np.arange(len(BX_ordered)), np.c_[BX_ordered, BY_ordered], axis=0, bc_type='periodic')
            t_fine = np.linspace(0, len(BX_ordered)-1, 300)
            x_fine, y_fine = cs(t_fine).T

            ax1.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)
            ax1.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)
            
            # -------------------------- 1. Build custom colour mapping (corresponds to MATLAB colormap(ax1, colorMap)) --------------------------
            # MATLAB's colorMap is [n_colors, 3] RGB matrix (0-1 floats), Python uses ListedColormap wrapper
            
            # -------------------------- 2. Create independent colour bar axis (corresponds to MATLAB colorbar(ax1) + position adjustment) --------------------------
            # Method: use plt.gcf() to get current figure, manually add independent axis for colour bar (avoid squeezing main plot)
            # MATLAB cbar.Position = [0.65, 0.52, 0.02, 0.15] → directly reuse normalised coordinates
            cbar_ax = plt.gcf().add_axes([0.67, 0.44, 0.02, 0.15])  # [left, bottom, width, height]
            # Create colour bar: bind scatter plot sc, custom colour mapping, independent axis cbar_ax
            cbar = plt.colorbar(
                mappable=sc,        # bind scatter plot (must be object with colour mapping, like PathCollection returned by scatter)
                cax=cbar_ax,        # independent axis for colour bar (corresponds to MATLAB ax1 association)
                cmap=custom_cmap    # bind custom colour mapping (corresponds to MATLAB colormap)
            )
            # -------------------------- 3. Set colour bar ticks and labels (corresponds to MATLAB cbar.Ticks + cbar.TickLabels) --------------------------
            # MATLAB: cbar.Ticks = 0:2:4; cbar.TickLabels = {'0','2','4'}
            # Python needs to set tick positions first, then labels (order cannot be reversed)
            cbar.set_ticks([0, 0.5, 1])                # tick positions (same as MATLAB 0:2:4)
            cbar.set_ticklabels(['0', '2', '4'])     # tick labels (same as MATLAB {'0','2','4'})
            # -------------------------- 4. Hide colour bar labels (corresponds to MATLAB cbar.Label.Visible = 'off') --------------------------
            cbar.set_label('')  # clear default colour bar label (or use cbar.label.set_visible(False))
            # -------------------------- 5. Add custom text on top of colour bar (corresponds to MATLAB text(...)) --------------------------
            # Python: add text on colour bar axis (cbar_ax) using normalised coordinates (units='normalized')
            # Process sampleName case: first letter uppercase + rest lowercase (corresponds to MATLAB upper(sampleName(1))+lower(sampleName(2:end)))
            sample_text = sample_name[0].upper() + sample_name[1:].lower()
            # Add text on top of colour bar (x=0.5: horizontal centre, y=1.05: above colour bar, avoid overlap)
            cbar_ax.text(
                x=0.5,                              # horizontal position (normalised, 0=left, 1=right)
                y=1.05,                             # vertical position (normalised, 1=top of colour bar, 1.05=outside top)
                s=sample_text,                      # text content (processed sampleName)
                horizontalalignment='center',       # horizontal centre (corresponds to MATLAB 'HorizontalAlignment','center')
                verticalalignment='bottom',         # vertical bottom alignment (corresponds to MATLAB 'VerticalAlignment','bottom')
                fontsize=10,                        # font size (corresponds to MATLAB 'FontSize',10)
                rotation=0                          # rotation angle (corresponds to MATLAB 'Rotation',0)
            )

            # -------------------------- 6. Adjust colour bar label alignment (corresponds to MATLAB cbar.Label.VerticalAlignment etc.) --------------------------
            # Since label was hidden earlier (cbar.set_label('')), if you need to show label later, add:
            # cbar.set_label('Density', fontsize=10)  # set label text first
            # cbar.label.set_verticalalignment('middle')   # vertical centre (corresponds to MATLAB cbar.Label.VerticalAlignment='middle')
            # cbar.label.set_horizontalalignment('right')  # horizontal right alignment (MATLAB 'HorizontalAlignment','center' adjust as needed)
            # cbar.label.set_color('black')                # label colour (corresponds to MATLAB cbar.Label.Color='k')


            # -------------------------- 7. Hide colour bar border (corresponds to MATLAB cbar.Box = 'off') --------------------------
            # MATLAB's cbar.Box controls border, Python needs to hide colour bar outline
            cbar.outline.set_visible(False)  # remove black border line outside colour bar

            ax2 = plt.subplot2grid((8, 8), (0, 0), rowspan=2, colspan=6)
            bin_edges = np.linspace(BX_ordered.min(), BX_ordered.max(), num_xbins+1)
            ax2.hist(pos_x, bins=bin_edges, color=hist_color, alpha=0.5, density=True)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim((0, 3))
            ax2.set_yticks(np.linspace(0, 3, 4))
            ax2.tick_params(axis='x', labelbottom=False)  # Only set labelbottom=False
            ax2.set_ylabel(
                'prob. dens.',          # Required: meaning of x-axis (e.g. "X coordinate", "density")
                fontsize=10,        # Optional: font size (default 10, adjust according to subplot size)
                fontweight='normal',  # Optional: font weight ('normal' default/'bold' bold)
                # color='b',    # Optional: font colour (dark grey #333333 softer than pure black)
                labelpad=5         # Optional: distance between label and x-axis (points, avoid overlap with ticks)
            )
            ax3 = plt.subplot2grid((8, 8), (2, 6), rowspan=6)
            bin_edges = np.linspace(BY_ordered.min(), BY_ordered.max(), num_ybins+1)
            ax3.hist(pos_y, bins=bin_edges, color=hist_color, alpha=0.5, density=True, orientation='horizontal')
            ax3.set_ylim(ax1.get_ylim())
            ax3.set_xlim((0,3))
            ax3.set_xticks(np.linspace(0, 3, 4))
            ax3.tick_params(axis='y', labelleft=False)  # Only set labelbottom=False
            ax3.set_xlabel(
                'prob. dens.',          # Required: meaning of x-axis (e.g. "X coordinate", "density")
                fontsize=10,        # Optional: font size (default 10, adjust according to subplot size)
                fontweight='normal',  # Optional: font weight ('normal' default/'bold' bold)
                # color='b',    # Optional: font colour (dark grey #333333 softer than pure black)
                labelpad=5         # Optional: distance between label and x-axis (points, avoid overlap with ticks)
            )
            # 2. Loop to hide "top border" and "right border" for each subplot
            axes_to_adjust = [ax1, ax2, ax3]
            for ax in axes_to_adjust:
                # Hide top border
                ax.spines['top'].set_visible(False)
                # Hide right border
                ax.spines['right'].set_visible(False)

            # plt.tight_layout()
            density_save_file_path = os.path.join(summary_folder, f'{sam_name}_scatter_density_{sample_name}.png')
            plt.subplots_adjust(
                left=0.15,    # Left margin (0.1→0.15, leave space for ax3 y-axis ticks)
                right=0.9,   # Right margin (0.9→0.85, leave space for colour bar)
                bottom=0.15,  # Bottom margin (0.1→0.15, leave space for ax1 x-axis labels)
                top=0.9,     # Top margin (0.9→0.85, leave space for ax2 title)
                hspace=1,   # Vertical spacing (between ax2 and ax1, 0.3→0.5, avoid overlap)
                wspace=0.3    # Horizontal spacing (between ax1 and ax3, 0.3→0.5, avoid crowding)
            )
            plt.savefig(density_save_file_path, dpi=1200, bbox_inches='tight')
            plt.show()
            plt.close()
