import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline
from matplotlib.colors import ListedColormap  # 用于自定义颜色映射
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
            # 确保数据长度一致
            if len(roi_info['isPositive']) != len(pos_x) or len(roi_info['isPositive']) != len(pos_y):
                raise ValueError("数据长度不匹配，请检查数据来源")

            # 直接用布尔索引绘图，不新建变量
            plt.figure()
            plt.grid(True)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'{folder} - {marker} 样品ROI配准后坐标与样品标定点')

            # 直接在绘图时使用条件筛选，一行完成
            plt.plot(pos_x[roi_info['isPositive'] == 1], pos_y[roi_info['isPositive'] == 1], 
                     'ro', markersize=8, label='pos-ROI', alpha=0.5)
            plt.plot(pos_x[roi_info['isPositive'] == 0], pos_y[roi_info['isPositive'] == 0], 
                     'bo', markersize=8, label='neg-ROI', alpha=0.5)

            plt.plot(BX, BY, 'kx', markersize=10, label='mapping anchors')
            plt.legend(loc='best')
            # 保存图片
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
    获取颜色组合（确保所有颜色值为 0-1 浮点数）
    """
    first_group_color = {
        # 正确：RGB 每个通道都是 0-1 浮点数
        'maxColor': [0.08, 0.08, 1.0],  # 蓝色
        'minColor': [1.0, 0.95, 0.1],  # 黄色
        'histColor': [0.5647, 0.7490, 0.9765],  # 正确：已在 0-1 范围
        'zeroColor': [0.5, 0.5, 0.5]  # 正确：中灰色（0-1 范围）
    }
    min_color = first_group_color['minColor']
    zero_color = first_group_color['zeroColor']
    max_color = first_group_color['maxColor']
    hist_color = first_group_color['histColor']
    return min_color, zero_color, max_color, hist_color 

def color_func_factory(color_list):
    """
    生成颜色插值函数（修复：用线性插值替代三次插值，避免边界导数错误）
    """
    # 颜色节点的横坐标：从0到1，数量与color_list一致
    x = np.linspace(0, 1, len(color_list))
    # 提取RGB三个通道的数值
    y1 = [c[0] for c in color_list]  # R通道
    y2 = [c[1] for c in color_list]  # G通道
    y3 = [c[2] for c in color_list]  # B通道
    # -------------------------- 核心修复：kind='linear' --------------------------
    # 线性插值（linear）：只需2个及以上数据点，无边界导数要求，稳定可靠
    interp_r = interp1d(x, y1, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_g = interp1d(x, y2, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_b = interp1d(x, y3, kind='linear', bounds_error=False, fill_value="extrapolate")
    def color_func(X):
        # 堆叠RGB通道，生成最终颜色数据（形状：[n_points, 3]）
        return np.stack([interp_r(X), interp_g(X), interp_b(X)], axis=-1)
    return color_func

def flxdensity2C(X, Y, XList, YList, minColor, maxColor, zeroColor):
    """
    二维核密度估计并生成颜色矩阵
    """    
    # 1. 生成网格与核密度估计
    XMesh, YMesh = np.meshgrid(XList, YList)
    XYi = np.vstack([XMesh.ravel(), YMesh.ravel()]).T  # 网格点坐标
    values = np.vstack([X, Y])  # 输入数据（用于KDE）
    
    # -------------------------- 新增：KDE计算异常捕获 --------------------------
    try:
        kde = gaussian_kde(values)  # 初始化核密度估计器
        F = kde(XYi.T)  # 计算每个网格点的密度值
    except Exception as e:
        print(f"警告：核密度估计失败 -> {str(e)}")
        return np.array([]), np.array([]), np.array([])
    
    # 2. 构建密度网格（确保与XMesh/YMesh尺寸一致）
    ZMesh = np.zeros_like(XMesh)
    # 避免F长度超过ZMesh（极端情况网格点数量异常）
    fill_length = min(len(F), ZMesh.size)
    ZMesh.ravel()[:fill_length] = F
    
    # 3. 插值得到每个原始数据点的密度值（替代interp2）
    from scipy.interpolate import griddata
    h = griddata(XYi, F, (X, Y), method='linear', fill_value=0)  # 线性插值更稳定
    
    # -------------------------- 修复：密度归一化（避免除零错误） --------------------------
    h_min = np.min(h)
    # 若所有密度值相同（如只有1个数据点），避免除以0，直接设为0
    if np.max(h) - h_min < 1e-8:
        h_norm = np.zeros_like(h)
    else:
        # 原逻辑：除以4（可根据需求调整，若需归一到0-1范围可改为 (h - h_min)/(np.max(h)-h_min)）
        h_norm = (h - h_min) / 4    
    # 4. 生成颜色数据（核心修复：使用线性插值替代三次插值）
    color_list = [minColor, zeroColor, maxColor]
    # 调用修改后的color_func_factory（使用linear插值）
    color_func = color_func_factory(color_list)
    CData = color_func(h_norm)
    
    # 5. 生成颜色映射矩阵（用于后续可能的colorbar，保持线性插值）
    # -------------------------- 保留colorMap（用于构建custom_cmap） --------------------------
    num_colors = 64
    x_color = np.linspace(0, 1, len([minColor, zeroColor, maxColor]))
    y1 = [c[0] for c in [minColor, zeroColor, maxColor]]
    y2 = [c[1] for c in [minColor, zeroColor, maxColor]]
    y3 = [c[2] for c in [minColor, zeroColor, maxColor]]
    colorMap = np.stack([
        np.interp(np.linspace(0, 1, num_colors), x_color, y1),
        np.interp(np.linspace(0, 1, num_colors), x_color, y2),
        np.interp(np.linspace(0, 1, num_colors), x_color, y3)
    ], axis=-1)
    
    # 返回 h_norm（用于映射颜色）、h（原始密度）、colorMap（自定义cmap）
    return h_norm, h, colorMap  # 核心：CData → 替换为 h_norm

def plot_summary_scatter(summary_folder, marker_names, order_list, num_xbins, num_ybins):
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
        for sample_name in marker_names:
            file_name = f'{sam_name}_summary_{sample_name}.csv'
            min_color, zero_color, max_color, hist_color = get_color_combination_d(marker_names, sample_name)
            summary_file_path = os.path.join(summary_folder, file_name)
            summary_data = pd.read_csv(summary_file_path, header=0)
            summary_data = summary_data.drop_duplicates()
            # 直接在需要的地方使用条件筛选，无需单独定义positive_idx
            standardized_x = summary_data['warpedROIvar1'] / avg_vg_size[0]
            standardized_y = summary_data['warpedROIvar2'] / avg_vg_size[1]

            # 直接通过条件获取pos_x和pos_y，省去中间变量
            pos_x = standardized_x[summary_data['isPositive'] == 1].values
            pos_y = standardized_y[summary_data['isPositive'] == 1].values

            xedges = np.linspace(-0.6, 0.6, num_xbins+1)
            yedges = np.linspace(-0.6, 0.6, num_ybins+1)
            h_norm, _, colorMap = flxdensity2C(pos_x, pos_y, xedges, yedges, min_color, max_color, zero_color)
            custom_cmap = ListedColormap(colorMap)  # 转为Matplotlib可识别的颜色映射
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot2grid((8, 8), (2, 0), colspan=6, rowspan=6)
            sc = ax1.scatter(
                pos_x, pos_y,
                c=h_norm,          # 传入数值（密度），而非直接RGB
                cmap=custom_cmap,  # 绑定自定义映射（蓝灰黄）
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
            
            # -------------------------- 1. 构建自定义颜色映射（对应 MATLAB colormap(ax1, colorMap)） --------------------------
            # MATLAB的colorMap是[n_colors, 3]的RGB矩阵（0-1浮点数），Python用ListedColormap封装
            
            # -------------------------- 2. 创建颜色条独立轴（对应 MATLAB colorbar(ax1) + 位置调整） --------------------------
            # 方法：用plt.gcf()获取当前图，手动添加颜色条的独立轴（避免挤压主图）
            # MATLAB cbar.Position = [0.65, 0.52, 0.02, 0.15] → 直接复用归一化坐标
            cbar_ax = plt.gcf().add_axes([0.67, 0.44, 0.02, 0.15])  # [left, bottom, width, height]
            # 创建颜色条：绑定散点图sc、自定义颜色映射、独立轴cbar_ax
            cbar = plt.colorbar(
                mappable=sc,        # 绑定散点图（必须是带颜色映射的对象，如scatter返回的PathCollection）
                cax=cbar_ax,        # 颜色条的独立轴（对应MATLAB的ax1关联）
                cmap=custom_cmap    # 绑定自定义颜色映射（对应MATLAB colormap）
            )
            # -------------------------- 3. 设置颜色条刻度与标签（对应 MATLAB cbar.Ticks + cbar.TickLabels） --------------------------
            # MATLAB: cbar.Ticks = 0:2:4; cbar.TickLabels = {'0','2','4'}
            # Python需先设置刻度位置，再设置标签（顺序不能反）
            cbar.set_ticks([0, 0.5, 1])                # 刻度位置（与MATLAB 0:2:4一致）
            cbar.set_ticklabels(['0', '2', '4'])     # 刻度标签（与MATLAB {'0','2','4'}一致）
            # -------------------------- 4. 隐藏颜色条标签（对应 MATLAB cbar.Label.Visible = 'off'） --------------------------
            cbar.set_label('')  # 清空颜色条默认标签（或用cbar.label.set_visible(False)）
            # -------------------------- 5. 颜色条顶部添加自定义文本（对应 MATLAB text(...)） --------------------------
            # Python：在颜色条轴（cbar_ax）上添加文本，用归一化坐标（units='normalized'）
            # 处理sampleName大小写：首字母大写 + 其余小写（对应MATLAB upper(sampleName(1))+lower(sampleName(2:end))）
            sample_text = sample_name[0].upper() + sample_name[1:].lower()
            # 在颜色条顶部添加文本（x=0.5：水平居中，y=1.05：在颜色条上方，避免重叠）
            cbar_ax.text(
                x=0.5,                              # 水平位置（归一化，0=左，1=右）
                y=1.05,                             # 垂直位置（归一化，1=颜色条顶部，1.05=顶部外侧）
                s=sample_text,                      # 文本内容（处理后的sampleName）
                horizontalalignment='center',       # 水平居中（对应MATLAB 'HorizontalAlignment','center'）
                verticalalignment='bottom',         # 垂直底部对齐（对应MATLAB 'VerticalAlignment','bottom'）
                fontsize=10,                        # 字体大小（对应MATLAB 'FontSize',10）
                rotation=0                          # 旋转角度（对应MATLAB 'Rotation',0）
            )

            # -------------------------- 6. 调整颜色条标签对齐（对应 MATLAB cbar.Label.VerticalAlignment 等） --------------------------
            # 由于前面已隐藏标签（cbar.set_label('')），若后续要显示标签，可添加以下设置：
            # cbar.set_label('Density', fontsize=10)  # 先设置标签文本
            # cbar.label.set_verticalalignment('middle')   # 垂直居中（对应MATLAB cbar.Label.VerticalAlignment='middle'）
            # cbar.label.set_horizontalalignment('right')  # 水平右对齐（MATLAB 'HorizontalAlignment','center' 按需调整）
            # cbar.label.set_color('black')                # 标签颜色（对应MATLAB cbar.Label.Color='k'）


            # -------------------------- 7. 隐藏颜色条边框（对应 MATLAB cbar.Box = 'off'） --------------------------
            # MATLAB的cbar.Box控制边框，Python需隐藏颜色条的outline（轮廓）
            cbar.outline.set_visible(False)  # 去掉颜色条外面的黑色框线

            ax2 = plt.subplot2grid((8, 8), (0, 0), rowspan=2, colspan=6)
            bin_edges = np.linspace(BX_ordered.min(), BX_ordered.max(), num_xbins+1)
            ax2.hist(pos_x, bins=bin_edges, color=hist_color, alpha=0.5, density=True)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim((0, 3))
            ax2.set_yticks(np.linspace(0, 3, 4))
            ax2.tick_params(axis='x', labelbottom=False)  # 仅设置 labelbottom=False
            ax2.set_ylabel(
                'prob. dens.',          # 必选：x轴的含义（如“X坐标”“密度”）
                fontsize=10,        # 可选：字体大小（默认10，根据子图大小调整）
                fontweight='normal',  # 可选：字体粗细（'normal'默认/'bold'加粗）
                # color='b',    # 可选：字体颜色（深灰色#333333比纯黑更柔和）
                labelpad=5         # 可选：标签与x轴的距离（单位：磅，避免与刻度重叠）
            )
            ax3 = plt.subplot2grid((8, 8), (2, 6), rowspan=6)
            bin_edges = np.linspace(BY_ordered.min(), BY_ordered.max(), num_ybins+1)
            ax3.hist(pos_y, bins=bin_edges, color=hist_color, alpha=0.5, density=True, orientation='horizontal')
            ax3.set_ylim(ax1.get_ylim())
            ax3.set_xlim((0,3))
            ax3.set_xticks(np.linspace(0, 3, 4))
            ax3.tick_params(axis='y', labelleft=False)  # 仅设置 labelbottom=False
            ax3.set_xlabel(
                'prob. dens.',          # 必选：x轴的含义（如“X坐标”“密度”）
                fontsize=10,        # 可选：字体大小（默认10，根据子图大小调整）
                fontweight='normal',  # 可选：字体粗细（'normal'默认/'bold'加粗）
                # color='b',    # 可选：字体颜色（深灰色#333333比纯黑更柔和）
                labelpad=5         # 可选：标签与x轴的距离（单位：磅，避免与刻度重叠）
            )
            # 2. 循环隐藏每个子图的“上边框”和“右边框”
            axes_to_adjust = [ax1, ax2, ax3]
            for ax in axes_to_adjust:
                # 隐藏上边框
                ax.spines['top'].set_visible(False)
                # 隐藏右边框
                ax.spines['right'].set_visible(False)

            # plt.tight_layout()
            density_save_file_path = os.path.join(summary_folder, f'{sam_name}_scatter_density_{sample_name}.png')
            plt.subplots_adjust(
                left=0.15,    # 左侧留白（从0.1→0.15，给ax3的y轴刻度留空间）
                right=0.9,   # 右侧留白（从0.9→0.85，给颜色条留空间）
                bottom=0.15,  # 底部留白（从0.1→0.15，给ax1的x轴标签留空间）
                top=0.9,     # 顶部留白（从0.9→0.85，给ax2的标题留空间）
                hspace=1,   # 垂直间距（ax2与ax1之间，从0.3→0.5，避免重叠）
                wspace=0.3    # 水平间距（ax1与ax3之间，从0.3→0.5，避免拥挤）
            )
            plt.savefig(density_save_file_path, dpi=1200, bbox_inches='tight')
            plt.show()
            plt.close()
