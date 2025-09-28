import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse  # 用于处理命令行参数
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, hex2color
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import UnivariateSpline

# 设置中文字体支持
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial"]
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 数据预处理
def preprocess_data(wt_data):
    """预处理数据，筛选阳性细胞并识别相关列"""
    # 筛选hasPositive为1的阳性细胞行
    positive_cells = wt_data[wt_data['hasPositive'] == 1].copy()
    print(f"筛选出阳性细胞 {len(positive_cells)} 行")
    
    # 识别所有xxx_isPositive列
    is_positive_cols = [col for col in positive_cells.columns if col.endswith('_isPositive')]
    print(f"找到 {len(is_positive_cols)} 个isPositive列: {is_positive_cols}")
    
    return positive_cells, is_positive_cols

def two_step_stratified_sample(df, target_size, random_num):
    # 筛选所有以_isPositive结尾的列
    positive_cols = [col for col in df.columns if col.endswith('_isPositive')]
    if not positive_cols:
        raise ValueError("未找到以_isPositive结尾的列")
    
    print(f"使用以下列进行分层抽样: {', '.join(positive_cols)}")
    
    # 生成类别标签（将多列组合为一个字符串键）
    df['stratify_key'] = df[positive_cols].apply(
        lambda row: '_'.join(row.astype(str)), axis=1
    )
    
    # 合并样本数少于2的小类别
    key_counts = df['stratify_key'].value_counts()
    small_keys = set(key_counts[key_counts < 2].index)
    if small_keys:
        print(f"合并{len(small_keys)}个小类别（样本数<2）为'other'")
        df['stratify_key'] = df['stratify_key'].apply(
            lambda x: 'other' if x in small_keys else x
        )
    
    total_samples = len(df)
    if total_samples <= target_size:
        print(f"原始样本数({total_samples})≤目标，返回所有数据")
        return df.drop(columns=['stratify_key']).copy()
    
    # 第一步：进行分层抽样，获取比目标稍多的样本（1.2倍目标，为第二步预留空间）
    # 确保分层抽样得到足够样本用于后续随机抽样
    # stratified_size = min(int(target_size * 1.2), total_samples)
    stratified_ratio = target_size / total_samples
    print(f"第一步：分层抽样至{target_size}个样本")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=stratified_ratio, random_state=random_num)
    try:
        _, stratified_idx = next(sss.split(df, df['stratify_key']))
    except ValueError:
        print("分层抽样失败，使用简单随机抽样获取中间样本")
        stratified_idx = np.random.choice(total_samples, size=target_size, replace=False)
    
    # 获取分层抽样结果
    stratified_df = df.iloc[stratified_idx].drop(columns=['stratify_key']).copy()
    print(f"分层抽样后实际样本数: {len(stratified_df)}")
    
    # 第二步：从分层抽样结果中随机抽样，精确达到目标数量
    if len(stratified_df) > target_size:
        result_df = stratified_df.sample(n=target_size, random_state=random_num)
        print(f"第二步：随机抽样至目标数量{target_size}")
    else:
        # 若分层抽样结果少于目标，直接使用（通常不会发生）
        result_df = stratified_df
        print(f"分层抽样结果不足，使用全部{len(result_df)}个样本")
    
    # 输出最终结果
    print(f"最终样本数: {len(result_df)}")
    return result_df

# 创建位置 bins
def create_position_bins(
    bx_min, bx_max, by_min, by_max, 
    num_xbins, num_ybins,
    target_x_range=(-0.6, 0.6),  # x方向目标范围
    target_y_range=(-0.6, 0.6)   # y方向目标范围
):
    """
    生成x_bins和y_bins，逻辑为：
    1. 在基础范围内生成指定数量的等间距基础分点
    2. 按基础分点的间距向两侧无限制扩展（允许超出目标范围）
    3. 最后将分点截断到目标范围内（仅首尾可能被截断）
    4. 确保所有分点严格等间距，仅边界可能被调整
    
    参数：
        bx_min/bx_max：x方向基础范围
        by_min/by_max：y方向基础范围
        num_xbins：x方向基础分箱数
        num_ybins：y方向基础分箱数
        target_x_range：x方向目标范围（默认-0.6~0.6）
        target_y_range：y方向目标范围（默认-0.6~0.6）
    
    返回：
        x_bins：等间距分点（边界被截断到target_x_range）
        y_bins：等间距分点（边界被截断到target_y_range）
    """
    # -------------------------- 工具函数：生成单个方向的分点 --------------------------
    def extend_and_cut(base_min, base_max, num_bins, target_range):
        """
        单个方向的分点生成逻辑：
        1. 生成基础分点 → 2. 等间距扩展 → 3. 截断到目标范围
        """
        target_min, target_max = target_range
        
        # 1. 生成基础分点并计算间距
        base_bins = np.linspace(base_min, base_max, num_bins + 1)
        bin_spacing = base_bins[1] - base_bins[0]  # 基础间距，扩展时保持不变
        if bin_spacing <= 0:
            raise ValueError("基础范围的最小值不能大于等于最大值")
        
        # 2. 计算需要扩展的步数（向左和向右）
        # 向左扩展：直到分点 ≤ target_min（多扩展1步确保覆盖）
        left_steps = int(np.ceil((base_min - target_min) / bin_spacing)) + 1
        # 向右扩展：直到分点 ≥ target_max（多扩展1步确保覆盖）
        right_steps = int(np.ceil((target_max - base_max) / bin_spacing)) + 1
        
        # 3. 生成扩展后的所有分点（基础分点 + 左右扩展分点）
        # 左扩展分点：从基础左端点向左按间距生成
        left_bins = base_min - np.arange(1, left_steps + 1) * bin_spacing
        # 右扩展分点：从基础右端点向右按间距生成
        right_bins = base_max + np.arange(1, right_steps + 1) * bin_spacing
        # 合并所有分点（左扩展 + 基础 + 右扩展）
        all_bins = np.concatenate([left_bins[::-1], base_bins, right_bins])
        
        # 4. 截断到目标范围（仅调整超出的分点）
        all_bins[all_bins < target_min] = target_min  # 左侧超出部分设为目标最小值
        all_bins[all_bins > target_max] = target_max  # 右侧超出部分设为目标最大值
        
        # 5. 去除重复分点（可能因截断产生）并保持排序
        unique_bins = np.unique(all_bins)
        
        # 6. 确保首尾严格为目标范围（避免空集或意外情况）
        if unique_bins[0] != target_min:
            unique_bins = np.insert(unique_bins, 0, target_min)
        if unique_bins[-1] != target_max:
            unique_bins = np.append(unique_bins, target_max)
        
        return unique_bins, bin_spacing

    # -------------------------- 生成x_bins和y_bins --------------------------
    x_bins, x_spacing = extend_and_cut(bx_min, bx_max, num_xbins, target_x_range)
    y_bins, y_spacing = extend_and_cut(by_min, by_max, num_ybins, target_y_range)
    
    # 打印验证信息
    print(f"x_bins：覆盖{target_x_range}，共{len(x_bins)}个分点，基础间距={x_spacing:.6f}")
    print(f"y_bins：覆盖{target_y_range}，共{len(y_bins)}个分点，基础间距={y_spacing:.6f}")
    
    return x_bins, y_bins

def count_positive_in_bins(data, is_positive_col, x_col, y_col, x_bins, y_bins):
    # 筛选出阳性的行
    positive_cell = data[data[is_positive_col] == 1].copy()
    if len(positive_cell) == 0:
        print(f"⚠️ {is_positive_col} 没有符合条件的细胞")
        return np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=int)
    
    # 从筛选后的数据中提取x和y坐标（修复：定义positive_x和positive_y）
    positive_x = positive_cell[x_col].values  # 提取x列的值
    positive_y = positive_cell[y_col].values  # 提取y列的值
    
    # 检查坐标是否在bins范围内（增加微小误差范围，避免边界问题）
    x_in_range = (positive_x >= x_bins[0]) & (positive_x <= x_bins[-1])
    y_in_range = (positive_y >= y_bins[0]) & (positive_y <= y_bins[-1])
    in_range_mask = x_in_range & y_in_range
    
    # 过滤超出范围的点
    positive_x = positive_x[in_range_mask]
    positive_y = positive_y[in_range_mask]
    
    # 计算二维直方图（统计每个bin中的数量）
    counts, _, _ = np.histogram2d(positive_y, positive_x, bins=[y_bins, x_bins], density=False)
    # counts = counts/len(data)
    return counts

def monte_carlo_simulation(positive_cells, is_positive_col, x_col, y_col, bx_bins, by_bins, 
                          n_simulations, save_base_dir='simulation_results'):
    """
    蒙特卡洛模拟函数（带检验量分布图保存）
    保存文件夹名规则：save_base_dir + (is_positive_col去掉末尾'_isPositive'后的字符串)
    """
    # 1. 处理文件夹名称：用户输入的基础目录 + 处理后的基因名
    # 提取is_positive_col去掉末尾"_isPositive"的字符串（如"geneA_isPositive"→"geneA"）
    gene_suffix = is_positive_col.rstrip('_isPositive')  # 确保仅移除末尾的"_isPositive"
    # 拼接最终保存目录（如save_base_dir="output"，gene_suffix="geneA"→"output_geneA"）
    final_save_dir = f"{save_base_dir}_{gene_suffix}"
    
    # 创建最终保存目录（若不存在则自动创建）
    #os.makedirs(final_save_dir, exist_ok=True)
    #print(f"📁 检验量分布图将保存至：{os.path.abspath(final_save_dir)}")

    # 2. 计算真实计数（原有逻辑保留）
    filtered_data = positive_cells[positive_cells[is_positive_col] != -1].copy()
    filtered_data[is_positive_col] = filtered_data[is_positive_col].astype(bool)
    real_counts = count_positive_in_bins(
        filtered_data, is_positive_col, x_col, y_col, bx_bins, by_bins
    )
    
    # 若真实计数为0，跳过模拟
    if np.sum(real_counts) == 0:
        print(f"⚠️ {is_positive_col} 真实计数为0，跳过模拟")
        return None

    # 3. 执行蒙特卡洛模拟（原有逻辑保留）
    sim_counts = np.zeros(
        (n_simulations, real_counts.shape[0], real_counts.shape[1]), 
        dtype=int
    )
    original_values = filtered_data[is_positive_col].values.copy()
    
    for i in tqdm(range(n_simulations), desc=f"模拟 {is_positive_col}"):
        np.random.seed(i)
        shuffled = original_values.copy()
        np.random.shuffle(shuffled)
        
        temp_df = filtered_data.copy()
        temp_df[is_positive_col] = shuffled
        
        sim_counts[i] = count_positive_in_bins(
            temp_df, is_positive_col, x_col, y_col, bx_bins, by_bins
        )

    # 4. 计算p值（原有逻辑保留）
    p_matrix = sim_counts >= real_counts
    p_values = (np.sum(p_matrix, axis=0)+1)/(n_simulations+1)
    """
    # 5. 绘制并保存检验量分布图（优化目录路径）
    rows, cols = real_counts.shape
    for i in range(rows):
        for j in range(cols):
            # 获取当前网格的模拟数据、真实值和p值
            bin_sim = sim_counts[:, i, j]
            bin_real = real_counts[i, j]
            bin_p = p_values[i, j]

            # 创建图形
            plt.figure(figsize=(6, 4))
            
            # 绘制模拟分布直方图（计数为整数， bins按整数对齐）
            bins = np.arange(sim_counts.min(), sim_counts.max() + 2) - 0.5
            plt.hist(bin_sim, bins=bins, alpha=0.7, color='#4CAF50', edgecolor='black', 
                     label=f'模拟分布（{n_simulations}次）')
            
            # 标注真实值（红色虚线，突出显示）
            plt.axvline(x=bin_real, color='#F44336', linestyle='--', linewidth=3, 
                       label=f'真实计数: {bin_real}', zorder=5)  # zorder确保线在直方图上方
            
            # 添加p值文本框（右上角，不遮挡数据）
            plt.text(0.98, 0.95, f'p值: {bin_p:.4f}', 
                    horizontalalignment='right', verticalalignment='top',
                    transform=plt.gca().transAxes, fontsize=5,
                    bbox=dict(facecolor='white', edgecolor='#2196F3', alpha=0.9, boxstyle='round,pad=0.5'))
            
            # 设置图形样式
            plt.title(f'网格 ({i+1},{j+1}) 阳性细胞计数分布\n（基因：{gene_suffix}）', 
                      fontsize=14, pad=20)
            plt.xlabel('阳性细胞计数（整数）', fontsize=5, labelpad=10)
            plt.ylabel('模拟次数（频数）', fontsize=5, labelpad=10)
            plt.legend(loc='upper right', fontsize=5)
            
            # x轴强制为整数（符合计数数据特性）
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            # 优化网格显示
            plt.grid(axis='y', alpha=0.3, linestyle='-')
            
            # 保存图形（路径为最终目录，文件名含网格坐标）
            plot_name = f'grid_{i+1}_{j+1}_count_distribution.png'
            plot_path = os.path.join(final_save_dir, plot_name)
            plt.tight_layout()  # 自动调整布局，避免标签截断
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')  # 300dpi确保高清
            plt.close()  # 关闭图形释放内存
    """
    # 6. 返回结果（新增final_save_dir字段，方便用户定位文件）
    return {
        'is_positive_col': is_positive_col,
        'gene': gene_suffix,
        'real_counts': real_counts,
        'sim_counts': sim_counts,
        'p_values': p_values,
        'corrected_p': None,
        'final_save_dir': final_save_dir,  # 返回最终保存目录的路径
        'final_save_dir_abspath': os.path.abspath(final_save_dir)  # 返回绝对路径，更直观
    }

def count_positive_in_x_bins(data: pd.DataFrame,
                             is_positive_col: str,
                             x_col: str,
                             x_bins: np.ndarray) -> np.ndarray:
    """
    基于 x 轴分 bin 计算阳性细胞计数 —— 向量化加速版
    参数
    ----
    data : DataFrame
    is_positive_col : str  阳性标记列（0/1 或 False/True）
    x_col : str           坐标列
    x_bins : 1-D array    左闭右开区间边界，长度 = k+1

    返回
    ----
    counts : 1-D int array, length = k
    """
    # 1. 先扔掉 -1（如果调用前没过滤）
    df = data[data[is_positive_col] != -1].copy()
    # 2. 一次性分箱，labels=False 返回 bin 编号 0..k-1
    bin_idx = pd.cut(df[x_col], bins=x_bins, right=False, labels=False)
    # 3. 按 bin 编号分组，对阳性列求和
    counts = df.groupby(bin_idx, sort=True)[is_positive_col].sum()
    # 4. 补全可能出现的空 bin（pd.cut 不会自动补 0）
    counts = counts.reindex(range(len(x_bins) - 1), fill_value=0)
    return counts.values

def x_monte(data, is_positive_col, x_col, x_bins, n_simulations):
    """
    仅对x轴分bin进行蒙特卡洛模拟计算显著性
    
    参数:
        data: 包含细胞数据的DataFrame
        is_positive_col: 指示细胞是否阳性的列名
        x_col: x轴坐标列名
        x_bins: x轴分箱边界
        n_simulations: 模拟次数
        
    返回:
        包含真实计数、模拟计数和p值的字典
    """
    # 过滤数据，排除标记为-1的值
    filtered_data = data[data[is_positive_col] != -1].copy()
    filtered_data[is_positive_col] = filtered_data[is_positive_col].astype(bool)
    
    # 计算真实计数（仅基于x轴）
    real_counts = count_positive_in_x_bins(
        filtered_data, is_positive_col, x_col, x_bins
    )
    
    # 若真实计数为0，跳过模拟
    if np.sum(real_counts) == 0:
        print(f"⚠️ {is_positive_col} 真实计数为0，跳过模拟")
        return None
    
    # 初始化模拟计数数组（仅x轴维度）
    sim_counts = np.zeros(
        (n_simulations, len(x_bins) - 1), 
        dtype=int
    )
    
    # 获取原始阳性标记值
    original_values = filtered_data[is_positive_col].values.copy()
    
    # 执行蒙特卡洛模拟
    for i in tqdm(range(n_simulations), desc=f"X轴模拟 {is_positive_col}"):
        # 打乱原始值（保持分布，仅随机重排）
        shuffled = original_values.copy()
        np.random.shuffle(shuffled)
        
        # 创建临时DataFrame并赋值打乱后的值
        temp_df = filtered_data.copy()
        temp_df[is_positive_col] = shuffled
        
        # 计算模拟计数（仅基于x轴）
        sim_counts[i] = count_positive_in_x_bins(
            temp_df, is_positive_col, x_col, x_bins
        )
    
    # 计算p值（模拟计数 >= 真实计数的比例）
    p_matrix = sim_counts >= real_counts
    p_values = (np.sum(p_matrix, axis=0)+1)/(n_simulations+1)
    
    return {
        'is_positive_col': is_positive_col,
        'gene': is_positive_col.replace('_isPositive', ''),
        'real_counts': real_counts,  # 1D数组，仅x轴分bin
        'sim_counts': sim_counts,    # 2D数组，[模拟次数, x轴分bin数]
        'p_values': p_values,        # 1D数组，仅x轴分bin的p值
        'corrected_p': None  # 可后续添加多重检验校正
    }

# 多重比较矫正
def correct_grid_pvalues(p_values, method='fdr'):
    p_flat = p_values.flatten()
    n = len(p_flat)  # n为假设检验的总次数
    original_indices = np.arange(n)
    if method == 'fdr':
        sorted_indices = np.argsort(p_flat)
        sorted_p = p_flat[sorted_indices]
        correction = np.arange(1, n+1) / n
        corrected_sorted_p = np.minimum(1.0, sorted_p / correction)
        
        for i in range(n-2, -1, -1):
            corrected_sorted_p[i] = min(corrected_sorted_p[i], corrected_sorted_p[i+1])
            
        corrected_p_flat = np.zeros(n)
        corrected_p_flat[sorted_indices] = corrected_sorted_p
    elif method == 'Holm-Bonferroni':  # bonferroni校正
        sorted_indices = np.argsort(p_flat)
        sorted_p = p_flat[sorted_indices]  # 排序后的p值数组（p₁ ≤ p₂ ≤ ... ≤ pₘ）
        
        # 4. 计算Holm-Bonferroni校正后的p值
        corrected_sorted_p = np.zeros_like(sorted_p)  # 存储排序后的校正p值
        for k in range(n):
            # 第k个排序p值（注意：k从0开始，对应逻辑中的k+1）
            # 逻辑中的“第k+1个p值”，对应的调整因子为 (m - (k+1) + 1) = m - k
            adjustment_factor = n - k  # 调整因子：m - k（替代Bonferroni的m）
            # 校正公式：p_corrected = min(p_raw × 调整因子, 1.0)
            corrected_sorted_p[k] = np.minimum(1.0, sorted_p[k] * adjustment_factor)
        
        # 5. 恢复校正p值到原数组的位置（按原始索引排序）
        # 创建空数组，按原始索引填入校正后的p值
        corrected_p_flat = np.zeros_like(p_flat)
        corrected_p_flat[original_indices[sorted_indices]] = corrected_sorted_p
    else:
        print("no method")
    return corrected_p_flat.reshape(p_values.shape)
    
# ----------------------
# 新增函数: 绘制校正后p值的热图
# ----------------------
"""
def plot_corrected_p_heatmap(result, bx_min, bx_max, by_min, by_max, BX, BY, output_folder, min_count, max_count, zero_color=[0.5, 0.5, 0.5]):
    os.makedirs(output_folder, exist_ok=True)
    gene = result['gene']
    corrected_p = result['corrected_p']
    real_counts = result['real_counts']
    
    # 获取网格的形状
    rows, cols = real_counts.shape
    
    # --------------------------
    # 绘制细胞计数热图并叠加标定点
    # --------------------------
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    
    # 绘制计数热图
    im1 = ax1.pcolormesh(
        np.linspace(bx_min, bx_max, cols),
        np.linspace(by_min, by_max, rows),
        real_counts/np.sum(real_counts),
        cmap='Blues',
        vmin=min_count,
        vmax=max_count
    )
    plt.colorbar(im1, ax=ax1, label='细胞计数比例')
    
    # 叠加标定点和连线
    order = [0, 7, 2, 6, 4, 5, 3, 1, 0]
    BX_ordered = BX[order]
    BY_ordered = BY[order]
    
    # 创建平滑曲线
    cs = CubicSpline(np.arange(len(BX_ordered)), np.c_[BX_ordered, BY_ordered], 
                    axis=0, bc_type='periodic')
    t_fine = np.linspace(0, len(BX_ordered)-1, 300)
    x_fine, y_fine = cs(t_fine).T
    
    # 绘制曲线和标定点
    ax1.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)
    ax1.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)
    
    # 设置坐标轴和标题
    ax1.set_xlim([bx_min, bx_max])
    ax1.set_ylim([by_min, by_max])
    plt.title(f'{gene} 阳性细胞计数热图')
    plt.xlabel('标准化X 网格')
    plt.ylabel('标准化Y 网格')
    
    plt.tight_layout()
    # 保存计数热图
    count_output_path = os.path.join(output_folder, f'{gene}_count.png')
    plt.savefig(count_output_path, dpi=1200)
    plt.close()
    print(f"已保存计数热图至: {count_output_path}")
    
    # --------------------------
    # 绘制校正后p值热图
    # --------------------------
    plt.figure(figsize=(8, 4))
    ax2 = plt.gca()
    
    # 绘制p值热图
    im2 = ax2.pcolormesh(
        np.linspace(bx_min, bx_max, cols),
        np.linspace(by_min, by_max, rows),
        corrected_p,
        cmap='Reds_r',
        vmin=0,
        vmax=0.05
    )
    plt.colorbar(im2, ax=ax2, label='校正后p值 (0-0.1)')
    
    ax2.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)
    ax2.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)

    # 设置坐标轴和标题
    ax2.set_xlim([bx_min, bx_max])
    ax2.set_ylim([by_min, by_max])
    plt.title(f'{gene} 校正后p值热图')
    plt.xlabel('标准化X 网格')
    plt.ylabel('标准化Y 网格')
    
    plt.tight_layout()
    # 保存p值热图
    p_output_path = os.path.join(output_folder, f'{gene}_p.png')
    plt.savefig(p_output_path, dpi=1200)
    plt.close()
    print(f"已保存p值热图至: {p_output_path}")
"""

def plot_corrected_p_heatmap(
    result,
    data,
    gene,
    bx_min, bx_max, by_min, by_max, 
    BX, BY, 
    output_folder, 
    min_count, max_count, 
    zero_color=[0.5, 0.5, 0.5],
    num_xbins=20,  # 新增：直方图分箱数（与原scatter函数一致）
    num_ybins=10,
    order_list=[0, 7, 2, 6, 4, 5, 3, 1, 0],  # 新增：标定点顺序（默认同原函数）
    custom_color = {'calb1': "gray", 'd': "#E3776B",'e': "#9AA218", 'b': "#1BAF73", 'a': "#26A2D5", 'c': "#B374AD"}
):
    os.makedirs(output_folder, exist_ok=True)
    gene = result['gene']
    corrected_p = result['corrected_p']
    real_counts = result['real_counts']
    rows, cols = real_counts.shape  # 网格维度（y行x列）

    hex_color = custom_color[gene]
    target_rgb = hex2color(hex_color)  # 返回 (r, g, b)，如(0.890, 0.466, 0.427)
    white_rgb = (1.0, 1.0, 1.0)  # 白色的RGB
    
    # 3.2 创建“白→目标色”的渐变色图（Colormap）
    # 定义渐变节点：位置0→白色，位置1→目标色
    # 生成自定义Colormap（名称为颜色名，N=256确保渐变平滑）
    custom_cmap = LinearSegmentedColormap.from_list(gene, [white_rgb, target_rgb], N=256)
    custom_cmap_r = LinearSegmentedColormap.from_list(gene, [target_rgb, white_rgb], N=256)
    # -------------------------- 1. 通用工具函数：创建子图布局与样式 --------------------------
    def create_subplot_layout(fig, main_cmap):
        """创建ax1（主热图）、ax2（顶部直方图）、ax3（右侧直方图）的布局"""
        # 主热图ax1（对应原scatter函数的ax1位置）
        ax1 = plt.subplot2grid((16, 8), (6, 0), rowspan=10, colspan=8)
        # 顶部直方图ax2（x方向分布）
        ax2 = plt.subplot2grid((16, 8), (0, 0), rowspan=6, colspan=8)
        # 右侧直方图ax3（y方向分布）
        #ax3 = plt.subplot2grid((8, 8), (2, 6), rowspan=6)

        # 隐藏所有子图的上边框和右边框（与原scatter一致）
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # 创建颜色条（位置与原scatter一致：[left, bottom, width, height]）
        # cbar_ax = fig.add_axes([0.64, 0.15, 0.01, 0.15])
        cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.15])
        # return ax1, ax2, ax3, cbar_ax
        return ax1, ax2, cbar_ax

    # -------------------------- 2. 绘制「细胞计数热图」（带ax2/ax3直方图） --------------------------
    fig1 = plt.figure(figsize=(6, 4))  # 保持原图尺寸
    # ax1_count, ax2_count, ax3_count, cbar_ax_count = create_subplot_layout(fig1, 'Blues')
    ax1_count, ax2_count, cbar_ax_count = create_subplot_layout(fig1, 'Blues')
    # 2.1 主图：细胞计数热图（替代原scatter位置）
    count_data = real_counts / np.sum(real_counts)  # 计数比例（与原函数一致）
    im_count = ax1_count.pcolormesh(
        np.linspace(bx_min, bx_max, cols),  # x轴网格（列数=cols）
        np.linspace(by_min, by_max, rows),  # y轴网格（行数=rows）
        count_data,
        cmap=custom_cmap,  # 原计数热图颜色映射
        vmin=min_count,
        vmax=max_count
    )
    # 2.2 叠加标定点与平滑曲线（与原函数颜色一致）
    BX_ordered = BX[order_list]
    BY_ordered = BY[order_list]

    cs = CubicSpline(np.arange(len(BX_ordered)), np.c_[BX_ordered, BY_ordered], 
                    axis=0, bc_type='periodic')
    t_fine = np.linspace(0, len(BX_ordered)-1, 300)
    x_fine, y_fine = cs(t_fine).T

    major_ticks = np.round(np.arange(x_fine.min()-0.05, x_fine.max()+0.05, 0.1), 1) 
    ax1_count.set_xticks(major_ticks)
    ax1_count.set_xticklabels([f'{x:g}' for x in major_ticks])
    ax1_count.set_xlim([x_fine.min()-0.05, x_fine.max()+0.05])
    ax1_count.set_ylim([by_min, by_max])
    ax1_count.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)  # 曲线
    ax1_count.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)  # 标定点

    # 2.5 颜色条（保持原热图颜色，样式与原scatter一致）
    cbar_count = plt.colorbar(im_count, cax=cbar_ax_count, cmap=custom_cmap)
    cbar_count.set_ticks([min_count, (min_count+max_count)/2, max_count])  # 刻度（中间值）
    cbar_count.set_ticklabels([f'{min_count:.2f}', f'{(min_count+max_count)/2:.2f}', f'{max_count:.2f}'])
    cbar_count.set_label('')
    cbar_count.outline.set_visible(False)  # 隐藏颜色条边框（与原scatter一致）
    # 颜色条顶部文本（对应原scatter的sample_text）
    cbar_ax_count.text(
        x=0.5, y=1.10, s=gene.capitalize(),
        ha='center', va='bottom', fontsize=10, rotation=0,transform=cbar_ax_count.transAxes
    )

    # 提取阳性细胞数据
    fig_data = data[data[f'{gene}_isPositive']==1].copy()
    # 2.3 顶部ax2：x方向计数分布直方图（按列求和，即每个x网格的总计数）
    bin_edges_x = np.linspace(x_fine.min(), x_fine.max(), num_xbins+1)
    ax2_count.hist(fig_data['standardized_x'], bins=bin_edges_x, color= cbar_count.cmap(0.5)[:3], alpha=0.5, density=True)
    # 2. 显示范围只露出 [-0.5, 0.5]
    # ax2_count.sharex(ax1_count)
    ax2_count.set_xlim([x_fine.min()-0.05, x_fine.max()+0.05])
    ax2_count.set_xticks(major_ticks)
    ax2_count.set_ylim((0, 3.5))  # 与原scatter直方图y轴范围一致
    ax2_count.set_yticks(np.linspace(0, 3, 4))
    ax2_count.tick_params(axis='x', labelbottom=False)  # 隐藏x轴刻度标签
    ax2_count.set_ylabel('prob. dens.', fontsize=10, labelpad=5)  # 与原scatter一致
    # ax2_count.set_title(f'{gene}-Grid Proportions Heatmap', pad=1)  # 主图标题

    # 2.4 右侧ax3：y方向计数分布直方图（按行求和，水平方向
    """
    fig_data = data[data[f'{gene}_isPositive']==1].copy()
    bin_edges_y = np.linspace(BY.min(), BY.max(), num_ybins+1)
    ax3_count.hist(fig_data['standardized_y'], bins=bin_edges_y, color= cbar_count.cmap(0.5)[:3], alpha=0.5, density=True, orientation='horizontal')
    ax3_count.set_ylim(ax1_count.get_ylim())
    ax3_count.set_xlim((0, 3))  # 与原scatter直方图x轴范围一致
    ax3_count.set_xticks(np.linspace(0, 3, 4))
    ax3_count.tick_params(axis='y', labelleft=False)  # 隐藏y轴刻度标签
    ax3_count.set_xlabel('prob. dens.', fontsize=10, labelpad=5)  # 与原scatter一致
    """
    # 4.1 保存计数热图
    count_output_path = os.path.join(output_folder, f'{gene}_count.png')
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.05, top=1,
        hspace=1, wspace=2  # 与原scatter子图间距一致
    )
    fig1.savefig(count_output_path, dpi=1200, bbox_inches='tight')
    plt.close(fig1) 
    print(f"已保存计数热图至: {count_output_path}")
    # -------------------------- 3. 绘制「校正后p值热图」（ax1_p占满画幅 + 自定义颜色条位置） --------------------------
    fig2 = plt.figure(figsize=(6, 2.9))  # 唯一绘图容器，尺寸保持8x4
    # 让ax1_p占据整个fig2画幅（替代create_subplot_layout，无ax2/ax3）
    ax1_p = plt.gca()  # 获取fig2的当前轴，即主图轴，默认占满整个图形
    ax1_p.spines['top'].set_visible(False)
    ax1_p.spines['right'].set_visible(False)
    # 3.1 主图：校正后p值热图（绘制在占满画幅的ax1_p上）
    im2 = ax1_p.pcolormesh(
        np.linspace(bx_min, bx_max, cols),
        np.linspace(by_min, by_max, rows),
        corrected_p,
        cmap=custom_cmap_r,
        vmin=0,
        vmax=0.05
    )

    # 3.2 自定义颜色条位置：使用指定的[left, bottom, width, height] = [0.64, 0.15, 0.01, 0.15]
    # 直接在fig2上创建颜色条轴，避免影响主图ax1_p的布局
    cbar_ax_p = fig2.add_axes([0.85, 0.15, 0.01, 0.15])
    # 2.5 颜色条（保持原热图颜色，样式与原scatter一致）
    cbar_p = plt.colorbar(im2, cax=cbar_ax_p)
    cbar_p.set_label('')
    cbar_p.outline.set_visible(False)  # 隐藏颜色条边框（与原scatter一致）
    # 颜色条顶部文本（对应原scatter的sample_text）
    cbar_ax_p.text(
        x=0.5, y=1.10, s=gene.capitalize() + ' q-val',
        ha='center', va='bottom', fontsize=10, rotation=0,transform=cbar_ax_p.transAxes
    )
    # 3.3 叠加标定点与平滑曲线（仍绘制在占满画幅的ax1_p上）
    ax1_p.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)
    ax1_p.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)
    # 3.4 设置主图坐标轴和标题（ax1_p占满画幅，参数直接作用于整个图形）
    ax1_p.set_xticks(major_ticks)
    ax1_p.set_xlim([x_fine.min()-0.05, x_fine.max()+0.05])
    ax1_p.set_ylim([by_min, by_max])
    # ax1_p.set_title(f'{gene}-FDR heatmap')
    # -------------------------- 4. 保存与关闭图片 --------------------------
    # 4.2 保存p值热图
    p_output_path = os.path.join(output_folder, f'{gene}_p.png')
    fig2.savefig(p_output_path, dpi=1200, bbox_inches='tight')
    plt.close(fig2)
    print(f"已保存p值热图至: {p_output_path}")

