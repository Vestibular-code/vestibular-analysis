import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, hex2color

plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial"]

def preprocess_data(wt_data):
    """Filter positive cells and identify relevant columns."""
    positive_cells = wt_data[wt_data['hasPositive'] == 1].copy()
    print(f"Filtered {len(positive_cells)} positive cells")
    
    is_positive_cols = [col for col in positive_cells.columns if col.endswith('_isPositive')]
    print(f"Found {len(is_positive_cols)} isPositive columns: {is_positive_cols}")
    
    return positive_cells, is_positive_cols

# 创建位置 bins
def create_position_bins(
    bx_min, bx_max, by_min, by_max, 
    num_xbins, num_ybins,
    target_x_range=(-0.6, 0.6),  # x方向目标范围
    target_y_range=(-0.6, 0.6)   # y方向目标范围
):
    """
    Generate x_bins and y_bins with the following logic:
    1. Generate equally-spaced base breakpoints within the base range
    2. Extend equally outward on both sides (allow exceeding target range)
    3. Finally clip breakpoints to target range (only ends may be clipped)
    4. Ensure all breakpoints are equally spaced, only boundaries may be adjusted
    
    Parameters:
        bx_min/bx_max: base x range
        by_min/by_max: base y range
        num_xbins: number of base x bins
        num_ybins: number of base y bins
        target_x_range: target x range (default -0.6~0.6)
        target_y_range: target y range (default -0.6~0.6)
    
    Returns:
        x_bins: equally-spaced breakpoints (clipped to target_x_range)
        y_bins: equally-spaced breakpoints (clipped to target_y_range)
    """
    def extend_and_cut(base_min, base_max, num_bins, target_range):
        """
        Single-dimension breakpoint generation logic:
        1. Generate base breakpoints → 2. Extend equally → 3. Clip to target range
        """
        target_min, target_max = target_range
        
        # 1. Generate base breakpoints and compute spacing
        base_bins = np.linspace(base_min, base_max, num_bins + 1)
        bin_spacing = base_bins[1] - base_bins[0]  # base spacing, kept during extension
        if bin_spacing <= 0:
            raise ValueError("Base range minimum must not be greater than or equal to maximum")
        
        # 2. Compute extension steps (left and right)
        # Left extension: until breakpoint ≤ target_min (extend 1 extra step for coverage)
        left_steps = int(np.ceil((base_min - target_min) / bin_spacing)) + 1
        # Right extension: until breakpoint ≥ target_max (extend 1 extra step for coverage)
        right_steps = int(np.ceil((target_max - base_max) / bin_spacing)) + 1
        
        # 3. Generate extended breakpoints (base + left + right extensions)
        # Left extension breakpoints: generate leftward from base left endpoint
        left_bins = base_min - np.arange(1, left_steps + 1) * bin_spacing
        # Right extension breakpoints: generate rightward from base right endpoint
        right_bins = base_max + np.arange(1, right_steps + 1) * bin_spacing
        # Merge all breakpoints (left extensions + base + right extensions)
        all_bins = np.concatenate([left_bins[::-1], base_bins, right_bins])
        
        all_bins[all_bins < target_min] = target_min
        all_bins[all_bins > target_max] = target_max
        
        unique_bins = np.unique(all_bins)
        if unique_bins[0] != target_min:
            unique_bins = np.insert(unique_bins, 0, target_min)
        if unique_bins[-1] != target_max:
            unique_bins = np.append(unique_bins, target_max)
        
        return unique_bins, bin_spacing

    # -------------------------- Generate x_bins and y_bins --------------------------
    x_bins, x_spacing = extend_and_cut(bx_min, bx_max, num_xbins, target_x_range)
    y_bins, y_spacing = extend_and_cut(by_min, by_max, num_ybins, target_y_range)
    
    # Print validation info
    print(f"x_bins: cover {target_x_range}, {len(x_bins)} breakpoints, base spacing={x_spacing:.6f}")
    print(f"y_bins: cover {target_y_range}, {len(y_bins)} breakpoints, base spacing={y_spacing:.6f}")
    
    return x_bins, y_bins

def count_positive_in_bins(data, is_positive_col, x_col, y_col, x_bins, y_bins):
    # Filter positive rows
    positive_cell = data[data[is_positive_col] == 1].copy()
    if len(positive_cell) == 0:
        print(f"⚠️ {is_positive_col} has no qualified cells")
        return np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=int)
    
    # Extract x and y coordinates from filtered data (fix: define positive_x and positive_y
    positive_x = positive_cell[x_col].values
    positive_y = positive_cell[y_col].values
    
    x_in_range = (positive_x >= x_bins[0]) & (positive_x <= x_bins[-1])
    y_in_range = (positive_y >= y_bins[0]) & (positive_y <= y_bins[-1])
    in_range_mask = x_in_range & y_in_range
    positive_x = positive_x[in_range_mask]
    positive_y = positive_y[in_range_mask]
    
    counts, _, _ = np.histogram2d(positive_y, positive_x, bins=[y_bins, x_bins], density=False)
    return counts

def monte_carlo_simulation(positive_cells, is_positive_col, x_col, y_col, bx_bins, by_bins, 
                          n_simulations, save_base_dir='simulation_results'):
    """
    Monte-Carlo simulation function (with test-statistic distribution plot saved)
    Save folder name rule: save_base_dir + (is_positive_col with '_isPositive' removed)
    """
    # 1. Handle folder name: user base directory + processed gene name
    # Extract string with '_isPositive' removed from the end (e.g. "geneA_isPositive" → "geneA")
    gene_suffix = is_positive_col.rstrip('_isPositive')
    final_save_dir = f"{save_base_dir}_{gene_suffix}"
    # 2. Compute real counts
    filtered_data = positive_cells[positive_cells[is_positive_col] != -1].copy()
    filtered_data[is_positive_col] = filtered_data[is_positive_col].astype(bool)
    real_counts = count_positive_in_bins(
        filtered_data, is_positive_col, x_col, y_col, bx_bins, by_bins
    )
    if np.sum(real_counts) == 0:
        print(f"⚠️ {is_positive_col} 真实计数为0，跳过模拟")
        return None

    # 3. Perform Monte-Carlo simulation
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

    # 4. Compute p-values (keep original logic)
    p_matrix = sim_counts >= real_counts
    p_values = (np.sum(p_matrix, axis=0)+1)/(n_simulations+1)
    
    # 6. Return results (add final_save_dir field for user convenience)
    return {
        'is_positive_col': is_positive_col,
        'gene': gene_suffix,
        'real_counts': real_counts,
        'sim_counts': sim_counts,
        'p_values': p_values,
        'corrected_p': None,
        'final_save_dir': final_save_dir,  # return final save directory path
        'final_save_dir_abspath': os.path.abspath(final_save_dir)  # return absolute path, more intuitive
    }

def count_positive_in_x_bins(data: pd.DataFrame,
                             is_positive_col: str,
                             x_col: str,
                             x_bins: np.ndarray) -> np.ndarray:
    """
    Count positive cells along x-axis bins — vectorised acceleration
    Parameters
    ----
    data : DataFrame
    is_positive_col : str  positive marker column (0/1 or False/True)
    x_col : str           coordinate column
    x_bins : 1-D array    left-closed right-open interval boundaries, length = k+1

    Returns
    ----
    counts : 1-D int array, length = k
    """
    # 1. Remove -1 values (if not filtered before calling)
    df = data[data[is_positive_col] != -1].copy()
    # 2. Bin in one go, labels=False returns bin indices 0..k-1
    bin_idx = pd.cut(df[x_col], bins=x_bins, right=False, labels=False)
    # 3. Group by bin index and sum positive marker
    counts = df.groupby(bin_idx, sort=True)[is_positive_col].sum()
    # 4. Fill missing bins (pd.cut doesn't auto-fill 0)
    counts = counts.reindex(range(len(x_bins) - 1), fill_value=0)
    return counts.values

def x_monte(data, is_positive_col, x_col, x_bins, n_simulations):
    """
    Monte-Carlo simulation along x-axis bins only
    
    Parameters:
        data: DataFrame containing cell data
        is_positive_col: column name indicating positive cells
        x_col: x-axis coordinate column name
        x_bins: x-axis bin boundaries
        n_simulations: number of simulations
        
    Returns:
        Dictionary containing real counts, simulated counts and p-values
    """
    # Filter data, exclude values marked as -1
    filtered_data = data[data[is_positive_col] != -1].copy()
    filtered_data[is_positive_col] = filtered_data[is_positive_col].astype(bool)
    
    # Compute real counts (only along x-axis)
    real_counts = count_positive_in_x_bins(
        filtered_data, is_positive_col, x_col, x_bins
    )
    
    # If real counts are 0, skip simulation
    if np.sum(real_counts) == 0:
        print(f"⚠️ {is_positive_col} real counts are 0, skipping simulation")
        return None
    
    # Initialize simulated counts array (only x-axis dimension)
    sim_counts = np.zeros(
        (n_simulations, len(x_bins) - 1), 
        dtype=int
    )
    
    # Get original positive marker values
    original_values = filtered_data[is_positive_col].values.copy()
    
    # Perform Monte-Carlo simulation
    for i in tqdm(range(n_simulations), desc=f"X-axis sim {is_positive_col}"):
        # Shuffle original values (keep distribution, only random reordering)
        shuffled = original_values.copy()
        np.random.shuffle(shuffled)
        
        # Create temporary DataFrame and assign shuffled values
        temp_df = filtered_data.copy()
        temp_df[is_positive_col] = shuffled
        
        # Compute simulated counts (only along x-axis)
        sim_counts[i] = count_positive_in_x_bins(
            temp_df, is_positive_col, x_col, x_bins
        )
    
    # Compute p-values (proportion of simulated counts >= real counts)
    p_matrix = sim_counts >= real_counts
    p_values = (np.sum(p_matrix, axis=0)+1)/(n_simulations+1)
    
    return {
        'is_positive_col': is_positive_col,
        'gene': is_positive_col.replace('_isPositive', ''),
        'real_counts': real_counts,  # 1D array, only x-axis bins
        'sim_counts': sim_counts,    # 2D array, [simulations, x-axis bins]
        'p_values': p_values,        # 1D array, p-values for x-axis bins only
        'corrected_p': None  # can add multiple testing correction later
    }

# Multiple comparison correction
def correct_grid_pvalues(p_values, method='fdr'):
    p_flat = p_values.flatten()
    n = len(p_flat)  # total number of hypothesis tests
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
    elif method == 'Holm-Bonferroni':  # Bonferroni correction
        sorted_indices = np.argsort(p_flat)
        sorted_p = p_flat[sorted_indices]  # sorted p-values (p₁ ≤ p₂ ≤ ... ≤ pₘ)
        
        # 4. Compute Holm-Bonferroni corrected p-values
        corrected_sorted_p = np.zeros_like(sorted_p)  # store corrected p-values after sorting
        for k in range(n):
            # For the k-th sorted p-value (note: k starts from 0, corresponds to k+1 in logic)
            # In logic "k+1-th p-value", adjustment factor is (m - (k+1) + 1) = m - k
            adjustment_factor = n - k  # adjustment factor: m - k (replaces Bonferroni m)
            # Correction formula: p_corrected = min(p_raw × adjustment_factor, 1.0)
            corrected_sorted_p[k] = np.minimum(1.0, sorted_p[k] * adjustment_factor)
        
        # 5. Restore corrected p-values to original array positions (sort by original indices)
        # Create empty array, fill corrected p-values by original indices
        corrected_p_flat = np.zeros_like(p_flat)
        corrected_p_flat[original_indices[sorted_indices]] = corrected_sorted_p
    else:
        print("no method")
    return corrected_p_flat.reshape(p_values.shape)
    
# ----------------------
# Plot corrected p-value heatmap
# ----------------------

def plot_corrected_p_heatmap(
    result,
    data,
    gene,
    bx_min, bx_max, by_min, by_max, 
    BX, BY, 
    output_folder, 
    min_count, max_count, 
    zero_color=[0.5, 0.5, 0.5],
    num_xbins=20,  # new: histogram bins (same as original scatter function)
    num_ybins=10,
    order_list=[0, 7, 2, 6, 4, 5, 3, 1, 0],  # new: landmark order (same as original function)
    custom_color = {'calb1': "gray", 'a': "#E3776B",'b': "#9AA218", 'c': "#1BAF73", 'd': "#26A2D5", 'e': "#B374AD"}
):
    os.makedirs(output_folder, exist_ok=True)
    gene = result['gene']
    corrected_p = result['corrected_p']a
    real_counts = result['real_counts']
    rows, cols = real_counts.shape  # grid dimensions (y rows, x columns)

    hex_color = custom_color[gene]
    target_rgb = hex2color(hex_color)  # returns (r, g, b), e.g. (0.890, 0.466, 0.427)
    white_rgb = (1.0, 1.0, 1.0)  # white RGB
    
    # 3.2 Create “white → target colour” gradient colormap
    # Define gradient nodes: position 0 → white, position 1 → target colour
    # Generate custom colormap (name is colour name, N=256 ensures smooth gradient)
    custom_cmap = LinearSegmentedColormap.from_list(gene, [white_rgb, target_rgb], N=256)
    custom_cmap_r = LinearSegmentedColormap.from_list(gene, [target_rgb, white_rgb], N=256)
    # -------------------------- 1. General utility: create subplot layout and style --------------------------
    def create_subplot_layout(fig, main_cmap):
        """Create ax1 (main heatmap), ax2 (top histogram), ax3 (right histogram) layout"""
        # Main heatmap ax1 (corresponds to ax1 position in original scatter function)
        ax1 = plt.subplot2grid((16, 8), (6, 0), rowspan=10, colspan=8)
        # Top histogram ax2 (x-direction distribution)
        ax2 = plt.subplot2grid((16, 8), (0, 0), rowspan=6, colspan=8)
        # Right histogram ax3 (y-direction distribution)
        #ax3 = plt.subplot2grid((8, 8), (2, 6), rowspan=6)

        # Hide all subplot top and right borders (same as original scatter)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Create colour bar (same position as original scatter: [left, bottom, width, height])
        # cbar_ax = fig.add_axes([0.64, 0.15, 0.01, 0.15])
        cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.15])
        # return ax1, ax2, ax3, cbar_ax
        return ax1, ax2, cbar_ax

    # -------------------------- 2. Plot「cell count heatmap」(with ax2/ax3 histograms) --------------------------
    fig1 = plt.figure(figsize=(6, 4))  # keep original figure size
    # ax1_count, ax2_count, ax3_count, cbar_ax_count = create_subplot_layout(fig1, 'Blues')
    ax1_count, ax2_count, cbar_ax_count = create_subplot_layout(fig1, 'Blues')
    # 2.1 Main panel: cell count heatmap (replace original scatter position)
    count_data = real_counts / np.sum(real_counts)  # count proportion (same as original function)
    im_count = ax1_count.pcolormesh(
        np.linspace(bx_min, bx_max, cols),  # x-axis grid (columns = cols)
        np.linspace(by_min, by_max, rows),  # y-axis grid (rows = rows)
        count_data,
        cmap=custom_cmap,  # original count heatmap colour map
        vmin=min_count,
        vmax=max_count
    )
    # 2.2 Overlay landmarks and smooth curve (same colour as original function)
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
    ax1_count.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)  # curve
    ax1_count.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)  # landmarks

    # 2.5 Colour bar (keep original heatmap colour, same style as original scatter)
    cbar_count = plt.colorbar(im_count, cax=cbar_ax_count, cmap=custom_cmap)
    cbar_count.set_ticks([min_count, (min_count+max_count)/2, max_count])  # ticks (middle value)
    cbar_count.set_ticklabels([f'{min_count:.2f}', f'{(min_count+max_count)/2:.2f}', f'{max_count:.2f}'])
    cbar_count.set_label('')
    cbar_count.outline.set_visible(False)  # hide colour bar border (same as original scatter)
    # Text on top of colour bar (corresponds to sample_text in original scatter)
    cbar_ax_count.text(
        x=0.5, y=1.10, s=gene.capitalize(),
        ha='center', va='bottom', fontsize=10, rotation=0,transform=cbar_ax_count.transAxes
    )

    # Extract positive cell data
    fig_data = data[data[f'{gene}_isPositive']==1].copy()
    # 2.3 Top ax2: x-direction count distribution histogram (sum by column, i.e. total count per x grid)
    bin_edges_x = np.linspace(x_fine.min(), x_fine.max(), num_xbins+1)
    ax2_count.hist(fig_data['standardized_x'], bins=bin_edges_x, color= cbar_count.cmap(0.5)[:3], alpha=0.5, density=True)
    # 2. Show range only [-0.5, 0.5]
    # ax2_count.sharex(ax1_count)
    ax2_count.set_xlim([x_fine.min()-0.05, x_fine.max()+0.05])
    ax2_count.set_xticks(major_ticks)
    ax2_count.set_ylim((0, 3.5))  # same y-axis range as original scatter histogram
    ax2_count.set_yticks(np.linspace(0, 3, 4))
    ax2_count.tick_params(axis='x', labelbottom=False)  # hide x-axis tick labels
    ax2_count.set_ylabel('prob. dens.', fontsize=10, labelpad=5)  # same as original scatter
    # ax2_count.set_title(f'{gene}-Grid Proportions Heatmap', pad=1)  # main plot title

    # 2.4 Right ax3: y-direction count distribution histogram (sum by row, horizontal direction
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
    # 4.1 Save count heatmap
    count_output_path = os.path.join(output_folder, f'{gene}_count.png')
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.05, top=1,
        hspace=1, wspace=2  # same subplot spacing as original scatter
    )
    fig1.savefig(count_output_path, dpi=1200, bbox_inches='tight')
    plt.close(fig1) 
    print(f"Count heatmap saved to: {count_output_path}")

    # -------------------------- 3. Plot「corrected p-value heatmap」(ax1_p fills entire figure + custom colour bar position) --------------------------
    fig2 = plt.figure(figsize=(6, 2.9))  # single plotting container, size kept 8x4
    # Let ax1_p occupy entire fig2 (instead of create_subplot_layout, no ax2/ax3)
    ax1_p = plt.gca()  # get current axis of fig2, i.e. main axis, fills entire figure by default
    ax1_p.spines['top'].set_visible(False)
    ax1_p.spines['right'].set_visible(False)

    # 3.1 Main panel: corrected p-value heatmap (drawn on full-figure ax1_p)
    im2 = ax1_p.pcolormesh(
        np.linspace(bx_min, bx_max, cols),
        np.linspace(by_min, by_max, rows),
        corrected_p,
        cmap=custom_cmap_r,
        vmin=0,
        vmax=0.05
    )

    # 3.2 Custom colour bar position: use specified [left, bottom, width, height] = [0.64, 0.15, 0.01, 0.15]
    # Create colour bar axis directly on fig2 to avoid affecting main ax1_p layout
    cbar_ax_p = fig2.add_axes([0.85, 0.15, 0.01, 0.15])
    # 2.5 Colour bar (keep original heatmap colours, style consistent with original scatter)
    cbar_p = plt.colorbar(im2, cax=cbar_ax_p)
    cbar_p.set_label('')
    cbar_p.outline.set_visible(False)  # hide colour bar border (consistent with original scatter)
    # Text on top of colour bar (corresponds to sample_text in original scatter)
    cbar_ax_p.text(
        x=0.5, y=1.10, s=gene.capitalize() + ' q-val',
        ha='center', va='bottom', fontsize=10, rotation=0, transform=cbar_ax_p.transAxes
    )

    # 3.3 Overlay landmarks and smooth curve (still drawn on full-figure ax1_p)
    ax1_p.plot(x_fine, y_fine, color=1-np.array(zero_color), linewidth=2)
    ax1_p.plot(BX, BY, '+', markersize=10, linewidth=2, color=np.array(zero_color)*0.6)

    # 3.4 Set main plot axes and title (ax1_p fills entire figure, parameters act on whole figure)
    ax1_p.set_xticks(major_ticks)
    ax1_p.set_xlim([x_fine.min()-0.05, x_fine.max()+0.05])
    ax1_p.set_ylim([by_min, by_max])
    # ax1_p.set_title(f'{gene}-FDR heatmap')

    # -------------------------- 4. Save and close figure --------------------------
    # 4.2 Save p-value heatmap
    p_output_path = os.path.join(output_folder, f'{gene}_p.png')
    fig2.savefig(p_output_path, dpi=1200, bbox_inches='tight')
    plt.close(fig2)
    print(f"p-value heatmap saved to: {p_output_path}")

