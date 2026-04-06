# /// script
# requires-python = "==3.12.*"
# dependencies = [
# "matplotlib>=3.10.8",
# "numpy>=2.4.4",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np
import os

# Data (min, mean, max)
scales = ['Nano', 'Small', 'Medium', 'Large', 'XLarge']
data = {
    'pf_seg': [
        [175.31, 178.02, 180.94], [133.13, 135.53, 138.24], [233.91, 242.42, 252.07], 
        [236.86, 241.22, 245.93], [407.25, 422.37, 439.75]
    ],
    'pf_det': [
        [131.92, 137.54, 143.86], [72.932, 75.296, 77.900], [122.83, 126.02, 129.52], 
        [184.64, 195.00, 205.75], [342.08, 358.87, 376.05]
    ],
    'p_seg': [
        [53.725, 55.733, 57.863], [83.623, 86.745, 90.183], [161.80, 165.66, 169.92], 
        [183.45, 187.57, 191.85], [386.06, 401.46, 417.96]
    ],
    'p_det': [
        [39.873, 41.508, 43.306], [55.710, 58.095, 60.747], [106.18, 109.37, 112.70], 
        [127.93, 132.41, 137.37], [236.90, 242.91, 249.31]
    ]
}

# Setup output dir
output_dir = 'benchmarks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_grid_plot():
    # Calculate global max for consistent y-axis
    global_max = 0
    for key in data:
        for d in data[key]:
            global_max = max(global_max, d[2]) # d[2] is the max value including error
    y_limit = global_max * 1.1

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor('white')
    
    plot_configs = [
        ('pf_seg', 'Prompt Free: Mask', axes[0, 0], '#e74c3c'),
        ('pf_det', 'Prompt Free: Detection', axes[0, 1], '#3498db'),
        ('p_seg', 'Promptable: Mask', axes[1, 0], '#2ecc71'),
        ('p_det', 'Promptable: Detection', axes[1, 1], '#f1c40f')
    ]
    
    for key, title, ax, color in plot_configs:
        means = [d[1] for d in data[key]]
        mins = [d[0] for d in data[key]]
        maxs = [d[2] for d in data[key]]
        
        # Calculate error values
        yerr = [
            [m - mi for m, mi in zip(means, mins)],
            [ma - m for m, ma in zip(means, maxs)]
        ]
        
        ax.set_facecolor('#fdfdfd')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc', zorder=0)
        ax.set_axisbelow(True)
        
        # Bar plot
        bars = ax.bar(scales, means, color=color, alpha=0.8, edgecolor='black', linewidth=0.8, zorder=3, width=0.6)
        
        # Error bars for deviation
        ax.errorbar(scales, means, yerr=yerr, fmt='none', ecolor='#333333', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        
        # Adding data labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (y_limit * 0.02), 
                    f'{yval:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

        ax.set_title(title, fontsize=16, pad=20, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, labelpad=10, fontweight='600', color='#34495e')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#2c3e50')
        ax.set_ylim(0, y_limit) # Use consistent global limit

    plt.tight_layout(pad=5.0)
    output_path = os.path.join(output_dir, 'benchmark_grid.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()

def create_individual_plots():
    # Calculate global max for consistent y-axis
    global_max = 0
    for key in data:
        for d in data[key]:
            global_max = max(global_max, d[2])
    y_limit = global_max * 1.1

    plot_configs = [
        ('pf_seg', 'Prompt Free: Mask', 'pf_seg.png', '#e74c3c'),
        ('pf_det', 'Prompt Free: Detection', 'pf_det.png', '#3498db'),
        ('p_seg', 'Promptable: Mask', 'p_seg.png', '#2ecc71'),
        ('p_det', 'Promptable: Detection', 'p_det.png', '#f1c40f')
    ]
    
    for key, title, filename, color in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        means = [d[1] for d in data[key]]
        mins = [d[0] for d in data[key]]
        maxs = [d[2] for d in data[key]]
        yerr = [[m - mi for m, mi in zip(means, mins)], [ma - m for m, ma in zip(means, maxs)]]
        
        ax.set_facecolor('#fdfdfd')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc', zorder=0)
        ax.set_axisbelow(True)
        bars = ax.bar(scales, means, color=color, alpha=0.8, edgecolor='black', linewidth=0.8, zorder=3, width=0.6)
        ax.errorbar(scales, means, yerr=yerr, fmt='none', ecolor='#333333', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (y_limit * 0.02), f'{yval:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, labelpad=10, fontweight='600')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, y_limit) # Use consistent global limit
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Created: benchmarks/{filename}")

create_grid_plot()
create_individual_plots()

print(f"\nSuccessfully generated 4 benchmark graphs in the '{output_dir}' directory.")
