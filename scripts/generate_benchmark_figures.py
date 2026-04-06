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
import re

RAW_BENCHMARK_DATA = """
Running benches/full_benchmarks.rs (target/release/deps/full_benchmarks-9fd2ebd2bf25f117)
Gnuplot not found, using plotters backend
full_predict/prompt_free/nano/seg
                        time:   [20.330 ms 20.397 ms 20.472 ms]
                        change: [−49.300% −48.653% −48.053%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) high mild
  5 (5.00%) high severe
full_predict/prompt_free/nano/det
                        time:   [18.012 ms 18.061 ms 18.122 ms]
                        change: [−3.9323% −2.2669% −0.8231%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 12 outliers among 100 measurements (12.00%)
  1 (1.00%) low mild
  2 (2.00%) high mild
  9 (9.00%) high severe
full_predict/promptable/nano/seg
                        time:   [6.0484 ms 6.0599 ms 6.0739 ms]
                        change: [−31.707% −30.933% −30.181%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe
full_predict/promptable/nano/det
                        time:   [4.8933 ms 4.9026 ms 4.9142 ms]
                        change: [−6.3849% −5.7212% −5.1746%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) high mild
  6 (6.00%) high severe
full_predict/prompt_free/small/seg
                        time:   [17.412 ms 17.465 ms 17.529 ms]
                        change: [−53.910% −52.998% −52.091%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  2 (2.00%) high mild
  5 (5.00%) high severe
full_predict/prompt_free/small/det
                        time:   [14.542 ms 14.576 ms 14.619 ms]
                        change: [−3.6260% −3.2866% −2.9376%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  10 (10.00%) high severe
full_predict/promptable/small/seg
                        time:   [8.1658 ms 8.1786 ms 8.1941 ms]
                        change: [−24.781% −23.838% −22.923%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) high severe
full_predict/promptable/small/det
                        time:   [6.4999 ms 6.5115 ms 6.5270 ms]
                        change: [−7.6469% −6.3621% −5.1696%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 8 outliers among 100 measurements (8.00%)
  8 (8.00%) high severe
full_predict/prompt_free/medium/seg
                        time:   [23.833 ms 23.902 ms 23.980 ms]
                        change: [−51.026% −50.023% −48.993%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 100 measurements (9.00%)
  1 (1.00%) high mild
  8 (8.00%) high severe
full_predict/prompt_free/medium/det
                        time:   [19.481 ms 19.529 ms 19.585 ms]
                        change: [−6.2923% −5.7704% −5.2567%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 11 outliers among 100 measurements (11.00%)
  11 (11.00%) high severe
full_predict/promptable/medium/seg
                        time:   [14.382 ms 14.406 ms 14.436 ms]
                        change: [−22.955% −21.665% −20.386%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  2 (2.00%) high mild
  8 (8.00%) high severe
full_predict/promptable/medium/det
                        time:   [10.940 ms 10.960 ms 10.984 ms]
                        change: [−3.5249% −2.8969% −2.3295%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  6 (6.00%) high mild
  4 (4.00%) high severe
full_predict/prompt_free/large/seg
                        time:   [25.990 ms 26.043 ms 26.102 ms]
change: [−46.447% −45.656% −44.888%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  5 (5.00%) high mild
  5 (5.00%) high severe
full_predict/prompt_free/large/det
                        time:   [21.218 ms 21.280 ms 21.360 ms]
                        change: [−1.4477% −0.7929% −0.2368%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 10 outliers among 100 measurements (10.00%)
  1 (1.00%) high mild
  9 (9.00%) high severe
full_predict/promptable/large/seg
                        time:   [16.675 ms 16.711 ms 16.758 ms]
                        change: [−15.426% −14.312% −13.240%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  5 (5.00%) high mild
  5 (5.00%) high severe
full_predict/promptable/large/det
                        time:   [13.220 ms 13.246 ms 13.278 ms]
                        change: [−2.1040% −1.6909% −1.3175%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  1 (1.00%) high mild
  9 (9.00%) high severe
Benchmarking full_predict/prompt_free/xlarge/seg: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.6s, or reduce sample count to 80.
full_predict/prompt_free/xlarge/seg
                        time:   [36.589 ms 36.738 ms 36.906 ms]
                        change: [−37.834% −36.389% −35.011%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 100 measurements (3.00%)
  1 (1.00%) high mild
  2 (2.00%) high severe
full_predict/prompt_free/xlarge/det
                        time:   [28.808 ms 28.873 ms 28.949 ms]
                        change: [−4.9240% −3.4595% −2.1694%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 100 measurements (9.00%)
  1 (1.00%) high mild
  8 (8.00%) high severe
full_predict/promptable/xlarge/seg
                        time:   [27.532 ms 27.587 ms 27.649 ms]
                        change: [−7.8955% −7.3543% −6.8738%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  8 (8.00%) high mild
  2 (2.00%) high severe
full_predict/promptable/xlarge/det
                        time:   [21.038 ms 21.080 ms 21.132 ms]
                        change: [−6.8979% −5.4039% −3.9997%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 100 measurements (9.00%)
  7 (7.00%) high mild
  2 (2.00%) high severe
"""


# ==========================================

def parse_benchmark_data(raw_text):
    scales = ['nano', 'small', 'medium', 'large', 'xlarge']
    parsed = {
        'pf_seg': {}, 'pf_det': {},
        'p_seg': {}, 'p_det': {}
    }

    def convert_to_ms(value, unit):
        if unit == 'µs': return value / 1000.0
        if unit == 'ns': return value / 1000000.0
        if unit == 's': return value * 1000.0
        return value  # already ms

    pattern = re.compile(
        r"full_predict/(?P<mode>prompt_free|promptable)/(?P<scale>\w+)/(?P<type>seg|det)\s+"
        r"time:\s+\[(?P<min_v>[\d.]+) (?P<min_u>\w+) (?P<mean_v>[\d.]+) (?P<mean_u>\w+) (?P<max_v>[\d.]+) (?P<max_u>\w+)\]"
    )

    matches = pattern.finditer(raw_text)
    for m in matches:
        mode = 'pf' if m.group('mode') == 'prompt_free' else 'p'
        scale = m.group('scale').lower()
        ttype = m.group('type')
        key = f"{mode}_{ttype}"

        times = [
            convert_to_ms(float(m.group('min_v')), m.group('min_u')),
            convert_to_ms(float(m.group('mean_v')), m.group('mean_u')),
            convert_to_ms(float(m.group('max_v')), m.group('max_u'))
        ]
        parsed[key][scale] = times

    final_data = {}
    for key in parsed:
        final_data[key] = [parsed[key].get(s, [0, 0, 0]) for s in scales]

    return final_data


# Parse the data
data = parse_benchmark_data(RAW_BENCHMARK_DATA)
scales = ['Nano', 'Small', 'Medium', 'Large', 'XLarge']

# Setup output dir
output_dir = '.github/benchmarks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_grid_plot():
    global_max = 0
    for key in data:
        for d in data[key]:
            global_max = max(global_max, d[2])
    y_limit = global_max * 1.15

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

        bars = ax.bar(scales, means, color=color, alpha=0.8, edgecolor='black',
                      linewidth=0.8, zorder=3, width=0.6)
        ax.errorbar(scales, means, yerr=yerr, fmt='none', ecolor='#333333', capsize=5,
                    capthick=1.5, elinewidth=1.5, zorder=4)

        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, yval + (y_limit * 0.02),
                        f'{yval:.1f}ms', ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color='#333333')

        ax.set_title(title, fontsize=16, pad=20, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, labelpad=10, fontweight='600',
                      color='#34495e')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        ax.set_ylim(0, y_limit)

    plt.tight_layout(pad=5.0)
    output_path = os.path.join(output_dir, 'benchmark_grid.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_individual_plots():
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
        yerr = [[m - mi for m, mi in zip(means, mins)],
                [ma - m for m, ma in zip(means, maxs)]]

        ax.set_facecolor('#fdfdfd')
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc', zorder=0)
        ax.set_axisbelow(True)
        bars = ax.bar(scales, means, color=color, alpha=0.8, edgecolor='black',
                      linewidth=0.8, zorder=3, width=0.6)
        ax.errorbar(scales, means, yerr=yerr, fmt='none', ecolor='#333333', capsize=5,
                    capthick=1.5, elinewidth=1.5, zorder=4)
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, yval + (y_limit * 0.02),
                        f'{yval:.1f}ms', ha='center', va='bottom', fontsize=10,
                        fontweight='bold')

        ax.set_title(title, fontsize=16, pad=20, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, labelpad=10, fontweight='600')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, y_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f"Created: benchmarks/{filename}")


if __name__ == "__main__":
    create_grid_plot()
    create_individual_plots()
    print(f"\nSuccessfully generated benchmark graphs in the '{output_dir}' directory.")
