import sys
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Flag to control whether to plot in separate subplots or single plot
SEPARATE_PLOTS = True
FIGURE_WIDTH = 4
FIGURE_HEIGHT = 4.5
FIGURE_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

TOPO_FONT_SIZE = 14
TIME_FONT_SIZE = 14
AXIS_FONT_SIZE = 14
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 12
XLABEL_FONT_SIZE = 15
YLABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12

# Marker size for all plot lines (circles, stars)
MARKER_SIZE = 6

# Font size for the ×10^X scale label on the objective y-axis
OBJECTIVE_SCALE_FONT_SIZE = 11

# Y-axis limits: (ymin, ymax) or None for auto
YLIM_AVG_HOPS = (3.5, 7.5)      # top subplot (average hops or non-integrality)
YLIM_OBJECTIVE = (0.0035, 0.0085)    # bottom subplot (Objective)
# Objective value y-axis major tick interval
OBJECTIVE_MAJOR_TICK = 0.001
# Average hops y-axis major tick interval (None for auto)
AVG_HOPS_MAJOR_TICK = 1

# MILP: skip rows with |objective| < this (do not plot near-zero objective)
MILP_OBJECTIVE_EPSILON = 1e-6

# X-axis limit lower bound: float or None for data minimum
XLIM_LOWER = -3

# Known-value label vertical offset: fraction of y-range above the line (0 = on line, >0 = above)
KNOWN_VALUES_LABEL_OFFSET = 0.01

# Sentinel for unconnected (very large avg_hops)
UNCONNECTED_HOPS = 2147483648


def detect_format(in_name):
    """Detect LP vs MILP from CSV header. Returns 'lp' or 'milp'."""
    with open(in_name, 'r') as inf:
        first = inf.readline().strip().lower()
    if 'elapsed_time' in first:
        return 'milp'
    return 'lp'


def load_lp_timeline(in_name):
    """Load LP format: time,n_opt_links,avg_hops,cur_asc_obj,cumul_non_integrality"""
    time_vals = []
    obj_vals = []
    avg_hop_vals = []
    n_links_vals = []
    non_int_vals = []
    n_unconnected = 0

    with open(in_name, 'r') as inf:
        for line in inf:
            if 'time' in line and 'n_opt_links' in line:
                continue
            split_line = line.strip().split(',')
            if len(split_line) < 5:
                continue
            time_val = float(split_line[0])
            n_links = int(split_line[1])
            avg_hops = float(split_line[2])
            obj_val = float(split_line[3])
            non_int = float(split_line[4])

            if avg_hops > 100:
                n_unconnected += 1

            time_vals.append(time_val / 60.0)
            obj_vals.append(obj_val)
            avg_hop_vals.append(avg_hops)
            n_links_vals.append(n_links)
            non_int_vals.append(non_int)

    t_start = time_vals[0]
    time_vals = [t - t_start for t in time_vals]
    return time_vals, obj_vals, avg_hop_vals, n_links_vals, non_int_vals, n_unconnected


def load_milp_timeline(in_name):
    """Load MILP format: elapsed_time,avg_hops,objective,best_bound,pct_diff"""
    time_vals = []
    obj_vals = []
    avg_hop_vals = []
    best_bound_vals = []
    n_unconnected = 0

    with open(in_name, 'r') as inf:
        for line in inf:
            if 'elapsed_time' in line:
                continue
            split_line = line.strip().split(',')
            if len(split_line) < 3:
                continue
            elapsed_sec = float(split_line[0])
            avg_hops = float(split_line[1])
            obj_val = float(split_line[2])
            best_bound = float(split_line[3]) if len(split_line) >= 4 else None

            if avg_hops > 100:
                n_unconnected += 1

            time_vals.append(elapsed_sec / 60.0)
            obj_vals.append(obj_val)
            avg_hop_vals.append(avg_hops)
            best_bound_vals.append(best_bound)

    return time_vals, obj_vals, avg_hop_vals, n_unconnected, best_bound_vals


def main():
    parser = argparse.ArgumentParser(description='Plot timeline (avg hops and objective) from LP or MILP logs.')
    parser.add_argument('timeline_csv', help='Path to timeline CSV (LP or MILP format)')
    parser.add_argument('known_values_file', nargs='?', default=None, help='Optional CSV: topology,avg_hops,approx_sc[,std_hops,std_sc]')
    parser.add_argument('--milp', action='store_true', help='Force MILP format (default: auto-detect from header)')
    parser.add_argument('--lp', action='store_true', help='Force LP format (default: auto-detect from header)')
    parser.add_argument('--no-avg-hops', action='store_true', help='Plot cumulative non-integrality instead of avg hops (LP only)')
    args = parser.parse_args()

    in_name = args.timeline_csv
    known_values_file = args.known_values_file

    if args.milp:
        fmt = 'milp'
    elif args.lp:
        fmt = 'lp'
    else:
        fmt = detect_format(in_name)

    if fmt == 'milp':
        time_vals, obj_vals, avg_hop_vals, n_unconnected, best_bound_vals = load_milp_timeline(in_name)
        n_links_vals = []
        non_int_vals = []
        plot_avg_hops = True  # MILP has no non_int
    else:
        time_vals, obj_vals, avg_hop_vals, n_links_vals, non_int_vals, n_unconnected = load_lp_timeline(in_name)
        best_bound_vals = []
        plot_avg_hops = not args.no_avg_hops

    base_in_name = in_name.replace('_timeline.csv', '').replace('_milp_timeline.csv', '').split('/')[-1]
    out_png_name = f'files/timeline_graphs/{base_in_name}_continuous_iter_timeline.png'

    # X range from data (skip first if needed for plotting)
    if plot_avg_hops:
        plot_time_hops = time_vals[1 + n_unconnected:] if len(time_vals) > 1 + n_unconnected else time_vals
        plot_hop_vals = avg_hop_vals[1 + n_unconnected:] if len(avg_hop_vals) > 1 + n_unconnected else avg_hop_vals
    else:
        plot_time_hops = time_vals[1:] if len(time_vals) > 1 else time_vals
        plot_hop_vals = non_int_vals[1:] if len(non_int_vals) > 1 else non_int_vals

    # Build objective (and for MILP, bounds) series; MILP: skip rows with |objective| near 0
    if fmt == 'milp':
        _time = time_vals[1:] if len(time_vals) > 1 else time_vals
        _obj = obj_vals[1:] if len(obj_vals) > 1 else obj_vals
        _bb = best_bound_vals[1:] if (best_bound_vals and len(best_bound_vals) > 1) else (best_bound_vals or [])
        _triples = [(t, o, b) for t, o, b in zip(_time, _obj, _bb) if abs(o) >= MILP_OBJECTIVE_EPSILON]
        plot_time_obj = [x[0] for x in _triples]
        plot_obj_vals = [x[1] for x in _triples]
        _sane = 1e10
        plot_time_bound = [x[0] for x in _triples if x[2] is not None and x[2] < _sane]
        plot_bound_vals = [x[2] for x in _triples if x[2] is not None and x[2] < _sane]
    else:
        plot_time_obj = time_vals[1:] if len(time_vals) > 1 else time_vals
        plot_obj_vals = obj_vals[1:] if len(obj_vals) > 1 else obj_vals
        plot_time_bound = []
        plot_bound_vals = []

    all_times = (plot_time_hops + plot_time_obj)
    x_min = min(all_times) if all_times else 0
    if XLIM_LOWER is not None:
        x_min = XLIM_LOWER
    x_max_data = max(all_times) if all_times else 1
    # Extend x-axis slightly beyond last data point
    x_margin = max((x_max_data - x_min) * 0.03, 0.01) if x_max_data > x_min else 0.01
    x_max = x_max_data + x_margin
    # Place known-value labels: left for LP, right for MILP
    x_range = (x_max_data - x_min) if x_max_data > x_min else 1
    if fmt == 'milp':
        x_label_pos = x_max_data - x_range * 0.02  # right side
    else:
        x_label_pos = x_min + x_range * 0.02  # left side

    # Choose exponent for objective y-axis so tick labels are single-digit (×10^X)
    if YLIM_OBJECTIVE is not None:
        obj_ymin, obj_ymax = YLIM_OBJECTIVE[0], YLIM_OBJECTIVE[1]
    else:
        obj_ymin = min(plot_obj_vals) if plot_obj_vals else 0
        obj_ymax = max(plot_obj_vals) if plot_obj_vals else 1
    obj_mag = max(abs(obj_ymin), abs(obj_ymax), 1e-20)
    obj_exponent = int(math.floor(math.log10(obj_mag)))

    def objective_tick_formatter(y, pos):
        scaled = y * (10 ** (-obj_exponent))
        return f'{scaled:.0f}' if scaled == int(scaled) else f'{scaled:.1f}'

    obj_ylabel = 'Objective (MCF)'
    obj_scale_text = f'$\\times 10^{{{obj_exponent}}}$'

    if SEPARATE_PLOTS:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
        if plot_avg_hops:
            ax1.plot(plot_time_hops, plot_hop_vals, marker='o', color='tab:blue', linewidth=3, markersize=MARKER_SIZE)
            ax1.set_ylabel('Average Hops', fontsize=YLABEL_FONT_SIZE, color='black')
            # ax1.set_title('Average Hops', fontsize=SUBTITLE_FONT_SIZE, pad=10, color='black')
        else:
            ax1.plot(plot_time_obj, plot_hop_vals, marker='o', color='tab:blue', linewidth=3, markersize=MARKER_SIZE)
            ax1.set_ylabel('Cumulative Non-Integrality', fontsize=YLABEL_FONT_SIZE, color='black')
            # ax1.set_title('Cumulative Non-Integrality', fontsize=SUBTITLE_FONT_SIZE, pad=10, color='black')
        ax1.tick_params(axis='y', labelsize=AXIS_FONT_SIZE, colors='black')
        ax1.grid(True, which='major', alpha=0.3)
        ax1.grid(True, which='minor', alpha=0.3)
        if AVG_HOPS_MAJOR_TICK is not None:
            ax1.yaxis.set_major_locator(mtick.MultipleLocator(AVG_HOPS_MAJOR_TICK))
        ax1.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))  # minor at 1/2 major increment
        ax1.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax1.set_xlim(x_min, x_max)
        if YLIM_AVG_HOPS is not None:
            ax1.set_ylim(YLIM_AVG_HOPS)

        ax2.plot(plot_time_obj, plot_obj_vals, marker='o', color='tab:red', linewidth=3,
                 label='Incumbent' if fmt == 'milp' else None)
        if plot_time_bound and plot_bound_vals:
            ax2.plot(plot_time_bound, plot_bound_vals, marker='*', color='red', linewidth=2, markersize=MARKER_SIZE+2,
                     label='Bounds' if fmt == 'milp' else None)
        if fmt == 'milp':
            ax2.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, frameon=True,ncol=2)
        ax2.set_ylabel(obj_ylabel, fontsize=YLABEL_FONT_SIZE, color='black')
        ax2.tick_params(axis='y', labelsize=AXIS_FONT_SIZE, colors='black')
        ax2.yaxis.set_major_locator(mtick.MultipleLocator(OBJECTIVE_MAJOR_TICK))
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(objective_tick_formatter))
        ax2.grid(True, which='major', alpha=0.3)
        ax2.grid(True, which='minor', alpha=0.3)
        ax2.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax2.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        # ax2.set_title('Objective (MCF)', fontsize=SUBTITLE_FONT_SIZE, pad=10, color='black')
        ax2.set_xlabel('Time (minutes)', fontsize=XLABEL_FONT_SIZE, color='black')
        ax2.tick_params(axis='x', labelsize=AXIS_FONT_SIZE, colors='black')
        ax2.set_xlim(x_min, x_max)
        if YLIM_OBJECTIVE is not None:
            ax2.set_ylim(YLIM_OBJECTIVE)
        # Draw ×10^X scale on the left (where the ticks are), at top of y-axis
        ytop = ax2.get_ylim()[1]
        ax2.text(-0.06, ytop, obj_scale_text, transform=ax2.get_yaxis_transform(),
                 va='bottom', ha='right', fontsize=OBJECTIVE_SCALE_FONT_SIZE, color='black')

        # fig.suptitle('Timeline Plot', fontsize=TITLE_FONT_SIZE, y=0.995, color='black')
    else:
        fig, ax1 = plt.subplots()

        if plot_avg_hops:
            ax1.plot(plot_time_hops, plot_hop_vals, marker='o', color='tab:blue', linewidth=3, markersize=MARKER_SIZE)
            ax1.set_ylabel('Average Hops', fontsize=YLABEL_FONT_SIZE, color='black')
        else:
            ax1.plot(plot_time_obj, plot_hop_vals, marker='o', color='tab:blue', linewidth=3, markersize=MARKER_SIZE)
            ax1.set_ylabel('Cumulative Non-Integrality', fontsize=YLABEL_FONT_SIZE, color='black')
        ax1.tick_params(axis='y', labelsize=AXIS_FONT_SIZE, colors='black')
        ax1.grid(True, which='major', alpha=0.3)
        ax1.grid(True, which='minor', alpha=0.3)
        if AVG_HOPS_MAJOR_TICK is not None:
            ax1.yaxis.set_major_locator(mtick.MultipleLocator(AVG_HOPS_MAJOR_TICK))
        ax1.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax1.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax1.set_xlim(x_min, x_max)
        if YLIM_AVG_HOPS is not None:
            ax1.set_ylim(YLIM_AVG_HOPS)

        ax2 = ax1.twinx()
        ax2.plot(plot_time_obj, plot_obj_vals, marker='*', color='tab:red', linewidth=3, markersize=MARKER_SIZE,
                 label='Incumbent' if fmt == 'milp' else None)
        if plot_time_bound and plot_bound_vals:
            ax2.plot(plot_time_bound, plot_bound_vals, marker='*', color='red', linewidth=2, markersize=MARKER_SIZE,
                     label='Bounds' if fmt == 'milp' else None)
        if fmt == 'milp':
            ax2.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE, frameon=True)
        ax2.set_ylabel(obj_ylabel, fontsize=YLABEL_FONT_SIZE, color='black')
        ax2.tick_params(axis='y', labelsize=AXIS_FONT_SIZE, colors='black')
        ax2.yaxis.set_major_locator(mtick.MultipleLocator(OBJECTIVE_MAJOR_TICK))
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(objective_tick_formatter))
        ax2.grid(True, which='major', alpha=0.3)
        ax2.grid(True, which='minor', alpha=0.3)
        ax2.yaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        ax2.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
        if YLIM_OBJECTIVE is not None:
            ax2.set_ylim(YLIM_OBJECTIVE)
        ytop = ax2.get_ylim()[1]
        ax2.text(-0.06, ytop, obj_scale_text, transform=ax2.get_yaxis_transform(),
                 va='bottom', ha='right', fontsize=OBJECTIVE_SCALE_FONT_SIZE, color='black')

        ax1.set_xlabel('Time (min)', fontsize=XLABEL_FONT_SIZE, color='black')
        ax1.tick_params(axis='x', labelsize=AXIS_FONT_SIZE, colors='black')
        ax1.set_xlim(x_min, x_max)
        # plt.title('Timeline Plot: Avg Hops and Objective', fontsize=TITLE_FONT_SIZE, color='black')

    # Y ranges for known values and labels
    if plot_avg_hops:
        y_min_hops = min(plot_hop_vals) if plot_hop_vals else 0
        y_max_hops = max(plot_hop_vals) if plot_hop_vals else 100
    else:
        y_min_hops = min(plot_hop_vals) if plot_hop_vals else 0
        y_max_hops = max(plot_hop_vals) if plot_hop_vals else 1
    y_min_sc = min(plot_obj_vals) if plot_obj_vals else 0
    y_max_sc = max(plot_obj_vals) if plot_obj_vals else 1.0

    # Read and plot known values if provided
    if known_values_file:
        try:
            with open(known_values_file, 'r') as inf:
                for line in inf:
                    if line.strip() == "":
                        continue
                    split_line = line.strip().split(',')
                    if len(split_line) >= 3:
                        try:
                            float(split_line[1])
                            float(split_line[2])
                        except ValueError:
                            continue

                        topology = split_line[0].strip()
                        avg_hops_val = float(split_line[1])
                        approx_sc_val = float(split_line[2])

                        avg_hops_std = 0.0
                        approx_sc_std = 0.0
                        if len(split_line) >= 4:
                            try:
                                avg_hops_std = float(split_line[3])
                            except (ValueError, IndexError):
                                pass
                        if len(split_line) >= 5:
                            try:
                                approx_sc_std = float(split_line[4])
                            except (ValueError, IndexError):
                                pass

                        if avg_hops_val <= 100:
                            if avg_hops_std > 0:
                                upper_hops = avg_hops_val + avg_hops_std
                                lower_hops = avg_hops_val - avg_hops_std
                                ax1.axhline(y=upper_hops, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.5)
                                ax1.axhline(y=lower_hops, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.5)
                                ax1.fill_between([x_min, x_max], lower_hops, upper_hops, color='tab:blue', alpha=0.15)
                                ax1.axhline(y=avg_hops_val, color='tab:blue', linestyle='--', linewidth=2.5, alpha=0.7)
                            else:
                                ax1.axhline(y=avg_hops_val, color='tab:blue', linestyle='--', linewidth=2.5, alpha=0.7)

                        if approx_sc_val <= 1.0 or approx_sc_val <= max(y_max_sc, 1.0) * 1.1:
                            if approx_sc_std > 0:
                                upper_sc = approx_sc_val + approx_sc_std
                                lower_sc = approx_sc_val - approx_sc_std
                                ax2.axhline(y=upper_sc, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.5)
                                ax2.axhline(y=lower_sc, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.5)
                                ax2.fill_between([x_min, x_max], lower_sc, upper_sc, color='tab:red', alpha=0.15)
                                ax2.axhline(y=approx_sc_val, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.7)
                            else:
                                ax2.axhline(y=approx_sc_val, color='tab:red', linestyle='--', linewidth=2.5, alpha=0.7)

                        # Labels: configurable height above the lines (offset 0 = on line)
                        y_range_h = (y_max_hops - y_min_hops) if (y_max_hops > y_min_hops) else 1
                        y_offset_h = KNOWN_VALUES_LABEL_OFFSET * y_range_h
                        y_range_sc = (y_max_sc - y_min_sc) if (y_max_sc > y_min_sc) else 0.1
                        y_offset_sc = KNOWN_VALUES_LABEL_OFFSET * y_range_sc
                        va = 'center' if KNOWN_VALUES_LABEL_OFFSET == 0 else 'bottom'

                        if avg_hops_val <= 100:
                            ax1.text(x_label_pos, avg_hops_val + y_offset_h, topology, fontsize=TOPO_FONT_SIZE, color='black',
                                     alpha=0.8, verticalalignment=va,
                                     horizontalalignment='left' if fmt == 'lp' else 'right')

                        if approx_sc_val <= 1.0 or approx_sc_val <= max(y_max_sc, 1.0) * 1.1:
                            ax2.text(x_label_pos, approx_sc_val + y_offset_sc, topology, fontsize=TOPO_FONT_SIZE, color='black',
                                     alpha=0.8, verticalalignment=va,
                                     horizontalalignment='left' if fmt == 'lp' else 'right')
        except Exception as e:
            print(f"Warning: Could not read known values file {known_values_file}: {e}")

    fig.tight_layout()
    plt.savefig(out_png_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f'Wrote to {out_png_name}')
    plt.show()


if __name__ == '__main__':
    main()
