import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.path import Path
from matplotlib.spines import Spine
plt.rcParams['svg.fonttype'] = 'none'  

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def draw_frame(self, x):
            if frame == 'circle':
                super().draw_frame(x)
            elif frame == 'polygon':
                verts = unit_poly_verts(theta)
                path = Path(verts)
                self.patch = plt.Polygon(verts, closed=True, edgecolor='k')
                self.patch.set_transform(self.transAxes)
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    def unit_poly_verts(theta):
        x0, y0, r = [0.5] * 3
        verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
        return verts

    register_projection(RadarAxes)
    return theta

def plot_multiple_connectivity_radar(csv_files, target_seeds):
    assert len(csv_files) == len(target_seeds), "CSV files and seeds must match in length"

    dfs = [pd.read_csv(csv, index_col=0).dropna() for csv in csv_files]

    fig = plt.figure(figsize=(8, 4 * len(csv_files)), dpi=300)

    for idx, (df, seed) in enumerate(zip(dfs, target_seeds)):
        labels = df.index.tolist()
        num_vars = len(labels)
        theta = radar_factory(num_vars, frame='polygon')

        ax = fig.add_subplot(len(csv_files), 1, idx + 1, projection='radar')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=10)

        values = df['Correlation'].tolist()

        for i, val in enumerate(values):
            theta_center = theta[i]
            theta_range = np.linspace(theta_center - 0.25, theta_center + 0.25, 100)
            r = (-1) + (val + 1) * np.sin(np.linspace(0, np.pi, 100))
            color = 'steelblue' if val > 0 else 'purple'

            ax.fill(theta_range, r, color=color, alpha=0.3, linewidth=0)
            ax.plot(theta_range, r, color=color, linewidth=1.5)

        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels([])

        for gl in ax.yaxis.get_gridlines():
            y = gl.get_ydata()[0]
            gl.set_linewidth(0.5)
            gl.set_color("black" if np.isclose(y, 0) else "gray")

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

        ax.set_title(f"{seed}", fontsize=12, weight='bold', pad=12)

    fig.suptitle("Radial Functional Connectivity Profile of Cortical Targets", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = "/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/visuals"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "radar-connectivity.svg")
    plt.savefig(output_file, format='svg')
    plt.savefig(output_file.replace('.svg', '.png'), format='png', dpi=300)
    print(f"Saved flower-style stacked radar plot to {output_file}")
    plt.show()

files = [
    "/Users/similovesyou/Desktop/qts/simian-brain/functional-connectivity/site-strasbourg/seed-to-LIP-connectivity/group_mean_correlation_subs-10_seed-to-LIP.csv",
    "/Users/similovesyou/Desktop/qts/simian-brain/functional-connectivity/site-strasbourg/seed-to-FEF-connectivity/group_mean_correlation_subs-10_seed-to-FEF.csv"
]

target_seeds = ["LIP", "FEF"]

plot_multiple_connectivity_radar(files, target_seeds)