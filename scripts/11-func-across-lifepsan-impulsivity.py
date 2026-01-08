import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  # they annoy me
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import f
import warnings
warnings.filterwarnings('ignore')

# need the age of the other monkeys!!!!

directory = "/Users/similovesyou/Desktop/qts/simian-behavior/data/py"
figure_me_out = "/Users/similovesyou/Desktop/qts/simian-brain/plots"
Path(figure_me_out).mkdir(parents=True, exist_ok=True)

rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo-min-attempts.csv")
tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo-min-attempts.csv")

#rhesus_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/rhesus-elo.csv")
#tonkean_table = pd.read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/tonkean-elo.csv")

#fc_dir = "/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-smoothed/site-strasbourg/PITd-seed-2-seed-connectivity"
#fc_dir = "/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-spatial-smoothed/site-strasbourg/attention-seed-2-seed-connectivity"
fc_dir = "/Users/similovesyou/Desktop/qts/simian-brain/final-functional-connectivity-2mm/site-strasbourg/attention-seed-2-seed-connectivity"

# Helper to get ROI pairs from first available matrix in a directory
def get_roi_pairs(fc_directory, subject_list):
    rois = []
    for name in subject_list:
        conn_path = os.path.join(fc_directory, f"{name}_seed_connectivity.csv")
        if os.path.exists(conn_path):
            mat = pd.read_csv(conn_path, index_col=0)
            rois = list(mat.index)
            break
    if not rois:
        print(f"No FC matrices found in {fc_directory}")
        return []
    pairs = [(a, b) for i, a in enumerate(rois) for j, b in enumerate(rois) if i < j]
    return pairs

# Get ROI pairs for both FC directories
seed2_pairs = get_roi_pairs(fc_dir, rhesus_table["name"])

# Add columns for seed2 pairs
for a, b in seed2_pairs:
    col = f"fc_{a}_{b}"
    rhesus_table[col] = pd.NA
    tonkean_table[col] = pd.NA

# Function to fill FC values for a dataframe from one directory and ROI pairs
def fill_fc(df, fc_directory, roi_pairs):
    for i, row in df.iterrows():
        name = row["name"]
        conn_path = os.path.join(fc_directory, f"{name}_seed_connectivity.csv")
        if not os.path.isfile(conn_path):
            print(f"-- Skipping {name}: no file in {fc_directory}")
            continue
        try:
            mat = pd.read_csv(conn_path, index_col=0)
            for a, b in roi_pairs:
                if a in mat.index and b in mat.columns:
                    df.at[i, f"fc_{a}_{b}"] = mat.at[a, b]
        except Exception as e:
            print(f"!! Error processing {name} in {fc_directory}: {e}")

# Fill values from seed-2 directory
fill_fc(rhesus_table, fc_dir, seed2_pairs)
fill_fc(tonkean_table, fc_dir, seed2_pairs)

# Scan dates for each subject
scan_dates = {
    "amidala": "2024-10-17",
    "arwen": "2024-03-08",
    "baal": "2024-05-24", # FEF coverage subpar but correlations look OK + good tSNR
    "berenice": "2025-02-14", # low tSNR
    "dory": "2024-09-27",
    # "eowyn": "2025-03-07", # primary visual + FEF coverage subpar
    "ficelle": "2025-01-17",
    "gabie": "2024-09-05", # FEF a bit off but look into matrix to decide - correlations strong
    "gandhi": "2025-02-28", # there's a second scan (done 2024-02-16)
    # "havana": "2024-04-05", # poor FEF coverage
    "horus": "2025-02-07", # second scan 2024-01-19
    "indigo": "2024-03-15",
    "iron": "2024-10-31",
    "isis": "2025-03-14",
    "jazz": "2025-01-31",
    "jipsy": "2024-10-18",
    "joy": "2025-03-13",
    "karma": "2024-11-01",
    "kenobi": "2025-02-21",
    "kenya": "2024-11-29",
    # "lassa": "2024-07-05", # check matrix 
    "marouchka": "2024-10-25", # V1 iffy but eh LIP-FEF good
    "natasha": "2024-04-19",
    "nema": "2025-01-24",
    # "patchouli": "2024-06-21", # FEF subpar early visual too
    "radja": "2024-10-24",
    "samael": "2024-09-26",
    "volga": "2024-05-17", # check why the correlation is so high
    "yannick": "2025-01-30", # bad early visual, but else good
    "yin": "2024-11-08",
    "patsy": "2024-11-22"
}

rhesus_table["scan_date"] = pd.NaT
rhesus_table["age_at_scan"] = pd.NA
rhesus_table["start_date"] = pd.to_datetime(rhesus_table["start_date"])

tonkean_table["scan_date"] = pd.NaT
tonkean_table["age_at_scan"] = pd.NA
tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])

# Fill in scan dates and compute age at scan
for i, row in rhesus_table.iterrows():
    name = row["name"]
    if name in scan_dates:
        scan_dt = pd.to_datetime(scan_dates[name])
        rhesus_table.at[i, "scan_date"] = scan_dt
        
        # Estimate age at scan using time delta from original age (assumed in years)
        if pd.notna(row["age"]) and pd.notna(row["start_date"]):
            days_diff = (scan_dt - row["start_date"]).days
            years_added = days_diff / 365.25
            rhesus_table.at[i, "age_at_scan"] = round(row["age"] + years_added, 2)

for i, row in tonkean_table.iterrows():
    name = row["name"]
    if name in scan_dates:
        scan_dt = pd.to_datetime(scan_dates[name])
        tonkean_table.at[i, "scan_date"] = scan_dt
        
        # Estimate age at scan using time delta from original age (assumed in years)
        if pd.notna(row["age"]) and pd.notna(row["start_date"]):
            days_diff = (scan_dt - row["start_date"]).days
            years_added = days_diff / 365.25
            tonkean_table.at[i, "age_at_scan"] = round(row["age"] + years_added, 2)

# start_date and end_date objects?
rhesus_table["start_date"] = pd.to_datetime(rhesus_table["start_date"])
rhesus_table["end_date"] = pd.to_datetime(rhesus_table["end_date"])
tonkean_table["start_date"] = pd.to_datetime(tonkean_table["start_date"])
tonkean_table["end_date"] = pd.to_datetime(tonkean_table["end_date"])

# Compute midpoint of task period
rhesus_table["task_midpoint"] = rhesus_table["start_date"] + (rhesus_table["end_date"] - rhesus_table["start_date"]) / 2
tonkean_table["task_midpoint"] = tonkean_table["start_date"] + (tonkean_table["end_date"] - tonkean_table["start_date"]) / 2

# Compute age at task (based on age at start_date)
rhesus_table["age_at_task"] = rhesus_table.apply(
    lambda row: round(row["age"] + (row["task_midpoint"] - row["start_date"]).days / 365.25, 2) # Optionally drop the midpoint helper column rhesus_table.drop(columns="task_midpoint", inplace=True)
    if pd.notna(row["age"]) and pd.notna(row["task_midpoint"]) else pd.NA,
    axis=1
)
tonkean_table["age_at_task"] = tonkean_table.apply(
    lambda row: round(row["age"] + (row["task_midpoint"] - row["start_date"]).days / 365.25, 2) # Optionally drop the midpoint helper column rhesus_table.drop(columns="task_midpoint", inplace=True)
    if pd.notna(row["age"]) and pd.notna(row["task_midpoint"]) else pd.NA,
    axis=1
)

# Let's see how the ages differ
summary_cols = ["name", "age", "age_at_task", "age_at_scan"]
summary_rhesus_table = rhesus_table[summary_cols].copy()
summary_tonkean_table = tonkean_table[summary_cols].copy()

summary_rhesus_table.sort_values(by="name", inplace=True)
summary_tonkean_table.sort_values(by="name", inplace=True)

print(summary_rhesus_table.to_string(index=False))
print(summary_tonkean_table.to_string(index=False))

# Convert relevant columns to numeric
for table in [rhesus_table, tonkean_table]:
    table['age_at_scan'] = pd.to_numeric(table['age_at_scan'], errors='coerce')
    
    # Convert all FC columns to numeric
    fc_cols = [col for col in table.columns if col.startswith('fc_')]
    for col in fc_cols:
        table[col] = pd.to_numeric(table[col], errors='coerce')



# plotting proportions across the lifespan
def p_lifespan(
    ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],
        x="age_at_scan",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.6,
        edgecolor="steelblue",
        linewidth=1,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_scan",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.6,
        edgecolor="purple",
        linewidth=1,
    )

    # sns.regplot(
    #     ax=ax,
    #     data=rhesus_table,
    #     x="age_at_scan",
    #     y=metric,
    #     color=colors["rhesus"],
    #     order=2,
    #     scatter=False,
    #     ci=None,
    #     line_kws={"linestyle": "--"},
    # )
    # sns.regplot(
    #     ax=ax,
    #     data=tonkean_table,
    #     x="age_at_scan",
    #     y=metric,
    #     color=colors["tonkean"],
    #     order=2,
    #     scatter=False,
    #     ci=None,
    #     line_kws={"linestyle": "--"},
    # )
    # sns.regplot(
    #     ax=ax,
    #     data=pd.concat([rhesus_table, tonkean_table]),
    #     x="age_at_scan",
    #     y=metric,
    #     color="black",
    #     order=2,
    #     scatter=False,
    #     ci=None,
    # )

    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0", "", "", "", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    # Remove bold x and y axes but keep thick tick marks
    ax.spines["bottom"].set_linewidth(1)  # Default thickness
    ax.spines["left"].set_linewidth(1)  # Default thickness
    ax.tick_params(
        axis="both", which="major", width=2.5, length=7
    )  # Thicker and longer ticks

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("final-functional Connectivity (r)", fontsize=14) # CHANGE NAME!!
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")


sns.set(style="whitegrid", font_scale=1.2, rc={"font.family": "DejaVu Sans"})

fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # Changed to 2 subplots

# Adjusted colors
colors = {"tonkean": "purple", "rhesus": "steelblue"}

def find_fc_column(a, b, table):
    direct = f"fc_{a}_{b}"
    flipped = f"fc_{b}_{a}"
    if direct in table.columns:
        return direct
    elif flipped in table.columns:
        return flipped
    else:
        raise ValueError(f"No FC column found for pair {a}–{b}")

# CHOOSE WHICH METRICS TO INCLUDE!! 
metrics = [
    find_fc_column("LIP", "FEF", rhesus_table),
    find_fc_column("LIP", "MT", rhesus_table)
]

titles  = ["LIP-FEF rs-fc", "LIP-MT rs-fc"]

for ax, metric, title, show_xlabel, show_ylabel in zip(
    axes, metrics, titles, [True] * 2, [True] + [False] * 1
):
    p_lifespan(
        ax, metric, title, rhesus_table, tonkean_table, colors, show_xlabel, show_ylabel
    )

handles = [
    plt.Line2D(
        [],
        [],
        color=colors["rhesus"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="steelblue",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ mulatta}$ (m)",
    ),
    plt.Line2D(
        [],
        [],
        color=colors["tonkean"],
        marker="o",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (f)",
    ),
    plt.Line2D(
        [],
        [],
        color="none",
        marker="o",
        markeredgecolor="purple",
        linestyle="None",
        markersize=8,
        label=r"$\it{Macaca\ tonkeana}$ (m)",
    ),
]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

fig.suptitle(
    "Resting State FC Across the Lifespan in Macaque Species",
    fontsize=25,
    fontweight="bold",
    fontname="DejaVu Sans",
    color="purple",
)

plt.tight_layout(rect=[0, 0.02, 1, 0.93])
plt.show()

combined_df = pd.concat(
    [  # combine both species into one df
        rhesus_table.assign(species="Rhesus"),
        tonkean_table.assign(species="Tonkean"),
    ]
).drop_duplicates(subset="name", ignore_index=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 2]}
    )

    # --- LEFT PLOT (Behavioral Data) ---
    jitter = 0.1
    colors = {"Rhesus": "steelblue", "Tonkean": "purple", "mulatta": "steelblue"}

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.6,
            size=10,
            ax=ax1,
        )
        sns.stripplot(
            x="species",
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax1,
        )

    for species in ["Rhesus", "Tonkean"]:
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age_at_task"].mean()
        std_age = subset["age_at_task"].std()
        ax1.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax1.set_ylabel("Age", fontsize=14)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )

    ax1.text(
        0.5,
        1.05,
        "Behavioral Data",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- RIGHT PLOT (Neuroimaging Data) ---
    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.6,
            size=10,
            ax=ax2,
        )

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax2,
        )

    for species in ["Rhesus", "Tonkean"]:
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age_at_scan"].mean()
        std_age = subset["age_at_scan"].std()
        ax2.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[std_age],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax2.set_ylabel("Age", fontsize=14)
    ax2.set_xticks(ticks=[0, 1])
    ax2.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )
    rhesus_count_scan = combined_df[(combined_df["species"] == "Rhesus") & (combined_df["age_at_scan"].notna())].shape[0]
    tonkean_count_scan = combined_df[(combined_df["species"] == "Tonkean") & (combined_df["age_at_scan"].notna())].shape[0]

    ax2.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={rhesus_count_scan})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={tonkean_count_scan})',
        ],
        fontsize=14,
    )

    ax2.text(
        0.5,
        1.05,
        "Neuroimaging Data",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- Shared Formatting ---
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("")
    ax2.set_yticklabels([])

    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_ylim(0, 25)
        ax.set_xlabel("")

    ax1.set_xlim(-0.5, 1.5)
    ax2.set_xlim(-0.5, 1.5)

    fig.suptitle(
        "Demographics",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
        y=1.02,
    )

    plt.tight_layout()

    filename_base = "demographics-neuroimaging"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()


def summarize_demographics(df, age_col, species_col="species", gender_col="gender"):
    summary = []
    for species in df[species_col].unique():
        subset = df[df[species_col] == species]
        females = subset[subset[gender_col] == 2]
        males = subset[subset[gender_col] != 2]

        age_vals = subset[age_col].dropna()
        if not age_vals.empty:
            age_mean = round(age_vals.mean(), 2)
            age_min = round(age_vals.min(), 2)
            age_max = round(age_vals.max(), 2)
        else:
            age_mean, age_min, age_max = [None] * 3

        female_pct = round(len(females) / len(subset) * 100, 1) if len(subset) > 0 else 0

        summary.append({
            "Species": species,
            "N": len(subset),
            "Mean Age": age_mean,
            "Min Age": age_min,
            "Max Age": age_max,
            "Female (%)": female_pct,
            "Male (%)": 100 - female_pct,
        })

    return pd.DataFrame(summary)

# Full behavioral cohort
behavior_summary = summarize_demographics(combined_df, "age_at_task")
print("\n--- Behavioral Cohort Summary ---")
print(behavior_summary.to_string(index=False))

# Imaging cohort (with valid scan dates)
neuro_df = combined_df[combined_df["age_at_scan"].notna()]
neuro_summary = summarize_demographics(neuro_df, "age_at_scan")
print("\n--- Neuroimaging Cohort Summary ---")
print(neuro_summary.to_string(index=False))

# Inclusion of non-behavioral subjects 
demographics_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/demographics/IRM-demographics.xlsx"
birthdates_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/demographics/IRM-birthdates.xlsx"

IRM_demographics_df = pd.read_excel(demographics_path)
IRM_birthdates_df = pd.read_excel(birthdates_path)

behavioral_subjects = set(rhesus_table['name']).union(set(tonkean_table['name']))
non_behavioral_subjects = [s for s in scan_dates.keys() if s not in behavioral_subjects]

non_behavioral_df = pd.DataFrame({
    'name': non_behavioral_subjects,
    'scan_date': pd.to_datetime([scan_dates[name] for name in non_behavioral_subjects])
})

# Use lowercase for name matching
IRM_demographics_df['name_lower'] = IRM_demographics_df['name'].str.lower()
IRM_birthdates_df['name_lower'] = IRM_birthdates_df['Individual'].str.lower()
non_behavioral_df['name_lower'] = non_behavioral_df['name'].str.lower()

# First merge with demographics for sex info
non_behavioral_df = non_behavioral_df.merge(
    IRM_demographics_df[['name_lower', 'sex']],
    on='name_lower',
    how='left'
)

# Then merge with birth dates
non_behavioral_df = non_behavioral_df.merge(
    IRM_birthdates_df[['name_lower', 'Date of birth']],
    on='name_lower',
    how='left'
)

# Clean up column names and ensure proper datetime format
non_behavioral_df.rename(columns={
    "sex": "gender", 
    "Date of birth": "birth_date"
}, inplace=True)

# Ensure both dates are datetime objects
non_behavioral_df['birth_date'] = pd.to_datetime(non_behavioral_df['birth_date'])
non_behavioral_df['scan_date'] = pd.to_datetime(non_behavioral_df['scan_date'])

# Merge species/group info from demographics
non_behavioral_df = non_behavioral_df.merge(
    IRM_demographics_df[['name_lower', 'group']],  # 'group' is species
    on='name_lower',
    how='left'
)

# Optional: rename 'group' to 'species' for clarity
non_behavioral_df.rename(columns={'group': 'species'}, inplace=True)

# Calculate precise age at scan in decimal years
def calculate_decimal_age(birth_date, scan_date):
    """Calculate age in decimal years between two dates"""
    if pd.isna(birth_date) or pd.isna(scan_date):
        return None
    
    # Calculate the difference in days
    days_diff = (scan_date - birth_date).days
    
    # Convert to decimal years (using 365.25 to account for leap years)
    age_years = days_diff / 365.25
    
    return round(age_years, 2)

# Apply the calculation
non_behavioral_df['age_at_scan'] = non_behavioral_df.apply(
    lambda row: calculate_decimal_age(row['birth_date'], row['scan_date']), 
    axis=1
)

# Drop temporary columns
non_behavioral_df = non_behavioral_df.drop(columns=['name_lower', 'birth_date'])

# Normalize gender values
gender_map = {"M": 1, "F": 2}
non_behavioral_df["gender"] = non_behavioral_df["gender"].map(gender_map)

roi_pairs = seed2_pairs 

for a, b in roi_pairs:
    non_behavioral_df[f"fc_{a}_{b}"] = pd.NA

for i, row in non_behavioral_df.iterrows():
    name = row["name"]
    conn_path = os.path.join(fc_dir, f"{name}_seed_connectivity.csv")
    
    if os.path.isfile(conn_path):
        try:
            mat = pd.read_csv(conn_path, index_col=0)
            for a, b in roi_pairs:
                if a in mat.index and b in mat.columns:
                    non_behavioral_df.at[i, f"fc_{a}_{b}"] = mat.at[a, b]
        except Exception as e:
            print(f"Error processing {name}: {e}")

print("\nSuccessfully processed subjects:")
print(non_behavioral_df[['name', 'gender', 'age_at_scan']].dropna())

def find_fc_column(a, b, table):
    direct = f"fc_{a}_{b}"
    flipped = f"fc_{b}_{a}"
    if direct in table.columns:
        return direct
    elif flipped in table.columns:
        return flipped
    else:
        raise ValueError(f"No FC column found for pair {a}–{b}")

def p_lifespan_all(
    ax, metric, title, rhesus_table, tonkean_table, non_behavioral_df, colors, show_xlabel, show_ylabel
):
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 1],  # Males
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="steelblue",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=rhesus_table[rhesus_table["gender"] == 2],  # Females
        x="age_at_scan",
        y=metric,
        color=colors["rhesus"],
        s=80,
        alpha=0.8,
        edgecolor="steelblue",
        linewidth=1,
    )

    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 1],
        x="age_at_scan",
        y=metric,
        facecolors="none",
        edgecolor="purple",
        s=80,
        linewidth=1,
    )
    sns.scatterplot(
        ax=ax,
        data=tonkean_table[tonkean_table["gender"] == 2],
        x="age_at_scan",
        y=metric,
        color=colors["tonkean"],
        s=80,
        alpha=0.8,
        edgecolor="purple",
        linewidth=1,
    )

    if metric in non_behavioral_df.columns:
        valid_df = non_behavioral_df.dropna(subset=["age_at_scan", metric])
        # Split by species and gender
        rhesus_males = valid_df[(valid_df["gender"] == 1) & (valid_df["species"] == "rhesus")]
        rhesus_females = valid_df[(valid_df["gender"] == 2) & (valid_df["species"] == "rhesus")]
        tonkean_males = valid_df[(valid_df["gender"] == 1) & (valid_df["species"] == "tonkean")]
        tonkean_females = valid_df[(valid_df["gender"] == 2) & (valid_df["species"] == "tonkean")]

        # Plot with same marker, just lower alpha
        sns.scatterplot(
            ax=ax,
            data=rhesus_males,
            x="age_at_scan",
            y=metric,
            facecolors="none",
            edgecolor=colors["rhesus"],
            s=80,
            linewidth=1,
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=rhesus_females,
            x="age_at_scan",
            y=metric,
            color=colors["rhesus"],
            s=80,
            alpha=0.3,
            edgecolor=colors["rhesus"],
            linewidth=1,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=tonkean_males,
            x="age_at_scan",
            y=metric,
            facecolors="none",
            edgecolor=colors["tonkean"],
            s=80,
            linewidth=1,
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            ax=ax,
            data=tonkean_females,
            x="age_at_scan",
            y=metric,
            color=colors["tonkean"],
            s=80,
            alpha=0.3,
            edgecolor=colors["tonkean"],
            linewidth=1,
            legend=False,
        )

    # Combine all tables
    all_subjects = pd.concat([rhesus_table, tonkean_table, non_behavioral_df], ignore_index=True)

    # Ensure metric exists in all_subjects
    if metric not in all_subjects.columns:
        all_subjects[metric] = np.nan

    # Convert to numeric
    all_subjects["age_at_scan"] = pd.to_numeric(all_subjects["age_at_scan"], errors="coerce")
    all_subjects[metric] = pd.to_numeric(all_subjects[metric], errors="coerce")

    # Keep only rows with valid values
    all_subjects = all_subjects.dropna(subset=["age_at_scan", metric])

    ax.set_title(title, fontsize=20, fontweight="bold", fontname="DejaVu Sans", color="black")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 25)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "", "0", "", "1"])
    ax.grid(True, which="both", linestyle="--")
    ax.grid(True, which="major", axis="x")

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", width=2.5, length=7)

    if show_xlabel:
        ax.set_xlabel("Age", fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Functional Connectivity (r)", fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

# Create figure with 4 subplots (2x2 grid)
#fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
#colors2 = {"tonkean": "purple", "rhesus": "steelblue"}
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 10)) 
colors2 = {"tonkean": "purple", "rhesus": "steelblue"}   

metrics2 = [     
    find_fc_column("LIP", "FEF", rhesus_table),     
    find_fc_column("LIP", "MT", rhesus_table),      
    find_fc_column("LIP", "MST", rhesus_table),     
    find_fc_column("LIP", "V1", rhesus_table),      
    find_fc_column("V2", "V4", rhesus_table),       
    find_fc_column("MT", "MST", rhesus_table)       
]  

titles2 = [     
    "LIP-FEF", "LIP-MT", "LIP-MST", "LIP-V1", "V2-V4", "MT-MST" 
]   

for m in metrics2:     
    if m not in non_behavioral_df.columns:         
        non_behavioral_df[m] = np.nan  

axes_flat = axes2.flatten()  
show_xlabel_flags = [False, False, False, True, True, True]  
show_ylabel_flags = [True, False, False, True, False, False]  

for i, (ax, metric, title, show_xlabel, show_ylabel) in enumerate(zip(     
    axes_flat, metrics2, titles2, show_xlabel_flags, show_ylabel_flags 
)):     
    # Call the full plotting function
    p_lifespan_all(         
        ax, metric, title, rhesus_table, tonkean_table, non_behavioral_df, 
        colors2, show_xlabel, show_ylabel     
    )
    
    ax.tick_params(labelsize=14)
    
    # Add regression lines
    all_data_frames = [rhesus_table, tonkean_table]
    if (metric in non_behavioral_df.columns and 'age_at_scan' in non_behavioral_df.columns):
        non_behavioral_subset = non_behavioral_df[['age_at_scan', metric]].dropna()
        if not non_behavioral_subset.empty:
            all_data_frames.append(non_behavioral_subset)
    
    all_data = pd.concat(all_data_frames, ignore_index=True)
    
    significant_fits = {
        "LIP-FEF": {"type": "quadratic", "significant": False, "better_than_linear": True, "r2": 0.1824, "direction": "inverted U", "p": 0.0347},
        "LIP-MT": {"type": None, "significant": False, "better_than_linear": False, "r2": 0.1438, "direction": "negative", "p": 0.0511},
        "LIP-MST": {"type": "linear", "significant": True, "better_than_linear": False, "r2": 0.1673, "direction": "negative", "p": 0.0341},
        "LIP-V1": {"type": "linear", "significant": True, "better_than_linear": False, "r2": 0.2080, "direction": "negative", "p": 0.0168},
        "V2-V4": {"type": "linear", "significant": True, "better_than_linear": False, "r2": 0.1962, "direction": "negative", "p": 0.0207},
        "MT-MST": {"type": None, "significant": False, "better_than_linear": False, "r2": 0.1430, "direction": "negative", "p": 0.0518},
    }
    
    # Add statistics text
    if title in significant_fits:
        fit_info = significant_fits[title]
        r2_val, direction, p_val = fit_info["r2"], fit_info["direction"], fit_info["p"]
        
        if fit_info["type"] == "quadratic":
            stats_text = f"R² = {r2_val:.3f}\n{direction}\np = {p_val:.4f}*"
        elif fit_info["type"] == "linear":
            r_val = np.sqrt(r2_val) * (-1 if "negative" in direction else 1)
            if p_val < 0.001:
                p_text = "p < 0.001***"
            elif p_val < 0.01:
                p_text = f"p = {p_val:.4f}**"
            elif p_val < 0.05:
                p_text = f"p = {p_val:.4f}*"
            else:
                p_text = f"p = {p_val:.4f}"
            stats_text = f"r = {r_val:.3f}\nR² = {r2_val:.3f}\n{p_text}"
        else:
            r_val = np.sqrt(r2_val) * (-1 if "negative" in direction else 1)
            stats_text = f"r = {r_val:.3f}\nR² = {r2_val:.3f}\np = {p_val:.4f}"
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                         edgecolor='lightgray', linewidth=0.5))
    
    # Add regression lines
    if title in significant_fits and significant_fits[title]["type"] is not None:
        fit_info = significant_fits[title]
        fit_type = fit_info["type"]
        
        clean_data = all_data[[metric, 'age_at_scan']].dropna().copy()
        clean_data['age_at_scan'] = pd.to_numeric(clean_data['age_at_scan'], errors='coerce')
        clean_data[metric] = pd.to_numeric(clean_data[metric], errors='coerce')
        clean_data = clean_data.dropna()
        
        if len(clean_data) > 3:
            line_color = "grey" if fit_info["better_than_linear"] else "black"
            line_kws = {"linewidth": 1.5, "alpha": 0.8, "linestyle": "-"}
            
            if fit_type == "quadratic":
                sns.regplot(ax=ax, data=clean_data, x="age_at_scan", y=metric, 
                           color=line_color, order=2, scatter=False, ci=None, line_kws=line_kws)
            elif fit_type == "linear":
                sns.regplot(ax=ax, data=clean_data, x="age_at_scan", y=metric, 
                           color=line_color, order=1, scatter=False, ci=None, line_kws=line_kws)
    
    ax.set_title(title, color='black', fontsize=18, fontweight='bold')

# FORCE CONSISTENT AXIS LABELS AT THE END
for i, ax in enumerate(axes_flat):
    if i < 3:  # Top row
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    else:  # Bottom row
        ax.set_xlabel('Age', fontsize=14)
    
    if i == 0 or i == 3:  # Left column
        ax.set_ylabel('rs-fc (r)', fontsize=16)
    else:  # Other columns
        ax.set_ylabel('')

handles2 = [     
    plt.Line2D([], [], color=colors2["rhesus"], marker="o", linestyle="None", markersize=8, label=r"$\it{Macaca\ mulatta}$ (f)"),     
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="steelblue", linestyle="None", markersize=8, label=r"$\it{Macaca\ mulatta}$ (m)"),     
    plt.Line2D([], [], color=colors2["tonkean"], marker="o", linestyle="None", markersize=8, label=r"$\it{Macaca\ tonkeana}$ (f)"),     
    plt.Line2D([], [], color="none", marker="o", markeredgecolor="purple", linestyle="None", markersize=8, label=r"$\it{Macaca\ tonkeana}$ (m)"), 
]  

legend = fig2.legend(handles=handles2, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=14)
fig2.suptitle("Resting State FC Across the Lifespan in Macaque Species", fontsize=25, fontweight="bold", color="black")
plt.tight_layout(rect=[0, 0.02, 1, 0.93])

filename_base = "final-functional-connectivity" 
save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg") 
save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")  
plt.savefig(save_path_svg, format="svg") 
plt.savefig(save_path_png, format="png", dpi=300)
plt.show()

def describe_neuroimaging_sample(
    df, age_col="age_at_scan", gender_col="gender", species_col="species", by_gender=True
):
    """
    Generate descriptive statistics for neuroimaging cohort.
    Includes count, mean, std, min, max by species, with optional gender split.

    Parameters:
        df (DataFrame): Input data (e.g., combined_df)
        age_col (str): Column name for age at scan
        gender_col (str): Column name for gender (1 = male, 2 = female)
        species_col (str): Column name for species
        by_gender (bool): If True, split by gender. If False, summarize species only.

    Returns:
        summary_df (DataFrame): Summary table
    """
    df = df[df[age_col].notna()].copy()
    summary = []

    if by_gender:
        groupings = [
            (species, gender_code, gender_label)
            for species in sorted(df[species_col].dropna().unique())
            for gender_code, gender_label in [(1, "Male"), (2, "Female")]
        ]
    else:
        groupings = [(species, None, None) for species in sorted(df[species_col].dropna().unique())]

    for species, gender_code, gender_label in groupings:
        if by_gender:
            subset = df[(df[species_col] == species) & (df[gender_col] == gender_code)]
        else:
            subset = df[df[species_col] == species]

        count = len(subset)
        ages = subset[age_col].dropna()

        if not ages.empty:
            mean_age = round(ages.mean(), 2)
            std_age = round(ages.std(), 2)
            min_age = round(ages.min(), 2)
            max_age = round(ages.max(), 2)
        else:
            mean_age = std_age = min_age = max_age = None

        summary.append({
            "Species": species,
            "Gender": gender_label if by_gender else "All",
            "N": count,
            "Mean Age": mean_age,
            "SD": std_age,
            "Min Age": min_age,
            "Max Age": max_age,
        })

    return pd.DataFrame(summary)

# Add species to non_behavioral_df if missing
if "species" not in non_behavioral_df.columns:
    if "group" in IRM_demographics_df.columns:
        group_map = dict(zip(IRM_demographics_df['name_lower'], IRM_demographics_df['group']))
        non_behavioral_df['species'] = non_behavioral_df['name'].str.lower().map(group_map)
    else:
        non_behavioral_df['species'] = pd.NA

combined_all_df = pd.concat([combined_df, non_behavioral_df], ignore_index=True)

def clean_species_column(df):
    df["species"] = df["species"].str.strip().str.capitalize()
    return df
# Clean both dataframes
combined_df = clean_species_column(combined_df)
non_behavioral_df = clean_species_column(non_behavioral_df)

# Combine for full descriptive stats
combined_all_df = pd.concat([combined_df, non_behavioral_df], ignore_index=True)

# Split by species & gender
neuro_summary_detailed = describe_neuroimaging_sample(combined_all_df, by_gender=True)

# Species-level only (no gender split)
neuro_summary_species = describe_neuroimaging_sample(combined_all_df, by_gender=False)

print("\n--- Neuroimaging Cohort (Non-Behaviorals Incl.) Detailed Summary ---")
print(neuro_summary_detailed.to_string(index=False))

print("\n--- Neuroimaging Cohort (Species-Level Only) ---")
print(neuro_summary_species.to_string(index=False))

# Print non-behavioral subjects
non_behavioral_df = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()]
non_behavioral_df["age_at_scan"] = pd.to_numeric(non_behavioral_df["age_at_scan"], errors='coerce')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 2]}
    )

    # --- LEFT PLOT (Behavioral Data - unchanged) ---
    jitter = 0.15
    colors = {"Rhesus": "steelblue", "Tonkean": "purple", "mulatta": "steelblue"}

    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2)
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2)
        ]

        sns.stripplot(
            x="species",
            y="age_at_task",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,
            size=10,
            ax=ax1,
        )
        sns.stripplot(
            x="species",
            y="age_at_task",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            ax=ax1,
        )

    for species in ["Rhesus", "Tonkean"]:
        subset = combined_df[combined_df["species"] == species]
        mean_age = subset["age_at_task"].mean()
        sem_age = subset["age_at_task"].std() / np.sqrt(len(subset))  # Standard Error of Mean
        ci_95 = 1.96 * sem_age  # 95% Confidence Interval
        ax1.errorbar(
            x=[species],
            y=[mean_age],
            yerr=[ci_95],
            fmt="none",
            ecolor=colors[species],
            capsize=5,
            elinewidth=2,
            zorder=10,
        )

    ax1.set_ylabel("Age", fontsize=14)
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={combined_df[combined_df["species"] == "Rhesus"].shape[0]})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={combined_df[combined_df["species"] == "Tonkean"].shape[0]})',
        ],
        fontsize=14,
    )

    ax1.text(
        0.5,
        1.05,
        "Behavioral Data",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- RIGHT PLOT (Neuroimaging Data with Non-Behavioral Subjects) ---
    
    # First, plot behavioral subjects (colored dots with higher alpha)
    for species in ["Rhesus", "Tonkean"]:
        males = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] != 2) & (combined_df["age_at_scan"].notna())
        ]
        females = combined_df[
            (combined_df["species"] == species) & (combined_df["gender"] == 2) & (combined_df["age_at_scan"].notna())
        ]

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=females,
            jitter=jitter,
            marker="o",
            color=colors[species],
            alpha=0.7,  # Higher alpha for behavioral subjects
            size=10,
            ax=ax2,
        )

        sns.stripplot(
            x="species",
            y="age_at_scan",
            data=males,
            jitter=jitter,
            marker="o",
            facecolors="none",
            edgecolor=colors[species],
            size=10,
            linewidth=1.5,
            alpha=0.7,  # Higher alpha for behavioral subjects
            ax=ax2,
        )

    # Add non-behavioral subjects (same colors but lower alpha)
    non_behavioral_valid = non_behavioral_df[non_behavioral_df["age_at_scan"].notna()].copy()
    
    if not non_behavioral_valid.empty:
        # Map species for non-behavioral subjects if needed
        if "species" not in non_behavioral_valid.columns or non_behavioral_valid["species"].isna().all():
            # Assuming all non-behavioral subjects are Rhesus based on your scan_dates
            non_behavioral_valid["species"] = "Rhesus"  # Adjust this mapping as needed
        
        # Plot non-behavioral subjects with same colors but lower alpha (0.3)
        males_non_behav = non_behavioral_valid[non_behavioral_valid["gender"] == 1]
        females_non_behav = non_behavioral_valid[non_behavioral_valid["gender"] == 2]
        
        for species in ["Rhesus", "Tonkean"]:
            females_species = females_non_behav[females_non_behav["species"] == species]
            males_species = males_non_behav[males_non_behav["species"] == species]
            
            if not females_species.empty:
                sns.stripplot(
                    x="species",
                    y="age_at_scan",
                    data=females_species,
                    jitter=jitter,
                    marker="o",
                    color=colors[species],  # Same color as behavioral
                    alpha=0.3,  # Lower alpha for non-behavioral subjects
                    size=10,
                    ax=ax2,
                )
            
            if not males_species.empty:
                sns.stripplot(
                    x="species",
                    y="age_at_scan",
                    data=males_species,
                    jitter=jitter,
                    marker="o",
                    facecolors="none",
                    edgecolor=colors[species],  # Same edge color as behavioral
                    size=10,
                    linewidth=1.5,
                    alpha=0.3,  # Lower alpha for non-behavioral subjects
                    ax=ax2,
                )

    # Calculate two sets of error bars with SEM
    for species in ["Rhesus", "Tonkean"]:
        # First error bar: behavioral subjects only (colored)
        behavioral_subset = combined_df[(combined_df["species"] == species) & (combined_df["age_at_scan"].notna())]
        
        if not behavioral_subset.empty:
            behavioral_ages = behavioral_subset["age_at_scan"].dropna().tolist()
            if behavioral_ages:
                mean_behavioral = np.mean(behavioral_ages)
                sem_behavioral = np.std(behavioral_ages, ddof=1) / np.sqrt(len(behavioral_ages)) if len(behavioral_ages) > 1 else 0
                ci_95_behavioral = 1.96 * sem_behavioral  # 95% Confidence Interval
                ax2.errorbar(
                    x=[species],
                    y=[mean_behavioral],
                    yerr=[ci_95_behavioral],
                    fmt="none",
                    ecolor=colors[species],
                    capsize=5,
                    elinewidth=2,
                    zorder=10,
                )
        
        # Second error bar: all subjects including non-behavioral
        non_behavioral_subset = non_behavioral_valid[non_behavioral_valid["species"] == species] if not non_behavioral_valid.empty else pd.DataFrame()
        
        # Combine ages from both groups
        all_ages = []
        if not behavioral_subset.empty:
            all_ages.extend(behavioral_subset["age_at_scan"].dropna().tolist())
        if not non_behavioral_subset.empty:
            all_ages.extend(non_behavioral_subset["age_at_scan"].dropna().tolist())
        
        if all_ages and len(all_ages) > len(behavioral_ages):  # Only add if there are additional non-behavioral subjects
            mean_all = np.mean(all_ages)
            sem_all = np.std(all_ages, ddof=1) / np.sqrt(len(all_ages)) if len(all_ages) > 1 else 0
            ci_95_all = 1.96 * sem_all  # 95% Confidence Interval
            # Offset slightly to the right so both error bars are visible
            offset = 0.05
            ax2.errorbar(
                x=[0 + offset if species == "Rhesus" else 1 + offset],
                y=[mean_all],
                yerr=[ci_95_all],
                fmt="none",
                ecolor=colors[species],  # Use species color instead of gray
                capsize=5,
                elinewidth=2,
                zorder=9,
                alpha=0.3,  # Match the alpha of non-behavioral subjects
            )

    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.set_xticks(ticks=[0, 1])
    
    # Calculate total counts including non-behavioral subjects
    rhesus_total = len(combined_df[(combined_df["species"] == "Rhesus") & (combined_df["age_at_scan"].notna())])
    tonkean_total = len(combined_df[(combined_df["species"] == "Tonkean") & (combined_df["age_at_scan"].notna())])
    
    if not non_behavioral_valid.empty:
        rhesus_total += len(non_behavioral_valid[non_behavioral_valid["species"] == "Rhesus"])
        tonkean_total += len(non_behavioral_valid[non_behavioral_valid["species"] == "Tonkean"])
    
    ax2.set_xticklabels(
        [
            f'$\\it{{Macaca\\ mulatta}}$\n(n={rhesus_total})',
            f'$\\it{{Macaca\\ tonkeana}}$\n(n={tonkean_total})',
        ],
        fontsize=14,
    )

    ax2.text(
        0.5,
        1.05,
        "Neuroimaging Data (All)",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    # --- Shared Formatting ---
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', labelleft=False)
    
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_ylim(0, 25)
        ax.set_xlabel("")

    ax1.set_xlim(-0.6, 1.6)
    ax2.set_xlim(-0.6, 1.6)

    fig.suptitle(
        "Demographics (Including Non-Behavioral Subjects)",
        fontsize=25,
        fontweight="bold",
        fontname="DejaVu Sans",
        color="purple",
        y=1.02,
    )

    plt.tight_layout()

    filename_base = "demographics-neuroimaging-all"
    save_path_svg = os.path.join(figure_me_out, f"{filename_base}.svg")
    save_path_png = os.path.join(figure_me_out, f"{filename_base}.png")

    plt.savefig(save_path_svg, format="svg")
    plt.savefig(save_path_png, format="png", dpi=300)
    plt.show()

def calculate_model_significance(y_true, y_pred, n_params, n_obs):
    """
    Calculate various significance metrics for a model
    """
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Degrees of freedom
    df_model = n_params - 1  # -1 because we don't count intercept in F-test
    df_error = n_obs - n_params
    
    if df_error <= 0:
        return None
    
    # F-statistic for overall model significance
    # F = (ESS/df_model) / (RSS/df_error)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    ss_explained = ss_total - ss_residual
    
    if ss_residual == 0:  # Perfect fit
        f_stat = np.inf
        p_value_f = 0.0
    else:
        f_stat = (ss_explained / df_model) / (ss_residual / df_error)
        p_value_f = 1 - f.cdf(f_stat, df_model, df_error)
    
    # AIC and BIC
    aic = n_obs * np.log(mse) + 2 * n_params
    bic = n_obs * np.log(mse) + np.log(n_obs) * n_params
    
    return {
        'r2': r2,
        'mse': mse,
        'aic': aic,
        'bic': bic,
        'f_stat': f_stat,
        'p_value': p_value_f,
        'df_model': df_model,
        'df_error': df_error,
        'significant': p_value_f < 0.05
    }

def likelihood_ratio_test(y_true, y_pred_simple, y_pred_complex, n_params_simple, n_params_complex, n_obs):
    """
    Likelihood ratio test comparing nested models
    Returns p-value for whether complex model is significantly better
    """
    # Calculate log-likelihoods (assuming normal errors)
    mse_simple = mean_squared_error(y_true, y_pred_simple)
    mse_complex = mean_squared_error(y_true, y_pred_complex)
    
    # Avoid log(0)
    if mse_simple <= 0 or mse_complex <= 0:
        return np.nan
    
    log_likelihood_simple = -n_obs/2 * (1 + np.log(2*np.pi) + np.log(mse_simple))
    log_likelihood_complex = -n_obs/2 * (1 + np.log(2*np.pi) + np.log(mse_complex))
    
    # Likelihood ratio statistic
    lr_stat = 2 * (log_likelihood_complex - log_likelihood_simple)
    df_diff = n_params_complex - n_params_simple
    
    if df_diff <= 0:
        return np.nan
    
    # Chi-square test
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    
    return p_value

def linear_vs_quadratic_testing(df, x_col, y_col, group_name=""):
    """
    Test linear vs quadratic models with significance testing
    """
    # Remove NaN values
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) < 5:
        return None
    
    X = pd.to_numeric(valid_data[x_col], errors='coerce').values
    y = pd.to_numeric(valid_data[y_col], errors='coerce').values

    # Remove any problematic values
    mask = np.isfinite(X) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) < 5:
        return None
    
    n = len(y)
    results = {}
    
    # 1. Linear model
    X_reshaped = X.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X_reshaped, y)
    y_pred_linear = linear_model.predict(X_reshaped)
    
    linear_sig = calculate_model_significance(y, y_pred_linear, 2, n)
    if linear_sig:
        linear_sig.update({
            'params': [linear_model.intercept_, linear_model.coef_[0]],
            'model_type': 'linear',
            'equation': f"y = {linear_model.coef_[0]:.4f}x + {linear_model.intercept_:.4f}",
            'vs_linear_p': np.nan  # Baseline comparison
        })
        results['linear'] = linear_sig
    
    # 2. Quadratic model
    try:
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X_reshaped)
        quad_model = LinearRegression()
        quad_model.fit(X_poly, y)
        y_pred_quad = quad_model.predict(X_poly)
        
        quad_sig = calculate_model_significance(y, y_pred_quad, 3, n)
        if quad_sig:
            # Test vs linear
            vs_linear_p = likelihood_ratio_test(y, y_pred_linear, y_pred_quad, 2, 3, n)
            
            quad_sig.update({
                'params': [quad_model.intercept_] + list(quad_model.coef_),
                'model_type': 'quadratic',
                'equation': f"y = {quad_model.coef_[1]:.6f}x² + {quad_model.coef_[0]:.4f}x + {quad_model.intercept_:.4f}",
                'vs_linear_p': vs_linear_p,
                'better_than_linear': vs_linear_p < 0.05 if not np.isnan(vs_linear_p) else False
            })
            results['quadratic'] = quad_sig
    except Exception as e:
        results['quadratic'] = None

    # Determine best model
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Best by AIC (lower is better)
        best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        # Best by R² (higher is better)
        best_r2 = max(valid_results.keys(), key=lambda k: valid_results[k]['r2'])
        # Best significant model by R²
        significant_models = {k: v for k, v in valid_results.items() if v.get('significant', False)}
        best_significant_r2 = max(significant_models.keys(), key=lambda k: significant_models[k]['r2']) if significant_models else None
        
        results['best_aic'] = best_aic
        results['best_r2'] = best_r2
        results['best_significant_r2'] = best_significant_r2
        results['n'] = n
        results['group'] = group_name
        results['n_significant'] = len(significant_models)
    
    return results

def print_linear_vs_quadratic_results(results_dict, metric_name):
    """
    Print linear vs quadratic model comparison results with detailed parameters
    """
    print(f"\n" + "="*100)
    print(f"LINEAR VS QUADRATIC MODEL TESTING: {metric_name}")
    print("="*100)
    
    for group_name, results in results_dict.items():
        if results is None:
            print(f"\n{group_name}: Insufficient data")
            continue
        
        print(f"\n{group_name} (n={results['n']}):")
        print("-" * 90)
        
        # Print detailed model information
        for model_name in ['linear', 'quadratic']:
            if model_name in results and results[model_name] is not None:
                model_info = results[model_name]
                
                # Format p-value
                p_val = model_info.get('p_value', np.nan)
                p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
                if p_val < 0.001:
                    p_str = "<0.001"
                elif p_val < 0.01:
                    p_str = f"{p_val:.3f}"
                
                # Significance markers
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                # Print model details
                print(f"\n{model_name.upper()} MODEL:")
                print(f"  Equation: {model_info['equation']}")
                print(f"  R² = {model_info['r2']:.4f}")
                print(f"  F-statistic = {model_info['f_stat']:.3f}")
                print(f"  p-value = {p_str}{sig_marker}")
                print(f"  AIC = {model_info['aic']:.1f}")
                print(f"  BIC = {model_info['bic']:.1f}")
                
                # Direction interpretation for linear model
                if model_name == 'linear':
                    slope = model_info['params'][1]
                    if slope > 0:
                        direction = "POSITIVE (increases with age)"
                    elif slope < 0:
                        direction = "NEGATIVE (decreases with age)"
                    else:
                        direction = "FLAT (no change with age)"
                    print(f"  Direction: {direction}")
                    print(f"  Slope = {slope:.6f} (change per year)")
                    
                # Direction interpretation for quadratic model
                elif model_name == 'quadratic':
                    linear_coef = model_info['params'][1]
                    quad_coef = model_info['params'][2]
                    
                    if quad_coef > 0:
                        curvature = "U-SHAPED (accelerating increase)"
                    elif quad_coef < 0:
                        curvature = "INVERTED U-SHAPED (peak then decline)"
                    else:
                        curvature = "LINEAR (no curvature)"
                    
                    print(f"  Curvature: {curvature}")
                    print(f"  Linear coefficient = {linear_coef:.6f}")
                    print(f"  Quadratic coefficient = {quad_coef:.8f}")
                    
                    # Find turning point if quadratic coefficient is significant
                    if abs(quad_coef) > 1e-8:  # Avoid division by near-zero
                        turning_point = -linear_coef / (2 * quad_coef)
                        print(f"  Turning point at age: {turning_point:.1f} years")
                
                # Compare to linear model
                vs_linear_p = model_info.get('vs_linear_p', np.nan)
                if not np.isnan(vs_linear_p) and model_name != 'linear':
                    vs_linear_str = ""
                    if vs_linear_p < 0.001:
                        vs_linear_str = f"{vs_linear_p:.6f}***"
                    elif vs_linear_p < 0.01:
                        vs_linear_str = f"{vs_linear_p:.4f}**"
                    elif vs_linear_p < 0.05:
                        vs_linear_str = f"{vs_linear_p:.4f}*"
                    else:
                        vs_linear_str = f"{vs_linear_p:.4f}"
                    
                    better = "YES" if model_info.get('better_than_linear', False) else "NO"
                    print(f"  vs Linear: p = {vs_linear_str}, Better = {better}")
        
        # Summary comparison
        print(f"\nSUMMARY:")
        valid_models = [k for k in ['linear', 'quadratic'] if k in results and results[k] is not None]
        
        if len(valid_models) > 1:
            print(f"  Best by AIC: {results['best_aic'].upper()}")
            print(f"  Best by R²:  {results['best_r2'].upper()}")
            if results['best_significant_r2']:
                print(f"  Best SIGNIFICANT by R²: {results['best_significant_r2'].upper()}")
            
            # Highlight improvements
            linear_r2 = results['linear']['r2'] if 'linear' in results else 0
            if 'quadratic' in results and results['quadratic'] is not None:
                quad_r2 = results['quadratic']['r2']
                r2_improvement = quad_r2 - linear_r2
                if r2_improvement > 0.05:  # 5% improvement
                    improvement_sig = results['quadratic'].get('better_than_linear', False)
                    sig_text = " (SIGNIFICANT)" if improvement_sig else " (not significant)"
                    print(f"  *** QUADRATIC IMPROVEMENT: +{r2_improvement:.3f} R² over linear{sig_text} ***")
                elif r2_improvement > 0.01:  # 1% improvement
                    improvement_sig = results['quadratic'].get('better_than_linear', False)
                    sig_text = " (SIGNIFICANT)" if improvement_sig else " (not significant)"
                    print(f"  Modest quadratic improvement: +{r2_improvement:.3f} R² over linear{sig_text}")
                else:
                    print(f"  Minimal difference between models (+{r2_improvement:.3f} R²)")
        
        print(f"\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05")

# Configuration - Change this to switch between metric sets
ANALYZE_IMPULSIVITY = False  # Set to True for impulsivity metrics, False for attention metrics


# Extended Linear vs Quadratic Analysis
print("LINEAR VS QUADRATIC MODEL TESTING - EXTENDED")
print("="*80)

SEED_NAMES = ["V1", "V2", "V3", "V4", "V4t", "MT", "MST", "FST", "LIP", "FEF"]

if ANALYZE_IMPULSIVITY:
    # Keep existing impulsivity metrics
    metrics_to_test = [
        find_fc_column("rIFG_44", "preSMA", rhesus_table),
        find_fc_column("rIFG_45", "preSMA", rhesus_table),
        find_fc_column("rIFG_45", "SMA", rhesus_table),
        find_fc_column("rIFG_45", "PMv", rhesus_table),
        find_fc_column("M1", "rIFG_44", rhesus_table),
        find_fc_column("M1", "preSMA", rhesus_table)
    ]
    
    metric_names = [
        "rIFG_44 - preSMA",
        "rIFG_45 - preSMA", 
        "rIFG_45 - SMA",
        "rIFG_45 - PMv",
        "M1 - rIFG_44",
        "M1 - preSMA"
    ]
else:
    # EXTENDED ATTENTION/VISUAL PROCESSING METRICS
    
    # 1. CORE ATTENTION NETWORK (keep existing + add)
    core_attention = [
        ("LIP", "FEF"),      # Primary attention control
        ("LIP", "MT"),       # Motion attention
        ("FEF", "V4"),       # Visual attention
        ("LIP", "V4"),       # Spatial-visual attention
    ]
    
    # 2. VISUAL HIERARCHY CONNECTIONS (developmental controls)
    visual_hierarchy = [
        ("V1", "V2"),        # Early visual processing
        ("V2", "V4"),        # Form processing pathway  
        ("V4", "V4t"),       # Ventral stream progression
        ("MT", "MST"),       # Motion processing hierarchy
        ("V4", "MT"),        # Form-motion integration
    ]
    
    # 3. TOP-DOWN CONTROL CONNECTIONS
    top_down = [
        ("FEF", "V1"),       # Top-down to earliest visual area
        ("FEF", "V2"),       # Top-down to early visual
        ("FEF", "MT"),       # Attention to motion areas
        ("LIP", "V1"),       # Spatial attention to early visual
        ("LIP", "MST"),      # Spatial attention to motion
    ]
    
    # 4. LONG-RANGE INTEGRATION
    integration = [
        ("FST", "LIP"),      # Superior temporal to parietal
        ("FST", "FEF"),      # Superior temporal to frontal
        ("V4t", "LIP"),      # Ventral temporal to parietal
    ]
    
    # Combine all connections
    all_connections = core_attention + visual_hierarchy + top_down + integration
    
    # Create metrics and names
    metrics_to_test = []
    metric_names = []
    
    for area1, area2 in all_connections:
        metric = find_fc_column(area1, area2, rhesus_table)
        if metric:  # Only add if connection exists
            metrics_to_test.append(metric)
            metric_names.append(f"{area1}-{area2} rs-fc")

# Ensure metrics exist in non_behavioral_df
for m in metrics_to_test:
    if m not in non_behavioral_df.columns:
        non_behavioral_df[m] = np.nan

# Group metrics by category for organized output
if not ANALYZE_IMPULSIVITY:
    print("\nANALYZING VISUAL-ATTENTION CONNECTIVITY PATTERNS")
    print("Categories:")
    print("  • Core Attention Network: LIP-FEF, LIP-MT, FEF-V4, LIP-V4")
    print("  • Visual Hierarchy: V1-V2, V2-V4, V4-V4t, MT-MST, V4-MT") 
    print("  • Top-Down Control: FEF→V1/V2/MT, LIP→V1/MST")
    print("  • Long-Range Integration: FST connections, V4t-LIP")
    print()

# Run analysis for each metric
for metric, metric_name in zip(metrics_to_test, metric_names):
    comp_results = {}
    
    # All subjects
    all_data = pd.concat([
        rhesus_table[['age_at_scan', metric]],
        tonkean_table[['age_at_scan', metric]],
        non_behavioral_df[['age_at_scan', metric]] if metric in non_behavioral_df.columns else pd.DataFrame()
    ], ignore_index=True)
    
    comp_results['All Subjects'] = linear_vs_quadratic_testing(all_data, 'age_at_scan', metric, 'All Subjects')
    
    # Optional: Add species-specific analyses for significant findings
    # if comp_results['All Subjects']['quadratic_better']:
    #     behavioral_data = pd.concat([
    #         rhesus_table[['age_at_scan', metric]],
    #         tonkean_table[['age_at_scan', metric]]
    #     ], ignore_index=True)
    #     comp_results['Behavioral Only'] = linear_vs_quadratic_testing(behavioral_data, 'age_at_scan', metric, 'Behavioral Only')
    
    print_linear_vs_quadratic_results(comp_results, metric_name)

print(f"\n" + "="*80)
print("EXTENDED LINEAR VS QUADRATIC ANALYSIS COMPLETE")
print("="*80)

