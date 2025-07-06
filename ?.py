import numpy as np
import matplotlib.pyplot as plt

# Paths to your timeseries text files
FEF_ts_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/connectivity/FEF_ts.txt"
LIP_ts_path = "/Users/similovesyou/Desktop/qts/simian-brain/data/site-strasbourg/derivatives/patchoulli/func/connectivity/LIP_ts.txt"

# Load timeseries
fe_ts = np.loadtxt(FEF_ts_path)
lip_ts = np.loadtxt(LIP_ts_path)

# Calculate Pearson correlation
corr = np.corrcoef(fe_ts, lip_ts)[0,1]
print(f"Pearson correlation between FEF and LIP time series: {corr:.3f}")

# Plot timeseries
plt.figure(figsize=(10, 5))
plt.plot(fe_ts, label='FEF', linewidth=1.5)
plt.plot(lip_ts, label='LIP', linewidth=1.5)
plt.title(f'FEF vs LIP Time Series (r = {corr:.3f})')
plt.xlabel('Timepoints')
plt.ylabel('Mean BOLD Signal')
plt.legend()
plt.tight_layout()
plt.show()
