import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wm', required=True)
parser.add_argument('--csf', required=True)
parser.add_argument('--mc', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--volterra', action='store_true')
args = parser.parse_args()

# Load regressors
wm = np.loadtxt(args.wm)
csf = np.loadtxt(args.csf)
mc = np.loadtxt(args.mc)

# Optionally apply Volterra expansion
if args.volterra:
    mc_sq = mc ** 2
    mc_int = []
    for i in range(mc.shape[1]):
        for j in range(i+1, mc.shape[1]):
            mc_int.append(mc[:, i] * mc[:, j])
    mc = np.column_stack([mc, mc_sq] + mc_int)

# Combine
confounds = np.column_stack([wm, csf, mc])
np.savetxt(args.output, confounds, fmt='%.6f')
