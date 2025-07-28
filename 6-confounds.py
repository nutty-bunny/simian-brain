import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mc', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--volterra', action='store_true', help="Include squares and interactions")
args = parser.parse_args()

# Load motion regressors: assumed shape (timepoints x 6)
mc = np.loadtxt(args.mc)

# Calculate derivatives (frame-to-frame differences)
mc_deriv = np.vstack([np.zeros(mc.shape[1]), np.diff(mc, axis=0)])

# Base regressors: motion + derivatives
regressors = np.column_stack([mc, mc_deriv])

if args.volterra:
    # Squares of motion + derivatives
    squares = regressors ** 2
    
    # Optional: add pairwise interactions between all regressors (can explode dims)
    interactions = []
    for i in range(regressors.shape[1]):
        for j in range(i+1, regressors.shape[1]):
            interactions.append(regressors[:, i] * regressors[:, j])
    if interactions:
        interactions = np.column_stack(interactions)
        regressors = np.column_stack([regressors, squares, interactions])
    else:
        regressors = np.column_stack([regressors, squares])
else:
    # Just add squares (classic Friston 24 style is with volterra=False)
    regressors = np.column_stack([regressors, regressors ** 2])

np.savetxt(args.output, regressors, fmt='%.6f')
