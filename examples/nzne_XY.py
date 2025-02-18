"""
Example: run NZNE on XY model data. 
"""
import json
import sys
from pathlib import Path

# Add the src directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

from nzne import nzne

data_file = "./data/examples/YX_XY_60.json"

with open(data_file, "r") as f:
    data = json.load(f)["fidelities_and_expvals_by_noise_strength"]
    data = {float(k): v for k, v in data.items()}  # Convert data keys to floats

extrapolated_value, extrapolated_log_expval, log_expvals, extrapolation_plot, sorted_noise_strengths = nzne.full_non_zero_noise_extrapolation_pipeline(
        fidelities_and_expvals_by_noise_strength=data, 
        target_noise_strength=0.002, 
        fidelity_threshold=0.99, 
        ignore_pure_state=False, 
        outlier_noise_strengths=[0.002],
        )

print(f"Extrapolated value: {extrapolated_value}")
