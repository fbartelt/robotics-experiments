# %%
# Processes a txt file containint lines like:
# Warning: case_113244 at t=0.9320 (step 932) – dist=1.1620e-01, euc_dist=0.0
# And convert it to a csv file with columns: case, time, step, dist, euc_dist
import re
import pandas as pd

def log2csv(log_file, csv_file):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'case_(\d+) at t=([\d.]+) \(step (\d+)\) – dist=([\deE.-]+), euc_dist=([\deE.-]+)', line)
            if match:
                case = int(match.group(1))
                time = float(match.group(2))
                step = int(match.group(3))
                dist = float(match.group(4))
                euc_dist = float(match.group(5))
                data.append((case, time, step, dist, euc_dist))

    df = pd.DataFrame(data, columns=['case', 'time', 'step', 'dist', 'euc_dist'])
    df.to_csv(csv_file, index=False)

log_file = "./experiment_log2.txt"
csv_file = "./experiment_log2.csv"
log2csv(log_file, csv_file)


