import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# -----------------------
# Load and prepare data
# -----------------------
df_main = pd.read_csv('summary_PILOT_500times2_output.csv')
df_mars = pd.read_csv('summary_PILOT_two_settings_250_output.csv')
df_mars = df_mars[df_mars['model'] == 'MARS']  # keep only MARS rows

df = pd.concat([df_main, df_mars], ignore_index=True)
df = df[df['model'] != 'TrueDGP']  # drop TrueDGP from analysis

models = list(df['model'].unique())
priority = ['PILOT', 'PILOT_Base']
models_ordered = [m for m in priority if m in models] + [m for m in models if m not in priority]
models = models_ordered

idxs = sorted(df['idx'].unique())

# -----------------------
# 1) Wilcoxon: PILOT vs. PILOT-k on nISE
# -----------------------
results_perf = []

for idx in idxs:
    df_idx = df[df['idx'] == idx]
    nise_pilot = df_idx[df_idx['model'] == 'PILOT_Base']['mse_test'].values
    nise_pilotk = df_idx[df_idx['model'] == 'PILOT']['mse_test'].values

    if len(nise_pilot) != len(nise_pilotk) or len(nise_pilot) < 2:
        print(f"[Performance] idx={idx}: unmatched or too few samples; skipping.")
        continue

    diff = nise_pilot - nise_pilotk
    stat, p_value = wilcoxon(diff, alternative='two-sided')
    mean_diff = np.mean(diff)  # sign indicates direction

    results_perf.append({
        'idx': idx,
        'Wilcoxon_stat': stat,
        'p_value': p_value,
        'mean_direction': mean_diff
    })

summary_perf = pd.DataFrame(results_perf)
print("\nWilcoxon per-idx results comparing PILOT vs. PILOT-k on nISE:")
print(summary_perf.to_string(index=False))
summary_perf.to_csv('wilcoxon_pilot_vs_pilotk_nise.csv', index=False)

# -----------------------
# 2) Wilcoxon: PILOT-k k-values (noise vs. clean) + median k on clean
# -----------------------
results_k = []

for idx in idxs:
    df_idx = df[(df['idx'] == idx) & (df['model'] == 'PILOT')]

    k_clean = df_idx[df_idx['n_useless'] == 0]['param_bic_factor'].values
    k_noise = df_idx[df_idx['n_useless'] == 1]['param_bic_factor'].values

    if len(k_clean) != len(k_noise) or len(k_clean) < 2:
        print(f"[Penalty] idx={idx}: unmatched or too few samples; skipping.")
        continue

    diff = k_noise - k_clean
    stat, p_value = wilcoxon(diff, alternative='greater')
    mean_diff = np.mean(diff)
    median_k_clean = np.median(k_clean)

    results_k.append({
        'idx': idx,
        'Wilcoxon_stat': stat,
        'p_value': p_value,
        'mean_direction': mean_diff,
        'median_k_clean': median_k_clean
    })

summary_k = pd.DataFrame(results_k)
print("\nWilcoxon per-idx results comparing PILOT-k penalty factor (noise vs. clean):")
print(summary_k.to_string(index=False))
summary_k.to_csv('wilcoxon_pilotk_kfactor_noise_vs_clean.csv', index=False)

# -----------------------
# 3) Mean ± std table: means + stds in parentheses on separate rows
# -----------------------
mean_std_rows = []

for idx in idxs:
    mean_row = {'idx': idx}
    std_row = {'idx': ""}
    df_idx = df[df['idx'] == idx]

    for model in models:
        nise = df_idx[df_idx['model'] == model]['mse_test'].values
        if len(nise) > 0:
            mean = np.mean(nise)
            std = np.std(nise)
            mean_row[model] = f"{mean:.4f}"
            std_row[model] = f"({std:.4f})"
        else:
            mean_row[model] = "NA"
            std_row[model] = "NA"

    mean_std_rows.append(mean_row)
    mean_std_rows.append(std_row)

mean_std_df = pd.DataFrame(mean_std_rows)
print("\nMean ± std nISE per model per idx (std shown in parentheses on row below):")
print(mean_std_df.to_string(index=False))
mean_std_df.to_csv('mean_std_nISE_per_model.csv', index=False)

# -----------------------
# 4) Percent increase from clean (0) to noise (1) table
# -----------------------
perc_noise_rows = []

for idx in idxs:
    row = {'idx': idx}
    for model in models:
        df_model_idx = df[(df['idx'] == idx) & (df['model'] == model)]

        nise_clean = df_model_idx[df_model_idx['n_useless'] == 0]['mse_test'].values
        nise_noise = df_model_idx[df_model_idx['n_useless'] == 1]['mse_test'].values

        if len(nise_clean) > 0 and len(nise_noise) > 0:
            mean_clean = np.mean(nise_clean)
            mean_noise = np.mean(nise_noise)

            if mean_clean == 0.0:
                perc_increase = "NA"  # avoid division by zero
            else:
                perc_increase = 100 * (mean_noise - mean_clean) / mean_clean
                perc_increase = f"{perc_increase:.1f}%"
        else:
            perc_increase = "NA"

        row[model] = perc_increase
    perc_noise_rows.append(row)

perc_noise_df = pd.DataFrame(perc_noise_rows)
print("\nPercent increase in nISE from clean (0) to noise (1) per model per idx:")
print(perc_noise_df.to_string(index=False))
perc_noise_df.to_csv('percent_increase_nISE_clean_to_noise.csv', index=False)
