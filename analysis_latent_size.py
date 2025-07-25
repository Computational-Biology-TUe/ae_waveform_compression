import glob
import pandas as pd
import matplotlib.pyplot as plt

from main_ae import autoencoder


#%% Define the runs

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
runs_short = pd.DataFrame(
    [[2, [178, 179, 180, 181, 182, 465, 471, 470, 478, 477]],
    [4, [189, 190, 191, 193, 198, 529, 467, 464, 481, 468]],
    [6, [199, 200, 201, 202, 208, 469, 482, 473, 483, 479]],
    [8, [209, 210, 211, 212, 218, 480, 474, 472, 476, 475]],
    [9, [219, 220, 221, 222, 228, 530, 491, 492, 531, 489]],
    [10, [229, 230, 231, 232, 239, 291, 289, 292, 293, 294]],
    [11, [183, 184, 185, 186, 187, 532, 487, 495, 496, 533]],
    [12, [170, 172, 174, 177, 176, 494, 498, 497, 486, 485]],
    [13, [246, 250, 249, 247, 248, 506, 534, 511, 505, 503]],
    [14, [192, 197, 194, 195, 196, 518, 517, 515, 516, 502]],
    [16, [203, 207, 204, 205, 206, 504, 501, 513, 514, 509]],
    [18, [213, 217, 214, 215, 216, 510, 500, 499, 507, 508]],
    [20, [223, 227, 224, 225, 226, 300, 296, 297, 301, 299]],
    [22, [233, 237, 235, 234, 244, 525, 526, 523, 535, 520]],
    [24, [240, 243, 242, 241, 238, 519, 521, 522, 528, 527]]],
    columns=['latent_size', 'run_nr'])

# Explode to create one row per run number
runs = runs_short.explode('run_nr')
runs.reset_index(drop=True, inplace=True)
# Add the 'seed' column by repeating the seeds list for each latent size
runs['seed'] = seeds * len(runs_short)


#%% Extract min validation loss to check for convergence

min_val_losses = []

for idx in range(len(runs)):
    run_nr = runs.at[idx, 'run_nr']
    model_path = glob.glob(f'results/networks_autoencoder/AUT-{run_nr}/model_min_val_loss-*.pth')[0]
    # Extract the numeric part from model_path
    min_val_losses.append(float(model_path.split('-')[-1].split('.pth')[0]))

# Add the min validation loss to the runs DataFrame
runs['min_val_loss'] = min_val_losses


#%% Run the test set results

results = []

for idx in range(len(runs)):
    run_nr = runs.at[idx, 'run_nr']
    latent_size = runs.at[idx, 'latent_size']
    seed = runs.at[idx, 'seed']
    custom_args = ['--mode', 'test', '--trained_model', f'AUT-{run_nr}', '--latent_size', str(latent_size), '--seed', str(seed), '--plotting', False]
    print(f"Test AUT-{run_nr}: latent size {latent_size}, seed={seed}")

    # Execute autoencoder model
    test_loss, rmse, _, prd, _, _, _, _ = autoencoder(custom_args)
    results.append([test_loss, rmse, prd])

# Efficient assignment after the loop
runs[['Test MSE Loss', 'RMSE', 'PRD']] = results

size = 320
runs['CR'] = size / runs.latent_size
runs['Quality Score'] = runs['CR'] / runs['PRD']


#%% Plot results

# Remove runs that dit not converge
runs_converged = runs.loc[runs.min_val_loss < 0.1]

# Group by 'latent_size' and calculate median and IQR
summary = runs_converged.groupby('latent_size').agg(
    rmse_median=('RMSE', 'median'),
    rmse_q1=('RMSE', lambda x: x.quantile(0.25)),
    rmse_q3=('RMSE', lambda x: x.quantile(0.75)),
    qs_median=('Quality Score', 'median'),
    qs_q1=('Quality Score', lambda x: x.quantile(0.25)),
    qs_q3=('Quality Score', lambda x: x.quantile(0.75))
).reset_index()

# Plot
fig, ax1 = plt.subplots(figsize=(10, 4))

# Plot RMSE with the first y-axis
ax1.plot(summary['latent_size'], summary['rmse_median'], marker='o', label=f'RMSE Median', color='blue')
# Plot IQR as shaded area
ax1.fill_between(summary['latent_size'], summary['rmse_q1'], summary['rmse_q3'], color='blue', alpha=0.2,
                 label='IQR (25th - 75th percentile)')

# Create a second y-axis for Quality score
ax2 = ax1.twinx()
ax2.plot(summary['latent_size'], summary['qs_median'], marker='s', color='red', label='Quality Score Median')
ax2.fill_between(summary['latent_size'], summary['qs_q1'], summary['qs_q3'], color='red', alpha=0.2,
                 label='IQR (25th - 75th percentile)')

# Add labels and legend
ax1.set_xlabel(r'Latent space size ($d$)')
ax1.set_xticks(summary['latent_size'])

ax1.set_ylabel('RMSE', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.set_ylabel('Quality Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

fig.savefig("analysis_latent_size.svg", format="svg")

plt.show()
