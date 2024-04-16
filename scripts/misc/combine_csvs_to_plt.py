import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(labels, metrics, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for label, csv_file in labels.items():
            df = pd.read_csv(csv_file)
            plt.plot(df['step'], df[metric], label=label)
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.title(f'Plot of {metric}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_folder}/{metric.replace("/", "_")}.png')
        plt.close()

# Example usage
labels = {
    'Full': 'tensorboard_data_full_baseline_eval.csv',
    'Habitat': 'tensorboard_data_hab_baseline_eval.csv',
    'Mul-Curr': 'tensorboard_data_mul_curr_final_eval.csv',
    'No GPS': 'tensorboard_data_no_gps_baseline_eval.csv',
    'Sparse-100': 'tensorboard_data_sparse_100_eval.csv',
}
# Eval Metrics
metrics = [
    'eval_metrics/robot_collisions.total_collisions',
    'eval_metrics/social_nav_stats.avg_robot_to_human_after_encounter_dis_over_epi',
    'eval_metrics/social_nav_stats.follow_human_steps_after_frist_encounter',
    'eval_metrics/social_nav_stats.frist_ecnounter_steps',
    'eval_metrics/social_nav_stats.has_found_human',
    'eval_metrics/num_agents_collide',
    'eval_reward/average_reward'
]
output_folder = 'plots'

plot_metrics(labels, metrics, output_folder)
