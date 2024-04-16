import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import re


import json


# with open('tb_temp/step_to_path_mapping.json') as f:
#     step_to_file_path = json.load(f)
    
# ckpt_idx_to_step = {}
# for step, file_path in step_to_file_path.items():
#     ckpt_idx_to_step[file_path.split('.')[-2]] = step
    
# print(ckpt_idx_to_step)

# Function to extract checkpoint number from the tensorboard log path
def extract_checkpoint_number(tb_path):
    match = re.search(r'/(\d+)/events\.out\.', tb_path)
    if match:
        return match.group(1)
    return None


# Function to extract data from event logs
def extract_data_from_event_logs(logdir, use_subdir=False):
    combined_data = {
        'step': [],
        'wall_time': [],
    }

    if use_subdir:
        for subdir in os.listdir(logdir):
            subdir_path = os.path.join(logdir, subdir)
            if os.path.isdir(subdir_path):
                event_acc = EventAccumulator(subdir_path)
                event_acc.Reload()

                # Get a list of all scalar tags
                scalar_tags = event_acc.Tags()['scalars']

                if scalar_tags:  # Check if scalar_tags is not empty
                    # Initialize data lists for each tag
                    for tag in scalar_tags:
                        if tag not in combined_data:
                            combined_data[tag] = []

                    for event in event_acc.Scalars(scalar_tags[0]):  # Using the first tag to get all steps
                        # ckpt_idx = subdir
                        # if ckpt_idx:
                            # step = ckpt_idx_to_step[ckpt_idx]
                            # if step:
                        step = event.step
                        combined_data['step'].append(int(step))
                        combined_data['wall_time'].append(event.wall_time)

                        for tag in scalar_tags:
                            values = event_acc.Scalars(tag)
                            value = next((value for value in values if value.step == event.step), None)
                            combined_data[tag].append(value.value if value is not None else None)
    else:
        if os.path.isdir(logdir):
            event_acc = EventAccumulator(logdir)
            event_acc.Reload()

            # Get a list of all scalar tags
            scalar_tags = event_acc.Tags()['scalars']

            if scalar_tags:  # Check if scalar_tags is not empty
                # Initialize data lists for each tag
                for tag in scalar_tags:
                    if tag not in combined_data:
                        combined_data[tag] = []

                for event in event_acc.Scalars(scalar_tags[0]):  # Using the first tag to get all steps
                    step = event.step
                    combined_data['step'].append(int(step))
                    combined_data['wall_time'].append(event.wall_time)

                    for tag in scalar_tags:
                        values = event_acc.Scalars(tag)
                        value = next((value for value in values if value.step == event.step), None)
                        combined_data[tag].append(value.value if value is not None else None)

    return combined_data

# Function to convert data to DataFrame
def event_logs_to_dataframe(logdir, use_subdir=False):
    data = extract_data_from_event_logs(logdir, use_subdir)
    df = pd.DataFrame(data)
    df = df.sort_values(by='step')
    return df

# Save the DataFrame to a CSV file
def save_dataframe_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

# Function to create and save plots for each column against "step"
def save_plots(df, output_directory):
    for col in df.columns:
        if col != 'step':
            plt.figure()
            plt.plot(df['step'], df[col])
            plt.title(f'{col} vs. step')
            plt.xlabel('step')
            plt.ylabel(col)
            plt.savefig(f'{output_directory}/{col.replace("/", "_")}_plot.png')
            plt.close()

# Example usage
if __name__ == "__main__":
    logdir = f'tb/socnav_ddppo_baseline_multi_gpu/seed_3/eval'
    df = event_logs_to_dataframe(logdir, use_subdir=False)
    output_file = f'tensorboard_data_hab_baseline_eval.csv'  # Specify the desired output file name
    save_dataframe_to_csv(df, output_file)
    # save_plots(df, './')
    print("DataFrame saved to", output_file)
    