import os
import torch
import json

# Define the directory containing the PyTorch checkpoints
experiment_name = "socnav_ddppo_baseline_multi_gpu_full"
checkpoint_dir = f'data/socnav_checkpoints/{experiment_name}'

# Define the path for the JSON file
json_output_path = f'step_to_path_mapping_{experiment_name}.json'


# Initialize an empty dictionary to store 'extra_state'['step'] to path mapping
extra_state_to_path_mapping = {}

# Iterate over the files in the checkpoint directory
for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if file.endswith('.pt') or file.endswith('.pth') and not file.startswith('.habitat-resume-path'):  # Assuming PyTorch checkpoint file extensions
            checkpoint_path = os.path.join(root, file)
            try:
                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path)

                # Check if the checkpoint has an 'extra_state' attribute
                if 'extra_state' in checkpoint:
                    # print(checkpoint.keys())
                    extra_state = checkpoint['extra_state']
                    # extra_state['wall_time'] = float(extra_state['wall_time'])

                    # Store the mapping of 'extra_state'['step'] to the checkpoint path
                    extra_state_to_path_mapping[int(extra_state['step'])] = checkpoint_path

            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")

# Save the 'extra_state'['step'] to path mapping as a JSON file
with open(json_output_path, 'w') as json_file:
    json.dump(extra_state_to_path_mapping, json_file)

print("'extra_state'['step'] to path mapping has been saved to", json_output_path)
