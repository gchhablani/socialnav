import json
import os

# Read JSON file
with open('/srv/flash1/gchhablani3/spring_2024/socialnav/step_to_path_mapping_socnav_ddppo_baseline_multi_gpu_full.json', 'r') as f:
    data = json.load(f)

# Sort checkpoints by integer value of keys
sorted_checkpoints = sorted(data.items(), key=lambda x: int(x[0]))

# Select 49 checkpoints at regular intervals
interval = len(sorted_checkpoints) // 50
selected_checkpoints = [sorted_checkpoints[i*interval][1] for i in range(50)]

# Add the last checkpoint to the selection
last_checkpoint = sorted_checkpoints[-1][1]
selected_checkpoints.append(last_checkpoint)

# Extract file names
checkpoint_file_names = [os.path.basename(path) for path in selected_checkpoints]

# Write selected checkpoint file names to a text file
with open('selected_checkpoints.txt', 'w') as f:
    for file_name in checkpoint_file_names:
        f.write("%s\n" % file_name)
