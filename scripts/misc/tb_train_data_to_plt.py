import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_tags_from_event_file(event_file):
    event_acc = event_accumulator.EventAccumulator(event_file)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    return tags

def select_tags(tags):
    selected_tags = []
    print("Available tags:")
    tags = sorted(tags)
    for i, tag in enumerate(tags):
        print(f"{i + 1}. {tag}")

    while True:
        try:
            selection = int(input("Enter the tag number to select (0 to finish): "))
            if selection == 0:
                break
            selected_tags.append(tags[selection - 1])
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid tag number.")

    return selected_tags

def combine_data_from_directories(directories, selected_tags):
    combined_data = {}
    for tag in tqdm(selected_tags):
        for directory in tqdm(directories):
            event_files = [file for file in os.listdir(directory) if file.startswith('events.out.tfevents')]
            current_plot_data = {'steps': [], 'values': []}
            if event_files:
                for event_file in event_files:
                    event_file_path = os.path.join(directory, event_file)
                    tags = get_tags_from_event_file(event_file_path)
                    if tag in tags:
                        try:
                            event_acc = event_accumulator.EventAccumulator(os.path.join(event_file_path))
                            event_acc.Reload()
                            data = event_acc.Scalars(tag)
                            steps = [event.step for event in data]
                            values = [event.value for event in data]
                            current_plot_data['steps'] += steps
                            current_plot_data['values'] += values
                        except:
                            print(f"Skipping for {directory}, {event_file}, {tag} due to missing tag in Reservoir.")
                            continue
            if tag not in combined_data:
                combined_data[tag] = {
                    directory: {
                        'steps': current_plot_data['steps'], 'values': current_plot_data['values']
                    }
                }
            else:
                combined_data[tag][directory] = {
                    'steps': current_plot_data['steps'], 'values': current_plot_data['values']
                }
    return combined_data

def plot_and_save_figures(combined_data, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for tag, directory_names in combined_data.items():
        plt.figure()
        for directory in directory_names:
            if 'full' in directory:
                label = 'full'
            elif 'no_gps' in directory:
                label = 'no_gps'
            else:
                label = 'hab'
            plt.plot(combined_data[tag][directory]['steps'], combined_data[tag][directory]['values'], label=label)
        plt.title(tag)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f'{tag.replace("/", "_")}_plot.png'))
        plt.close()

def main():
    # directories = input("Enter the directories (comma-separated): ").split(',')
    # output_directory = input("Enter the output directory: ")
    
    directories = [
        'tb_temp/socnav_ddppo_baseline_multi_gpu/seed_1',
        'tb_temp/socnav_ddppo_baseline_multi_gpu_full/seed_2',
        'tb_temp/socnav_ddppo_baseline_multi_gpu_no_gps_train/seed_2'
    ]
    
    output_directory = "pyplots"

    # all_tags = set()
    # for directory in directories:
    #     event_files = [file for file in os.listdir(directory) if file.startswith('events.out.tfevents')]
    #     if event_files:
    #         event_file = os.path.join(directory, event_files[0])
    #         tags = get_tags_from_event_file(event_file)
    #         all_tags.update(tags)
    # selected_tags = select_tags(list(all_tags))
    
    selected_tags = [
        'metrics/robot_collisions.total_collisions',
        'metrics/social_nav_stats.avg_robot_to_human_after_encounter_dis_over_epi',
        'metrics/social_nav_stats.follow_human_steps_after_frist_encounter',
        'metrics/social_nav_stats.frist_ecnounter_steps',
        'metrics/social_nav_stats.has_found_human',
        'metrics/num_agents_collide',
        'reward'
    ]
    combined_data = combine_data_from_directories(directories, selected_tags)
    plot_and_save_figures(combined_data, output_directory)

if __name__ == "__main__":
    main()

# tb_temp/socnav_ddppo_baseline_multi_gpu/seed_1,tb_temp/socnav_ddppo_baseline_multi_gpu_full/seed_2,tb_temp/socnav_ddppo_baseline_multi_gpu_no_gps_train/seed_2
# 27, 30, 35, 40, 49