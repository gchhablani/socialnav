# socialnav

## Setup
- `conda create -n hab_mt -y python=3.9`
- `conda activate socnav`
- `conda install habitat-sim withbullet headless -c conda-forge -c aihabitat`
- In another directory:
    - `git clone -b v0.3.0 https://github.com/facebookresearch/habitat-lab.git`
    - `cd habitat-lab`
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`
- `python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab3_bench_assets`
- `cd` back to project directory
- `pip install -e .`
- For missing objects:
    - `python -m habitat_sim.utils.datasets_download --uids hab_spot_arm --data-path data/`
    - ``


## Training
- ``

## Evaluation
- ``

## References
- https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines#social-navigation
- Additional objects: https://huggingface.co/datasets/ai-habitat/OVMM_objects
