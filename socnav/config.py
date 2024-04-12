from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import HabitatConfig, LabSensorConfig
from habitat.config.default_structured_configs import HabitatBaseConfig
from typing import List

from dataclasses import dataclass, field

@dataclass
class StepIDSensorConfig(LabSensorConfig):
    type: str = "StepIDSensor"


@dataclass
class SparseGpsHabitatConfig(HabitatConfig):
    gps_available_every_x_steps: int = 5
    last_gps: bool = False

@dataclass
class CurriculumConfig(HabitatBaseConfig):
    last_gps: bool = False
    additive: bool = False
    dynamic_additive: bool = False
    update_curriculum_every_x_steps: int = 1
    warmup_steps: int = 25000000
    curriculum_upper_threshold: float = 0.9
    curriculum_lower_threshold: float = 0.8
    dynamic_increment_baseline_score: float = 0.8
    dynamic_increment_scaling_factor: float = 0.02
    dynamic_decrement_scaling_factor: float = 0.1
    use_dynamic_lower_threshold: bool = False
    dynamic_lower_threshold: List[List[float]] = field(default_factory=lambda: [[1e8, 0.8], [2e8, 0.85], [3e8, 0.9]])
    add_increment: int = 10
    add_decrement: int = 5
    mult_increment: float = 2
    mult_decrement: float = 2
    staircase_increment: int = 50


@dataclass
class CurriculumHabitatConfig(SparseGpsHabitatConfig):
    curriculum_config: CurriculumConfig = CurriculumConfig()


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()

cs.store(group="habitat", name="sparse_gps_habitat_config", node=SparseGpsHabitatConfig)
cs.store(group="habitat", name="curriculum_habitat_config", node=CurriculumHabitatConfig)

cs.store(
    package="habitat.task.lab_sensors.step_id_sensor",
    group="habitat/task/lab_sensors",
    name="step_id_sensor",
    node=StepIDSensorConfig,
)


# @dataclass
# class CacheImageGoalSensorConfig(LabSensorConfig):
#     type: str = "CacheImageGoalSensor"
#     cache: str = "/srv/flash1/gchhablani3/spring_2024/vlm-task/data/datasets/vc1_embeddings/"

# cs.store(
#     package=f"habitat.task.lab_sensors.cache_imagegoal_sensor",
#     group="habitat/task/lab_sensors",
#     name="cache_imagegoal_sensor",
#     node=CacheImageGoalSensorConfig,
# )

# class HabitatConfigPlugin(SearchPathPlugin):
#     def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
#         search_path.append(
#             provider="habitat",
#             path="pkg://config/experiments/",
#         )
