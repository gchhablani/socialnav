from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import HabitatConfig, LabSensorConfig

from dataclasses import dataclass

@dataclass
class StepIDSensorConfig(LabSensorConfig):
    type: str = "StepIDSensor"


@dataclass
class SparseGpsHabitatConfig(HabitatConfig):
    gps_available_every_x_steps: int = 5

@dataclass
class CurriculumHabitatConfig(HabitatConfig):
    gps_available_every_x_steps: int = 5
    curriculum_lower_threshold: float = 0.85
    curriculum_upper_threshold: float = 0.90



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