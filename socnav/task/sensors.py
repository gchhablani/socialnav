from typing import Any

import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes


@registry.register_sensor
class StepIDSensor(Sensor):
    cls_uuid: str = "step_id"
    curr_ep_id: str = ""
    _elapsed_steps: int = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,1),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self.curr_ep_id != episode_uniq_id:
            self.curr_ep_id = episode_uniq_id
            self._elapsed_steps = 0
        else:
            self._elapsed_steps += 1
        return np.array(self._elapsed_steps).reshape(1, 1)