from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle


class MultiAgentRoundaboutEnv(RoundaboutEnv):
    """
    A multi-agent roundabout environment.
    
    The ego-vehicles are driving on a roundabout with several other vehicles.
    They should avoid collision with other vehicles and reach their destinations.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [0, 8, 16]
                }
            },
            "controlled_vehicles": 2,  # Number of ego-vehicles
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 11,  # [s]
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "normalize_reward": True
        })
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(self._agent_reward(action, vehicle) 
                  for vehicle in self.controlled_vehicles) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: MDPVehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward 
                    for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                              [self.config["collision_reward"], self.config["high_speed_reward"]],
                              [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _agent_rewards(self, action: int, vehicle: MDPVehicle) -> dict:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(vehicle.speed, [0, self.config["high_speed_reward"]], [0, 1])
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        """The episode is over if any vehicle crashed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles)

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) 
                                     for vehicle in self.controlled_vehicles)
        info["agents_terminated"] = tuple(vehicle.crashed 
                                        for vehicle in self.controlled_vehicles)
        return info

    def _spawn_vehicle(self, longitudinal: float = 0, position_deviation: float = 1.0, 
                      speed_deviation: float = 1.0, spawn_probability: float = 0.6,
                      go_straight: bool = False) -> None:
        """Spawn a vehicle on the roundabout."""
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("se", "ex", 0),
            longitudinal=(longitudinal + 5 + self.np_random.normal() * position_deviation),
            speed=8 + self.np_random.normal() * speed_deviation
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("ex")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane.
        Adapted for multiple controlled vehicles.
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"]))
             for _ in range(self.config["simulation_frequency"])]

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("se", "ex", 0))
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60)
            )
            
            # Set random route
            destination = "ex"
            ego_vehicle.plan_route_to(destination)
            
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    continue

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)