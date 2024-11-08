from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class IntersectionEnv(AbstractEnv):
    """
    十字路口环境。
    包含一个四向十字路口,每个方向都有直行、左转和右转车道。
    车辆需要在此环境中安全通过路口。

    道路优先级规则:
    - 水平方向直行和右转车道优先级为3
    - 垂直方向直行和右转车道优先级为1
    - 水平方向左转车道优先级为2
    - 垂直方向左转车道优先级为0

    道路网络节点编码规则:
    (o:外部 | i:内部 + [r:右, l:左]) + (0:南 | 1:西 | 2:北 | 3:东)
    例如: "o0"表示南向外部节点, "ir1"表示西向内部右侧节点
    """
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    """动作空间定义: 减速、保持、加速"""
    
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """动作名称到索引的映射"""

    @classmethod
    def default_config(cls) -> dict:
        """
        默认配置参数

        配置内容包括:
        1. 观察空间配置:
           - 类型: 运动学状态
           - 观察车辆数量: 15
           - 特征: 位置、速度、朝向等
           - 特征值范围
           - 是否使用绝对坐标
           - 是否观察其他车辆意图

        2. 动作空间配置:
           - 类型: 离散元动作
           - 纵向控制: 是
           - 横向控制: 否
           - 目标速度选项: [0, 4.5, 9]m/s

        3. 环境参数:
           - 持续时间: 13秒
           - 目的地: "o1"(西向出口)
           - 控制车辆数: 1
           - 初始车辆数: 10
           - 车辆生成概率: 0.6
           - 视图参数
           - 奖励参数
        """
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",                # 观察类型:运动学状态
                    "vehicles_count": 15,                # 观察车辆数量
                    "features": [
                        "presence",                      # 是否存在
                        "x", "y",                        # 位置坐标
                        "vx", "vy",                      # 速度分量
                        "cos_h", "sin_h"                 # 朝向角的三角函数值
                    ],
                    "features_range": {                  # 特征值范围
                        "x": [-100, 100],                # x坐标范围
                        "y": [-100, 100],                # y坐标范围
                        "vx": [-20, 20],                 # x方向速度范围
                        "vy": [-20, 20],                 # y方向速度范围
                    },
                    "absolute": True,                    # 使用绝对坐标
                    "flatten": False,                    # 不压平观察空间
                    "observe_intentions": False,         # 不观察其他车辆意图
                },
                "action": {
                    "type": "DiscreteMetaAction",       # 动作类型:离散元动作
                    "longitudinal": True,                # 启用纵向控制
                    "lateral": False,                    # 禁用横向控制
                    "target_speeds": [0, 4.5, 9],       # 目标速度选项[m/s]
                },
                "duration": 13,                         # 场景持续时间[s]
                "destination": "o1",                    # 目的地(西向出口)
                "controlled_vehicles": 1,               # 控制车辆数量
                "initial_vehicle_count": 10,            # 初始车辆数量
                "spawn_probability": 0.6,               # 车辆生成概率
                "screen_width": 600,                    # 屏幕宽度
                "screen_height": 600,                   # 屏幕高度
                "centering_position": [0.5, 0.6],       # 视角中心位置
                "scaling": 5.5 * 1.3,                   # 缩放比例
                "collision_reward": -5,                 # 碰撞惩罚
                "high_speed_reward": 1,                 # 高速行驶奖励
                "arrived_reward": 1,                    # 到达目的地奖励
                "reward_speed_range": [7.0, 9.0],      # 速度奖励范围[m/s]
                "normalize_reward": False,              # 不归一化奖励
                "offroad_terminal": False,              # 驶出道路不终止
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        计算所有受控车辆的平均奖励值。用于多智能体协作场景。
        
        :param action: 执行的动作
        :return: 所有受控车辆的平均奖励值
        """
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """
        计算多目标奖励。用于多智能体协作场景。
        
        对每个奖励目标:
        1. 计算每个智能体的奖励值
        2. 求所有智能体该目标的平均值
        
        :param action: 执行的动作
        :return: 包含各奖励目标平均值的字典
        """
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        计算单个智能体的奖励值。
        
        奖励计算过程:
        1. 获取各项奖励分量
        2. 根据配置权重进行加权求和
        3. 如果到达目的地,使用到达奖励替代
        4. 如果不在道路上,奖励归零
        5. 根据配置进行归一化
        
        :param action: 执行的动作
        :param vehicle: 目标车辆
        :return: 该车辆的总奖励值
        """
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """
        计算单个智能体的各项奖励分量。
        
        奖励分量包括:
        1. collision_reward: 碰撞惩罚
        2. high_speed_reward: 高速行驶奖励(速度映射到[0,1])
        3. arrived_reward: 到达目的地奖励
        4. on_road_reward: 在道路上的奖励
        
        :param action: 执行的动作
        :param vehicle: 目标车辆
        :return: 包含各奖励分量的字典
        """
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        """
        判断回合是否结束。
        
        以下任一条件满足时结束:
        1. 任一受控车辆发生碰撞
        2. 所有受控车辆都到达目的地
        3. 配置了驶出道路终止且车辆驶出道路
        
        :return: 是否结束回合
        """
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """
        判断单个智能体是否结束。
        
        当发生碰撞或到达目的地时结束。
        
        :param vehicle: 目标车辆
        :return: 该车辆是否结束
        """
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """
        判断是否截断回合。
        
        当超过配置的持续时间时截断。
        
        :return: 是否截断回合
        """
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        """
        获取环境信息。
        
        除基类信息外,还包含:
        1. 每个智能体的奖励
        2. 每个智能体的终止状态
        
        :param obs: 观察值
        :param action: 执行的动作
        :return: 包含环境信息的字典
        """
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        """
        重置环境。
        
        包括:
        1. 创建道路网络
        2. 创建初始车辆
        """
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        环境步进一步。
        
        除了基类的步进操作外,还:
        1. 清除不需要的车辆
        2. 根据概率生成新车辆
        
        :param action: 执行的动作
        :return: (观察值, 奖励值, 是否终止, 是否截断, 信息字典)
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        创建十字路口道路网络。
        
        道路结构:
        1. 四个方向的直行道路
        2. 每个方向的左转弯道(圆弧)
        3. 每个方向的右转弯道(圆弧)
        
        道路参数:
        - lane_width: 车道宽度(默认值)
        - right_turn_radius: 右转弯半径 = 车道宽度 + 5m
        - left_turn_radius: 左转弯半径 = 右转弯半径 + 车道宽度
        - outer_distance: 外部距离 = 右转弯半径 + 车道宽度/2
        - access_length: 接入道路长度 = 50 + 50m
        
        优先级规则:
        - 水平方向(东西向)直行和右转优先级为3
        - 垂直方向(南北向)直行和右转优先级为1
        - 水平方向左转优先级为2
        - 垂直方向左转优先级为0
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m]
        left_turn_radius = right_turn_radius + lane_width  # [m]
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        
        # 遍历四个方向创建道路
        for corner in range(4):
            angle = np.radians(90 * corner)  # 当前方向的角度
            is_horizontal = corner % 2        # 是否为水平方向(东西向)
            priority = 3 if is_horizontal else 1  # 设置优先级
            
            # 计算旋转矩阵
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            
            # 创建入口道路
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )
            
            # 创建右转弯道路
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            
            # 创建左转弯道路
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,  # 左转优先级降低
                    speed_limit=10,
                ),
            )
            
            # 创建直行道路
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10
                ),
            )
            
            # 创建出口道路
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        在道路上创建车辆。
        
        创建的车辆包括:
        1. 随机车辆:
           - 数量: n_vehicles - 1
           - 位置: 均匀分布在0-80m范围内
           - 每创建一辆车后模拟几步,避免初始拥堵
        
        2. 挑战车辆:
           - 数量: 1
           - 位置: 距路口60m处
           - 特点: 直行通过路口
           - 位置偏差: 0.1m
           - 速度偏差: 0m/s
        
        3. 受控车辆(ego vehicles):
           - 数量: 由config["controlled_vehicles"]指定
           - 初始位置: 距路口60m处,带5m随机偏差
           - 初始速度: 道路限速
           - 目的地: 由config["destination"]指定或随机选择
        
        车辆配置:
        - 期望车距: 7m
        - 最大加速度: 6m/s²
        - 最小加速度: -3m/s²
        
        :param n_vehicles: 初始车辆数量
        """
        # 配置车辆参数
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # 创建随机车辆
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # 创建挑战车辆
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # 创建受控车辆(ego vehicles)
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                (f"o{ego_id % 4}", f"ir{ego_id % 4}", 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            speed=8 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )
        return config
