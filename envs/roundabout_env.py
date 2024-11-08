from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle


class RoundaboutEnv(AbstractEnv):
    """
    环形交叉路口环境。
    
    环境特点:
    1. 包含一个双车道环形交叉路口和四个方向的出入口道路
    2. 车辆需要在此环境中安全通行
    3. 支持多车辆交互
    4. 提供基于运动学状态的观察
    5. 使用离散元动作控制
    """
    @classmethod
    def default_config(cls) -> dict:
        """
        默认配置参数。
        
        配置内容包括:
        1. 观察空间配置:
           - 类型: 运动学状态
           - 使用绝对坐标
           - 特征值范围(x,y,vx,vy的观察范围)
        
        2. 动作空间配置:
           - 类型: 离散元动作
           - 目标速度选项: [0, 8, 16]m/s
        
        3. 环境参数:
           - 进入车辆的目的地: 可配置或随机
           - 奖励参数:
             * collision_reward: 碰撞惩罚(-1)
             * high_speed_reward: 高速行驶奖励(0.2)
             * right_lane_reward: 右车道行驶奖励(0)
             * lane_change_reward: 变道惩罚(-0.05)
           - 视图参数:
             * 屏幕尺寸: 600x600
             * 视角中心位置: [0.5, 0.6]
           - 场景持续时间: 11秒
           - 是否归一化奖励: 是
        
        :return: 包含默认配置的字典
        """
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",  # 观察类型:运动学状态
                    "absolute": True,       # 是否使用绝对坐标
                    "features_range": {     # 特征值范围
                        "x": [-100, 100],   # x坐标范围
                        "y": [-100, 100],   # y坐标范围
                        "vx": [-15, 15],    # x方向速度范围
                        "vy": [-15, 15],    # y方向速度范围
                    },
                },
                "action": {
                    "type": "DiscreteMetaAction",     # 动作类型:离散元动作
                    "target_speeds": [0, 8, 16]       # 目标速度选项[m/s]
                },
                "incoming_vehicle_destination": None,  # 进入车辆的目的地
                "collision_reward": -1,               # 碰撞惩罚
                "high_speed_reward": 0.2,            # 高速行驶奖励
                "right_lane_reward": 0,              # 右车道行驶奖励
                "lane_change_reward": -0.05,         # 变道惩罚
                "screen_width": 600,                 # 屏幕宽度
                "screen_height": 600,                # 屏幕高度
                "centering_position": [0.5, 0.6],    # 视角中心位置
                "duration": 11,                      # 场景持续时间[s]
                "normalize_reward": True,            # 是否归一化奖励
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        计算当前步骤的奖励值。
        
        奖励计算过程:
        1. 获取各项奖励分量
        2. 根据配置权重进行加权求和
        3. 如果配置了归一化,将奖励值映射到[0,1]区间
        4. 如果车辆驶出道路,奖励值归零
        
        :param action: 执行的动作
        :return: 计算得到的奖励值
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            # 将奖励值映射到[0,1]区间
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["high_speed_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]  # 如果车辆驶出道路,奖励值归零
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        """
        计算各项奖励分量。
        
        奖励分量包括:
        1. collision_reward: 碰撞惩罚
           - 发生碰撞时为1,否则为0
        
        2. high_speed_reward: 高速行驶奖励
           - 基于车辆当前速度索引计算
           - 速度越高奖励越大
           - 归一化到[0,1]区间
        
        3. lane_change_reward: 变道惩罚
           - 执行变道动作(action为0或2)时为1
           - 其他动作为0
        
        4. on_road_reward: 在道路上的奖励
           - 在道路上为1
           - 驶出道路为0
        
        :param action: 执行的动作
        :return: 包含各奖励分量的字典
        """
        return {
            "collision_reward": self.vehicle.crashed,                # 碰撞惩罚
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)  # 高速奖励
            / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],                 # 变道惩罚
            "on_road_reward": self.vehicle.on_road,                 # 是否在道路上
        }

    def _is_terminated(self) -> bool:
        """
        判断回合是否结束。
        
        终止条件:
        - 车辆发生碰撞时结束
        
        :return: 是否结束回合
        """
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """
        判断是否截断回合。
        
        截断条件:
        - 当超过配置的持续时间(duration)时截断
        
        :return: 是否截断回合
        """
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        """
        重置环境到初始状态。
        
        重置过程:
        1. 调用_make_road()创建道路网络
           - 包括环形主路和四个方向的出入口
        
        2. 调用_make_vehicles()创建车辆
           - 包括自车和其他车辆
           - 设置它们的初始位置、速度和行为
        """
        self._make_road()           # 创建道路
        self._make_vehicles()       # 创建车辆

    def _make_road(self) -> None:
        """
        创建环形交叉路口道路网络。
        
        道路结构:
        1. 环形主路:
           - 双车道设计(内外两条车道)
           - 由8段圆弧车道组成(每个象限2段)
           - 内圈半径20m,外圈半径24m
           - 车道线类型:
             * 内圈: 内侧连续线,外侧虚线
             * 外圈: 内侧无线,外侧连续线
        
        2. 四个方向的出入口道路:
           - 每个方向包含一条入口和一条出口
           - 入口由直线段和正弦曲线段组成
           - 出口同样由正弦曲线段和直线段组成
           - 道路参数:
             * 道路总长: 170m
             * 偏移距离: 85m
             * 正弦曲线振幅: 5m
             * 起点偏移: 17m(0.2*85m)
        
        道路网络节点编码:
        - 环形主路节点:
          * se: 东南象限入口
          * ex: 东象限
          * ee: 东象限出口
          * nx: 北象限
          * ne: 东北象限入口
          * wx: 西象限
          * we: 西象限出口
          * sx: 南象限
        
        - 出入口节点:
          * [方向]er: 入口远端 (如ser)
          * [方向]es: 入口近端 (如ses)
          * [方向]xs: 出口近端 (如sxs)
          * [方向]xr: 出口远端 (如sxr)
        
        参数说明:
        :param center: 环形交叉路口中心坐标 [0, 0]
        :param radius: 内圈半径 20m
        :param alpha: 入口道路切入角度 24度
        :param access: 出入口道路长度 170m
        :param dev: 道路偏移量 85m
        :param a: 正弦曲线振幅 5m
        :param delta_st: 起点偏移量 17m
        """
        # 环形交叉路口参数
        center = [0, 0]  # 圆心坐标 [m]
        radius = 20      # 内圈半径 [m]
        alpha = 24       # 入口道路切入角度 [deg]

        net = RoadNetwork()
        
        # 定义车道线类型
        radii = [radius, radius + 4]  # 内外圈半径
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]       # 内外圈的车道线类型
        
        # 创建8段圆弧车道(每个象限2段)
        for lane in [0, 1]:  # 内外两条车道
            # 东南象限第一段
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 东南象限第二段
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 东北象限第一段
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 东北象限第二段
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 西北象限第一段
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 西北象限第二段
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 西南象限第一段
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 西南象限第二段
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # 创建四个方向的出入口道路
        access = 170  # 道路长度 [m]
        dev = 85      # 道路偏移量 [m]
        a = 5         # 正弦曲线振幅 [m]
        delta_st = 0.2 * dev  # 起点偏移 [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev  # 正弦曲线频率
        
        # 南向入口道路
        net.add_lane(
            "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2],
                [2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        # 南向出口道路
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [-2 - a, -dev / 2 + delta_en],
                [-2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        # 东向入口道路
        net.add_lane(
            "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        # 东向出口道路
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en, 2 + a],
                [dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        # 北向入口道路
        net.add_lane(
            "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        # 北向出口道路
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a, dev / 2 - delta_en],
                [2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        # 西向入口道路
        net.add_lane(
            "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a],
                [-dev / 2 + delta_st, 2 + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        # 西向出口道路
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en, -2 - a],
                [-dev / 2, -2 - a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        在道路上创建车辆。该函数负责初始化环境中的所有车辆。
        
        创建的车辆包括:
        1. 自车(ego-vehicle):
           - 从南向入口进入
           - 初始速度为8m/s
           - 目标为北向出口("nxs")
        
        2. 其他车辆(other vehicles):
           a) 一辆即将进入环岛的车辆:
              - 位于西向入口的外车道
              - 初始速度16m/s(带随机偏差)
              - 随机选择东/南/北向出口
              - 具有随机化的行为特征
           
           b) 两辆在西向入口的车辆:
              - 位于西向入口的内车道
              - 间隔20m
              - 初始速度16m/s(带随机偏差)
              - 随机选择出口方向
              - 具有随机化的行为特征
           
           c) 一辆正在进入环岛的车辆:
              - 位于东向入口
              - 距入口50m处
              - 初始速度16m/s(带随机偏差)
              - 随机选择出���方向
              - 具有随机化的行为特征

        随机参数:
        - position_deviation: 位置的随机偏移标准差，2m
        - speed_deviation: 速度的随机偏移标准差，2m/s
        
        注意:
        1. 所有车辆的初始状态都带有随机性，以增加环境的多样性
        2. 其他车辆都具有随机化的行为特征(通过randomize_behavior()实现)
        3. 可以通过config["incoming_vehicle_destination"]指定第一辆其他车辆的出口
        """
        position_deviation = 2  # 位置随机偏移量 [m]
        speed_deviation = 2     # 速度随机偏移量 [m/s]

        # 创建自车(ego-vehicle)
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))  # 获取南向入口车道
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125, 0),  # 在距入口125m处创建
            speed=8,                    # 初始速度8m/s
            heading=ego_lane.heading_at(140),  # 车头朝向
        )
        try:
            ego_vehicle.plan_route_to("nxs")  # 设置目标为北向出口
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 创建一个即将进入环岛的车辆(位于西向入口外侧车道)
        destinations = ["exr", "sxr", "nxr"]  # 可选的出口: 东/南/北
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("we", "sx", 1),  # 西向入口外侧车道
            longitudinal=5 + self.np_random.normal() * position_deviation,  # 距入口5m(带随机偏差)
            speed=16 + self.np_random.normal() * speed_deviation,  # 初始速度16m/s(带随机偏差)
        )

        # 确定该车辆的目的地
        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()  # 随机化车辆行为特征
        self.road.vehicles.append(vehicle)

        # 创建两辆在西向入口内侧车道的车辆
        for i in list(range(1, 2)) + list(range(-1, 0)):  # 创建两辆车，间隔20m
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("we", "sx", 0),  # 西向入口内侧车道
                longitudinal=20 * i + self.np_random.normal() * position_deviation,  # 间隔20m
                speed=16 + self.np_random.normal() * speed_deviation,  # 初始速度16m/s(带随机偏差)
            )
            vehicle.plan_route_to(self.np_random.choice(destinations))  # 随机选择出口
            vehicle.randomize_behavior()  # 随机化车辆行为特征
            self.road.vehicles.append(vehicle)

        # 创建一个正在进入环岛的车辆(位于东向入口)
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("eer", "ees", 0),  # 东向入口
            longitudinal=50 + self.np_random.normal() * position_deviation,  # 距入口50m
            speed=16 + self.np_random.normal() * speed_deviation,  # 初始速度16m/s(带随机偏差)
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))  # 随机选择出口
        vehicle.randomize_behavior()  # 随机化车辆行为特征
        self.road.vehicles.append(vehicle)
