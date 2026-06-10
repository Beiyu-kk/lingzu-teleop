# Lingzu 遥操作平台

本项目是一个独立的 Lingzu EL-A3 遥操作与数据采集平台。

项目内部已经包含机械臂 SDK 代码和 URDF/mesh/config 资源。SDK 负责 CAN 通信、运动学/动力学、200Hz 控制循环、关节限位、重力补偿、`MoveJ`、`MoveL`、`EndPoseCtrl` 等核心功能；平台层负责遥操作、数据采集、相机接入、LeRobot 数据格式适配和模型动作执行。

## 功能定位

- Xbox 手柄遥操作
- 轨迹、状态和动作记录
- 相机接入扩展接口
- LeRobot 风格数据帧映射
- 记录动作或模型动作回放
- 后续训练数据采集平台的项目骨架

## 安装

推荐使用 conda 创建固定环境，然后使用 `pyproject.toml` 安装当前项目核心依赖。

```bash
conda create -n lingzu-teleop python=3.12 -y
conda activate lingzu-teleop
pip install -e .
```

## 环境依赖说明

`pyproject.toml` 只保留遥操作平台和内置 SDK 的核心依赖，避免安装当前阶段不需要的大型视觉、数据集和测试工具。

当前核心依赖包括：

- `imageio`
- `imageio-ffmpeg`
- `numpy`
- `opencv-python`
- `pandas`
- `pillow`
- `pyarrow`
- `pyyaml`
- `pyserial`
- `pin`

按需手动安装：

```bash
# RealSense 相机采集
pip install pyrealsense2

# 官方 LeRobot 工具链
pip install lerobot

# 运行测试
pip install pytest
```

## Xbox 遥操作

启动前请先配置 CAN：

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 128
```

启动手柄遥操作：

```bash
python scripts/xbox_control.py --can can0 --js /dev/input/js0
```

常用参数：

```bash
python scripts/xbox_control.py --can can1
python scripts/xbox_control.py --js /dev/input/js1
python scripts/xbox_control.py --profile auto
python scripts/xbox_control.py --dump-input
python scripts/xbox_control.py --no-zero-init
```

## DROID 风格数据采集

该流程参考 DROID 的 episode 采集方式：启动脚本后先进入待机状态，用户通过键盘决定什么时候开始一条 episode、什么时候结束这条 episode，并可以在同一个会话中重复采集多条数据。每条 episode 开始时先写入 `failure/YYYY-MM-DD/<episode>`，每个采样周期同步保存机械臂状态、最近一次手柄动作、腕部相机图像和主视角相机图像；结束后根据用户确认保留为 `failure` 或移动到 `success`。

当前相机约定：

- `wrist`：腕部相机，Intel RealSense D405
- `front`：主视角相机，Intel RealSense D455

默认启动采集会连接腕部 D405 和主视角 D455：

```bash
python scripts/collect_xbox_dataset.py \
  --can can0 \
  --js /dev/input/js0 \
  --wrist-serial <D405_SERIAL> \
  --front-serial <D455_SERIAL> \
  --task "把物体放入盒子" \
  --output-dir data
```

脚本启动后不会立刻写入数据，进入待机状态后使用键盘控制采集：

```text
s + Enter    开始采集一条 episode
e + Enter    结束当前 episode
y + Enter    将刚结束的 episode 标记为 success
n + Enter    将刚结束的 episode 标记为 failure
q + Enter    退出整个采集会话
```

如果接入了相机，默认会打开实时预览窗口。预览窗口里也可以直接按 `s/e/y/n/q` 控制采集。

没有连接相机时，可以只测试手柄遥操作、机械臂状态和动作采集：

```bash
python scripts/collect_xbox_dataset.py \
  --can can0 \
  --js /dev/input/js0 \
  --no-cameras \
  --task "无相机遥操作测试" \
  --output-dir data
```

常用参数：

```bash
python scripts/collect_xbox_dataset.py --rate 15
python scripts/collect_xbox_dataset.py --duration 60
python scripts/collect_xbox_dataset.py --image-format png
python scripts/collect_xbox_dataset.py --success
python scripts/collect_xbox_dataset.py --failure
python scripts/collect_xbox_dataset.py --no-cameras
python scripts/collect_xbox_dataset.py --no-preview
python scripts/collect_xbox_dataset.py --preview-rate 10 --preview-scale 0.5
```

输出目录示例：

```text
data/
  success/
    2026-06-05/
      20260605_143012/
        trajectory.jsonl
        metadata.json
        images/
          wrist/
            000000.jpg
          front/
            000000.jpg
  failure/
    2026-06-05/
      ...
```

`trajectory.jsonl` 中每条样本包含：

- `observation`：7 维关节状态、关节速度、力矩、末端位姿和机械臂状态码
- `action`：最近一次 Xbox 遥操作发送给机械臂的 7 维动作，包含夹爪角度
- 夹爪角度范围：`0°~108.5°`，对应 `0~1.893682 rad`
- `camera_refs`：`wrist` / `front` 图像文件相对路径
- `camera_metadata`：相机时间戳、frame number、内参等信息
- `controller_info`：手柄 profile、是否有有效动作、是否请求退出

安装 RealSense 依赖：

```bash
pip install pyrealsense2
```

如果没有填写相机序列号，RealSense 枚举顺序可能不稳定，建议固定传入 `--wrist-serial` 和 `--front-serial`。

## 转换为 LeRobot 格式

采集脚本输出的是项目内部的 `droid_style_jsonl_v1` 中间格式。转换脚本可以导出 LeRobot `2.1` 风格目录、`3.0` 风格目录，或一次性导出两份。

默认同时导出 2.1 和 3.0：

```bash
python scripts/convert_to_lerobot.py \
  --input data/success/2026-06-06/20260606_143012 \
  --output lerobot_datasets/pick_place
```

批量转换 `data/` 下所有 success episode：

```bash
python scripts/convert_to_lerobot.py \
  --input data \
  --output lerobot_datasets/pick_place \
  --format both \
  --fps 15 \
  --overwrite
```

只导出 2.1：

```bash
python scripts/convert_to_lerobot.py \
  --input data \
  --output lerobot_datasets/pick_place_v21 \
  --format 2.1
```

只导出 3.0：

```bash
python scripts/convert_to_lerobot.py \
  --input data \
  --output lerobot_datasets/pick_place_v30 \
  --format 3.0
```

2.1 输出目录示例：

```text
lerobot_datasets/pick_place/v2.1/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    stats.json
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.images.wrist/
        episode_000000.mp4
      observation.images.front/
        episode_000000.mp4
```

3.0 输出目录示例：

```text
lerobot_datasets/pick_place/v3.0/
  meta/
    info.json
    tasks.jsonl
    episodes/
      chunk-000/
        file-000.parquet
    episodes_stats/
      chunk-000/
        file-000.parquet
    stats.json
  data/
    chunk-000/
      file-000.parquet
  videos/
    observation.images.wrist/
      chunk-000/
        file-000.mp4
    observation.images.front/
      chunk-000/
        file-000.mp4
```

默认只转换 success episode。如果需要把 failure 也一起导出：

```bash
python scripts/convert_to_lerobot.py \
  --input data \
  --output lerobot_datasets/all_episodes \
  --include-failure
```

## 低维轨迹记录

记录低维状态和动作：

```bash
python scripts/record_trajectory.py \
  --can can0 \
  --duration 10 \
  --rate 20 \
  --output recordings/test.jsonl
```

如果只是想先记录当前关节状态，并把当前状态也作为动作保存：

```bash
python scripts/record_trajectory.py \
  --can can0 \
  --duration 10 \
  --rate 20 \
  --action-from-state \
  --output recordings/test.jsonl
```

JSONL 中每条样本包含：

- `observation`：机械臂状态、关节位置、速度、力矩、末端位姿、状态码
- `action`：最近一次发送给机械臂的动作
- `camera_refs`：后续接入图像、视频或外部相机文件时使用
- `metadata`：任务、场景、标注等扩展信息

## 回放动作

回放 JSONL 中记录的动作：

```bash
python scripts/run_actions.py --can can0 --input recordings/test.jsonl --rate 20
```

也可以回放一个 JSON 数组，例如：

```json
[
  [0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.2],
  [0.0, 0.6, -0.4, 0.0, 0.0, 0.0, 0.4]
]
```

默认会对模型动作做限幅，降低策略输出异常时的风险。如果确认输入已经安全，可以使用：

```bash
python scripts/run_actions.py --can can0 --input actions.json --rate 20 --no-clip
```

## 项目结构

```text
scripts/xbox_control.py
  Xbox 手柄遥操作启动脚本

scripts/collect_xbox_dataset.py
  参考 DROID 流程的数据采集启动脚本，支持 Xbox 遥操作和双 RealSense 相机

scripts/convert_to_lerobot.py
  将 droid_style_jsonl_v1 episode 转换为 LeRobot 2.1/3.0 风格数据集

scripts/record_trajectory.py
  轨迹和状态记录启动脚本

scripts/run_actions.py
  记录动作或模型动作回放启动脚本

src/lingzu_teleop
  项目唯一核心包目录，包含平台层、SDK、相机、记录、回放和 LeRobot 适配

lingzu_teleop.robot
  机械臂应用层封装，内部使用本项目自带的 lingzu_teleop.sdk.ELA3Interface

lingzu_teleop.teleop.xbox
  Xbox 笛卡尔遥操作控制器

lingzu_teleop.recording.trajectory
  独立于 GUI 的轨迹记录器，支持 observation/action/camera_refs/metadata

lingzu_teleop.recording.action_runner
  执行记录动作或模型策略输出

lingzu_teleop.recording.droid_style
  DROID 风格 episode writer，负责 success/failure 目录、图像和 trajectory.jsonl

lingzu_teleop.camera
  相机接入抽象，以及 RealSenseProvider / MultiCameraProvider

lingzu_teleop.lerobot_adapter
  将本地 observation/action/images 映射为 LeRobot 风格数据帧

lingzu_teleop.lerobot_convert
  将本项目采集的 episode 转换为 LeRobot parquet/video/meta 目录

lingzu_teleop.lerobot_robot_lingzu_ela3
  面向后续 LeRobot 自定义 Robot 的适配包骨架

lingzu_teleop.lerobot_teleoperator_lingzu_xbox
  面向后续 LeRobot 自定义 Teleoperator 的适配包骨架

lingzu_teleop.sdk
  内置机械臂 SDK，包含 CAN/SLCAN 驱动、运动学、轨迹规划、RealSense 工具等

lingzu_teleop.sdk/resources
  内置 URDF、mesh、惯量参数、相机标定配置等资源
```

## SDK 核心能力保留方式

本项目不会重新实现机械臂控制核心，而是内置并复用 `lingzu_teleop.sdk`。平台层通过 `ELA3Robot.arm` 暴露 SDK 实例。

已封装的常用能力：

- `ELA3Robot.connect()` / `disconnect()`
- `ELA3Robot.get_observation()`
- `ELA3Robot.send_joint_action()`
- `ELA3Robot.move_joints()`，内部调用 SDK `MoveJ`
- `ELA3Robot.set_gripper()`
- `ELA3Robot.zero_torque()`
- `ELA3Robot.emergency_stop()`
- `ELA3Robot.kinematics()`

仍可直接调用 SDK 的完整接口：

```python
robot.arm.MoveJ(...)
robot.arm.MoveL(...)
robot.arm.EndPoseCtrl(...)
robot.arm.GetArmEndPoseMsgs()
robot.arm.GetJacobian()
robot.arm.GetMassMatrix()
robot.kinematics().forward_kinematics(...)
robot.kinematics().inverse_kinematics(...)
```

## LeRobot 数据方向

建议第一版学习数据契约保持简单：

- `observation.state`：7 维关节状态，包含夹爪
- `action`：7 维关节目标，包含夹爪
- `observation.images.<camera_name>`：相机 RGB 图像

Xbox 遥操作可以继续使用笛卡尔空间控制，但写入训练数据时，建议保存最终发送给机械臂的关节目标。这样更方便做回放、安全限幅和模型部署。

## 验证

当前已通过：

```bash
python -m compileall src tests
```

并手动验证了：

- 轨迹 JSONL 保存与加载
- JSONL 动作读取
- LeRobot 风格 frame 映射
- LeRobot features 推断
- `scripts/*.py --help`

安装完成后可以运行测试：

```bash
pytest -q
```
