# demo_fixed_cube.py
# Isaac Lab 2.2.0  /  Isaac Sim 5.0.0
from isaaclab.app import AppLauncher


# 1) 启动 Isaac Sim 实例（窗口/无头均可）
sim_app = AppLauncher({"headless": False})

from isaaclab.sim import utils as sim_utils
from isaaclab.sim.spawners import shapes
from isaaclab.sim.schemas import RigidBodyPropertiesCfg

from isaaclab.sim.spawners.shapes import  CuboidCfg
# 2) 创建 /World 根节点（spawn_xform 会自动加 /World）
# sim_utils.create_world_prim()

# 3) 在 (0.0, 0.0, 1.0 m) 位置生成一个 0.2 m 立方体
prim_path = "/World/fixed_cube"
cube_size = (0.20, 0.20, 0.20)  # XYZ 尺寸 (米)
translation = (0.0, 0.0, 1.0)  # 悬空 1 米
orientation = (0.0, 0.0, 0.0, 1.0)  # 四元数 (w 最后)

# 3-A) 先用 spawn_cuboid 造一个 USDGeomCube
cube_prim = shapes.spawn_cuboid(
    prim_path=prim_path,
    cfg=CuboidCfg(size=cube_size),
    translation=translation,
    orientation=orientation,
)

# 3-B) 给它加上「固定刚体」属性：kinematic + 关闭重力
rigid_cfg = RigidBodyPropertiesCfg(
    kinematic_enabled=True,  # 固定在世界坐标
    disable_gravity=True,  # 不受重力
)
# sim_utils.apply_rigid_body_properties(cube_prim, rigid_cfg)

# 4) 启动仿真主循环
while True:
    sim_app.update()  # 渲染 + 物理
sim_app.close()
