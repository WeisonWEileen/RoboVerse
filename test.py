from pxr import Usd

file_path = "/home/panwei/RoboVerse/roboverse_data/robots/T1/T1_test_1.usd"
stage = Usd.Stage.Open(file_path)

# 找到 T1
t1 = stage.GetPrimAtPath("/Root/T1")
if not t1.IsValid():
    raise RuntimeError("❌ 没找到 /Root/T1")

# 把 T1 移动到世界根节点
stage.DefinePrim("/T1", t1.GetTypeName())
stage.GetEditTarget()
stage.GetRootLayer().TransferContent(t1.GetPath(), "/T1")

# 设置 defaultPrim = /T1
prim = stage.GetPrimAtPath("/T1")
stage.SetDefaultPrim(prim)

# 删除 Root
root = stage.GetPrimAtPath("/Root")
if root.IsValid():
    stage.RemovePrim("/Root")

# 保存
stage.GetRootLayer().Save()
print("✅ 已把 /Root/T1 提升为 /T1，并删除 Root")
