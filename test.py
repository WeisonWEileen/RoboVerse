import torch

# 载入 .pt 文件
data = torch.load("/home/panwei/RoboVerse/outputs/active_vision/2025_0928_024821/model_14300.pt", map_location="cuda")

# 打印文件里的顶层信息
print("文件类型:", type(data))

if isinstance(data, dict):
    print("顶层键:", data.keys())
    
    # 检查是否有 model_state_dict
    if 'model_state_dict' in data:
        print("\n=== 模型权重信息 ===")
        model_weights = data['model_state_dict']
        print("模型权重类型:", type(model_weights))
        print("模型层数量:", len(model_weights))
        
        # 显示前几个权重层的信息
        print("\n前5个权重层:")
        for i, (key, value) in enumerate(list(model_weights.items())[:5]):
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)} (非张量)")
        
        # 显示所有层的名称
        print(f"\n所有层名称 (共{len(model_weights)}层):")
        for key in model_weights.keys():
            print(f"  - {key}")
    
    # 检查其他信息
    if 'iter' in data:
        print(f"\n训练迭代次数: {data['iter']}")
    
    if 'infos' in data:
        print(f"\n其他信息: {data['infos']}")
        
else:
    print("内容是完整的模型对象:", data)
