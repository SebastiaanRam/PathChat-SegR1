import os
import json
#############################################################################################
#############################################################################################
# 读取最原始的json标注，转化成标准格式（还没有加text)
#############################################################################################
import os
import json

def convert_points_to_lisa_format(output_dir: str) -> None:
    """
    将指定目录下的 _points.json 文件转换为 LISA 格式的 JSON 文件。
    
    Args:
        output_dir (str): 包含 _points.json 文件的输出目录
    """
    # 创建转换后的输出目录
    convert_dir = os.path.join(output_dir, "converted")
    os.makedirs(convert_dir, exist_ok=True)
    
    # 遍历输出目录
    for fname in os.listdir(output_dir):
        if not fname.endswith("_points.json"):
            continue
            
        # 获取基本信息
        base = fname[:-12]  # 移除 "_points.json" 后缀
        json_path = os.path.join(output_dir, fname)
        img_name = base + ".png"
        
        try:
            # 读取原始坐标
            with open(json_path, 'r') as f:
                raw = json.load(f)
            
            # 构造 shapes 列表
            shapes = [
                {
                    "label": "target",
                    "labels": ["target"],
                    "shape_type": "polygon",
                    "image_name": img_name,
                    "points": pts,
                    "group_id": None,
                    "group_ids": [None],
                    "flags": {}
                }
                for region_id, pts in raw.items()
            ]
            
            # 构建 LISA 格式结构
            lisa_entry = {
                "text": [],
                "is_sentence": True,
                "shapes": shapes
            }
            
            # 写入新 JSON 文件
            out_path = os.path.join(convert_dir, base + ".json")
            with open(out_path, 'w') as f:
                json.dump(lisa_entry, f, indent=2, ensure_ascii=False)
            
            print(f"已转换 {fname} → {out_path}")
            
        except Exception as e:
            print(f"[错误] 处理 {fname} 失败: {str(e)}")
            continue
    
    print("全部转换完成！")

if __name__ == "__main__":
    output_dir = ""
    convert_dir = os.path.join(output_dir, "converted")
    convert_points_to_lisa_format(output_dir)