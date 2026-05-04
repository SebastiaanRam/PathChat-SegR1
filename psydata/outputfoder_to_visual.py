import os
import json
import cv2
import numpy as np
from PIL import Image
#############################################################################################
#############################################################################################
# 刚切好的1024×1024图片，读取配对JSON文件中的多边形坐标并直接保存标注结果图，用于gpt文本生成
#############################################################################################
def visualize_points(patch_path, mask_path, points_json_path, root_dir):
    """
    可视化JSON文件中的多边形坐标并直接保存标注结果图
    参数：
        patch_path: 1024×1024图片路径（例如，patch_0001.png）
        mask_path: 保留参数位（实际未使用）
        points_json_path: 对应JSON路径（例如，patch_0001_points.json）
        output_dir: 可视化结果保存目录
    """
    # 创建输出目录
    visual_dir = os.path.join(root_dir, "visual")  # 构造新的目录路径
    os.makedirs(visual_dir, exist_ok=True)  # 创建目录
    
    # 读取原始图片
    patch_img = np.array(Image.open(patch_path).convert("RGB"))
    
    # 读取标注点数据
    with open(points_json_path, "r") as f:
        points_dict = json.load(f)
    
    # 创建标注覆盖层
    overlay_img = patch_img.copy()
    
    # 绘制多边形标注
    for ann_name, points in points_dict.items():
        # 坐标格式转换
        points_int = [(int(round(x)), int(round(y))) for x, y in points]
        if len(points_int) < 3:
            print(f"警告：{ann_name} 点数量不足，已跳过")
            continue
        
        # 绘制多边形边界（绿色）
        points_array = np.array(points_int, np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay_img, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 绘制特征点（红色实心圆）
        for x, y in points_int:
            cv2.circle(overlay_img, (x, y), 5, (255, 0, 0), -1)
    
    # 直接保存结果图像
    output_filename = f"vis_{os.path.basename(patch_path)}"
    output_path = os.path.join(visual_dir, output_filename)
    Image.fromarray(overlay_img).save(output_path)
    print(f"标注结果已保存至：{output_path}")

def output_to_visual(input_dir, root_dir):
    """
    批量处理目录中的图像文件
    参数：
        input_dir: 包含patch_XXXX.png和patch_XXXX_points.json的目录
        output_dir: 结果保存目录
    """
    for file in os.listdir(input_dir):
        if file.endswith(".png") and not file.endswith("_mask.png"):
            base_name = file
            patch_path = os.path.join(input_dir, base_name)
            json_path = os.path.join(input_dir, base_name.replace(".png", "_points.json"))
            
            if os.path.exists(json_path):
                print(f"正在处理：{base_name}")
                visualize_points(patch_path, None, json_path, root_dir)
            else:
                print(f"缺失JSON文件：{base_name}")

if __name__ == "__main__":
    
    root_dir = ""
    output_dir = os.path.join(root_dir, "output")  # 输入目录
    visual_dir = os.path.join(root_dir, "visual")
    output_to_visual(output_dir, root_dir)