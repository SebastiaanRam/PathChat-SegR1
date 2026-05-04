import os
import numpy as np
from PIL import Image
#############################################################################################
#############################################################################################
# 计算mask文件中白色区域（值=255）的占比
# 筛选patch文件，删除白色区域占比低于阈值的patch、mask和points文件
# 输入的是刚切完的哪个文件夹
#############################################################################################
#############################################################################################
def calculate_white_ratio(mask_path):
    """
    计算mask文件中白色区域（值=255）的占比。
    参数：
        mask_path: mask文件路径（例如，patch_XXXX_mask.png）
    返回：
        float: 白色区域占比（0到1）
    """
    try:
        # 读取mask（灰度图）
        mask = np.array(Image.open(mask_path).convert("L"))
        # 计算白色像素（255）的数量
        white_pixels = np.sum(mask == 255)
        # 总像素数
        total_pixels = mask.size  # 1024×1024 = 1048576
        # 计算占比
        white_ratio = white_pixels / total_pixels
        return white_ratio
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return 0.0

def filter_output_folder(input_dir, white_ratio_threshold=0.1):
    """
    筛选patch文件，删除白色区域占比低于阈值的patch、mask和points文件。
    参数：
        input_dir: 包含patch_XXXX.png, patch_XXXX_mask.png, patch_XXXX_points.json的目录
        white_ratio_threshold: 白色区域占比阈值（默认0.3，即30%）
    """
    # 统计信息
    deleted_count = 0
    kept_count = 0
    
    # 遍历目录
    for file in os.listdir(input_dir):
        if file.endswith("_mask.png"):
            mask_path = os.path.join(input_dir, file)
            patch_name = file.replace("_mask.png", "")
            patch_path = os.path.join(input_dir, f"{patch_name}.png")
            points_path = os.path.join(input_dir, f"{patch_name}_points.json")
            
            # 计算白色占比
            white_ratio = calculate_white_ratio(mask_path)
            print(f"{patch_name}: White ratio = {white_ratio:.4f}")
            
            # 检查是否低于阈值
            if white_ratio < white_ratio_threshold:
                # 删除文件
                try:
                    if os.path.exists(patch_path):
                        os.remove(patch_path)
                        print(f"Deleted {patch_path}")
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                        print(f"Deleted {mask_path}")
                    if os.path.exists(points_path):
                        os.remove(points_path)
                        print(f"Deleted {points_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting files for {patch_name}: {e}")
            else:
                print(f"Kept {patch_name}")
                kept_count += 1
    
    # 打印总结
    print(f"\nSummary:")
    print(f"Deleted {deleted_count} patches (white ratio < {white_ratio_threshold})")
    print(f"Kept {kept_count} patches (white ratio >= {white_ratio_threshold})")

if __name__ == "__main__":
    # 示例路径（需要替换）
    root_dir = ""
    output_dir = os.path.join(root_dir, "output")  # 构造新的目录路径
    # 执行筛选
    filter_output_folder(output_dir, white_ratio_threshold=0.05)