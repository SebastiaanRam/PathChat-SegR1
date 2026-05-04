import os
import shutil
#############################################################################################
#############################################################################################
# 将 处理好的完整的json 文件（converted）和对应的图片文件复制到目标文件夹(train)下
#############################################################################################
#############################################################################################
def ensure_train_dir(train_dir):
    """确保目标文件夹存在"""
    os.makedirs(train_dir, exist_ok=True)

def copy_paired_files(convert_dir, output_dir, train_dir):
    """复制配对的JSON和PNG文件到目标文件夹"""
    ensure_train_dir(train_dir)
    copied_count = 0
    skipped_count = 0

    # 遍历json文件夹中的所有.json文件
    for filename in os.listdir(convert_dir):
        if filename.endswith(".json"):
            # 构建完整json文件路径
            json_path = os.path.join(convert_dir, filename)
            
            # 获取不带扩展名的前缀
            prefix = os.path.splitext(filename)[0]
            
            # 构建对应的图片文件名和路径
            image_name = prefix + ".png"
            image_path = os.path.join(output_dir, image_name)
            
            # 检查对应的图片是否存在
            if os.path.exists(image_path):
                # 定义目标路径下的目标文件路径
                target_json_path = os.path.join(train_dir, filename)
                target_image_path = os.path.join(train_dir, image_name)
                
                # 复制文件
                shutil.copy(json_path, target_json_path)
                shutil.copy(image_path, target_image_path)
                
                print(f"✅ 已复制: {filename} 和 {image_name}")
                copied_count += 1
            else:
                print(f"❌ 未找到对应图片: {image_name}，跳过该文件对。")
                skipped_count += 1
    
    return copied_count, skipped_count

def create_train(convert_dir, output_dir, train_dir):
    """主函数：处理JSON和PNG文件配对复制"""
    copied_count, skipped_count = copy_paired_files(convert_dir, output_dir, train_dir)
    print(f"🎉 所有文件复制完成。共复制 {copied_count} 对文件，跳过 {skipped_count} 对文件。")
    return copied_count, skipped_count

if __name__ == "__main__":
    root_dir= ""
    output_dir = ""
    convert_dir = os.path.join(output_dir, "converted")
    train_dir = os.path.join(root_dir, "train")
    
    create_train(convert_dir, output_dir, train_dir)