import json
import os
#############################################################################################
#############################################################################################
# 读取两个JSON文件，填充outputs字段
# 第一个文件是train.json，第二个文件是output_explanations.json
# 第一个文件的outputs字段需要填充，第二个文件的outputs字段是需要填充到第一个文件中的内容
# 需要将第二个文件中的outputs字段的markdown格式去掉（去掉**）
# 例如：将**a**替换为a
# 需要将第二个文件中的outputs字段的markdown格式去掉（去掉*）
# 例如：将*a*替换为a
#############################################################################################
#############################################################################################
def load_json_file(file_path):
    """加载JSON文件并返回数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_output_map(output_data):
    """创建从图像名到outputs的映射字典"""
    return {item['image']: item['outputs'] for item in output_data}

def update_train_data(train_data, output_map):
    """更新train数据中的outputs字段"""
    for item in train_data:
        image_name = item['image']
        if image_name in output_map:
            # 获取outputs并移除markdown格式（去掉**和*）
            outputs = output_map[image_name]
            raw_cleaned_outputs = outputs.replace('**', '')
            cleaned_outputs = raw_cleaned_outputs.replace('*', '')
            item['outputs'] = cleaned_outputs
    return train_data

def save_json_file(data, output_path):
    """保存数据到JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_train_json(final_json_path, gptjson_path):
    """主函数：更新train.json中的outputs字段"""
    # 加载JSON文件
    train_data = load_json_file(final_json_path)
    output_data = load_json_file(gptjson_path)
    
    # 创建输出映射
    output_map = create_output_map(output_data)
    
    # 更新train数据
    train_data = update_train_data(train_data, output_map)
    
    # 保存更新后的train.json
    save_json_file(train_data, final_json_path)
    
    return "Outputs have been successfully matched and updated in the train.json file."

if __name__ == "__main__":
    root_dir= ""
    output_dir = ""
    convert_dir = os.path.join(output_dir, "converted")
    train_dir = os.path.join(root_dir, "train")
    final_dir = os.path.join(root_dir, "final")
    final_json_path = os.path.join(final_dir, "train.json")
    visual_path = os.path.join(root_dir, "visual")
    gpt_path = os.path.join(root_dir, "language")
    gptjson_path = os.path.join(gpt_path, "output_explanations.json")
    
    message = update_train_json(final_json_path, gptjson_path)
    print(message)