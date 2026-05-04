import os
import json
#############################################################################################
#############################################################################################
# 读取文件夹下的所有已经转换成标准格式的 .json 文件，添加指定句子到 text 字段
#############################################################################################



def add_sentences_to_json(folder_path: str) -> None:
    """
    向指定文件夹中所有 JSON 文件的 'text' 字段添加预定义的句子列表。
    
    Args:
        folder_path (str): 包含 JSON 文件的目标文件夹路径
    """
    # 定义要添加的句子列表
    sentences_to_add = [
        "Please indicate the regions in the image that likely correspond to tumor tissue.",
        "Could you mark the areas of the slide displaying cells with an elevated nucleus-to-cytoplasm ratio and hyperchromatic nuclei?",
        "Which regions show cells with nuclear pleomorphism and prominent nucleoli? Please segment these areas.",
        "Identify and segment the regions where cells exhibit nuclear atypia and irregular nuclear membranes.",
        "Please highlight the areas containing tumor cells with abnormally high nucleus-to-cytoplasm ratios and deep nuclear staining.",
        "Which sections display cells characterized by a high nucleus-to-cytoplasm ratio and hyperchromasia? Please delineate them."
    ]
    
    # 遍历文件夹下的 JSON 文件
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新 text 字段
            if isinstance(data.get("text"), list):
                data["text"].extend(sentences_to_add)
            else:
                data["text"] = sentences_to_add
            
            # 写回 JSON 文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"已更新文件: {filename}")
            
        except Exception as e:
            print(f"[错误] 处理 {filename} 失败: {str(e)}")
            continue
    
    print("✅ 所有文件已更新完毕。")

if __name__ == "__main__":
    output_dir = ""
    convert_dir = os.path.join(output_dir, "converted")
    add_sentences_to_json(convert_dir)