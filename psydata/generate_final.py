import os
import json
import random
#############################################################################################
#############################################################################################
# 读取train底下的patch_xxxx.png和json文件，生成explanatroy底下的文件，但是没有Outputs字段
#############################################################################################
#############################################################################################
def get_base_name(filename):
    """Extract base name from filename without extension."""
    return filename.rsplit(".", 1)[0]

def get_paired_files(train_dir):
    """Get paired PNG and JSON files from directory."""
    # Get all files
    files = os.listdir(train_dir)
    
    # Filter PNG and JSON files
    png_files = [f for f in files if f.endswith(".png")]
    json_files = [f for f in files if f.endswith(".json")]
    
    # Convert to sets for matching
    png_set = set(png_files)
    json_set = set(json_files)
    
    return png_set, json_set

def process_paired_data(train_dir, png_set, json_set):
    """Process paired files and create data structure."""
    paired_data = []
    
    for png_file in png_set:
        base_name = get_base_name(png_file)
        corresponding_json = base_name + ".json"
        
        if corresponding_json in json_set:
            # Read JSON file
            json_path = os.path.join(train_dir, corresponding_json)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    text_list = data.get("text", [])
                    query = random.choice(text_list) if isinstance(text_list, list) and text_list else "No query available"
            except json.JSONDecodeError:
                query = "Invalid JSON format"
            
            paired_data.append({
                "query": query,
                "outputs": "",
                "image": png_file,
                "json": corresponding_json
            })
    
    return paired_data

def save_paired_data(paired_data, output_json_path):
    """Save paired data to JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Write to JSON file
    with open(output_json_path, "w") as f:
        json.dump(paired_data, f, indent=2)
    
    return len(paired_data)

def process_image_json_directory(train_dir, output_json_path):
    """Main function to process image and JSON files."""
    # Get paired files
    png_set, json_set = get_paired_files(train_dir)
    
    # Process paired data
    paired_data = process_paired_data(train_dir, png_set, json_set)
    
    # Save results
    pair_count = save_paired_data(paired_data, output_json_path)
    
    return pair_count

if __name__ == "__main__":
    root_dir= ""
    output_dir = ""
    convert_dir = os.path.join(output_dir, "converted")
    train_dir = os.path.join(root_dir, "train")
    final_dir = os.path.join(root_dir, "final")
    final_json_path = os.path.join(final_dir, "train.json")
    
    pair_count = process_image_json_directory(train_dir, final_json_path)
    print(f"成功生成 {final_json_path}，共找到 {pair_count} 对文件。")