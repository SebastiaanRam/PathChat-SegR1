import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI
#############################################################################################
#############################################################################################
# 使用gpt40输入可视化的结果，和给定的问题，输出的是只有output和image的json文件
#############################################################################################
#############################################################################################
# 初始化 OpenAI 客户端
import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI

def visual_to_gptjson_language(root_dir: str, patient: str, model: str = "gpt-4o") -> None:
    """
    Process pathology images and analyze cancerous regions using OpenAI's API.
    
    Args:
        root_dir (str): Root directory containing visual and language subdirectories
        patient (str): Patient identifier
        model (str): OpenAI model to use (default: gpt-4o)
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key="sk-eonCogPD68iCkrUd681f9d6a137e4cE5B351517948F52c02",
        base_url="https://api.mjdjourney.cn/v1"
    )
    
    # Define prompt
    PROMPT = (
        "Analyze the region marked by the red boundary in the image and explain the pathological differences between the cancerous and normal areas. "
        "Focus on identifying key morphological features that distinguish the marked region from the surrounding normal tissue. "
        "Key indicators may include nuclear pleomorphism, high nucleus-to-cytoplasm ratio, hyperchromatic nuclei, and irregular nuclear membranes. "
        "Based on your observation, explain why this region may be consistent with malignant transformation or tumor presence, compared to the adjacent normal tissue. "
        "Additionally, describe the spatial location of the cancerous region within the image (e.g., upper left, central, lower-right, diffused, or scattered). "
        "If the cancerous area is not confined to one region, explain its distribution across the image. "
        "Respond concisely in plain text only, without using Markdown formatting. Avoid redundant phrases—be direct and focused on the key features."
    )
    
    # Set paths
    visual_path = os.path.join(root_dir, "visual")
    gpt_path = os.path.join(root_dir, "language")
    os.makedirs(gpt_path, exist_ok=True)
    gptjson_path = os.path.join(gpt_path, "output_explanations.json")
    
    # Initialize results list
    results = []
    
    # Process images
    for fname in tqdm(sorted(os.listdir(visual_path))):
        if not fname.startswith("vis_") or not fname.endswith(".png"):
            continue
            
        image_path = os.path.join(visual_path, fname)
        print(f"Processing: {image_path}")
        
        # Analyze image
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{os.path.splitext(image_path)[1][1:]};base64,{image_base64}"
                        }
                    }
                ]
            }]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000
            )
            
            if response and response.choices[0].message.content:
                cleaned_name = fname.replace("vis_", "")
                results.append({
                    "outputs": response.choices[0].message.content,
                    "image": cleaned_name
                })
                
        except Exception as e:
            print(f"[ERROR] {image_path}: {str(e)}")
            continue
    
    # Save results
    with open(gptjson_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完成，共处理 {len(results)} 张图像。结果已保存至 {gptjson_path}")

if __name__ == "__main__":
    root_dir = ""
    visual_path = os.path.join(root_dir, "visual")
    gpt_path = os.path.join(root_dir, "language")
    gptjson_path = os.path.join(gpt_path, "output_explanations.json")
    patient = "002_level6"
    visual_to_gptjson_language(root_dir, patient)
