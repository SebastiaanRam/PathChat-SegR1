import os

# 遍历患者编号,根据实际情况修改
for i in range(3, 112):
    patient_number = f"{i:03d}"  # 格式化为3位数字，例如003, 004等
    wsi_path = f""
    xml_path = f""
    patient = f"{patient_number}_level6"
    root_dir = f""
    os.makedirs(root_dir, exist_ok=True)
    # 执行preprocess.py
    from preprocess import preprocess_cancer_regions_sliding
    preprocess_cancer_regions_sliding(wsi_path, xml_path, root_dir, patient)

    # 执行fliter_output.py
    output_dir = os.path.join(root_dir, "output")  
    from fliter_output import  filter_output_folder
    filter_output_folder(output_dir, white_ratio_threshold=0.05)

    # 执行outputfoder_to_visual.py
    from outputfoder_to_visual import output_to_visual
    output_to_visual(output_dir, root_dir)

    # 执行outputfolder_to_formatjson.py
    from outputfolder_to_formatjson import convert_points_to_lisa_format
    convert_points_to_lisa_format(output_dir)

    # 执行visual_to_gptjson.py
    from visual_to_gptjson import visual_to_gptjson_language
    visual_to_gptjson_language(root_dir, patient)

    # 执行creat_train_folder.py
    convert_dir = os.path.join(output_dir, "converted")
    train_dir = os.path.join(root_dir, "train")
    from creat_train_folder import create_train
    create_train(convert_dir, output_dir, train_dir)

    # 执行generate_final.py
    final_dir = os.path.join(root_dir, "final")
    final_json_path = os.path.join(final_dir, "train.json")
    from generate_final import process_image_json_directory
    pair_count = process_image_json_directory(train_dir, final_json_path)
    print(f"成功生成 {final_json_path}，共找到 {pair_count} 对文件。")

    # 执行fill_final_json_with_gptjson.py
    gpt_path = os.path.join(root_dir, "language")
    gptjson_path = os.path.join(gpt_path, "output_explanations.json")
    from fill_final_json_with_gptjson import update_train_json
    message = update_train_json(final_json_path, gptjson_path)
    print(message)