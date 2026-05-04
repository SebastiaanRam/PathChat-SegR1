import os
import json
import numpy as np
from PIL import Image
import openslide
from lxml import etree
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from skimage.draw import polygon
from scipy.interpolate import splprep, splev

def parse_asap_xml(xml_path, spline_points=100):
    """
    解析ASAP格式的XML文件，提取多边形和样条曲线的点坐标。
    返回：[{name: str, points: [(x1,y1), ...], polygon: Shapely Polygon}, ...]
    参数：
        xml_path: XML文件路径
        spline_points: 样条插值的点数（默认100）
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    annotations = []
    
    for annotation in root.xpath("//Annotation[@Type='Polygon' or @Type='Spline']"):
        name = annotation.get("Name")
        ann_type = annotation.get("Type")
        points = []
        
        # 提取坐标
        for coord in annotation.xpath(".//Coordinate"):
            x = float(coord.get("X"))
            y = float(coord.get("Y"))
            points.append((x, y))
        
        if len(points) < 3:
            print(f"Warning: Insufficient points for {name} ({ann_type}), skipping")
            continue
        
        try:
            if ann_type == "Spline":
                # 样条插值生成密集点集
                x, y = zip(*points)
                tck, u = splprep([x, y], s=0, per=True)  # 闭合样条
                u_new = np.linspace(0, 1, spline_points)
                x_new, y_new = splev(u_new, tck)
                points = list(zip(x_new, y_new))
            
            # 创建Shapely Polygon
            poly = Polygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)  # 修复无效多边形
                if not poly.is_valid:
                    print(f"Warning: Invalid polygon for {name} after repair, skipping")
                    continue
            
            annotations.append({"name": name, "points": points, "polygon": poly})
        except Exception as e:
            print(f"Error processing {name} ({ann_type}): {e}")
    
    return annotations

def get_intersecting_polygons(patch_bbox, annotations):
    """
    检查哪些多边形与1024×1024框有交集，返回裁剪后的多边形。
    参数：
        patch_bbox: 框的边界 (x_min, y_min, x_max, y_max)
        annotations: 所有多边形信息
    返回：[{name, clipped_polygon, clipped_points}, ...]
    """
    x_min, y_min, x_max, y_max = patch_bbox
    patch_poly = box(x_min, y_min, x_max, y_max)
    intersecting = []
    
    for ann in annotations:
        poly = ann["polygon"]
        if patch_poly.intersects(poly):
            try:
                clipped_poly = poly.intersection(patch_poly)  # 使用Shapely的intersection替代clip_by_rect
                if clipped_poly.is_empty:
                    continue
                
                # 提取裁剪后的点
                if clipped_poly.geom_type == "Polygon":
                    clipped_points = list(clipped_poly.exterior.coords)[:-1]
                elif clipped_poly.geom_type == "MultiPolygon":
                    clipped_points = []
                    for sub_poly in clipped_poly.geoms:
                        clipped_points.extend(list(sub_poly.exterior.coords)[:-1])
                else:
                    continue
                
                intersecting.append({
                    "name": ann["name"],
                    "clipped_polygon": clipped_poly,
                    "clipped_points": clipped_points
                })
            except Exception as e:
                print(f"Error clipping polygon {ann['name']}: {e}")
    
    return intersecting

def generate_patch(wsi, patch_bbox, intersecting_polygons, patch_size=1024):
    """
    生成1024×1024的图片、mask和图片级别坐标。
    参数：
        wsi: OpenSlide对象
        patch_bbox: 框的边界 (x_min, y_min, x_max, y_max)
        intersecting_polygons: 框内的裁剪多边形
        patch_size: 图片尺寸（默认1024）
    返回：
        image: PIL.Image
        mask: np.array
        points_png: {name: [(x1,y1), ...], ...}
    """
    x_min, y_min, x_max, y_max = patch_bbox
    
    # 提取1024×1024图片
    try:
        image = wsi.read_region((int(x_min), int(y_min)), 0, (patch_size, patch_size))
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error reading region at ({x_min}, {y_min}): {e}")
        return None, None, None
    
    # 创建mask
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    
    # 存储图片级别坐标
    points_png = {}
    
    # 绘制每个裁剪多边形
    for poly in intersecting_polygons:
        points = poly["clipped_points"]
        # 转换坐标到图片级别（0-1024）
        points_clipped = [
            (
                (x - x_min) / (x_max - x_min) * patch_size,
                (y - y_min) / (y_max - y_min) * patch_size
            )
            for x, y in points
        ]
        
        # 绘制mask
        x_coords = np.array([p[0] for p in points_clipped])
        y_coords = np.array([p[1] for p in points_clipped])
        x_coords = np.clip(x_coords, 0, patch_size - 1)
        y_coords = np.clip(y_coords, 0, patch_size - 1)
        try:
            rr, cc = polygon(y_coords, x_coords, mask.shape)
            mask[rr, cc] = 255
        except Exception as e:
            print(f"Error drawing polygon {poly['name']}: {e}")
        
        # 保存图片级别坐标
        points_png[poly["name"]] = points_clipped
    
    return image, mask, points_png

def preprocess_cancer_regions_sliding(wsi_path, xml_path, root_dir, patient, patch_size=1024, stride=256):
    """
    使用滑窗扫描WSI，生成包含癌变区域的1024×1024图片、mask和坐标。
    参数：
        wsi_path: WSI文件路径
        xml_path: XML标注文件路径
        output_dir: 输出目录
        patch_size: 图片尺寸（默认1024）
        stride: 滑窗步长（默认256，增加覆盖率）
    """
    # 创建输出目录
    output_dir = os.path.join(root_dir, "output")  # 构造新的目录路径
    os.makedirs(output_dir, exist_ok=True)  # 创建目录
    
    # 读取WSI
    try:
        wsi = openslide.OpenSlide(wsi_path)
        level0_width, level0_height = wsi.level_dimensions[0]
        print(f"Level 0 dimensions: {level0_width} x {level0_height}")
    except Exception as e:
        print(f"Error opening WSI: {e}")
        return
    
    # 解析XML
    annotations = parse_asap_xml(xml_path)
    if not annotations:
        print("No valid annotations found in XML")
        return
    print(f"Found {len(annotations)} cancer regions")
    
    # 验证坐标范围
    for ann in annotations:
        x_coords = [p[0] for p in ann["points"]]
        y_coords = [p[1] for p in ann["points"]]
        print(f"Annotation {ann['name']}: x_range=({min(x_coords)}, {max(x_coords)}), y_range=({min(y_coords)}, {max(y_coords)})")
        if max(x_coords) > level0_width or max(y_coords) > level0_height:
            print(f"Warning: Annotation {ann['name']} coordinates exceed WSI dimensions")
    
    # 滑窗扫描
    patch_count = 0
    for y in range(0, level0_height - patch_size + 1, stride):
        for x in range(0, level0_width - patch_size + 1, stride):
            patch_bbox = (x, y, x + patch_size, y + patch_size)
            
            # 检查框是否包含癌变区域
            intersecting_polygons = get_intersecting_polygons(patch_bbox, annotations)
            if intersecting_polygons:
                # 生成图片、mask和坐标
                image, mask, points_png = generate_patch(wsi, patch_bbox, intersecting_polygons, patch_size)
                if image is None:
                    continue
                
                # 保存输出
                patch_name = f"{patient}_patch_{patch_count:04d}"
                image.save(os.path.join(output_dir, f"{patch_name}.png"))
                Image.fromarray(mask).save(os.path.join(output_dir, f"{patch_name}_mask.png"))
                with open(os.path.join(output_dir, f"{patch_name}_points.json"), "w") as f:
                    json.dump(points_png, f, indent=4)
                
                print(f"Saved patch {patch_name} at ({x}, {y})")
                patch_count += 1
    
    print(f"Total patches generated: {patch_count}")

if __name__ == "__main__":
    # 示例文件路径（需要替换）
    wsi_path = ""
    xml_path = ""
    patient = "002_level6"
    root_dir = ""
    # 执行
    preprocess_cancer_regions_sliding(wsi_path, xml_path, root_dir, patient)