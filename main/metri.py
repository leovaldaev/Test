import os
import cv2
import numpy as np
from typing import List, Union, Dict
from ultralytics import YOLO
import torch_test
import matplotlib.pyplot as plt


# Загружаем модели YOLOv8
def load_YOLO_pt():
    name_model1 = "runs/detect/yolo11s.pt_100_1280_b_0.8_18_0_4080/weights/best.pt"
    name_model2 = "runs/detect/yolo11s.pt_2_1280_b_0.8_18_0_4080/weights/best.pt"
    model_pt = YOLO(name_model1)        # Модель .pt
    model_pt2 = YOLO(name_model2)        # Модель .pt

def export_pt_to_engine():
    #model_pt.export(format="engine", imgsz = "1280", half = True)  # creates 'yolo11n.engine'
    #model_pt2 = YOLO("best1.engine") # Экспортированная модель .engine
    pass

def iou(box1, box2) -> float:
    """Функция для вычисления пересечения (IoU) между двумя рамками."""
    x1, y1, w1, h1 = box1['xc'] - box1['w'] / 2, box1['yc'] - box1['h'] / 2, box1['w'], box1['h']
    x2, y2, w2, h2 = box2['xc'] - box2['w'] / 2, box2['yc'] - box2['h'] / 2, box2['w'], box2['h']
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def apply_nms(boxes: List[dict], iou_threshold: float = 0.1, score_threshold: float = 0.3) -> List[dict]:
    """Функция для применения NMS и удаления пересекающихся рамок с учетом порога вероятности."""
    # Фильтруем рамки по вероятности
    boxes = [box for box in boxes if box['score'] >= score_threshold]
    
    if not boxes:
        return []
    
    # Сортируем рамки по вероятности в порядке убывания
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    selected_boxes = []
    
    while boxes:
        best_box = boxes.pop(0)
        selected_boxes.append(best_box)
        boxes = [box for box in boxes if iou(best_box, box) < iou_threshold]
    
    return selected_boxes

def infer_image_bbox(model, image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении.

    Args:
        model: YOLO модель для инференса.
        image (np.ndarray): Изображение для инференса.

    Returns:
        List[dict]: Список словарей с координатами рамок и оценками.
    """
    res_list = []
    result = model.predict(source=image, imgsz=1280, device=0,rect=True)
    
    # Преобразуем результаты в numpy массивы
    for res in result:
        for box in res.boxes:
            xc = box.xywhn[0][0]
            yc = box.xywhn[0][1]
            w = box.xywhn[0][2]
            h = box.xywhn[0][3]
            conf = box.conf[0].item()

            formatted = {
                'xc': xc,
                'yc': yc,
                'w': w,
                'h': h,
                'label': 0,  # здесь можно добавить лейблы, если они доступны
                'score': conf
            }
            res_list.append(formatted)
    res_list = apply_nms(res_list)

    return res_list

def draw_bboxes(image: np.ndarray, bboxes: List[dict], color=(0, 255, 0)) -> np.ndarray:
    """Отображение рамок на изображении.

    Args:
        image (np.ndarray): Изображение.
        bboxes (List[dict]): Список координат рамок.
        color (tuple): Цвет рамок.

    Returns:
        np.ndarray: Изображение с нарисованными рамками.
    """
    h, w = image.shape[:2]
    for box in bboxes:
        xc, yc, bw, bh = box['xc'], box['yc'], box['w'], box['h']
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
        cv2.putText(image, f"{box['score']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def read_ground_truth_boxes(txt_path: str) -> List[dict]:
    """Чтение файла .txt с координатами реальных рамок.
    
    Args:
        txt_path (str): Путь к файлу с координатами рамок.

    Returns:
        List[dict]: Список рамок в виде словарей с координатами и размерами.
    """
    bboxes = []
    if not os.path.getsize(txt_path):  # Проверка, если файл пустой
        return bboxes
    
    with open(txt_path, "r") as file:
        for line in file:
            # Обработка строки, если формат соблюдён
            parts = line.strip().split()
            if len(parts) == 5:
                label, xc, yc, w, h = map(float, parts)
                bboxes.append({'label': int(label), 'xc': xc, 'yc': yc, 'w': w, 'h': h})
    
    return bboxes


def draw_ground_truth_centers(image: np.ndarray, bboxes: List[dict], color=(0, 255, 0)) -> np.ndarray:
    h, w = image.shape[:2]
    for box in bboxes:
        xc, yc = box['xc'], box['yc']
        center_x, center_y = int(xc * w), int(yc * h)
        cv2.circle(image, (center_x, center_y), 10, color, -1)
    return image

def calculate_metrics(pred_bboxes: List[dict], gt_bboxes: List[dict], iou_threshold: float = 0.93) -> Dict[str, int]:
    """Функция для вычисления TP, FP и FN.

    Args:
        pred_bboxes (List[dict]): Предсказанные рамки.
        gt_bboxes (List[dict]): Истинные рамки.
        iou_threshold (float): Порог IoU для корректного совпадения.

    Returns:
        Dict[str, int]: Словарь с TP, FP, FN.
    """
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    
    for pred_box in pred_bboxes:
        is_match = False
        for i, gt_box in enumerate(gt_bboxes):
            if i in matched_gt:
                continue  # Эта рамка уже была учтена как TP
            if iou(pred_box, gt_box) >= iou_threshold:
                is_match = True
                matched_gt.add(i)
                break
        if is_match:
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_bboxes) - len(matched_gt)
    
    return {"tp": tp, "fp": fp, "fn": fn}


def main():
    # Путь к папке с изображениями
    image_folder = "../public/images"
    label_folder = "../public/labels"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    
    total_tp, total_fp, total_fn = 0, 0, 0

    for image_file in image_files:
        # Загружаем изображение
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        label_path = os.path.join(label_folder, image_file.replace(".jpg", ".txt"))
        ground_truth_bboxes = read_ground_truth_boxes(label_path)
        
        # Инференс с использованием .pt модели
        bboxes_pt = infer_image_bbox(model_pt, image)


        image_with_bboxes_pt = draw_bboxes(image.copy(), bboxes_pt, color=(0, 255, 0))
        image_with_bboxes_pt = draw_ground_truth_centers(image_with_bboxes_pt, ground_truth_bboxes, color=(0, 0, 255))

        bboxes_pt2 = infer_image_bbox(model_pt2, image)
        image_with_bboxes_pt2 = draw_bboxes(image.copy(), bboxes_pt2, color=(0, 255, 0))
        image_with_bboxes_pt2 = draw_ground_truth_centers(image_with_bboxes_pt2, ground_truth_bboxes, color=(0, 0, 255))

        
        # if (len(bboxes_pt) != len(ground_truth_bboxes)) or (len(bboxes_pt2) != len(ground_truth_bboxes)):

        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(image_with_bboxes_pt, cv2.COLOR_BGR2RGB))
        # plt.title(f"{image_file} - {name_model1}(detected : {len(bboxes_pt)}/{len(ground_truth_bboxes)})")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(cv2.cvtColor(image_with_bboxes_pt2, cv2.COLOR_BGR2RGB))
        # plt.title(f"{image_file} - {name_model2} (detected : {len(bboxes_pt2)}/{len(ground_truth_bboxes)})")
        # plt.axis("off")

        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        # plt.show()

        metrics = calculate_metrics(bboxes_pt, ground_truth_bboxes)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        print(f"Файл: {image_file}")
        print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        print("-" * 50)
    
    # Итоговые метрики
    print("Итоговая статистика:")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")


if __name__ == "__main__":
    main()
