import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, GaussNoise,
    HorizontalFlip, RandomCrop, Rotate, ShiftScaleRotate
)

# 데이터 로딩 및 전처리 함수
def load_and_preprocess_data(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_path = os.path.join(image_dir, data['image_name'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    annotations = data['Annotation']
    boxes = []
    labels = []
    for ann in annotations:
        boxes.append(ann['data'])
        labels.append(ann['class_name'])
    
    return image, np.array(boxes), labels

# 박스 정규화 함수
def normalize_boxes(boxes, image_size):
    return boxes / np.array([image_size[1], image_size[0], image_size[1], image_size[0]])

# 데이터 증강 함수
def augment_data(image, boxes, labels):
    augmentations = Compose([
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        GaussNoise(p=0.3),
        HorizontalFlip(p=0.5),
        RandomCrop(height=image.shape[0], width=image.shape[1], p=0.3),
        Rotate(limit=10, p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    augmented = augmentations(image=image, bboxes=boxes, labels=labels)
    return augmented['image'], np.array(augmented['bboxes']), augmented['labels']

# YOLO 형식으로 레이블 변환
def convert_to_yolo_format(boxes, labels, image_size):
    yolo_labels = []
    for box, label in zip(boxes, labels):
        x_center = (box[0] + box[2]) / 2 / image_size[1]
        y_center = (box[1] + box[3]) / 2 / image_size[0]
        width = (box[2] - box[0]) / image_size[1]
        height = (box[3] - box[1]) / image_size[0]
        class_id = class_mapping[label]
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return yolo_labels

# 데이터셋 준비 함수
def prepare_dataset(json_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    data = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            image, boxes, labels = load_and_preprocess_data(json_path, image_dir)
            
            # 데이터 증강
            aug_image, aug_boxes, aug_labels = augment_data(image, boxes, labels)
            
            # YOLO 형식으로 변환
            yolo_labels = convert_to_yolo_format(aug_boxes, aug_labels, aug_image.shape[:2])
            
            # 이미지 저장
            image_filename = f"aug_{json_file[:-5]}.jpg"
            cv2.imwrite(os.path.join(output_dir, 'images', image_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            # 레이블 저장
            label_filename = f"aug_{json_file[:-5]}.txt"
            with open(os.path.join(output_dir, 'labels', label_filename), 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            data.append(image_filename)
    
    return data

# 클래스 매핑 정의
class_mapping = {
    'car': 0, 'bus': 1, 'truck': 2, 'special vehicle': 3,
    'motorcycle': 4, 'bicycle': 5, 'personal mobility': 6,
    'person': 7, 'Traffic_light': 8, 'Traffic_sign': 9
}

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 준비
    train_data = prepare_dataset('path/to/train/json', 'path/to/train/images', 'path/to/output/train')
    val_data = prepare_dataset('path/to/val/json', 'path/to/val/images', 'path/to/output/val')

    # 데이터 분할
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # YAML 설정 파일 생성
    yaml_content = f"""
    train: path/to/output/train/images
    val: path/to/output/val/images

    nc: {len(class_mapping)}
    names: {list(class_mapping.keys())}
    """

    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)

    # 모델 로드 및 학습
    model = YOLO('yolov5s.pt')  # 사전 학습된 YOLOv5s 모델 로드

    # 모델 학습
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov5_autonomous_driving'
    )

    # 모델 평가
    results = model.val()

    # 테스트 데이터에 대한 예측
    test_results = model('path/to/test/images')

    # 결과 저장
    test_results.save('path/to/save/results')