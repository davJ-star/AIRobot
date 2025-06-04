# 🚗 2D Semantic Segmentation - DeepLabv3+ 기반 자율주행 도로 영역 분할
> **DeepLabv3+ 기반 자율주행을 위한 실시간 도로 영역 의미적 분할 시스템**

본 프로젝트는 DeepLabv3+ 모델을 활용하여 자율주행 시스템의 도로 영역을 실시간으로 의미적으로 분할하는 핵심 모듈을 개발하고, 논문 구현 및 실험 결과를 체계적으로 정리한 보고서입니다[1].

---

## 📌 프로젝트 개요

- **개발 기간:** 2025.01 - 2025.06 (진행 중)
- **팀 구성:** 개인 프로젝트
- **개발 환경:** Python 3.12, PyTorch 2.4.0, CUDA 12.1

**주요 성과**
- 24 에포크 학습으로 안정적 수렴 달성
- Albumentations 기반 고급 데이터 증강 파이프라인 구축
- 자율주행 경진대회 참가를 위한 핵심 모듈 개발

---

## 🧐 문제 정의 및 배경

- **문제 배경:** 자율주행 시스템에서 정확한 도로 영역 인식은 안전한 주행을 위한 필수 기능입니다.
- **기존 한계:** 전통적인 컴퓨터 비전 기법은 날씨 변화 및 조명 조건 변화에 대한 강건성이 떨어집니다.
- **프로젝트 목표:** 25개 클래스에 대한 정확한 의미적 분할을 통한 실시간 도로 환경 이해

**데이터셋 정보**
- **출처:** 자율주행 경진대회 제공 데이터셋
- **규모:** Training 폴더 내 이미지/라벨 쌍, 25개 클래스(도로, 인도, 차량, 보행자 등)
- **특성:** RGB 이미지 및 JSON 형태의 폴리곤 어노테이션
- **학습 방식:** 지도학습 (Supervised Learning)

---

## 🛠️ 기술 스택 및 개발 환경

- **언어:** Python 3.12
- **IDE:** Jupyter Notebook
- **버전 관리:** Git, GitHub

**주요 라이브러리**

```python
# 핵심 프레임워크
torch==2.4.0
torchvision==0.19.0

# 데이터 처리
opencv-python==4.10.0.84
albumentations==1.4.14
numpy==1.26.4

# 시각화
matplotlib
```

- **하드웨어:** CUDA 지원 GPU 환경
- **모델 저장:** .pth 형태의 체크포인트 저장

---

## 🏗️ 모델 설계 및 구현

- **모델 타입:** DeepLabv3+ with MobileNetV3-Large Backbone
- **구조**
  - Input Layer: (3, 256, 256) RGB 이미지
  - Backbone: MobileNetV3-Large (사전 훈련됨)
  - Decoder: DeepLabv3+ 디코더
  - Output Layer: 25개 클래스 분할 마스크
- **총 파라미터 수:** MobileNetV3 기반 경량화

**모델 핵심 코드**

```python
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# 모델 정의
model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=25)

# 클래스 매핑
class_mapping = {
    'road': 1, 'sidewalk': 2, 'road roughness': 3, 
    'vehicle': 17, 'pedestrian': 16, 'sky': 23
    # ... 총 25개 클래스
}
```

**하이퍼파라미터**
- 학습률: 0.001
- 배치 크기: 16
- 에포크: 24 (Early Stopping 적용)
- 옵티마이저: Adam
- 손실 함수: CrossEntropyLoss

---

## 🧪 데이터 분석 및 전처리

**고급 데이터 증강 파이프라인**

```python
train_transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.OneOf([
        A.RandomFog(), A.RandomRain(), A.RandomShadow()
    ], p=0.3),
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 📈 실험 결과 및 성능 평가

- **Epoch 1:** Train Loss: 0.4663, Val Loss: 0.3685
- **Epoch 2:** Train Loss: 0.3349, Val Loss: 0.3555
- **최종 성과:** 24 에포크까지 안정적 수렴

**성능 향상 기법**
- 데이터 증강: 원본+증강 데이터로 데이터셋 2배 확장
- Early Stopping: 과적합 방지
- 모델 체크포인팅: 3 에포크마다 모델 저장

---

## ⚠️ 한계점 및 개선 방향

**현재 한계점**
- 검증 데이터 부족(훈련 데이터를 검증에도 사용)
- 클래스 불균형 가능성
- 실시간 추론 속도 최적화 필요

**향후 개선 방향**
- 데이터셋 분할(Train/Validation/Test)
- IoU, mIoU 등 세분화 평가 지표 도입
- 모델 경량화로 실시간 추론 최적화

---

## 📚 논문 분석 및 구현

- **논문 제목:** DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
- **저자:** Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
- **발행년도:** 2018 (ECCV)
- **연구 분야:** Computer Vision, Semantic Segmentation

**핵심 아이디어**
> DeepLabv3+는 atrous convolution을 활용한 인코더-디코더 구조로 정확하고 세밀한 객체 경계를 가진 의미적 분할을 수행하는 모델입니다.

**구현 특징**
- DeepLabv3+ 아키텍처 구현 (MobileNetV3 백본으로 경량화)
- 자율주행 도메인 25개 클래스에 맞춘 커스터마이징
- Adam optimizer, 0.001 learning rate, Cross-Entropy Loss, CUDA 환경

---

## 💡 개인적 분석 및 인사이트

**논문의 강점**
- Atrous convolution과 인코더-디코더 구조의 효과적 결합
- 다양한 데이터셋에서의 성능 검증
- 실시간 추론이 가능한 효율적 구조

**구현 도전과제**
- "Premature end of JPEG file" 오류 등 데이터 로딩 최적화 필요
- JSON 어노테이션을 픽셀 단위 마스크로 변환하는 복잡한 전처리
- 실시간 추론을 위한 추가 경량화

---

## 📝 학습 내용 및 성장 포인트

- **Albumentations**: 전문적인 컴퓨터 비전 데이터 증강 라이브러리 활용
- **DeepLabv3+**: SOTA 의미적 분할 모델 구현
- **PyTorch Segmentation**: torchvision.models.segmentation 모듈 활용

**프로젝트 회고**
- 잘한 점: 체계적인 데이터 증강 파이프라인, Early Stopping 통한 안정적 학습
- 아쉬운 점: 검증 데이터셋 분리 부족, 정량적 성능 평가 지표 부재
- 배운 교훈: 실제 데이터셋에서의 전처리 파이프라인 구축의 중요성

---

이 프로젝트는 자율주행 시스템의 핵심 컴포넌트인 의미적 분할을 DeepLabv3+ 모델로 구현하여 실무 경험을 쌓는 중요한 학습 과정이었습니다. 향후 실시간 추론 최적화와 정량적 성능 평가를 통해 더욱 완성도 높은 시스템으로 발전시킬 계획입니다[1].

---

> **문의 및 협업 제안은 언제든 환영합니다!**

---

[프로젝트 상세 코드 및 실험 노트북 참고](#)

---
