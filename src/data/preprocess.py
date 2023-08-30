"""
데이터를 전처리하고 전처리된 데이터를 저장합니다.
작동방식:
    - raw에서 메타데이터 csv와 image/gt를 불러옵니다.
    - 데이터를 변환할 transform을 정의합니다.
    - 
"""

import albumentations as alb
from albumentations.pytorch import ToTensorV2

base_transform = alb.Compose(
    [   
        alb.Resize(224, 224),
        alb.Normalize(),
        ToTensorV2()
    ])