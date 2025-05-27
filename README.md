# MFNR: Multi-embedding Fusion Network for Recommendation

본 레포지토리에서는 리뷰 기반 추천 시스템 모델인 MFNR을 구현하였습니다. 
MFNR은 사용자의 리뷰에서 추출한 텍스트 임베딩(BERT 및 RoBERTa 기반)을 기존 사용자/아이템 ID 임베딩과 융합하여 사용자-아이템 간의 복합 표현 학습을 수행하고, 평점 예측 정확도를 개선합니다.

---

## 파일 설명

- **main.ipynb**
  
  전체 실행 흐름을 포함한 Jupyter Notebook입니다.
  데이터 로딩 → 전처리 → 텍스트 임베딩(BERT & RoBERTa) 추출 → 모델 학습 및 평가 순서로 구성되어 있습니다.

- **Utils/utils.py**
  
  데이터 처리 및 리뷰 임베딩 추출을 위한 함수들이 정의되어 있습니다.

  GZIP JSONL 로딩

  사용자/아이템 기준 데이터 분할

  리뷰 텍스트 → BERT/RoBERTa 임베딩 추출 ([CLS] 토큰)

- **Utils/model.py**
  
  MFNR 모델 클래스 및 학습/검증/테스트 루프 함수 정의 파일입니다.

- **Utils/path.py**
  
  데이터 경로 설정 등 프로젝트 내 상수 경로 정의

- **Data/Subscription_Boxes.jsonl.gz**
  
  Amazon 제품 리뷰 기반의 추천용 데이터셋입니다.
  포함 컬럼: reviewerID, asin, overall, reviewText
  리뷰 기반 사용자/아이템 표현 학습에 활용됩니다.

---

## 모델 구조

- **입력**: 사용자 ID, 아이템 ID, 사용자 리뷰 임베딩(BERT + RoBERTa), 아이템 리뷰 임베딩(BERT + RoBERTa)
- **구성**:
  - 사용자/아이템 ID Embedding
  - 사용자/아이템 리뷰 → BERT, RoBERTa 임베딩 → MLP 통과
  - 모든 feature를 결합한 후, MLP 기반 회귀 예측 수행
- **출력**: 예측 평점 (실수값)

---

## 데이터

- 사용 데이터: Amazon Product Reviews  
- 파일 경로: `Data/Subscription_Boxes.jsonl.gz`
- 다운로드: [Amazon Review Dataset](https://amazon-reviews-2023.github.io/) 에서 `Subscription Boxes` 선택 가능

---

## 코드 실행 환경

- Python 3.8+
- TensorFlow 2.x
- PyTorch (transformers)
- transformers (Huggingface)
- 기타: pandas, numpy, sklearn, tqdm 등

---

## 실행 예시

```bash
# Jupyter 환경에서 실행 권장
# 또는 ipynb를 .py로 변환하여 실행

jupyter notebook main.ipynb
```


## Reference

- Lim, H., Li, Q., Yang, S., & Kim, J. (2025). A BERT‐Based Multi‐Embedding Fusion Method Using Review Text for Recommendation. Expert Systems
