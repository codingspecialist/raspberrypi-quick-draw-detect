# 🖌️ 숫자 그림판 - PyTorch AI 숫자 인식

PyTorch + Flask로 동작하는 초간단 웹 숫자 인식기입니다.

## ✨ 주요 기능
- 마우스로 0~9 숫자를 그리면 AI가 어떤 숫자인지 예측해줍니다
- 파일 최소화: app.py(서버/모델/예측), templates/index.html(그림판 UI)만 필요
- 최초 실행 시 MNIST 데이터 자동 다운로드 및 PyTorch CNN 모델 학습
- 입력 이미지는 중앙정렬/패딩/이진화 등으로 MNIST 스타일로 보정

## 🚀 사용법
1. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
2. 서버 실행
   ```bash
   python app.py
   ```
3. 브라우저에서 접속
   ```
   http://localhost:5000
   ```
4. 0~9 숫자를 그리고 "분석하기" 클릭!

## 📁 파일 구조
```
app.py                # 서버, 모델, 예측 모두 포함
templates/
  └── index.html      # 웹 그림판 UI
requirements.txt      # 필요한 패키지 목록
mnist_cnn.pt          # 학습된 PyTorch CNN 모델(최초 실행 시 생성, 이후 캐싱)
data/
  └── mnist/
      └── raw/        # MNIST 원본 데이터셋 캐시(최초 실행 시 자동 다운로드)
```

- **mnist_cnn.pt**: 학습된 모델의 가중치가 저장된 파일로, 재실행 시 빠른 예측을 위해 사용됩니다.
- **data/mnist/raw/**: torchvision에서 MNIST 데이터셋을 다운로드할 때 임시로 저장하는 폴더입니다. 서비스/배포에는 필요 없지만, 최초 실행 시 자동 생성됩니다.

## ⚠️ 참고
- 최초 실행 시 MNIST 데이터셋 다운로드 및 PyTorch CNN 모델 학습(수 분 소요)
- 이후에는 빠르게 동작
- 숫자(0~9)만 인식 가능
- 불필요한 파일/폴더는 모두 삭제됨

---
MIT License

실행 중 오류가 발생했습니다:

```
ImportError: `fetch_openml` requires pandas.
```

### 원인
- scikit-learn의 `fetch_openml` 함수는 내부적으로 pandas가 필요합니다.
- requirements.txt에 pandas가 누락되어 있습니다.

---

## 해결 방법

1. **requirements.txt에 pandas 추가**
2. 아래 명령어로 설치
   ```bash
   pip install -r requirements.txt
   ```

---

바로 requirements.txt에 pandas를 추가하고, 실행까지 자동화해드릴까요? 