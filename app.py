from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

app = Flask(__name__)
MODEL_PATH = "mnist_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 입력 이미지를 MNIST 스타일로 보정 (중앙정렬, 패딩, 외곽선 추출)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.point(lambda x: 0 if x < 30 else 255, 'L')
    np_img = np.array(image)
    coords = np.column_stack(np.where(np_img > 0))
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        image = image.crop((x0, y0, x1+1, y1+1))
    max_side = max(image.size)
    new_img = Image.new('L', (max_side, max_side), 0)
    new_img.paste(image, ((max_side - image.size[0]) // 2, (max_side - image.size[1]) // 2))
    image = new_img.resize((28, 28), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.reshape(1, 1, 28, 28)
    return arr

# PyTorch CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(64)  # 입력 크기 자동 추론
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model():
    model = SimpleCNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    print("Downloading MNIST & training PyTorch CNN model...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(5):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    return model

model = get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': '이미지 데이터가 없습니다'}), 400
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        arr = preprocess_image(image)
        arr = torch.tensor(arr, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            output = model(arr)
            pred = int(output.argmax(dim=1).cpu().numpy()[0])
        return jsonify({'result': f'이 숫자는 "{pred}"(와/과) 가장 유사합니다!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 