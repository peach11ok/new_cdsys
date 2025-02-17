from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision
import sys
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# 配置文件夹和路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 模型配置
MODEL_CONFIGS = {
    'TFIFNet': {
        'base_path': "/mnt/zt/FIFFST",
        'weights_path': "/mnt/zt/FIFFST_server2/second_point/original/best.pth",
    },
    'SOSCDNet': {
        'base_path': "/mnt/zt/FIFFST_server2",
        'weights_path': "/mnt/zt/FIFFST_server2/second_point/guass_and_kullback_leibler1/best.pth",
    }
}

def get_model(model_name):
    """根据模型名称获取对应的模型实例"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    base_path = config['base_path']
    
    try:
        # 添加基础路径到系统路径
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        
        print(f"基础路径: {base_path}")
        print(f"系统路径: {sys.path}")
        
        # 导入模型
        from models.full_module import Encoder
        
        # 创建模型实例并加载权重
        model = Encoder().cuda()
        model.load_state_dict(torch.load(config['weights_path']))
        return model
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class CustomTestData(Dataset):
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path
        
    def __len__(self):
        return 1

    def __getitem__(self, index):
        img1 = Image.open(self.image1_path).convert('RGB')
        img2 = Image.open(self.image2_path).convert('RGB')

        im1, im2 = self.transform(img1, img2)
        return im1, im2

    def transform(self, img1, img2):
        transform_img = transforms.Compose([
            transforms.Resize((224, 224)),  # 确保图片尺寸正确
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        im1 = transform_img(img1)
        im2 = transform_img(img2)
        return im1, im2

def post_process(image_path):
    """后处理函数"""
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    return binary

def detect_changes(image1_path, image2_path, timestamp, model_name):
    print(f"开始变化检测过程，使用模型: {model_name}...")
    
    try:
        # 获取对应的模型
        model = get_model(model_name)
        model.eval()
        print(f"模型 {model_name} 加载成功")

        # 创建数据加载器
        test_dataset = CustomTestData(image1_path, image2_path)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        print(f"数据加载成功，图片1路径: {image1_path}, 图片2路径: {image2_path}")

        # 创建输出路径
        temp_output = os.path.join(OUTPUT_FOLDER, f'temp_{timestamp}.png')
        final_output = os.path.join(OUTPUT_FOLDER, f'detection_{timestamp}.png')

        # 模型预测
        with torch.no_grad():
            for data in test_loader:
                im1, im2 = data
                print(f"输入图片1的形状: {im1.shape}")
                print(f"输入图片2的形状: {im2.shape}")
                
                im1 = im1.cuda()
                im2 = im2.cuda()
                print("图片已成功转移到GPU")

                outputs = model(im1, im2)
                print(f"模型输出的形状: {outputs[0].shape}")
                outputs = outputs[0][0]
                result = outputs[0].unsqueeze(0)

                # 保存临时结果
                torchvision.utils.save_image(result, temp_output)
                print(f"临时结果已保存到: {temp_output}")

                # 后处理
                binary_result = post_process(temp_output)
                cv2.imwrite(final_output, binary_result)
                print(f"最终结果已保存到: {final_output}")
                break

        return f"/outputs/detection_{timestamp}.png"
    
    except Exception as e:
        print(f"模型处理错误: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': '请上传两张图片'}), 400
    
    image1 = request.files['image1']
    image2 = request.files['image2']
    model_name = request.form.get('model', 'TFIFNet')
    
    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    timestamp = str(int(time.time()))
    result = {}
    
    try:
        # 保存上传的图片，添加 t1/t2 标识
        image_pairs = [
            (image1, 'image1', 't1'),
            (image2, 'image2', 't2')
        ]
        
        for image, key, time_id in image_pairs:
            if image and allowed_file(image.filename):
                # 使用时间标识构建新的文件名
                filename = secure_filename(image.filename)
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_{time_id}{ext}"
                filepath = os.path.join(UPLOAD_FOLDER, new_filename)
                image.save(filepath)
                result[key] = f"/uploads/{new_filename}"
                print(f"图片{key}已保存，路径: {filepath}")
        
        # 执行变化检测
        detection_result = detect_changes(
            os.path.join(UPLOAD_FOLDER, secure_filename(f"{os.path.splitext(image1.filename)[0]}_t1{os.path.splitext(image1.filename)[1]}")),
            os.path.join(UPLOAD_FOLDER, secure_filename(f"{os.path.splitext(image2.filename)[0]}_t2{os.path.splitext(image2.filename)[1]}")),
            timestamp,
            model_name
        )
        
        if detection_result:
            result['detection_result'] = detection_result
        else:
            return jsonify({'error': '检测失败'}), 500
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
