import React, { useState, useEffect } from 'react';
import ModelSelector from '../components/ModelSelector';
import ImageUploader from '../components/ImageUploader';
import ResultDisplay from '../components/ResultDisplay';
import { DetectionResult, DetectionRecord, StoredUser } from '../types';
import { Science, CloudUpload, Compare } from '@mui/icons-material';
import ProcessingModal from '../components/ProcessingModal';
import ResultModal from '../components/ResultModal';
import { useAuth } from '../contexts/AuthContext';

interface SelectedImages {
  image1: File | null;
  image2: File | null;
}

interface ImagePreviews {
  image1: string;
  image2: string;
}

interface DetectionTime {
  modelName: string;
  timestamp: string;
  duration: number;
}

const Detection: React.FC = () => {
  const { user } = useAuth();
  const [selectedImages, setSelectedImages] = useState<SelectedImages>({ image1: null, image2: null });
  const [imagePreviews, setImagePreviews] = useState<ImagePreviews>({ image1: '', image2: '' });
  const [imageFiles, setImageFiles] = useState<SelectedImages>({ image1: null, image2: null });
  const [selectedModels, setSelectedModels] = useState({
    detectionModel: 'TFIFNet',
    segmentationModel: 'DeepLabV3'
  });
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [detectionStartTime, setDetectionStartTime] = useState<number | null>(null);

  const handleImageUpload = (type: 'image1' | 'image2', file: File | null) => {
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    
    setImagePreviews(prev => ({
      ...prev,
      [type]: imageUrl
    }));

    setSelectedImages(prev => ({
      ...prev,
      [type]: file
    }));

    setImageFiles(prev => ({
      ...prev,
      [type]: file
    }));
  };

  const handleDetectionStart = async () => {
    if (!selectedImages?.image1 || !selectedImages?.image2 || !user) return;
    
    const startTime = Date.now();
    setDetectionStartTime(startTime);
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('image1', selectedImages.image1);
      formData.append('image2', selectedImages.image2);
      formData.append('model', selectedModels.detectionModel);
      
      console.log('开始上传图片:', {
        image1: selectedImages.image1.name,
        image2: selectedImages.image2.name,
        model: selectedModels.detectionModel
      });
      
      const uploadResponse = await fetch('http://10.16.39.70:8080/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.error || '图片上传失败');
      }

      const uploadResult = await uploadResponse.json();
      console.log('服务器返回结果:', uploadResult);

      // 构建检测结果，使用服务器返回的图片路径
      const detectionResult: DetectionResult = {
        changedAreas: ['检测到的变化区域'],
        confidence: 0.95,
        segmentationData: {
          image1: ['建筑', '道路'],
          image2: ['建筑', '绿地']
        },
        // 使用完整的服务器URL
        changeDetectionImage: `http://10.16.39.70:8080${uploadResult.detection_result}`,
        segmentationImages: {
          image1: uploadResult.image1 ? `http://10.16.39.70:8080${uploadResult.image1}` : '',
          image2: uploadResult.image2 ? `http://10.16.39.70:8080${uploadResult.image2}` : ''
        }
      };

      const endTime = Date.now();
      const duration = Number(((endTime - startTime) / 1000).toFixed(3));

      // 保存检测时间记录
      const timeRecord: DetectionTime = {
        modelName: selectedModels.detectionModel,
        timestamp: new Date().toISOString(),
        duration: duration
      };

      // 存储到 localStorage
      const existingTimes = JSON.parse(localStorage.getItem('detectionTimes') || '[]');
      existingTimes.push(timeRecord);
      localStorage.setItem('detectionTimes', JSON.stringify(existingTimes));

      setResult(detectionResult);
      setShowResult(true);

      // 保存检测记录
      const newRecord: DetectionRecord = {
        id: Math.random().toString(),
        userId: user.id,
        timestamp: new Date().toISOString(),
        inputImages: {
          image1: `http://10.16.39.70:8080${uploadResult.image1}`,
          image2: `http://10.16.39.70:8080${uploadResult.image2}`
        },
        models: selectedModels,
        results: {
          changeDetectionImage: `http://10.16.39.70:8080${uploadResult.detection_result}`,
          segmentationImages: {
            image1: uploadResult.image1 ? `http://10.16.39.70:8080${uploadResult.image1}` : '',
            image2: uploadResult.image2 ? `http://10.16.39.70:8080${uploadResult.image2}` : ''
          },
          changedAreas: ['检测到的变化区域'],
          changeTypes: ['变化类型']
        }
      };

      // 这里可以添加保存记录到数据库的逻辑
      console.log('检测记录已创建:', newRecord);

    } catch (error) {
      console.error('检测失败:', error);
      setDetectionStartTime(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerateReport = () => {
    const currentStartTime = detectionStartTime;
    if (!currentStartTime) return;

    const endTime = Date.now();
    const duration = Number(((endTime - currentStartTime) / 1000).toFixed(3));

    // 保存检测时间记录
    const timeRecord: DetectionTime = {
      modelName: selectedModels.detectionModel,
      timestamp: new Date().toISOString(),
      duration: duration
    };

    // 存储到 localStorage
    const existingTimes = JSON.parse(localStorage.getItem('detectionTimes') || '[]');
    existingTimes.push(timeRecord);
    localStorage.setItem('detectionTimes', JSON.stringify(existingTimes));

    // 重置开始时间
    setDetectionStartTime(null);
  };

  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow p-6">
        {/* 模型选择部分 */}
        <div className="mb-8">
          <div className="flex items-center mb-6">
            <Science className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">选择模型</h2>
          </div>
          <div className="space-y-6">
            <div>
              <label className="block text-lg font-medium text-gray-700 mb-2">
                变化检测模型
              </label>
              <ModelSelector
                type="detection"
                value={selectedModels.detectionModel}
                onChange={(model) => setSelectedModels(prev => ({ ...prev, detectionModel: model }))}
              />
            </div>
            <div>
              <label className="block text-lg font-medium text-gray-700 mb-2">
                语义分割模型
              </label>
              <ModelSelector
                type="segmentation"
                value={selectedModels.segmentationModel}
                onChange={(model) => setSelectedModels(prev => ({ ...prev, segmentationModel: model }))}
                disabled={true}
              />
            </div>
          </div>
        </div>

        {/* 图片上传部分 */}
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <CloudUpload className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">上传双时态图片</h2>
          </div>
          <div className="grid grid-cols-2 gap-8">
            <ImageUploader
              label="上传图片1"
              onChange={(file) => handleImageUpload('image1', file)}
            />
            <ImageUploader
              label="上传图片2"
              onChange={(file) => handleImageUpload('image2', file)}
            />
          </div>
        </div>

        {/* 检测按钮 */}
        <div className="flex justify-center mb-8">
          <button
            className="relative flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleDetectionStart}
            disabled={
              !selectedModels.detectionModel || 
              !selectedModels.segmentationModel || 
              !selectedImages?.image1 || 
              !selectedImages?.image2 || 
              isProcessing
            }
          >
            {isProcessing ? (
              <>
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                </div>
                <span className="opacity-0">检测中...</span>
              </>
            ) : (
              <>
                <Compare className="h-5 w-5 mr-2" />
                开始检测
              </>
            )}
          </button>
        </div>
      </div>

      {/* 处理中弹窗 */}
      <ProcessingModal isOpen={isProcessing} />

      {/* 结果弹窗 */}
      {result && (
        <ResultModal
          isOpen={showResult}
          onClose={() => setShowResult(false)}
          result={result}
          selectedImages={imagePreviews}
          selectedModels={selectedModels}
          onGenerateReport={handleGenerateReport}
        />
      )}
    </div>
  );
};

export default Detection; 