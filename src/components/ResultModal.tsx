import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { DetectionResult, DetectionRecord, StoredUser } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  result: DetectionResult | null;
  selectedImages?: { image1: string; image2: string; } | null;
  selectedModels?: { detectionModel: string; segmentationModel: string; };
  onGenerateReport: () => void;
}

interface RecordProps {
  record: DetectionRecord;
}


const ResultModal: React.FC<Props> = ({ isOpen, onClose, result, selectedImages, selectedModels, onGenerateReport }) => {
  const { user } = useAuth();
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);

  useEffect(() => {
    console.log('ResultModal result:', result);
    if (result?.segmentationImages) {
      console.log('Segmentation Images:', result.segmentationImages);
    }
  }, [result]);

  const handleGenerateReport = async () => {
    if (!user || !result || !selectedImages || !selectedModels) return;
    
    setIsGeneratingReport(true);
    try {
      // 获取用户信息
      const users = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
      const currentUser = users.find((u: StoredUser) => u.id === user.id);

      // 创建新的检测记录
      const newRecord: DetectionRecord = {
        id: Math.random().toString(),
        userId: user.id,
        username: currentUser?.username,
        timestamp: new Date().toISOString(),
        inputImages: selectedImages,
        models: selectedModels,
        results: {
          changeDetectionImage: result.changeDetectionImage,
          segmentationImages: result.segmentationImages,
          changedAreas: result.changedAreas,
          changeTypes: result.segmentationData.image2.filter(
            (type, index) => type !== result.segmentationData.image1[index]
          )
        }
      };

      // 保存到历史记录
      const histories = JSON.parse(localStorage.getItem('detectionHistories') || '[]');
      localStorage.setItem('detectionHistories', JSON.stringify([...histories, newRecord]));

      // 更新用户检测次数
      const updatedUsers = users.map((u: StoredUser) => {
        if (u.id === user.id) {
          return {
            ...u,
            detectionCount: (u.detectionCount || 0) + 1,
            updatedAt: new Date().toISOString()
          };
        }
        return u;
      });
      localStorage.setItem('registeredUsers', JSON.stringify(updatedUsers));

      setReportGenerated(true);
      onGenerateReport();
    } catch (error) {
      console.error('生成报告失败:', error);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleViewReport = () => {
    if (!result || !selectedImages || !selectedModels) return;

    const reportWindow = window.open('', '_blank');
    if (reportWindow) {
      reportWindow.document.write(`
        <html>
          <head>
            <title>变化检测报告</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 20px; max-width: 1200px; margin: 0 auto; }
              img { max-width: 100%; height: auto; margin: 10px 0; }
              .section { margin-bottom: 30px; }
              .image-container { background: #f3f4f6; padding: 20px; border-radius: 8px; }
              .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
              h1 { color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
              h2 { color: #374151; margin-top: 20px; }
            </style>
          </head>
          <body>
            <h1>变化检测报告</h1>
            <div class="section">
              <h2>检测时间</h2>
              <p>${new Date().toLocaleString('zh-CN')}</p>
            </div>
            <div class="section">
              <h2>使用模型</h2>
              <p>变化检测模型：${selectedModels.detectionModel}</p>
              <p>语义分割模型：${selectedModels.segmentationModel}</p>
            </div>
            <div class="section">
              <h2>输入图片</h2>
              <div class="grid">
                <div class="image-container">
                  <h3>图片1</h3>
                  <img src="${selectedImages.image1}" alt="输入图片1" />
                </div>
                <div class="image-container">
                  <h3>图片2</h3>
                  <img src="${selectedImages.image2}" alt="输入图片2" />
                </div>
              </div>
            </div>
            <div class="section">
              <h2>检测结果</h2>
              <div>
                <h3>变化区域展示</h3>
                <div class="image-container">
                  <img src="${result.changeDetectionImage}" alt="变化区域展示" />
                </div>
              </div>
              <div>
                <h3>语义类别展示</h3>
                <div class="grid">
                  <div class="image-container">
                    <h4>图片1的语义类别</h4>
                    <img src="${result.segmentationImages.image1}" alt="图片1的语义类别" />
                  </div>
                  <div class="image-container">
                    <h4>图片2的语义类别</h4>
                    <img src="${result.segmentationImages.image2}" alt="图片2的语义类别" />
                  </div>
                </div>
              </div>
              <div>
                <h3>变化分析</h3>
                <p>变化区域：${result.changedAreas.join('、')}</p>
                <p>变化类型：${result.segmentationData.image2.filter(
                  (type, index) => type !== result.segmentationData.image1[index]
                ).join('、')}</p>
              </div>
            </div>
          </body>
        </html>
      `);
      reportWindow.document.close();
    }
  };

  const handleDownloadReport = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!result) return;

    const reportContent = {
      timestamp: new Date().toISOString(),
      result: {
        changeDetectionImage: result.changeDetectionImage,
        segmentationImages: result.segmentationImages,
        changedAreas: result.changedAreas,
        confidence: result.confidence
      }
    };

    const blob = new Blob([JSON.stringify(reportContent, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `detection-report-${new Date().toISOString()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!isOpen || !result) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-6">检测结果</h2>
        
        {/* 变化检测结果 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">变化区域展示</h3>
          <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
            <img 
              src={result.changeDetectionImage} 
              alt="变化区域展示" 
              className="max-w-full h-auto object-contain max-h-[400px]"
            />
          </div>
        </div>

        {/* 语义分割结果 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">语义类别展示</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-base font-medium mb-2">图片1的语义类别</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={result.segmentationImages.image1}
                  alt="图片1的语义类别" 
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
            <div>
              <h4 className="text-base font-medium mb-2">图片2的语义类别</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={result.segmentationImages.image2}
                  alt="图片2的语义类别" 
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
          </div>
        </div>

        {/* 按钮部分 */}
        <div className="flex justify-end space-x-4 pt-4 border-t">
          {!reportGenerated ? (
            <button
              onClick={handleGenerateReport}
              disabled={isGeneratingReport}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isGeneratingReport ? '生成报告中...' : '生成报告'}
            </button>
          ) : (
            <>
              <button
                onClick={handleViewReport}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                查看报告
              </button>
              <button
                onClick={handleDownloadReport}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                下载报告
              </button>
            </>
          )}
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultModal; 