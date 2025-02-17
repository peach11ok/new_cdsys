import React from 'react';
import { DetectionRecord } from '../types';

interface Props {
  record: DetectionRecord;
  isOpen: boolean;
  onClose: () => void;
}

const RecordModal: React.FC<Props> = ({ record, isOpen, onClose }) => {
  const handleViewReport = () => {
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
              <p>${new Date(record.timestamp).toLocaleString()}</p>
            </div>
            <div class="section">
              <h2>使用模型</h2>
              <p>变化检测模型：${record.models.detectionModel}</p>
              <p>语义分割模型：${record.models.segmentationModel}</p>
            </div>
            <div class="section">
              <h2>输入图片</h2>
              <div class="grid">
                <div class="image-container">
                  <h3>图片1</h3>
                  <img src="${record.inputImages.image1}" alt="输入图片1" />
                </div>
                <div class="image-container">
                  <h3>图片2</h3>
                  <img src="${record.inputImages.image2}" alt="输入图片2" />
                </div>
              </div>
            </div>
            <div class="section">
              <h2>检测结果</h2>
              <div>
                <h3>变化检测结果</h3>
                <div class="image-container">
                  <img src="${record.results.changeDetectionImage}" alt="变化检测结果" />
                </div>
              </div>
              <div>
                <h3>语义分割结果</h3>
                <div class="grid">
                  <div class="image-container">
                    <h4>图片1语义分割</h4>
                    <img src="${record.results.segmentationImages.image1}" alt="语义分割结果1" />
                  </div>
                  <div class="image-container">
                    <h4>图片2语义分割</h4>
                    <img src="${record.results.segmentationImages.image2}" alt="语义分割结果2" />
                  </div>
                </div>
              </div>
              <div>
                <h3>变化分析</h3>
                <p>变化区域：${record.results.changedAreas.join('、')}</p>
                <p>变化类型：${record.results.changeTypes.join('、')}</p>
              </div>
            </div>
          </body>
        </html>
      `);
      reportWindow.document.close();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">变化检测报告</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <span className="sr-only">关闭</span>
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 基本信息 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">基本信息</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-gray-500">检测时间</p>
              <p className="mt-1">{new Date(record.timestamp).toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500">使用模型</p>
              <p className="mt-1">{record.models.detectionModel}</p>
            </div>
          </div>
        </div>

        {/* 输入图片 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">输入图片</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-base font-medium mb-2">时相1</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={record.inputImages.image1}
                  alt="输入图片1"
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
            <div>
              <h4 className="text-base font-medium mb-2">时相2</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={record.inputImages.image2}
                  alt="输入图片2"
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
          </div>
        </div>

        {/* 变化检测结果 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">变化检测结果</h3>
          <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
            <img 
              src={record.results.changeDetectionImage} 
              alt="变化检测结果" 
              className="max-w-full h-auto object-contain max-h-[400px]"
            />
          </div>
        </div>

        {/* 语义分割结果 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">语义分割结果</h3>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-base font-medium mb-2">图片1语义分割</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={record.results.segmentationImages.image1} 
                  alt="语义分割结果1" 
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
            <div>
              <h4 className="text-base font-medium mb-2">图片2语义分割</h4>
              <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
                <img 
                  src={record.results.segmentationImages.image2} 
                  alt="语义分割结果2" 
                  className="max-w-full h-auto object-contain max-h-[300px]"
                />
              </div>
            </div>
          </div>
        </div>

        {/* 变化分析 */}
        <div className="mb-8">
          <h3 className="text-lg font-medium mb-4 border-b pb-2">变化分析</h3>
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium text-gray-500">变化区域</p>
              <p className="mt-1">{record.results.changedAreas.join('、')}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500">变化类型</p>
              <p className="mt-1">{record.results.changeTypes.join('、')}</p>
            </div>
          </div>
        </div>

        {/* 按钮部分 */}
        <div className="flex justify-end space-x-4 pt-4 border-t">
          <button
            onClick={handleViewReport}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            查看报告
          </button>
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

export default RecordModal; 