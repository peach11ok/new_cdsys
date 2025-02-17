import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { DetectionRecord } from '../types';
import { Info } from '@mui/icons-material';
import Modal from '../components/Modal';

interface DetailModalProps {
  record: DetectionRecord;
  onClose: () => void;
}
const MODEL_NAMES = {
  detection: {
    'model1': 'TFIFNet',
    'model2': 'TFIFNetPro'
  },
  segmentation: {
    'seg1': 'DeepLabV3'
  }
};

const DetailModal: React.FC<DetailModalProps> = ({ record }) => {
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-6">检测详情</h2>
      <div className="space-y-6">
        {/* 检测时间 */}
        <div className="border-b pb-4">
          <label className="text-sm font-medium text-gray-500">检测时间</label>
          <p className="mt-1 text-lg font-medium text-gray-900">
            {new Date(record.timestamp).toLocaleString('zh-CN')}
          </p>
        </div>

        {/* 变化检测模型 */}
        <div className="border-b pb-4">
          <label className="text-sm font-medium text-gray-500">变化检测模型</label>
          <p className="mt-1 text-lg font-medium text-gray-900">
            {MODEL_NAMES.detection[record.models.detectionModel as keyof typeof MODEL_NAMES.detection] || record.models.detectionModel}
          </p>
        </div>

        {/* 语义分割模型 */}
        <div className="border-b pb-4">
          <label className="text-sm font-medium text-gray-500">语义分割模型</label>
          <p className="mt-1 text-lg font-medium text-gray-900">
            {MODEL_NAMES.segmentation[record.models.segmentationModel as keyof typeof MODEL_NAMES.segmentation] || record.models.segmentationModel}
          </p>
        </div>
      </div>
    </div>
  );
};

const History: React.FC = () => {
  const { user, getDetectionHistories } = useAuth();
  const [records, setRecords] = useState<DetectionRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<DetectionRecord | null>(null);

  useEffect(() => {
    if (user) {
      const histories = getDetectionHistories();
      const filteredRecords = user.role === 'root'
        ? histories
        : histories.filter((record: DetectionRecord) => record.userId === user.id);
      
      const sortedRecords = filteredRecords.sort((a: DetectionRecord, b: DetectionRecord) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      
      setRecords(sortedRecords);
    }
  }, [user, getDetectionHistories]);

  const handleViewReport = (record: DetectionRecord) => {
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
              <p>${new Date(record.timestamp).toLocaleString('zh-CN')}</p>
            </div>
            <div class="section">
              <h2>使用模型</h2>
              <p>变化检测模型：${MODEL_NAMES.detection[record.models.detectionModel as keyof typeof MODEL_NAMES.detection] || record.models.detectionModel}</p>
              <p>语义分割模型：${MODEL_NAMES.segmentation[record.models.segmentationModel as keyof typeof MODEL_NAMES.segmentation] || record.models.segmentationModel}</p>
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
              <div class="image-container">
                <h3>变化区域展示</h3>
                <img src="${record.results.changeDetectionImage}" alt="变化区域展示" />
              </div>
              <div class="grid">
                <div class="image-container">
                  <h3>图片1的语义类别</h3>
                  <img src="${record.results.segmentationImages.image1}" alt="图片1的语义类别" />
                </div>
                <div class="image-container">
                  <h3>图片2的语义类别</h3>
                  <img src="${record.results.segmentationImages.image2}" alt="图片2的语义类别" />
                </div>
              </div>
              <div class="section">
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

  if (records.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-2xl font-bold mb-6">历史检测记录</h2>
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <div className="flex justify-center mb-4">
            <Info className="h-12 w-12 text-gray-400" />
          </div>
          <p className="text-gray-500">暂无检测记录</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">历史检测记录</h2>
      <div className="grid gap-6">
        {records.map((record) => (
          <div key={record.id} className="bg-white rounded-lg shadow p-6">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">输入图片 1</h3>
                <div className="w-48 h-48 bg-gray-100 rounded-lg overflow-hidden">
                  <img 
                    src={record.inputImages.image1} 
                    alt="输入图片 1" 
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">输入图片 2</h3>
                <div className="w-48 h-48 bg-gray-100 rounded-lg overflow-hidden">
                  <img 
                    src={record.inputImages.image2} 
                    alt="输入图片 2" 
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            </div>

            <div className="border-t pt-4">
              <div className="flex justify-between items-center">
                <div className="space-y-1">
                  <div className="text-sm text-gray-500">
                    检测时间: {new Date(record.timestamp).toLocaleString('zh-CN')}
                  </div>
                  {user?.role === 'root' && record.username && (
                    <div className="text-sm text-gray-500">
                      检测用户: {record.username}
                    </div>
                  )}
                </div>
                <div className="space-x-3">
                  <button
                    onClick={() => handleViewReport(record)}
                    className="px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    查看报告
                  </button>
                  <button
                    onClick={() => setSelectedRecord(record)}
                    className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    查看详情
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 详情弹窗 */}
      <Modal isOpen={!!selectedRecord} onClose={() => setSelectedRecord(null)}>
        {selectedRecord && <DetailModal record={selectedRecord} onClose={() => setSelectedRecord(null)} />}
      </Modal>
    </div>
  );
};

export default History; 