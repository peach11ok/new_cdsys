import React, { useEffect, useState } from 'react';
import { Satellite, Person, Architecture, ArrowForward, ArrowBack, Timer } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TooltipItem,
  ChartOptions,
  Scale,
  ScriptableContext
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface DetectionTime {
  modelName: string;
  timestamp: string;
  duration: number;
}

const Dashboard: React.FC = () => {
  const { getAllUsers } = useAuth();
  const [userCount, setUserCount] = useState(0);
  const [currentModelIndex, setCurrentModelIndex] = useState(0);
  const [detectionTimes, setDetectionTimes] = useState<DetectionTime[]>([]);

  const models = [
    {
      name: 'TFIFNet',
      description: 'TFIFNet(Transformer with feature interaction and fusion for remote sensing image change detection)是一个专门针对遥感图像变化检测设计的深度学习模型。该模型基于Swin Transformer和孪生网络架构，结合特征交互和特征融合的思想，旨在解决小变化区域漏检问题和配准误差问题。',
      image: '/images/TFIFNet.jpg',
    },
    {
      name: 'SOSCDNet',
      description: 'SOSCDNet(Sobel operator and similarity combined remote sensing image change detection network)是一个旨在缓解变化检测边缘不完整的模型。该模型结合传统方法和深度学习方法以更好的识别变化区域，并设计一个边缘感知模块增强边缘信息。',
      image: '/images/SOFSNet.png',
    }
  ];

  useEffect(() => {
    const fetchUserCount = async () => {
      const users = await getAllUsers();
      setUserCount(users.length);
    };
    fetchUserCount();

    // 自动轮播
    const interval = setInterval(() => {
      setCurrentModelIndex((prev) => (prev === 0 ? 1 : 0));
    }, 5000);

    // 更新检测时间数据
    const updateDetectionTimes = () => {
      const times = JSON.parse(localStorage.getItem('detectionTimes') || '[]') as DetectionTime[];
      setDetectionTimes(times);
    };

    updateDetectionTimes();
    const timeUpdateInterval = setInterval(updateDetectionTimes, 1000);

    return () => {
      clearInterval(interval);
      clearInterval(timeUpdateInterval);
    };
  }, [getAllUsers]);

  const handlePrevModel = () => {
    setCurrentModelIndex((prev) => (prev === 0 ? models.length - 1 : prev - 1));
  };

  const handleNextModel = () => {
    setCurrentModelIndex((prev) => (prev === models.length - 1 ? 0 : prev + 1));
  };

  const getChartData = (modelName: string) => {
    const modelTimes = detectionTimes
      .filter(time => time.modelName === modelName)
      .slice(-10);

    return {
      labels: modelTimes.map((_, index) => `第${index + 1}次`),
      datasets: [
        {
          label: '检测时间',
          data: modelTimes.map(time => time.duration),
          borderColor: '#1e88e5',
          backgroundColor: 'rgba(30, 136, 229, 0.1)',
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.4,
          fill: true,
        },
      ],
    };
  };

  // 添加一个函数来检查是否有数据
  const hasModelData = (modelName: string) => {
    return detectionTimes.some(time => time.modelName === modelName);
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        titleColor: '#000',
        bodyColor: '#666',
        borderColor: '#ddd',
        borderWidth: 1,
        padding: 10,
        displayColors: false,
        callbacks: {
          label: function(context: TooltipItem<'line'>) {
            const value = context.parsed.y;
            return `检测用时: ${value.toFixed(3)}秒`;
          },
        },
      },
    },
    scales: {
      y: {
        type: 'linear' as const,
        beginAtZero: true,
        grid: {
          color: '#e0e0e0',
          lineWidth: 1,
        },
        border: {
          display: false,
        },
        ticks: {
          padding: 10,
          callback: function(tickValue: number | string, index: number) {
            if (typeof tickValue === 'number') {
              return tickValue.toFixed(3) + '秒';
            }
            return tickValue;
          },
        },
      },
      x: {
        type: 'category' as const,
        grid: {
          display: false,
        },
        border: {
          display: false,
        },
        ticks: {
          padding: 10,
        },
      },
    },
  };

  return (
    <div className="p-6 space-y-8">
      {/* 顶部空白区域和系统信息 */}
      <div className="h-32 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow-lg flex items-center justify-between px-8">
        <div className="flex items-center">
          <Satellite className="h-12 w-12 text-white mr-4" />
          <div>
            <h1 className="text-3xl font-bold text-white">遥感图像变化检测系统</h1>
            <p className="text-blue-100 mt-2">当前系统总用户数：{userCount}</p>
          </div>
        </div>
      </div>

      {/* 模型介绍部分 */}
      <div className="space-y-8">
        <div className="flex items-center">
          <Architecture className="h-8 w-8 text-blue-600 mr-2" />
          <h2 className="text-2xl font-bold text-gray-900">变化检测模型介绍</h2>
        </div>

        {/* 模型轮播展示 */}
        <div className="relative bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold text-gray-800">{models[currentModelIndex].name}</h3>
              <div className="flex items-center space-x-2">
                <button 
                  onClick={handlePrevModel}
                  className="p-2 rounded-full hover:bg-gray-100"
                >
                  <ArrowBack className="h-6 w-6 text-gray-600" />
                </button>
                <button 
                  onClick={handleNextModel}
                  className="p-2 rounded-full hover:bg-gray-100"
                >
                  <ArrowForward className="h-6 w-6 text-gray-600" />
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-4">
                <p className="text-gray-600 leading-relaxed">
                  {models[currentModelIndex].description}
                </p>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="flex items-center mb-4">
                    <Timer className="h-5 w-5 text-blue-600 mr-2" />
                    <h4 className="font-semibold text-blue-800">报告生成时间统计</h4>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    {hasModelData(models[currentModelIndex].name) ? (
                      <div className="h-[200px]">
                        <Line 
                          options={chartOptions} 
                          data={getChartData(models[currentModelIndex].name)}
                        />
                      </div>
                    ) : (
                      <div className="h-[200px] flex items-center justify-center">
                        <div className="text-center text-gray-500">
                          <p className="mb-2">暂无数据</p>
                          <p className="text-sm">该模型尚未进行过检测</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <img 
                  src={models[currentModelIndex].image}
                  alt={`${models[currentModelIndex].name}框架图`}
                  className="w-full h-auto object-contain"
                />
                <p className="text-sm text-gray-500 mt-2 text-center">
                  {models[currentModelIndex].name} 模型架构图
                </p>
              </div>
            </div>
          </div>

          {/* 轮播指示器 */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
            {models.map((_, index) => (
              <button
                key={index}
                className={`w-2 h-2 rounded-full ${
                  index === currentModelIndex ? 'bg-blue-600' : 'bg-gray-300'
                }`}
                onClick={() => setCurrentModelIndex(index)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 