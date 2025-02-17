import React from 'react';
import { Analytics } from '@mui/icons-material';

interface Props {
  isOpen: boolean;
}

const ProcessingModal: React.FC<Props> = ({ isOpen }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4">
        <div className="fixed inset-0 bg-black opacity-50"></div>
        <div className="relative bg-white rounded-lg p-8 flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
          <p className="mt-4 text-lg font-medium text-gray-700">正在检测中...</p>
        </div>
      </div>
    </div>
  );
};

export default ProcessingModal; 