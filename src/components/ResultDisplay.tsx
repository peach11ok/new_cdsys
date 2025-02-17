import React from 'react';
import { DetectionRecord, DetectionResult } from '../types';

interface Props {
  result: DetectionResult;
}

interface RecordProps {
  record:DetectionRecord
}

const ResultDisplay: React.FC<Props> = ({ result }) => {
  return (
    <div className="space-y-4">
      {/* <div>
        <h3 className="font-medium text-gray-900">变化区域：</h3>
        <ul className="mt-2 list-disc list-inside text-gray-600">
          {result.changedAreas.map((area, index) => (
            <li key={index}>{area}</li>
          ))}
        </ul>
      </div> */}
      {/* <div>
        <h3 className="font-medium text-gray-900">检测结果：</h3>
        <div className="mt-2 flex justify-center">
          {result.changeDetectionImage && (
            <img 
              src={result.changeDetectionImage}
              alt="变化检测结果"
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          )}
        </div>
      </div> */}
    </div>
  );
};

export default ResultDisplay; 