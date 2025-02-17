import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { StoredUser } from '../types';
import { Person, AccessTime, Analytics } from '@mui/icons-material';

const Profile: React.FC = () => {
  const { user, getDetectionCount } = useAuth();
  const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
  const userDetails = registeredUsers.find((u: StoredUser) => u.id === user?.id);

  if (!user || !userDetails) return null;

  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-8 flex items-center">
          <Person className="h-8 w-8 text-blue-500 mr-2" />
          个人信息
        </h2>
        
        <div className="space-y-6">
          {/* 用户基本信息 */}
          <div className="border-b pb-4">
            <label className="text-sm font-medium text-gray-500">用户名</label>
            <p className="mt-1 text-lg font-medium text-gray-900">{user.username}</p>
          </div>

          {/* 用户角色 */}
          <div className="border-b pb-4">
            <label className="text-sm font-medium text-gray-500">用户角色</label>
            <p className="mt-1 text-lg font-medium text-gray-900">
              {user.role === 'root' ? '管理员' : '普通用户'}
            </p>
          </div>

          {/* 注册时间 */}
          <div className="border-b pb-4">
            <label className="text-sm font-medium text-gray-500 flex items-center">
              <AccessTime className="h-4 w-4 text-blue-500 mr-1" />
              注册时间
            </label>
            <p className="mt-1 text-lg font-medium text-gray-900">
              {new Date(userDetails.createdAt).toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
              })}
            </p>
          </div>

          {/* 检测统计 */}
          <div>
            <label className="text-sm font-medium text-gray-500 flex items-center">
              <Analytics className="h-4 w-4 text-blue-500 mr-1" />
              检测统计
            </label>
            <p className="mt-1 text-lg font-medium text-gray-900">
              总检测次数：{getDetectionCount(user.id)} 次
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile; 