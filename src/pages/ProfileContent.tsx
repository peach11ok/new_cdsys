import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { StoredUser } from '../types';
import { Person, AccessTime, Analytics } from '@mui/icons-material';

interface Props {
  onClose: () => void;
  onEdit: () => void;
}

const ProfileContent: React.FC<Props> = ({ onEdit }) => {
  const { user, getDetectionCount } = useAuth();
  const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
  const userDetails = registeredUsers.find((u: StoredUser) => u.id === user?.id);

  if (!user || !userDetails) return null;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-8 flex items-center">
        <Person className="h-8 w-8 text-blue-500 mr-2" />
        个人信息
      </h2>
      
      <div className="space-y-6">
        {/* 头像部分 */}
        <div className="flex justify-center mb-6">
          <button
            onClick={onEdit}
            className="relative group"
          >
            <div className="w-24 h-24 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden">
              {userDetails.avatar ? (
                <img
                  src={userDetails.avatar}
                  alt="用户头像"
                  className="w-full h-full object-cover"
                />
              ) : (
                <Person className="h-12 w-12 text-gray-400" />
              )}
            </div>
            <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 rounded-full flex items-center justify-center transition-opacity">
              <span className="text-white opacity-0 group-hover:opacity-100">修改头像</span>
            </div>
          </button>
        </div>

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

        {/* 修改资料按钮 */}
        <div className="flex justify-center pt-4">
          <button
            onClick={onEdit}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            修改资料
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProfileContent; 