import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Person, Lock } from '@mui/icons-material';

interface Props {
  onClose: () => void;
}

const EditProfile: React.FC<Props> = ({ onClose }) => {
  const { user, updateProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    currentPassword: '',
    newPassword: '',
    avatar: null as File | null
  });

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFormData(prev => ({ ...prev, avatar: file }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await updateProfile({
        currentPassword: formData.currentPassword,
        newPassword: formData.newPassword,
        avatar: formData.avatar || undefined
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新失败');
    } finally {
      setLoading(false);
    }
  };

  const defaultAvatar = (
    <div className="w-24 h-24 rounded-full bg-blue-100 flex items-center justify-center">
      <Person className="h-12 w-12 text-blue-500" />
    </div>
  );

  return (
    <div className="bg-white rounded-lg p-6 w-[28rem]">
      <h2 className="text-xl font-bold text-gray-900 flex items-center mb-4">
        <Person className="h-6 w-6 text-blue-500 mr-2" />
        修改资料
      </h2>

      {error && (
        <div className="mb-3 bg-red-50 border-l-4 border-red-400 p-3 rounded text-sm">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* 头像修改部分 */}
        <div className="bg-gray-50 p-3 rounded-lg">
          <h3 className="text-base font-medium text-gray-900 mb-3">头像设置</h3>
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              {formData.avatar ? (
                <img
                  src={URL.createObjectURL(formData.avatar)}
                  alt="新头像预览"
                  className="w-20 h-20 rounded-full object-cover"
                />
              ) : defaultAvatar}
            </div>
            <div className="flex-1">
              <input
                type="file"
                accept="image/*"
                onChange={handleAvatarChange}
                className="block w-full text-sm text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <p className="mt-1 text-xs text-gray-500">
                支持 JPG、PNG 格式，文件大小不超过 2MB
              </p>
            </div>
          </div>
        </div>

        {/* 密码修改部分 */}
        <div className="bg-gray-50 p-3 rounded-lg">
          <h3 className="text-base font-medium text-gray-900 mb-3">密码修改</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                当前密码
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="password"
                  value={formData.currentPassword}
                  onChange={e => setFormData(prev => ({ ...prev, currentPassword: e.target.value }))}
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                新密码
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  type="password"
                  value={formData.newPassword}
                  onChange={e => setFormData(prev => ({ ...prev, newPassword: e.target.value }))}
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-3 pt-3 border-t mt-3">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 text-gray-700 hover:text-gray-900"
          >
            返回
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? '保存中...' : '保存修改'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default EditProfile; 