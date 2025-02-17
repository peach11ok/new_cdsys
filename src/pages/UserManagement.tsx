import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { StoredUser } from '../types';
import { Person, Delete, Add, Edit, AdminPanelSettings, Group } from '@mui/icons-material';
import Modal from '../components/Modal';

interface AddUserModalProps {
  onClose: () => void;
  onAdd: (username: string, password: string) => void;
}

const AddUserModal: React.FC<AddUserModalProps> = ({ onClose, onAdd }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('用户名和密码不能为空');
      return;
    }
    onAdd(username, password);
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-6">添加用户</h2>
      {error && (
        <div className="mb-4 text-red-500 text-sm">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            用户名
          </label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            密码
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>
        <div className="flex justify-end space-x-3 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            添加
          </button>
        </div>
      </form>
    </div>
  );
};

interface ChangePasswordModalProps {
  username: string;
  onClose: () => void;
  onSubmit: (userId: string, newPassword: string) => Promise<void>;
  userId: string;
}

const ChangePasswordModal: React.FC<ChangePasswordModalProps> = ({ username, onClose, onSubmit, userId }) => {
  const [newPassword, setNewPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPassword) {
      setError('密码不能为空');
      return;
    }
    setLoading(true);
    try {
      await onSubmit(userId, newPassword);
      onClose();
    } catch (error) {
      setError(error instanceof Error ? error.message : '修改失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-6">修改密码 - {username}</h2>
      {error && (
        <div className="mb-4 text-red-500 text-sm">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            新密码
          </label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
          />
        </div>
        <div className="flex justify-end space-x-3 pt-4">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? '修改中...' : '确认修改'}
          </button>
        </div>
      </form>
    </div>
  );
};

const UserManagement: React.FC = () => {
  const { getAllUsers, deleteUser, createUser, updateUser } = useAuth();
  const [users, setUsers] = useState<StoredUser[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<StoredUser | null>(null);
  const [showPasswordModal, setShowPasswordModal] = useState(false);

  // 加载用户列表
  useEffect(() => {
    const loadUsers = async () => {
      const userList = await getAllUsers();
      setUsers(userList);
    };
    loadUsers();
  }, [getAllUsers]);

  // 按角色分组用户
  const adminUsers = users.filter(user => user.role === 'root');
  const normalUsers = users.filter(user => user.role === 'normal');

  const handleDeleteUser = async (userId: string) => {
    try {
      await deleteUser(userId);
      const updatedUsers = await getAllUsers();
      setUsers(updatedUsers);
    } catch (error) {
      console.error('删除用户失败:', error);
    }
  };

  const handleAddUser = async (username: string, password: string) => {
    try {
      await createUser({ username, password, role: 'normal' });
      const updatedUsers = await getAllUsers();
      setUsers(updatedUsers);
      setShowAddModal(false);
    } catch (error) {
      console.error('添加用户失败:', error);
    }
  };

  const handleMakeAdmin = async (userId: string) => {
    try {
      await updateUser({ id: userId, makeRoot: true });
      const updatedUsers = await getAllUsers();
      setUsers(updatedUsers);
    } catch (error) {
      console.error('设置管理员失败:', error);
    }
  };

  const handleChangePassword = async (userId: string, newPassword: string) => {
    try {
      await updateUser({ id: userId, password: newPassword });
      const updatedUsers = await getAllUsers();
      setUsers(updatedUsers);
    } catch (error) {
      console.error('修改密码失败:', error);
      throw error;
    }
  };

  const UserGroup: React.FC<{ title: string; icon: React.ReactNode; users: StoredUser[] }> = ({ title, icon, users }) => (
    <div className="mb-8">
      <div className="flex items-center mb-4">
        {icon}
        <h3 className="text-lg font-semibold ml-2">{title}</h3>
      </div>
      <div className="grid gap-4">
        {users.map((user) => (
          <div key={user.id} className="bg-white rounded-lg shadow p-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center space-x-4">
                <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                  <Person className="h-6 w-6 text-gray-400" />
                </div>
                <div>
                  <h3 className="font-medium">{user.username}</h3>
                  <p className="text-sm text-gray-500">
                    注册时间：{new Date(user.createdAt).toLocaleString('zh-CN')}
                  </p>
                  <p className="text-sm text-gray-500">
                    检测次数：{user.detectionCount || 0}
                  </p>
                </div>
              </div>
              {user.role !== 'root' && (
                <div className="flex space-x-4">
                  <div className="flex flex-col items-center">
                    <button
                      onClick={() => {
                        setSelectedUser(user);
                        setShowPasswordModal(true);
                      }}
                      className="p-2 text-green-600 hover:text-green-800"
                      title="修改密码"
                    >
                      <Edit className="h-5 w-5" />
                    </button>
                    <span className="text-xs text-gray-500">修改密码</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <button
                      onClick={() => handleMakeAdmin(user.id)}
                      className="p-2 text-blue-600 hover:text-blue-800"
                      title="设为管理员"
                    >
                      <AdminPanelSettings className="h-5 w-5" />
                    </button>
                    <span className="text-xs text-gray-500">设为管理员</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <button
                      onClick={() => handleDeleteUser(user.id)}
                      className="p-2 text-red-600 hover:text-red-800"
                      title="删除用户"
                    >
                      <Delete className="h-5 w-5" />
                    </button>
                    <span className="text-xs text-gray-500">删除用户</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">用户管理</h2>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          <Add className="h-5 w-5 mr-1" />
          添加用户
        </button>
      </div>

      {/* 管理员组 */}
      <UserGroup
        title="管理员"
        icon={<AdminPanelSettings className="h-6 w-6 text-blue-500" />}
        users={adminUsers}
      />

      {/* 普通用户组 */}
      <UserGroup
        title="普通用户"
        icon={<Group className="h-6 w-6 text-green-500" />}
        users={normalUsers}
      />

      <Modal isOpen={showAddModal} onClose={() => setShowAddModal(false)}>
        <AddUserModal
          onClose={() => setShowAddModal(false)}
          onAdd={handleAddUser}
        />
      </Modal>

      {/* 添加密码修改弹窗 */}
      <Modal isOpen={showPasswordModal} onClose={() => setShowPasswordModal(false)}>
        {selectedUser && (
          <ChangePasswordModal
            username={selectedUser.username}
            userId={selectedUser.id}
            onClose={() => setShowPasswordModal(false)}
            onSubmit={handleChangePassword}
          />
        )}
      </Modal>
    </div>
  );
};

export default UserManagement; 