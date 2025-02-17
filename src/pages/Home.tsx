import React, { useState } from 'react';
import { useNavigate, Link, Routes, Route, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Modal from '../components/Modal';
import ProfileContent from './ProfileContent';
import EditProfile from '../components/EditProfile';
import Dashboard from './Dashboard';
import Detection from './Detection';
import History from './History';
import UserManagement from './UserManagement';
import {
  Satellite,
  Home as HomeIcon,
  Compare,
  History as HistoryIcon,
  Logout,
  Person
} from '@mui/icons-material';

const Home: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isEditProfileOpen, setIsEditProfileOpen] = useState(false);

  const handleEditProfile = () => {
    setIsProfileOpen(false);
    setIsEditProfileOpen(true);
  };

  const handleEditComplete = () => {
    setIsEditProfileOpen(false);
    setIsProfileOpen(true);
  };

  const getNavLinkClass = (path: string) => {
    const isActive = location.pathname === path;
    return `flex items-center px-3 py-2 ${
      isActive 
        ? 'text-blue-600 font-medium' 
        : 'text-gray-700 hover:text-blue-600'
    }`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 顶部导航栏 */}
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center space-x-8">
              <div className="flex items-center">
                <Satellite className="h-8 w-8 text-blue-600" />
                <h1 className="ml-2 text-2xl font-bold text-gray-900">变化检测系统</h1>
              </div>
              <div className="flex items-center space-x-4">
                <Link
                  to="/"
                  className={getNavLinkClass('/')}
                >
                  <HomeIcon className={`h-5 w-5 mr-1 ${location.pathname === '/' ? 'text-blue-600' : ''}`} />
                  首页
                </Link>
                <Link
                  to="/detection"
                  className={getNavLinkClass('/detection')}
                >
                  <Compare className={`h-5 w-5 mr-1 ${location.pathname === '/detection' ? 'text-blue-600' : ''}`} />
                  变化检测
                </Link>
                <Link
                  to="/history"
                  className={getNavLinkClass('/history')}
                >
                  <HistoryIcon className={`h-5 w-5 mr-1 ${location.pathname === '/history' ? 'text-blue-600' : ''}`} />
                  历史记录
                </Link>
                {user?.role === 'root' && (
                  <Link
                    to="/users"
                    className={getNavLinkClass('/users')}
                  >
                    <Person className={`h-5 w-5 mr-1 ${location.pathname === '/users' ? 'text-blue-600' : ''}`} />
                    用户管理
                  </Link>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setIsProfileOpen(true)}
                className="flex items-center gap-2 text-gray-700 hover:text-blue-600"
              >
                <Person className="h-5 w-5" />
                <span>{user?.username}</span>
              </button>
              <button
                onClick={logout}
                className="flex items-center px-3 py-2 text-sm text-red-600 hover:text-red-800"
              >
                <Logout className="h-5 w-5 mr-1" />
                退出登录
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* 个人资料弹窗 */}
      <Modal isOpen={isProfileOpen} onClose={() => setIsProfileOpen(false)}>
        <ProfileContent 
          onClose={() => setIsProfileOpen(false)} 
          onEdit={handleEditProfile}
        />
      </Modal>

      {/* 修改资料弹窗 */}
      <Modal isOpen={isEditProfileOpen} onClose={() => setIsEditProfileOpen(false)}>
        <EditProfile onClose={handleEditComplete} />
      </Modal>

      {/* 主要内容区域 */}
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/detection" element={<Detection />} />
          <Route path="/history" element={<History />} />
          {user?.role === 'root' && (
            <Route path="/users" element={<UserManagement />} />
          )}
        </Routes>
      </div>
    </div>
  );
};

export default Home; 