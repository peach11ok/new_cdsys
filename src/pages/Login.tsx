import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Lock, Person, AccountCircle } from '@mui/icons-material';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [userExists, setUserExists] = useState<boolean | null>(null);

  // 检查用户是否存在
  const checkUserExists = (username: string) => {
    const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
    return registeredUsers.some((user: any) => user.username === username);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const username = formData.username.trim();
      if (username) {
        const exists = checkUserExists(username);
        setUserExists(exists);
        if (!exists) {
          setError('用户不存在，即将跳转到注册页面...');
          setTimeout(() => {
            navigate('/register', { state: { username } });
          }, 1500);
        } else {
          setError(null);
          // 如果用户存在，将焦点移动到密码输入框
          const passwordInput = document.getElementById('password');
          if (passwordInput) {
            passwordInput.focus();
          }
        }
      }
    }
  };

  // 处理密码输入
  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({ ...prev, password: e.target.value }));
    if (userExists) {
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // 先检查用户名是否存在
    const username = formData.username.trim();
    if (username) {
      const exists = checkUserExists(username);
      if (!exists) {
        setError('用户不存在，即将跳转到注册页面...');
        setTimeout(() => {
          navigate('/register', { state: { username } });
        }, 1500);
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      await login(formData);
      navigate('/');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '登录失败';
      if (errorMessage === '密码错误') {
        setError('密码错误，请重新输入');
      } else {
        setError(errorMessage);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="max-w-md w-full m-4">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center">
            <AccountCircle className="mx-auto h-16 w-16 text-blue-500" />
            <h2 className="mt-4 text-center text-3xl font-extrabold text-gray-900">
              登录系统
            </h2>
          </div>
          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            {error && (
              <div className={`rounded-md p-4 ${
                error.includes('即将跳转') ? 'bg-blue-50 text-blue-700' : 'bg-red-50 text-red-700'
              }`}>
                <div className="text-sm">{error}</div>
              </div>
            )}
            <div className="space-y-5">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                  用户名
                </label>
                <div className="relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Person className="h-5 w-5 text-blue-500" />
                  </div>
                  <input
                    id="username"
                    name="username"
                    type="text"
                    required
                    value={formData.username}
                    onChange={(e) => setFormData(prev => ({ ...prev, username: e.target.value }))}
                    onKeyPress={handleKeyPress}
                    className="appearance-none block w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="请输入用户名"
                  />
                </div>
              </div>
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                  密码
                </label>
                <div className="relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-blue-500" />
                  </div>
                  <input
                    id="password"
                    name="password"
                    type="password"
                    required
                    value={formData.password}
                    onChange={handlePasswordChange}
                    className="appearance-none block w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                    placeholder="请输入密码"
                  />
                </div>
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={loading}
                className="group relative w-full flex justify-center py-2.5 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition-colors"
              >
                <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                  <Person className="h-5 w-5 text-blue-400 group-hover:text-blue-300" />
                </span>
                {loading ? '登录中...' : '登录'}
              </button>
            </div>

            <div className="text-sm text-center">
              <Link
                to="/register"
                className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
              >
                还没有账号？立即注册
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;