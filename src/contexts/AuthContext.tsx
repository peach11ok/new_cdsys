import React, { createContext, useContext, useState, useEffect } from 'react';
import { User, LoginForm, RegisterForm } from '../types/auth';
import { DetectionRecord, StoredUser } from '../types/index';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  login: (data: LoginForm) => Promise<void>;
  register: (data: RegisterForm) => Promise<void>;
  logout: () => void;
  updateProfile: (data: UpdateProfileData) => Promise<void>;
  getAllUsers: () => Promise<StoredUser[]>;
  createUser: (data: { username: string; password: string; role: 'normal' }) => Promise<void>;
  deleteUser: (userId: string) => Promise<void>;
  getDetectionCount: (userId: string) => number;
  updateUser: (data: { id: string; username?: string; password?: string; makeRoot?: boolean }) => Promise<void>;
  getDetectionHistories: () => DetectionRecord[];
}

const AuthContext = createContext<AuthContextType | null>(null);

interface UpdateProfileData {
  username?: string;
  currentPassword?: string;
  newPassword?: string;
  avatar?: File;
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 清除所有本地存储数据
    const clearAllData = () => {
      // 清除所有用户相关的数据
      localStorage.clear();  // 这会清除所有的 localStorage 数据

      // 创建默认的 root 用户
      const rootUser: StoredUser = {
        id: 'root-' + Math.random().toString(),
        username: 'root',
        password: 'root123',  // 默认密码
        role: 'root',
        avatar: '',
        detectionCount: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      // 保存 root 用户
      localStorage.setItem('registeredUsers', JSON.stringify([rootUser]));
      
      // 重置加载状态
      setLoading(false);
      setUser(null);
    };

    clearAllData();
  }, []); // 只在组件首次加载时执行

  // 获取已注册用户列表
  const getRegisteredUsers = (): StoredUser[] => {
    const users = localStorage.getItem('registeredUsers');
    return users ? JSON.parse(users) : [];
  };

  // 保存用户列表
  const saveRegisteredUsers = (users: StoredUser[]) => {
    localStorage.setItem('registeredUsers', JSON.stringify(users));
  };

  const register = async (data: RegisterForm) => {
    try {
      setLoading(true);
      setError(null);

      const registeredUsers = getRegisteredUsers();

      // 检查用户名是否已存在
      if (registeredUsers.some(user => user.username === data.username)) {
        throw new Error('用户名已存在');
      }

      // 创建新用户（包含密码）
      const newUser: StoredUser = {
        id: Math.random().toString(),
        username: data.username,
        password: data.password,
        role: data.username.toLowerCase() === 'root' ? 'root' : 'normal',
        avatar: '',
        detectionCount: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      // 添加到注册用户列表
      const newUserList = [...registeredUsers, newUser];
      localStorage.setItem('registeredUsers', JSON.stringify(newUserList));
      
      return;
    } catch (err) {
      setError(err instanceof Error ? err.message : '注册失败');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const login = async (data: LoginForm) => {
    try {
      setLoading(true);
      setError(null);

      const registeredUsers = getRegisteredUsers();

      // 先检查用户名是否存在
      const storedUser = registeredUsers.find(u => u.username === data.username);
      
      if (!storedUser) {
        throw new Error('用户不存在，请先注册');
      }

      // 再检查密码是否正确
      if (storedUser.password !== data.password) {
        throw new Error('密码错误');
      }

      // 登录成功后只存储必要信息（不包含密码）
      const userWithoutPassword: User = {
        id: storedUser.id,
        username: storedUser.username,
        role: storedUser.role,
        avatar: storedUser.avatar
      };

      setUser(userWithoutPassword);
      localStorage.setItem('user', JSON.stringify(userWithoutPassword));
    } catch (err) {
      setError(err instanceof Error ? err.message : '登录失败');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  const updateProfile = async (data: UpdateProfileData) => {
    try {
      setLoading(true);
      setError(null);

      const registeredUsers = getRegisteredUsers();
      
      // 检查用户名是否已存在（如果要更改用户名）
      if (data.username && data.username !== user?.username) {
        if (registeredUsers.some(u => u.username === data.username)) {
          throw new Error('用户名已存在');
        }
      }

      // 验证当前密码（如果要更改密码）
      if (data.newPassword) {
        const currentUser = registeredUsers.find(u => u.id === user?.id);
        if (!currentUser || currentUser.password !== data.currentPassword) {
          throw new Error('当前密码错误');
        }
      }

      // 更新用户信息
      const updatedUsers = registeredUsers.map(u => {
        if (u.id === user?.id) {
          return {
            ...u,
            username: data.username || u.username,
            password: data.newPassword || u.password,
            avatar: data.avatar ? URL.createObjectURL(data.avatar) : u.avatar,
            updatedAt: new Date().toISOString()
          };
        }
        return u;
      });

      // 保存更新后的用户列表
      saveRegisteredUsers(updatedUsers);

      // 更新当前用户信息
      const updatedUser = updatedUsers.find(u => u.id === user?.id);
      if (updatedUser) {
        const userWithoutPassword = {
          id: updatedUser.id,
          username: updatedUser.username,
          role: updatedUser.role,
          avatar: updatedUser.avatar
        };
        setUser(userWithoutPassword);
        localStorage.setItem('user', JSON.stringify(userWithoutPassword));
      }

      // 如果修改了密码，登出用户
      if (data.newPassword) {
        logout();
        throw new Error('密码已更新，请重新登录');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新失败');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getAllUsers = async () => {
    const users = getRegisteredUsers();
    return users;
  };

  const createUser = async (data: { username: string; password: string; role: 'normal' }) => {
    const users = getRegisteredUsers();
    
    if (users.some(u => u.username === data.username)) {
      throw new Error('用户名已存在');
    }

    const newUser: StoredUser = {
      id: Math.random().toString(),
      username: data.username,
      password: data.password,
      role: data.role,
      avatar: '',
      detectionCount: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    saveRegisteredUsers([...users, newUser]);
  };

  const deleteUser = async (userId: string) => {
    try {
      // 删除用户数据
      const users = getRegisteredUsers();
      const updatedUsers = users.filter(u => u.id !== userId);
      saveRegisteredUsers(updatedUsers);

      // 删除用户的检测记录
      const histories = JSON.parse(localStorage.getItem('detectionHistories') || '[]');
      const updatedHistories = histories.filter((record: DetectionRecord) => record.userId !== userId);
      localStorage.setItem('detectionHistories', JSON.stringify(updatedHistories));

      // 触发更新
      setUser(prev => prev); // 触发重新渲染
    } catch (error) {
      console.error('删除用户失败:', error);
      throw error;
    }
  };

  const getDetectionCount = (userId: string) => {
    const histories = localStorage.getItem('detectionHistories');
    if (!histories) return 0;
    return JSON.parse(histories).filter((h: DetectionRecord) => h.userId === userId).length;
  };

  const updateUser = async (data: { id: string; username?: string; password?: string; makeRoot?: boolean }) => {
    const users = getRegisteredUsers();
    
    // 检查用户名是否已存在（如果要更改用户名）
    if (data.username) {
      const existingUser = users.find(u => u.username === data.username && u.id !== data.id);
      if (existingUser) {
        throw new Error('用户名已存在');
      }
    }

    const updatedUsers = users.map(u => {
      if (u.id === data.id) {
        return {
          ...u,
          username: data.username || u.username,  // 如果提供了新用户名则更新
          password: data.password || u.password,  // 如果提供了新密码则更新
          role: data.makeRoot ? 'root' : u.role,  // 只改变角色，不改变用户名
          updatedAt: new Date().toISOString()
        };
      }
      return u;
    });

    saveRegisteredUsers(updatedUsers);

    // 如果修改了密码，且修改的是当前登录用户，则登出
    if (data.password && data.id === user?.id) {
      logout();
      throw new Error('密码已更新，请重新登录');
    }
  };

  const getDetectionHistories = () => {
    return JSON.parse(localStorage.getItem('detectionHistories') || '[]') as DetectionRecord[];
  };

  const value = {
    user,
    isAuthenticated: !!user,
    loading,
    error,
    login,
    register,
    logout,
    updateProfile,
    getAllUsers,
    createUser,
    deleteUser,
    getDetectionCount,
    updateUser,
    getDetectionHistories
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 