import axios from 'axios';
import { ElMessage } from 'element-plus';
import type { UserInfo } from '@/types/user';
import { useRouter } from 'vue-router';
import router from '@/router/index';

let baseUrl = import.meta.env.VITE_API_BASE_URL;
let timeout = import.meta.env.VITE_TIMEOUT;

const apiClient = axios.create({
  baseURL: baseUrl,
  timeout: timeout,
});

const URL_WHITELIST = ['/login', '/register', '/public-api'];

// 请求拦截器
apiClient.interceptors.request.use(
  config => {
    // 检查是否为白名单
    if (config.url && URL_WHITELIST.some(path => config.url!.includes(path))) {
      return config;
    }

    // 检查 token 是否过期
    let userInfo: UserInfo | null = null;
    try {
      const userInfoStr = localStorage.getItem('userInfo');
      if (userInfoStr) {
        userInfo = JSON.parse(userInfoStr) as UserInfo;
      }
    } catch (error) {
      console.error('解析用户信息失败：', error);
      localStorage.removeItem('userInfo');
      router.push('/login');
      ElMessage.error('用户信息异常，请重新登录');
      return Promise.reject(new Error('用户信息异常'));
    }

    const tokenExpiration = userInfo?.tokenExpiration ?? 0;
    if (tokenExpiration && new Date().getTime() > tokenExpiration) {
      localStorage.removeItem('userInfo');
      router.push('/');
      ElMessage.error('登录已过期，请重新登录');
      return Promise.reject(new Error('登录已过期，请重新登录'));
    }

    if (!userInfo) {
      router.push('/login');
      ElMessage.error('用户信息异常，请重新登录');
      return Promise.reject(new Error('用户信息异常'));
    }

    // 在请求前头中添加 token
    const token = userInfo.token;
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  response => {
    const data = response?.data;

    if (data?.success === false) {
      ElMessage.error(`${data?.message}`);
    }
    return response.data;
  },
  error => {
    ElMessage.error(`${error.response?.data?.message || error.message}`);
    return Promise.reject(error);
  }
);

const api = {
  baseUrl,
  login(username: string, password: string) {
    return apiClient.post('/api/users/login', { username, password });
  },
  register(username: string, password: string) {
    return apiClient.post('/api/users/register', { username, password });
  },
};

export default api;
