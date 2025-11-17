import { createRouter, createWebHistory } from 'vue-router';

const title = import.meta.env.VITE_WEB_NAME;

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/home',
    },
    {
      path: '/login',
      name: 'Login',
      component: () => import('@/pages/LoginPage/LoginPage.vue'),
      meta: {
        title: '登录',
        isLoginPage: true,
      },
    },
    {
      path: '/home',
      name: 'Home',
      component: () => import('@/pages/HomePage/HomePage.vue'),
    },
  ],
});

/**
 * 路由前置守卫：处理登录状态校验、路由权限控制
 * @param to - 目标路由
 * @param from - 源路由
 * @param next - 路由跳转函数
 * @returns void
 */
router.beforeEach((to, from, next) => {
  // 设置页面标题
  document.title = to.meta.title || title;
  const userInfoStr = localStorage.getItem('userInfo');
  const userInfo = userInfoStr ? JSON.parse(userInfoStr) : null;
  const token = userInfo?.token ?? '';
  const tokenExpiration = userInfo?.tokenExpiration ?? 0;
  const isLoginPage = to.meta.isLoginPage;

  // 如果用户已经登录，且目标路由是登录页面，跳转到主页
  if (token && isLoginPage) {
    next({ path: '/' });
    return;
  }

  // 检查token是否有效
  const isAuthenticated = isAuth(token, tokenExpiration);

  // 已登录，检查 token 是否过期,
  // 进入条件：非登录页面 且 token 过期
  if (!isLoginPage && !isAuthenticated) {
    next({ path: '/login' });
    return;
  }

  next();
});

/**
 * 检查 token 是否有效
 * @param token
 * @param tokenExpiration - token 过期时间
 * @returns boolean
 */
const isAuth = (token: string, tokenExpiration: number) => {
  if (!token) {
    return false;
  }

  const now = new Date().getTime();
  if (tokenExpiration && now > tokenExpiration) {
    localStorage.clear();
    return false;
  }

  return true;
};

export default router;
