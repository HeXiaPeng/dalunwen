import { createRouter, createWebHistory } from 'vue-router';

const title = import.meta.env.VITE_WEB_NAME;

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/', redirect: '/dashboard',
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
      path: '/register',
      name: 'Register',
      component: () => import('@/pages/RegisterPage/RegisterPage.vue'),
      meta: {
        title: '注册',
        isLoginPage: true,
      },
    },
    {
      path: '/dashboard',
      component: () => import('@/pages/DashBoard/DashBoard.vue'),
      children: [
        { path: '', redirect: 'dashboard/clinicaltrialstotal' }, 
        {
          path: 'clinicaltrialstotal',
          name: 'clinicaltrialsTotal',
          component: () => import('@/pages/DashBoard/components/ClinicaltrialsTotal.vue'),
          meta: { breadcrumb: '试验管理 > 临床试验汇总' }
        },
        {
          path: 'clinicaltrials',
          name: 'clinicaltrials.gov',
          component: () => import('@/pages/DashBoard/components/ClinicaltrialsGov.vue'),
          meta: { breadcrumb: '试验管理 > clinicaltrials.gov' }
        },
        {
          path: 'chictrorgcn',
          name: 'chictr.org.cn',
          component: () => import('@/pages/DashBoard/components/ChictrOrgCn.vue'),
          meta: { breadcrumb: '试验管理 > chictr.org.cn' }
        },
        {
          path: 'ct-rep',
          name: 'ct-rep',
          component: () => import('@/pages/DashBoard/components/WarehousePage/WarehousePage.vue'),
          meta: { breadcrumb: 'CT仓库' }
        },
        {
          path: 'ct-rep/:warehouseId',
          name: 'WarehouseCTPage',
          component: () => import('@/pages/DashBoard/components/WarehouseCTPage/WarehouseCTPage.vue'),
          props: true,
          meta: { 
            breadcrumb: 'CT仓库 - 详情',
          },
      },
      {
        path: 'CT_match1',
        name: 'CT_match1',
        component: () => import('@/pages/DashBoard/components/PatientMatch.vue'),
        meta: { breadcrumb: '试验匹配 > 患者匹配临床试验' }
      },
      {
        path: 'CT_match2',
        name: 'CT_match2',
        component: () => import('@/pages/DashBoard/components/ClinicalTrialMatch.vue'),
        meta: { breadcrumb: '试验匹配 > 临床试验匹配临床试验' }
      },
      {
        path: 'ct-gen',
        name: 'ct-gen',
        component: () => import('@/pages/DashBoard/components/ClinicalTrialGenerator.vue'),
        meta: { breadcrumb: '临床试验方案生成' }
      },
      {
        path: 'ct-trend',
        name: 'ct-trend',
        component: () => import('@/pages/DashBoard/components/ClinicalTrialsTrend.vue'),
        meta: { breadcrumb: '智能分析与可视化' }
      }
      ]
    }
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
  void from;
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
