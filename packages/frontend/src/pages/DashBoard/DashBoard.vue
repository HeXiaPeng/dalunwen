<template>
  <div class="dashboard-container">
    <!-- 主体布局：左侧边栏 + 右侧内容 -->
    <div class="dashboard-content">
      <!-- 左侧边栏 -->
      <div class="sidebar">
        <el-menu
          active-text-color="#ffd04b"
          background-color="#545c64"
          class="el-menu-vertical-demo"
          :default-active="activeMenuKey"
          text-color="#fff"
          router
        >
          <!-- 边栏菜单 -->
          <el-sub-menu index="clinicaltrials-group">
            <template #title>
              <el-icon><Menu /></el-icon>
              <span>试验管理</span>
            </template>
            <el-menu-item-group title="">
              <el-menu-item index="/dashboard/clinicaltrialstotal">临床试验汇总</el-menu-item>
              <el-menu-item index="/dashboard/clinicaltrials">clinicaltrials.gov</el-menu-item>
              <el-menu-item index="/dashboard/chictrorgcn">chictr.org.cn</el-menu-item>
            </el-menu-item-group>
          </el-sub-menu>
          
          <el-menu-item index="/dashboard/ct-rep">
            <el-icon><Document /></el-icon>
            <span>CT仓库</span>
          </el-menu-item> 

          <el-sub-menu index="CT-match">
            <template #title>
              <el-icon><Menu /></el-icon>
              <span>试验匹配</span>
            </template>
            <el-menu-item-group title="">
              <el-menu-item index="/dashboard/CT_match1">患者匹配临床试验</el-menu-item>
              <el-menu-item index="/dashboard/CT_match2">临床试验匹配临床试验</el-menu-item>
            </el-menu-item-group>
          </el-sub-menu>
          
          <el-menu-item index="/dashboard/ct-gen">
            <el-icon><Document /></el-icon>
            <span>临床试验方案生成</span>
          </el-menu-item>

          <el-menu-item index="/dashboard/ct-trend">
            <el-icon><Document /></el-icon>
            <span>智能分析与可视化</span>
          </el-menu-item>
        </el-menu>
      </div>

      <!-- 右侧内容区域（子页面渲染在这里） -->
      <div class="main">
        <DashboardHeader :currentPath="currentBreadcrumb as string" />
        <!-- 子路由渲染出口：右侧显示 exam1、exam2 对应的组件 -->
        <div class="main-content">
          <router-view />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Menu, Document } from "@element-plus/icons-vue";
import { useRoute } from "vue-router";
import { ref, watch } from "vue";
import DashboardHeader from '@/pages/DashBoard/DashboardHeader.vue';

const route = useRoute();

// 面包屑
const currentBreadcrumb = ref(route.meta.breadcrumb);

// 菜单选中状态
const getActiveMenuKey = (path: string) => {
  const basePath = path.replace("/dashboard/", "");
  // 对于 ct-rep 子路由，应该激活主菜单项
  if (basePath.startsWith('ct-rep')) {
    return 'ct-rep';
  }
  // 对于 chictrorgcn 和 clinicaltrials 子路由，激活对应的主菜单项
  if (basePath.startsWith('chictrorgcn') || basePath.startsWith('clinicaltrials')) {
    return 'clinicaltrials-group';
  }
  return basePath;
};

const activeMenuKey = ref(getActiveMenuKey(route.path));

// 监听路由变化，自动更新面包屑和菜单选中状态
watch(
  () => route.path,
  (newPath) => {
    activeMenuKey.value = getActiveMenuKey(newPath);
    currentBreadcrumb.value = route.meta.breadcrumb;
  },
  { immediate: true }
);

</script>

<style scoped>
/* 布局样式：左侧边栏固定，右侧自适应 */
.dashboard-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: #f5f5f5;
}

.dashboard-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 200px;
  background-color: #545c64;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.el-menu-vertical-demo {
  width: 100%;
  height: 100%;
  background-color: #545c64;
  border: 0;
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #fff;
  overflow-y: auto;
}

.breadcrumb {
  margin-bottom: 20px;
}

.main-content {
  flex: 1;
}
</style>
