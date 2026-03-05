<template>
  <div class="simple-login-container">
    <div class="login-card">
      <h2 class="login-title">系统登录</h2>

      <el-form
        ref="loginFormRef"
        :model="loginForm"
        :rules="loginRules"
        label-width="0"
        class="login-form"
      >
        <el-form-item prop="username">
          <el-input
            v-model="loginForm.username"
            placeholder="请输入用户名"
            class="form-input"
          >
            <template #prefix>
              <el-icon><User /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="loginForm.password"
            type="password"
            placeholder="请输入密码"
            class="form-input"
            @keydown.enter="handleLogin"
          >
            <template #prefix>
              <el-icon><Lock /></el-icon>
            </template>
          </el-input>
        </el-form-item>

        <!-- 没有账号，点我注册 -->
        <el-form-item>
          <el-link type="primary" @click="handleToRegisterPage">没有账号，点我注册</el-link>
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            class="login-btn"
            :loading="isLoading"
            @click="handleLogin"
          >
            登录
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { User, Lock } from '@element-plus/icons-vue';
import api from '@/utils/api/index';
import type { LoginParams } from '@/types/user';
import type { FormInstance, FormRules } from 'element-plus';

const router = useRouter();
const loginFormRef = ref<FormInstance | null>(null);
const isLoading = ref(false);

const loginForm = reactive<LoginParams>({
  username: '',
  password: ''
});

const loginRules = reactive<FormRules>({
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
});

const handleLogin = async () => {
  if (!loginFormRef.value) return;

  try {
    await loginFormRef.value.validate();
    isLoading.value = true;

    const res: any = await api.login(loginForm.username, loginForm.password);
    if (res?.code === 200) {
      localStorage.setItem('userInfo', JSON.stringify(res.data));
      ElMessage.success('登录成功');
      router.push('/');
    } else {
      ElMessage.error(res.data.message || '登录失败');
    }
  } catch (error) {
    ElMessage.error('登录失败');
    console.error('登录失败：', error);
  } finally {
    isLoading.value = false;
  }
};

const handleToRegisterPage = () => {
  router.push('/register');
};
</script>

<style scoped lang="less">
.simple-login-container {
  width: 100%;
  min-height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
}

.login-card {
  width: 100%;
  max-width: 380px;
  background:
    linear-gradient(180deg, rgba(18, 24, 54, 0.72) 0%, rgba(10, 12, 30, 0.56) 100%) padding-box,
    linear-gradient(135deg, rgba(96, 165, 250, 0.75) 0%, rgba(34, 211, 238, 0.35) 35%, rgba(168, 85, 247, 0.7) 75%, rgba(59, 130, 246, 0.55) 100%) border-box;
  border-radius: 14px;
  padding: 36px 32px;
  box-shadow:
    0 26px 80px rgba(0, 0, 0, 0.45),
    0 10px 22px rgba(0, 0, 0, 0.25);
  border: 1px solid transparent;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  position: relative;
  overflow: hidden;
}

.login-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(700px 260px at 20% 0%, rgba(255, 255, 255, 0.18) 0%, rgba(255, 255, 255, 0) 58%),
    radial-gradient(520px 220px at 100% 10%, rgba(34, 211, 238, 0.18) 0%, rgba(34, 211, 238, 0) 55%);
  pointer-events: none;
  z-index: 0;
}

.login-card > * {
  position: relative;
  z-index: 1;
}

.login-title {
  font-size: 20px;
  font-weight: 600;
  color: rgba(235, 244, 255, 0.92);
  text-align: center;
  margin-bottom: 28px;
  letter-spacing: 0.5px;
  text-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.login-form {
  width: 100%;

  .form-input {
    height: 44px;    
    border-radius: 6px;
  }

  .form-extra {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    color: #4e5969;
  }

  .login-btn {
    width: 100%;
    height: 44px;
    font-size: 16px;
    border-radius: 6px;
    border: 0;
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.95) 0%, rgba(59, 130, 246, 0.92) 40%, rgba(6, 182, 212, 0.92) 100%);
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.25);
  }
}

.login-card :deep(.el-form-item) {
  margin-bottom: 14px;
}

.login-card :deep(.el-input__wrapper) {
  height: 44px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: none;
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.login-card :deep(.el-input__wrapper.is-focus) {
  border-color: rgba(34, 211, 238, 0.55);
  box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.16);
}

.login-card :deep(.el-input__inner) {
  color: rgba(235, 244, 255, 0.92);
}

.login-card :deep(.el-input__inner::placeholder) {
  color: rgba(235, 244, 255, 0.55);
}

.login-card :deep(.el-link) {
  color: rgba(235, 244, 255, 0.78);
}

.login-card :deep(.el-link:hover) {
  color: rgba(34, 211, 238, 0.92);
}

@media (max-width: 375px) {
  .login-card {
    padding: 28px 24px;
  }

  .form-input,
  .login-btn {
    height: 40px;
  }
}
</style>
