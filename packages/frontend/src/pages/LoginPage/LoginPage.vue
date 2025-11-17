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
import api from '@/utils/api';
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

    const res = await api.login(loginForm.username, loginForm.password);
    if (res.code === 200) {
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
</script>

<style scoped lang="less">
.simple-login-container {
  width: 100vw;
  height: 100vh;
  background-color: #f5f7fa;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.login-card {
  width: 100%;
  max-width: 380px;
  background: #fff;
  border-radius: 8px;
  padding: 36px 32px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
}

.login-title {
  font-size: 20px;
  font-weight: 600;
  color: #1d2129;
  text-align: center;
  margin-bottom: 28px;
}

.login-form {
  width: 100%;

  .form-input {
    height: 44px;
    margin-bottom: 20px;
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
  }
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
