<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();

const isAuthPage = computed(() => Boolean(route.meta?.isLoginPage));
</script>

<template>
  <div id="app">
    <div v-if="isAuthPage" class="auth-layout">
      <div class="auth-layout__content">
        <router-view />
      </div>
    </div>
    <router-view v-else />
  </div>
</template>

<style scoped>
.auth-layout {
  min-height: 100vh;
  padding: 32px 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  background-color: #070b1d;
  background-image:
    url("https://coreva-normal.trae.ai/api/ide/v1/text_to_image?prompt=high-end%20futuristic%20medical%20technology%20background%2C%20abstract%20gradient%20waves%2C%20soft%20bokeh%2C%20subtle%20hexagon%20grid%2C%20dna%20helix%20silhouette%2C%20cyan%20blue%20purple%2C%20cinematic%20lighting%2C%20depth%20of%20field%2C%20ultra%20detailed%2C%204k%2C%20no%20text%2C%20no%20logo%2C%20clean&image_size=landscape_16_9");
  background-size: cover;
  background-position: center;
}

.auth-layout::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(1200px 600px at 10% 10%, rgba(76, 110, 245, 0.55) 0%, rgba(76, 110, 245, 0) 55%),
    radial-gradient(900px 520px at 90% 20%, rgba(13, 230, 204, 0.35) 0%, rgba(13, 230, 204, 0) 60%),
    radial-gradient(900px 700px at 55% 90%, rgba(170, 0, 255, 0.28) 0%, rgba(170, 0, 255, 0) 62%),
    linear-gradient(180deg, rgba(7, 11, 29, 0.65) 0%, rgba(7, 11, 29, 0.82) 55%, rgba(7, 11, 29, 0.92) 100%);
  pointer-events: none;
  z-index: 0;
}

.auth-layout::after {
  content: '';
  position: absolute;
  inset: -40% -30% -40% -30%;
  background:
    radial-gradient(circle at 25% 35%, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 42%),
    radial-gradient(circle at 70% 60%, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0) 48%);
  filter: blur(28px);
  opacity: 0.55;
  animation: authFloat 16s ease-in-out infinite;
  pointer-events: none;
  z-index: 0;
}

.auth-layout__content {
  position: relative;
  width: 100%;
  display: flex;
  justify-content: center;
  z-index: 1;
}

@keyframes authFloat {
  0% {
    transform: translate3d(-2%, -1%, 0) rotate(0deg);
  }
  50% {
    transform: translate3d(2%, 1.5%, 0) rotate(2deg);
  }
  100% {
    transform: translate3d(-2%, -1%, 0) rotate(0deg);
  }
}

@media (max-width: 640px) {
  .auth-layout {
    padding: 20px 12px;
  }
}
</style>
