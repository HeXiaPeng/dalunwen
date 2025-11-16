import { createApp } from 'vue'
import App from './App.vue'
import '@/assets/css/normalize.css'
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import router from './router';

const app = createApp(App);
app.use(router);
app.mount('#app');
app.use(ElementPlus);

