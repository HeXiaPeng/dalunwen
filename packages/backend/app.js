require('dotenv').config(); // 加载环境变量
const Koa = require('koa');
const cors = require('@koa/cors'); 
const bodyParser = require('koa-bodyparser');
require('./config/db');
const errorHandler = require('./middleware/errorHandler');
const globalAuth = require('./middleware/globalAuth');
const userRouter = require('./router/userRouter');

// 初始化 Koa 实例
const app = new Koa();
const PORT = process.env.PORT || 3000;

// 全局错误处理中间件
app.use(errorHandler);

// 跨域中间件
app.use(cors({
  origin: process.env.NODE_ENV === 'development' ? 'http://localhost:5173' : '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowHeaders: ['Content-Type', 'Authorization'],
  credentials: false
}));

// 解析请求体（支持 JSON、form-data）
app.use(bodyParser({
  enableTypes: ['json', 'form'],
  jsonLimit: '100mb'
}));

// 全局鉴权中间件
app.use(globalAuth);

//  路由注册（所有用户相关接口前缀 /api/users）
app.use(userRouter.routes()).use(userRouter.allowedMethods());


// 启动服务器
app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
});