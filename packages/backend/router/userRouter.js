const Router = require('koa-router');
const userController = require('../controller/userController');

// 创建路由实例，接口前缀 /api/users
const router = new Router({ prefix: '/api/users' });

// 注册接口
router.post('/register', userController.register);

// 登录接口
router.post('/login', userController.login);

// 获取当前用户信息
router.get('/me', userController.getCurrentUser);

module.exports = router;