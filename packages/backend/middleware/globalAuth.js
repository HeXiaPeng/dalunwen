const jwt = require('jsonwebtoken');
const User = require('../model/User');

// 路由白名单：无需登录即可访问的接口
const WHITE_LIST = [
  '/api/users/register', // 注册接口
  '/api/users/login'     // 登录接口
  // 后续新增公开接口，直接添加到这里即可
];

// 全局鉴权中间件：所有请求都会经过
module.exports = async (ctx, next) => {
  try {
    // 1. 判断当前请求路径是否在白名单 → 白名单内直接放行
    const currentPath = ctx.path;
    
    if (WHITE_LIST.includes(currentPath)) {
      return await next();
    }

    // 2. 不在白名单 → 执行 Token 验证（原 auth 中间件逻辑）
    const authHeader = ctx.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new Error('UNAUTHORIZED');
    }

    // 解析 Token
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    // 验证用户是否存在
    const user = await User.findByPk(decoded.userId);
    if (!user) throw new Error('USER_NOT_FOUND');
    
    // 挂载用户信息
    ctx.user = {
      userId: user.id,
      username: user.username
    };

    await next();
  } catch (err) {
    throw err;
  }
};
