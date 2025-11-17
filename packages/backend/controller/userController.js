const jwt = require('jsonwebtoken');
const User = require('../model/User');

// 生成 JWT Token（逻辑不变）
const generateToken = (userId) => {
  return jwt.sign(
    { userId },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN }
  );
};

// 1. 用户注册
exports.register = async (ctx) => {
  const { username, password } = ctx.request.body;

  // 检查用户名是否已存在（Sequelize 语法：findOne + where）
  const existingUser = await User.findOne({ where: { username } });
  if (existingUser) {
    ctx.status = 400;
    throw new Error('用户名已被占用');
  }

  const user = await User.create({ username, password });

  // 生成 Token
  const token = generateToken(user.id); 
  const tokenExpiration = user.token_expiration; // 下划线格式（因模型配置 underscored: true）

  // 返回结果（格式不变，前端无需适配）
  ctx.body = {
    code: 200,
    message: '注册成功',
    data: {
      userId: user.id,
      username: user.username,
      token,
      tokenExpiration
    }
  };
};

// 2. 用户登录（修改数据库操作部分）
exports.login = async (ctx) => {
  const { username, password } = ctx.request.body;

  // 检查用户是否存在（强制查询 password 字段，Sequelize 语法：attributes.include）
  const user = await User.findOne({
    where: { username },
    attributes: { include: ['password'] }
  });
  if (!user) {
    ctx.status = 401;
    throw new Error('USER_NOT_FOUND');
  }
  // 验证密码
  const isPasswordValid = await user.comparePassword(password);
  if (!isPasswordValid) {
    ctx.status = 401;
    throw new Error('USER_NOT_FOUND');
  }

  // 生成新 Token + 更新过期时间
  const token = generateToken(user.id);
  const tokenExpiration = Date.now() + 7 * 24 * 60 * 60 * 1000;

  // 返回结果
  ctx.body = {
    code: 200,
    message: '登录成功',
    data: {
      userId: user.id,
      username: user.username,
      token,
      tokenExpiration
    }
  };
};

// 3. 获取当前用户信息（修改数据库操作部分）
exports.getCurrentUser = async (ctx) => {
  // 从 JWT 解析出 userId，查询用户
  const user = await User.findByPk(ctx.user.userId);
  if (!user) {
    throw new Error('USER_NOT_FOUND');
  }

  ctx.body = {
    code: 200,
    message: '获取用户信息成功',
    data: {
      userId: user.id,
      username: user.username,
      tokenExpiration: user.token_expiration
    }
  };
};