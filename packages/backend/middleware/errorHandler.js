const errMsg = require('../config/errMsg');
/**
 * 错误处理中间件，用于捕获并处理应用程序中的错误，同时兜底所有异常。
 * @param {Object} ctx - Koa 上下文对象
 * @param {Function} next - 下一个中间件函数
 */
module.exports = async (ctx, next) => {
  try {
    await next();
    if (ctx.status === 404) {
      ctx.status = 404;
      ctx.body = { code: 404, message: '接口不存在' };
    }
  } catch (err) {
    let { statusCode, code, message } = errMsg[err.message] || errMsg.DEFAULT;
    ctx.status = statusCode || 500; 
    ctx.body = {
      code: code,
      message: message || '服务器内部错误'
    };

    console.error('请求错误:', {
      errMsg: err.message,
      method: ctx.method,
      status: ctx.status,
      code: code,
      path: ctx.path,
      error: err.stack,
    });
  }
};