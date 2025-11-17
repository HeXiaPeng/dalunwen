// 业务错误码
module.exports = {
  DEFAULT: {
    statusCode: 500,
    code: 50001,
    message: '服务端错误',
  },
  USER_NOT_FOUND: {
    statusCode: 401,
    code: 40001,
    message: '用户名或密码错误',
  },
  UNAUTHORIZED: {
    statusCode: 401,
    code: 40002,
    message: '未登录，请先登录！',
  },
};
