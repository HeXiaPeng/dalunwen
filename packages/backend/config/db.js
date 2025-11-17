// config/db.js（新增文件，管理 MySQL 连接）
const { Sequelize } = require('sequelize');

// 从环境变量读取配置
const config = {
  host: process.env.MYSQL_HOST,
  port: process.env.MYSQL_PORT,
  user: process.env.MYSQL_USER,
  password: process.env.MYSQL_PASSWORD,
  database: process.env.MYSQL_DB
};

// 创建 Sequelize 实例（连接 MySQL）
const sequelize = new Sequelize(
  config.database, // 数据库名
  config.user, // 用户名
  config.password, // 密码
  {
    host: config.host,
    port: config.port,
    dialect: 'mysql', // 指定数据库类型（mysql）
    timezone: '+08:00', // 时区（重要！避免时间存储偏差）
    logging: process.env.NODE_ENV === 'development' ? console.log : false // 开发环境打印 SQL 日志
  }
);


// 测试连接
sequelize.authenticate()
  .then(() => console.log('MySQL 连接成功'))
  .catch(err => console.error('MySQL 连接失败：', err));

module.exports = sequelize;