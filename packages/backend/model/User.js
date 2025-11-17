const { DataTypes } = require('sequelize');
const bcrypt = require('bcryptjs');
const sequelize = require('../config/db');

// 定义 User 模型（对应 MySQL 中的 users 表，Sequelize 会自动 pluralize 表名）
const User = sequelize.define('User', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true,
    comment: '用户ID'
  },
  username: {
    type: DataTypes.STRING(50),
    allowNull: false,
    unique: true,
    trim: true,
    validate: {
      min: [3, '用户名长度不能少于3个字符'],
      max: [20, '用户名长度不能超过20个字符']
    },
    comment: '用户名'
  },
  // 密码（必填、加密存储）
  password: {
    type: DataTypes.STRING(100),
    allowNull: false,
    validate: {
      min: [6, '密码长度不能少于6个字符']
    },
    comment: '密码（加密存储）'
  }
}, {
  tableName: 'users', // 手动指定表名（避免 Sequelize 自动复数化）
  timestamps: true, // 自动添加 createdAt（创建时间）和 updatedAt（更新时间）字段
  underscored: true, // 字段名自动转为下划线格式（如 tokenExpiration → token_expiration）
  hooks: {
    // 密码加密钩子
    beforeSave: async (user) => {
      // 只有密码修改时才加密（避免更新其他字段时重复加密）
      if (user.changed('password')) {
        user.password = await bcrypt.hash(user.password, 10);
      }
    }
  }
});

// 实例方法：验证密码
User.prototype.comparePassword = async function(enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password);
};

// 同步模型到数据库（开发环境：不存在表则创建，存在则不修改；生产环境注释掉）
if (process.env.NODE_ENV === 'development') {
  User.sync({ alter: true }); // alter: true 表示更新表结构（不删除数据）
}

module.exports = User;