const { tool } = require('@openai/agents');
const { z } = require('zod');

// 定义时间工具
const getTime = tool({
  name: 'get_current_time',
  description: '获取当前的准确时间',
  parameters: z.object({}), // 无参数
  execute: async () => {
    return new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' });
  },
});

module.exports = { getTime };
