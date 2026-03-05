// packages/backend/mcp/weather_tool.js
const { tool } = require('@openai/agents');
const { z } = require('zod');

// 定义天气工具
const getWeather = tool({
  name: 'get_weather',
  description: '获取指定地点的当前天气情况',
  parameters: z.object({
    location: z.string().describe('城市名称，例如：上海、北京'),
  }),
  execute: async ({ location }) => {
    // 模拟天气查询逻辑
    console.log(`[MCP Tool] 正在查询 ${location} 的天气...`);
    // 在实际项目中，这里可以调用真实的天气 API
    return `${location} 的天气是晴朗，气温 25°C。`;
  },
});

module.exports = { getWeather };
