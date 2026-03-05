// packages/backend/mcp/agent_service.js
const { Agent, run, setDefaultModelProvider, OpenAIProvider } = require('@openai/agents');
const { allTools } = require('./tools'); // 引入工具集合
const dotenv = require('dotenv');

dotenv.config();

// 1. 配置 AI Provider (适配阿里云 Qwen)
try {
  setDefaultModelProvider(new OpenAIProvider({
    useResponses: false, // 强制使用 Chat Completions API
    apiKey: process.env.ALI_LLM_API_KEY, // 从 .env 读取 ALI_LLM_API_KEY
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", // 阿里云 Base URL
  }));
} catch (e) {
  // 如果 Provider 已经设置过，忽略错误
  // console.log('ModelProvider setup:', e.message);
}

// 2. 封装调用函数
async function processMessage(userMessage) {
  // 每次请求创建一个新的 Agent 实例（无状态）
  const agent = new Agent({
    name: 'SmartAssistant',
    instructions: '你是一个智能助手，可以回答用户的问题。你可以查询天气，也可以查询当前时间。请用中文回答。',
    tools: allTools, // 挂载所有工具
    model: 'qwen-plus',  // 使用阿里云模型
  });
  
  try {
    const result = await run(agent, userMessage);
    return result.finalOutput;
  } catch (error) {
    console.error('Agent execution error:', error);
    throw new Error('AI 服务暂时不可用');
  }
}

module.exports = { processMessage };
