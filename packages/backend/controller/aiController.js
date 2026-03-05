const aiClient = require("../utils/aiClient");
const { ALI_LLM_CONFIG } = require("../config/aiConfig");
const { processMessage } = require("../mcp/agent_service");

// Initialize OpenAI client with Aliyun endpoint
// const openai = new OpenAI({
//   apiKey: "sk-a5dd51f65116498ea6270abb19c4d979", // Ideally from env
//   baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
// });

async function generateProtocol(ctx) {
  try {
    const { messages } = ctx.request.body;

    if (!messages || !Array.isArray(messages)) {
      ctx.status = 400;
      ctx.body = { code: 400, msg: "Invalid messages format" };
      return;
    }

    const completion = await aiClient.chat.completions.create({
      model: ALI_LLM_CONFIG.MODEL, // Using qwen-max as seen in previous analysis files
      messages: messages,
    });

    const content = completion.choices[0].message.content;
    console.log("AI Generation Content:", content);

    ctx.status = 200;
    ctx.body = {
      code: 200,
      msg: "Success",
      data: content,
    };
  } catch (error) {
    console.error("AI Generation Error:", error);
    ctx.status = 500;
    ctx.body = {
      code: 500,
      msg: `Generation failed: ${error.message}`,
    };
  }
}

async function chatWithAgent(ctx) {
  try {
    const { message } = ctx.request.body;

    if (!message) {
      ctx.status = 400;
      ctx.body = { code: 400, msg: "Message is required" };
      return;
    }

    const reply = await processMessage(message);

    ctx.status = 200;
    ctx.body = {
      code: 200,
      msg: "Success",
      data: reply,
    };
  } catch (error) {
    console.error("Agent Error:", error);
    ctx.status = 500;
    ctx.body = {
      code: 500,
      msg: `Agent failed: ${error.message}`,
    };
  }
}

module.exports = {
  generateProtocol,
  chatWithAgent,
};
