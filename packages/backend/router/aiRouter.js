const Router = require('koa-router');
const aiController = require('../controller/aiController');

const router = new Router({ prefix: '/api/ai' });

// Generate clinical trial protocol
router.post('/generate', aiController.generateProtocol);

// Chat with Agent (MCP enabled)
router.post('/agent', aiController.chatWithAgent);

module.exports = router;

