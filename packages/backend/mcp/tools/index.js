const { getWeather } = require('./weather');
const { getTime } = require('./time');

// 导出所有工具的数组
// 以后新增工具，只需要在这里引入并加入数组即可
const allTools = [
  getWeather,
  getTime,
];

module.exports = { allTools };
