const path = require('path');
module.exports = {
  entry: './lib/es6/src/main.js',
  output: {
    path: path.resolve(__dirname, '..', 'static', 'js'),
    filename: 'main.js'
  }
};
