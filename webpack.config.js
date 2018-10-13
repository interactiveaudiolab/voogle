const HtmlWebPackPlugin = require("html-webpack-plugin");
module.exports = {
  entry: __dirname + '/static/js/index.js',
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader"
        }
      },
      {
        test: /\.html$/,
        use: [
          {
            loader: "html-loader"
          }
        ]
      }
    ]
  },
  plugins: [
    new HtmlWebPackPlugin({
      template: "./static/index.html",
      filename: "index.html"
    })
  ]
};
