const OptimizeCssAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const CleanWebpackPlugin = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const merge = require('webpack-merge');
const webpack = require('webpack');
const path = require('path');

const paths = {
  static: path.resolve(__dirname, 'static'),
  build: path.resolve(__dirname, 'build')
}

const htmlConfig = {
  template: path.join(paths.static, 'index.html'),
  minify : {
    collapseWhitespace: true
  }
}

const common = {
  devServer: {
    contentBase: path.join(__dirname, 'build'),
  },
  entry: path.join(paths.static, 'js', 'index.js'),
  resolve: {
    extensions: ['.js', '.jsx']
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /(node_modules)/,
        use: {
          loader: 'babel-loader'
        }
      },
      {
        test: /\.(css)$/,
        use: [ 'style-loader', 'css-loader' ]
      },
      {
        test: /\.(png|jpg|gif|wav|mp3)$/,
        exclude: /(node_modules)/,
        use: [
          {
            loader: 'file-loader',
            options: {}
          }
        ]
      }
    ]
  },
  plugins: [
    new CleanWebpackPlugin([paths.build]),
    new HtmlWebpackPlugin(htmlConfig),
  ]
};

const devSettings = {
  devtool: 'eval-source-map',
  devServer: {
    historyApiFallback: true,
  },
  output: {
    path: paths.build,
    filename: 'bundle.[hash].js',
    publicPath: '/'
  },
  plugins: [
    new webpack.HotModuleReplacementPlugin(),
    new CleanWebpackPlugin([paths.build]),
  ]
}

const prodSettings = {
  output: {
    path: paths.build,
    filename: 'bundle.[hash].js',
    publicPath: '/build/'
  },
  optimization: {
    minimize: true
  },
  plugins: [
    new webpack.DefinePlugin({ 'process.env': {
      NODE_ENV: JSON.stringify('production')
    }}),
    new OptimizeCssAssetsPlugin(),
    new webpack.optimize.OccurrenceOrderPlugin(),
    new webpack.DefinePlugin({'process.env.NODE_ENV': '"production"'})
  ]
}

const TARGET = process.env.npm_lifecycle_event;
process.env.BABEL_ENV = TARGET;

module.exports = env => {
  console.log(env)

  const modeSettings = env.production ? prodSettings : devSettings;
  const interfaceSettings = {
    plugins: [new webpack.DefinePlugin({'process.env.interface': JSON.stringify(env.interface)})]
  };

  return merge(common, modeSettings, interfaceSettings);
}
