const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Treat model artifacts as bundled binary assets (not JS modules).
config.resolver.assetExts.push('onnx', 'task');

module.exports = config;
