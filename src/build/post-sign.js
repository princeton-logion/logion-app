// Script for codesigning macOS app after packaging
// Codesign using electron-osx-sign
const { sign } = require('@electron/osx-sign');
const path = require('path');
const fs = require('fs');

exports.default = async function(context) {
  const appPath = context.appOutDir + '/' + context.packager.appInfo.productFilename + '.app';
  const identity = "Developer ID Application: NAME (TEAM_ID)";
  const keychain = "path/to/login/keychain";
  const entitlementsPath = 'path/to/entitlements'

  const signOpts = {
    app: appPath,
    identity: identity,
    type: 'distribution',
    platform: 'darwin',
    hardenedRuntime: false,
    entitlements: entitlementsPath,
    entitlementsInherit: entitlementsPath,
    gatekeeperAssess: false,
    keychain: keychain
  };

  try {
    await sign(signOpts);
    console.log(`Application signed! Signed app at: ${appPath}`);
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};