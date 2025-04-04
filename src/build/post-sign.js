// Script for codesigning macOS app after packaging
// Codesign using electron-osx-sign
const { sign } = require('@electron/osx-sign');
const path = require('path');
const fs = require('fs');
const appleID = process.env.APPLE_DEV_ID;
const keychainPath = process.env.KEYCHAIN;

exports.default = async function(context) {
  const appPath = context.appOutDir + '/' + context.packager.appInfo.productFilename + '.app';
  const identity = appleID;
  const keychain = keychainPath;
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
    console.log(`App signed at ${appPath}`);
  } catch (error) {
    console.error(error);
    throw error;
  }
};
