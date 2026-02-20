import { randomUUID } from "crypto";
import { FRONT_END, CLIENT_FEATURS_REGISTER } from "../constants/oauth.js"

function generateCreatedAt() {
  const now = new Date();
  const isoString = now.toISOString();
  const nanos = String(now.getMilliseconds()).padStart(3, '0') + '000000';
  return isoString.replace(/\.\d{3}Z$/, `.${nanos}Z`);
}

export function buildClientRegister(token) {
  return {
    appName: "codeium-language-server",
    instanceId: token.instanceId,
    connectionId: randomUUID(),
    sdkVersion: "unleash-client-go:4.5.0",
    strategies: [
      "default",
      "applicationHostname",
      "gradualRolloutRandom",
      "gradualRolloutSessionId",
      "gradualRolloutUserId",
      "remoteAddress",
      "userWithId",
      "flexibleRollout"
    ],
    started: generateCreatedAt(),
    interval: 60,
    platformVersion: "go1.26-20260115-RC01 cl/856841426 +532e320349 X:boringcrypto,simd",
    platformName: "go",
    yggdrasilVersion: null,
    specVersion: "4.3.1"
  }
}

export function buildFrontEnd(token) {
  return {
    context: {
      sessionId: "",
      appName: "codeium-extension",
      environment: "default",
      userId: token.access_token,
      properties: {
        ide: "antigravity",
        ideVersion: "1.18.3",
        extensionVersion: "",
        disableTelemetry: "false",
        invocationId: randomUUID(),
        devMode: "false"
      }
    }
  }
}

export function buildClientRegisterHeaders(token) {
  const unleashToken = CLIENT_FEATURS_REGISTER[0];
  return {
    "Host": "antigravity-unleash.goog",
    "User-Agent": "codeium-language-server",
    "Content-Length": "562",
    "Authorization": `*:${unleashToken}`,
    "Content-Type": "application/json",
    "Unleash-Appname": "codeium-language-server",
    "Unleash-Connection-Id": randomUUID(),
    "Unleash-Instanceid": `${token.instanceId}`,
    "Unleash-Interval": "60000",
    "Unleash-Sdk": "unleash-client-go:4.5.0",
    "Accept-Encoding": "gzip"
  }
}

export function buildFrontEndHeaders(token) {
  const unleashToken = FRONT_END;
  return {
    "Host": "antigravity-unleash.goog",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Antigravity/1.107.0 Chrome/142.0.7444.175 Electron/39.2.3 Safari/537.36",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US",
    "Authorization": `*:${unleashToken}`,
    "Cache-Control": "max-age=0",
    "Content-Type": "application/json",
    "Origin": "vscode-file://vscode-app",
    "Priority": "u=1, i",
    "Sec-Ch-Ua": `"Not_A Brand";v="99", "Chromium";v="142"`,
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "Windows",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "Unleash-Appname": "codeium-extension",
    "Unleash-Connection-Id": randomUUID(),
    "Unleash-Sdk": "unleash-client-js:3.7.8",
    "Content-Length": "532",
  }
}

export function buildClientFeatrueHeaders(token) {
  const unleashToken = CLIENT_FEATURS_REGISTER[0];
  return {
    "Host": "antigravity-unleash.goog",
    "User-Agent": "codeium-language-server",
    "Authorization": `*:${unleashToken}`,
    "Unleash-Appname": "codeium-language-server",
    "Unleash-Client-Spec": "4.3.1",
    "Unleash-Connection-Id": randomUUID(),
    "Unleash-Instanceid": token.instanceId,
    "Unleash-Interval": "60000",
    "Unleash-Sdk": "unleash-client-go:4.5.0",
    "Accept-Encoding": "gzip",
  }
}
