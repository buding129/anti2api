import axios from 'axios';
import dns from 'dns';
import http from 'http';
import https from 'https';
import { Readable } from 'stream';
import config from '../config/config.js';

// ==================== DNS & 代理统一配置 ====================

// 强制 IPv4 DNS 解析
function ipv4OnlyLookup(hostname, options, callback) {
  dns.lookup(hostname, { ...options, family: 4 }, callback);
}

// 自定义 DNS 解析：优先 IPv4，失败则回退 IPv6
function preferIPv4Lookup(hostname, options, callback) {
  dns.lookup(hostname, { ...options, family: 4 }, (err4, address4, family4) => {
    if (!err4 && address4) {
      return callback(null, address4, family4);
    }
    dns.lookup(hostname, { ...options, family: 6 }, (err6, address6, family6) => {
      if (!err6 && address6) {
        return callback(null, address6, family6);
      }
      callback(err4 || err6);
    });
  });
}

// 根据配置选择 DNS 解析策略
const lookupFn = config.forceIPv4 ? ipv4OnlyLookup : preferIPv4Lookup;

// 使用选定 DNS 解析策略的 Agent
const httpAgent = new http.Agent({
  lookup: lookupFn,
  keepAlive: true
});

const httpsAgent = new https.Agent({
  lookup: lookupFn,
  keepAlive: true
});

// 统一构建代理配置
function buildProxyConfig() {
  if (!config.proxy) return false;
  try {
    const proxyUrl = new URL(config.proxy);
    return {
      protocol: proxyUrl.protocol.replace(':', ''),
      host: proxyUrl.hostname,
      port: parseInt(proxyUrl.port, 10)
    };
  } catch {
    return false;
  }
}

// 将数据转换为流以启用 chunked 编码
function createChunkedStream(data) {
  const jsonStr = typeof data === 'string' ? data : JSON.stringify(data);
  return Readable.from([jsonStr]);
}

// 为 axios 构建统一请求配置
export function buildAxiosRequestConfig({
  method = 'POST',
  url,
  headers,
  data = null,
  timeout = config.timeout,
  responseType,
  useChunked = false
}) {
  const axiosConfig = {
    method,
    url,
    headers: { ...headers },
    timeout,
    httpAgent,
    httpsAgent,
    proxy: buildProxyConfig(),
    // 禁用自动设置 Content-Length，让 axios 使用 Transfer-Encoding: chunked
    maxContentLength: Infinity,
    maxBodyLength: Infinity
  };

  if (responseType) axiosConfig.responseType = responseType;
  
  if (data !== null) {
    if (useChunked) {
      // 使用流式数据以启用 chunked 编码
      axiosConfig.data = createChunkedStream(data);
      // 删除 Content-Length 头，强制使用 chunked
      delete axiosConfig.headers['Content-Length'];
    } else {
      axiosConfig.data = data;
    }
  }
  return axiosConfig;
}

// 简单封装 axios 调用，方便后续统一扩展（重试、打点等）
export async function httpRequest(configOverrides) {
  // 默认启用 chunked 编码以匹配官方客户端行为
  const axiosConfig = buildAxiosRequestConfig({ ...configOverrides, useChunked: true });
  return axios(axiosConfig);
}

// 流式请求封装
export async function httpStreamRequest(configOverrides) {
  // 默认启用 chunked 编码以匹配官方客户端行为
  const axiosConfig = buildAxiosRequestConfig({ ...configOverrides, useChunked: true });
  axiosConfig.responseType = 'stream';
  return axios(axiosConfig);
}
