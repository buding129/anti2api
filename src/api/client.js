import tokenManager from '../auth/token_manager.js';
import config from '../config/config.js';
import AntigravityRequester from '../AntigravityRequester.js';
import { saveBase64Image } from '../utils/imageStorage.js';
import logger from '../utils/logger.js';
import memoryManager from '../utils/memoryManager.js';
import { httpRequest, httpStreamRequest } from '../utils/httpClient.js';
import { MODEL_LIST_CACHE_TTL } from '../constants/index.js';
import { createApiError } from '../utils/errors.js';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import {
  getLineBuffer,
  releaseLineBuffer,
  parseAndEmitStreamChunk,
  convertToToolCall,
  registerStreamMemoryCleanup
} from './stream_parser.js';
import { setSignature, shouldCacheSignature, isImageModel } from '../utils/thoughtSignatureCache.js';

// 请求客户端：优先使用 AntigravityRequester，失败则自动降级到 axios
let requester = null;
let useAxios = false;

// 初始化请求客户端
if (config.useNativeAxios === true) {
  useAxios = true;
  logger.info('使用原生 axios 请求');
} else {
  try {
    requester = new AntigravityRequester();
  } catch (error) {
    logger.warn('AntigravityRequester 初始化失败，自动降级使用 axios:', error.message);
    useAxios = true;
  }
}

// ==================== 调试：最终请求/原始响应完整输出（单文件追加模式） ====================
const DEBUG_DUMP_FILE = path.join(process.cwd(), 'data', 'debug-dump.log');

function isDebugDumpEnabled() {
  return config.debugDumpRequestResponse === true;
}

// 确保目录存在
let dumpDirEnsured = false;
async function ensureDumpDir() {
  if (dumpDirEnsured) return;
  await fs.mkdir(path.dirname(DEBUG_DUMP_FILE), { recursive: true });
  dumpDirEnsured = true;
}

// 生成时间戳
function getTimestamp() {
  const now = new Date();
  const pad2 = (n) => String(n).padStart(2, '0');
  const pad3 = (n) => String(n).padStart(3, '0');
  return `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())} ` +
    `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}.${pad3(now.getMilliseconds())}`;
}

// 生成唯一请求 ID
function createDumpId(prefix = 'dump') {
  const rand = Math.random().toString(16).slice(2, 10);
  return `${prefix}-${rand}`;
}

// 追加写入日志文件
async function appendDumpLog(content) {
  await ensureDumpDir();
  await fs.appendFile(DEBUG_DUMP_FILE, content, 'utf8');
}

// 创建流式响应收集器
function createStreamCollector() {
  return { chunks: [] };
}

// 收集流式响应块
function collectStreamChunk(collector, chunk) {
  if (collector) collector.chunks.push(chunk);
}

// 写入请求体
async function dumpFinalRequest(dumpId, requestBody) {
  if (!isDebugDumpEnabled()) return;
  try {
    const json = JSON.stringify(requestBody, null, 2);
    const header = `\n${'='.repeat(80)}\n[${getTimestamp()}] REQUEST ${dumpId}\n${'='.repeat(80)}\n`;
    await appendDumpLog(header + json + '\n');
    logger.warn(`[DEBUG_DUMP ${dumpId}] 已写入请求体到: ${DEBUG_DUMP_FILE}`);
  } catch (e) {
    logger.error(`[DEBUG_DUMP ${dumpId}] 写入请求体失败:`, e?.message || e);
  }
}

// 写入流式响应（将所有 chunk 解析为 JSON 数组）
async function dumpStreamResponse(dumpId, collector) {
  if (!isDebugDumpEnabled() || !collector) return;
  try {
    const header = `\n${'-'.repeat(80)}\n[${getTimestamp()}] RESPONSE ${dumpId} (STREAM)\n${'-'.repeat(80)}\n`;
    
    // 解析 SSE 格式的流式响应，提取 JSON 数据
    const rawContent = collector.chunks.join('');
    const jsonObjects = [];
    const lines = rawContent.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) {
        const dataStr = trimmed.slice(5).trim();
        if (dataStr && dataStr !== '[DONE]') {
          try {
            const parsed = JSON.parse(dataStr);
            jsonObjects.push(parsed);
          } catch {
            // 非 JSON 数据，保留原始内容
            jsonObjects.push({ raw: dataStr });
          }
        }
      }
    }
    
    // 以 JSON 数组格式写入
    const jsonOutput = JSON.stringify(jsonObjects, null, 2);
    const footer = `\n[${getTimestamp()}] END ${dumpId}\n`;
    
    await appendDumpLog(header + jsonOutput + footer);
    logger.warn(`[DEBUG_DUMP ${dumpId}] 响应记录完成 (${jsonObjects.length} 条数据)`);
  } catch (e) {
    logger.error(`[DEBUG_DUMP ${dumpId}] 写入流式响应失败:`, e?.message || e);
  }
}

// 写入非流式响应（一次性写入完整响应）
async function dumpFinalRawResponse(dumpId, rawText, ext = 'txt') {
  if (!isDebugDumpEnabled()) return;
  try {
    const header = `\n${'-'.repeat(80)}\n[${getTimestamp()}] RESPONSE ${dumpId} (NO-STREAM)\n${'-'.repeat(80)}\n`;
    const footer = `\n[${getTimestamp()}] END ${dumpId}\n`;
    await appendDumpLog(header + (rawText ?? '') + footer);
    logger.warn(`[DEBUG_DUMP ${dumpId}] 响应记录完成`);
  } catch (e) {
    logger.error(`[DEBUG_DUMP ${dumpId}] 写入响应失败:`, e?.message || e);
  }
}

// ==================== 模型列表缓存（智能管理） ====================
const getModelCacheTTL = () => {
  return config.cache?.modelListTTL || MODEL_LIST_CACHE_TTL;
};

let modelListCache = null;
let modelListCacheTime = 0;

// 默认模型列表（当 API 请求失败时使用）
// 使用 Object.freeze 防止意外修改，并帮助 V8 优化
const DEFAULT_MODELS = Object.freeze([
  'claude-opus-4-5',
  'claude-opus-4-5-thinking',
  'claude-sonnet-4-5-thinking',
  'claude-sonnet-4-5',
  'gemini-3-pro-high',
  'gemini-2.5-flash-lite',
  'gemini-3-pro-image',
  'gemini-3-pro-image-4K',
  'gemini-3-pro-image-2K',
  'gemini-2.5-flash-thinking',
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'gemini-3-pro-low',
  'chat_20706',
  'rev19-uic3-1p',
  'gpt-oss-120b-medium',
  'chat_23310'
]);

// 生成默认模型列表响应
function getDefaultModelList() {
  const created = Math.floor(Date.now() / 1000);
  return {
    object: 'list',
    data: DEFAULT_MODELS.map(id => ({
      id,
      object: 'model',
      created,
      owned_by: 'google'
    }))
  };
}


// 注册对象池与模型缓存的内存清理回调
function registerMemoryCleanup() {
  // 由流式解析模块管理自身对象池大小
  registerStreamMemoryCleanup();

  // 统一由内存清理器定时触发：仅清理“已过期”的模型列表缓存
  memoryManager.registerCleanup(() => {
    const ttl = getModelCacheTTL();
    const now = Date.now();
    if (modelListCache && (now - modelListCacheTime) > ttl) {
      modelListCache = null;
      modelListCacheTime = 0;
    }
  });
}

// 初始化时注册清理回调
registerMemoryCleanup();

// ==================== 辅助函数 ====================

function buildHeaders(token) {
  return {
    'Host': config.api.host,
    'User-Agent': config.api.userAgent,
    'Authorization': `Bearer ${token.access_token}`,
    'Content-Type': 'application/json',
    'Accept-Encoding': 'gzip'
  };
}

function buildRequesterConfig(headers, body = null) {
  const reqConfig = {
    method: 'POST',
    headers,
    timeout_ms: config.timeout,
    proxy: config.proxy
  };
  if (body !== null) reqConfig.body = JSON.stringify(body);
  return reqConfig;
}


// 统一错误处理
async function handleApiError(error, token, dumpId = null) {
  const status = error.response?.status || error.status || error.statusCode || 500;
  let errorBody = error.message;
  
  if (error.response?.data?.readable) {
    const chunks = [];
    for await (const chunk of error.response.data) {
      chunks.push(chunk);
    }
    errorBody = Buffer.concat(chunks).toString();
  } else if (typeof error.response?.data === 'object') {
    errorBody = JSON.stringify(error.response.data, null, 2);
  } else if (error.response?.data) {
    errorBody = error.response.data;
  }

  if (dumpId) {
    await dumpFinalRawResponse(dumpId, String(errorBody ?? ''), 'error.txt');
  }
  
  if (status === 403) {
    if (JSON.stringify(errorBody).includes("The caller does not")){
      throw createApiError(`超出模型最大上下文。错误详情: ${errorBody}`, status, errorBody);
    }
    tokenManager.disableCurrentToken(token);
    throw createApiError(`该账号没有使用权限，已自动禁用。错误详情: ${errorBody}`, status, errorBody);
  }
  
  throw createApiError(`API请求失败 (${status}): ${errorBody}`, status, errorBody);
}


// ==================== 导出函数 ====================

export async function generateAssistantResponse(requestBody, token, callback) {
  
  const headers = buildHeaders(token);
  const dumpId = isDebugDumpEnabled() ? createDumpId('stream') : null;
  const streamCollector = dumpId ? createStreamCollector() : null;
  if (dumpId) {
    await dumpFinalRequest(dumpId, requestBody);
  }

  // 在 state 中临时缓存思维链签名，供流式多片段复用，并携带 session 与 model 信息以写入全局缓存
  const state = {
    toolCalls: [],
    reasoningSignature: null,
    sessionId: requestBody.request?.sessionId,
    model: requestBody.model
  };
  const lineBuffer = getLineBuffer(); // 从对象池获取
  
  const processChunk = (chunk) => {
    // 收集流式响应用于后续 JSON 格式化输出
    collectStreamChunk(streamCollector, chunk);
    const lines = lineBuffer.append(chunk);
    for (let i = 0; i < lines.length; i++) {
      parseAndEmitStreamChunk(lines[i], state, callback);
    }
  };
  
  try {
    if (useAxios) {
      const response = await httpStreamRequest({
        method: 'POST',
        url: config.api.url,
        headers,
        data: requestBody
      });
      
      // 使用 Buffer 直接处理，避免 toString 的内存分配
      response.data.on('data', chunk => {
        processChunk(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
      });
      
      await new Promise((resolve, reject) => {
        response.data.on('end', () => {
          releaseLineBuffer(lineBuffer); // 归还到对象池
          resolve();
        });
        response.data.on('error', reject);
      });
    } else {
      const streamResponse = requester.antigravity_fetchStream(config.api.url, buildRequesterConfig(headers, requestBody));
      let errorBody = '';
      let statusCode = null;

      await new Promise((resolve, reject) => {
        streamResponse
          .onStart(({ status }) => { statusCode = status; })
          .onData((chunk) => {
            if (statusCode !== 200) {
              errorBody += chunk;
              // 错误响应也收集
              collectStreamChunk(streamCollector, chunk);
            } else {
              processChunk(chunk);
            }
          })
          .onEnd(() => {
            releaseLineBuffer(lineBuffer); // 归还到对象池
            if (statusCode !== 200) {
              reject({ status: statusCode, message: errorBody });
            } else {
              resolve();
            }
          })
          .onError(reject);
      });
    }

    // 流式响应结束后，以 JSON 格式写入日志
    if (dumpId) {
      await dumpStreamResponse(dumpId, streamCollector);
    }
  } catch (error) {
    releaseLineBuffer(lineBuffer); // 确保归还
    await handleApiError(error, token, dumpId);
  }
}

// 内部工具：从远端拉取完整模型原始数据
async function fetchRawModels(headers, token) {
  try {
    if (useAxios) {
      const response = await httpRequest({
        method: 'POST',
        url: config.api.modelsUrl,
        headers,
        data: {}
      });
      return response.data;
    }
    const response = await requester.antigravity_fetch(config.api.modelsUrl, buildRequesterConfig(headers, {}));
    if (response.status !== 200) {
      const errorBody = await response.text();
      throw { status: response.status, message: errorBody };
    }
    return await response.json();
  } catch (error) {
    await handleApiError(error, token);
  }
}

export async function getAvailableModels() {
  // 检查缓存是否有效（动态 TTL）
  const now = Date.now();
  const ttl = getModelCacheTTL();
  if (modelListCache && (now - modelListCacheTime) < ttl) {
    return modelListCache;
  }
  
  const token = await tokenManager.getToken();
  if (!token) {
    // 没有 token 时返回默认模型列表
    logger.warn('没有可用的 token，返回默认模型列表');
    return getDefaultModelList();
  }
  
  const headers = buildHeaders(token);
  const data = await fetchRawModels(headers, token);
  if (!data) {
    // fetchRawModels 里已经做了统一错误处理，这里兜底为默认列表
    return getDefaultModelList();
  }

  const created = Math.floor(Date.now() / 1000);
  const modelList = Object.keys(data.models || {}).map(id => ({
    id,
    object: 'model',
    created,
    owned_by: 'google'
  }));
  
  // 添加默认模型（如果 API 返回的列表中没有）
  const existingIds = new Set(modelList.map(m => m.id));
  for (const defaultModel of DEFAULT_MODELS) {
    if (!existingIds.has(defaultModel)) {
      modelList.push({
        id: defaultModel,
        object: 'model',
        created,
        owned_by: 'google'
      });
    }
  }
  
  const result = {
    object: 'list',
    data: modelList
  };
  
  // 更新缓存
  modelListCache = result;
  modelListCacheTime = now;
  const currentTTL = getModelCacheTTL();
  logger.info(`模型列表已缓存 (有效期: ${currentTTL / 1000}秒, 模型数量: ${modelList.length})`);
  
  return result;
}

// 清除模型列表缓存（可用于手动刷新）
export function clearModelListCache() {
  modelListCache = null;
  modelListCacheTime = 0;
  logger.info('模型列表缓存已清除');
}

export async function getModelsWithQuotas(token) {
  const headers = buildHeaders(token);
  const data = await fetchRawModels(headers, token);
  if (!data) return {};

  const quotas = {};
  Object.entries(data.models || {}).forEach(([modelId, modelData]) => {
    if (modelData.quotaInfo) {
      quotas[modelId] = {
        r: modelData.quotaInfo.remainingFraction,
        t: modelData.quotaInfo.resetTime
      };
    }
  });
  
  return quotas;
}

export async function generateAssistantResponseNoStream(requestBody, token) {
  
  const headers = buildHeaders(token);
  const dumpId = isDebugDumpEnabled() ? createDumpId('no_stream') : null;
  if (dumpId) await dumpFinalRequest(dumpId, requestBody);
  let data;
  
  try {
    if (useAxios) {
      if (dumpId) {
        const resp = await httpRequest({
          method: 'POST',
          url: config.api.noStreamUrl,
          headers,
          data: requestBody,
          responseType: 'text'
        });
        const rawText = typeof resp.data === 'string' ? resp.data : JSON.stringify(resp.data, null, 2);
        await dumpFinalRawResponse(dumpId, rawText, 'json');
        data = JSON.parse(rawText);
      } else {
        data = (await httpRequest({
          method: 'POST',
          url: config.api.noStreamUrl,
          headers,
          data: requestBody
        })).data;
      }
    } else {
      const response = await requester.antigravity_fetch(config.api.noStreamUrl, buildRequesterConfig(headers, requestBody));
      if (response.status !== 200) {
        const errorBody = await response.text();
        if (dumpId) await dumpFinalRawResponse(dumpId, errorBody, 'txt');
        throw { status: response.status, message: errorBody };
      }
      const rawText = await response.text();
      if (dumpId) await dumpFinalRawResponse(dumpId, rawText, 'json');
      data = JSON.parse(rawText);
    }
  } catch (error) {
    await handleApiError(error, token, dumpId);
  }
  //console.log(JSON.stringify(data));
  // 解析响应内容
  const parts = data.response?.candidates?.[0]?.content?.parts || [];
  let content = '';
  let reasoningContent = '';
  let reasoningSignature = null;
  let lastSeenSignature = null;
  const toolCalls = [];
  const imageUrls = [];
  
  for (const part of parts) {
    if (part.thoughtSignature) {
      lastSeenSignature = part.thoughtSignature;
    }
    if (part.thought === true) {
      // 思维链内容 - 使用 DeepSeek 格式的 reasoning_content
      reasoningContent += part.text || '';
      if (part.thoughtSignature) {
        // 以“最新出现”的签名为准（有些响应会在末尾才给签名）
        reasoningSignature = part.thoughtSignature;
      }
    } else if (part.text !== undefined) {
      content += part.text;
    } else if (part.functionCall) {
      const toolCall = convertToToolCall(part.functionCall, requestBody.request?.sessionId, requestBody.model);
      const sig = part.thoughtSignature || lastSeenSignature || null;
      if (sig) toolCall.thoughtSignature = sig;
      toolCalls.push(toolCall);
    } else if (part.inlineData) {
      // 保存图片到本地并获取 URL
      const imageUrl = saveBase64Image(part.inlineData.data, part.inlineData.mimeType);
      imageUrls.push(imageUrl);
    }
  }

  // 若本轮未在 thought part 上拿到签名，则回退使用“最后出现”的签名（Gemini 等可能只在 functionCall part 上给签名）
  if (!reasoningSignature && lastSeenSignature) {
    reasoningSignature = lastSeenSignature;
  }
  
  // 提取 token 使用统计
  const usage = data.response?.usageMetadata;
  const usageData = usage ? {
    prompt_tokens: usage.promptTokenCount || 0,
    completion_tokens: usage.candidatesTokenCount || 0,
    total_tokens: usage.totalTokenCount || 0
  } : null;
  
  // 将新的签名和思考内容写入全局缓存（按 model），供后续请求兜底使用
  const sessionId = requestBody.request?.sessionId;
  const model = requestBody.model;
  const hasTools = toolCalls.length > 0;
  const isImage = isImageModel(model);
  
  // 判断是否应该缓存签名
  if (sessionId && model && shouldCacheSignature({ hasTools, isImageModel: isImage })) {
    // 获取最终使用的签名（优先使用工具签名，回退到思维签名）
    let finalSignature = reasoningSignature;
    
    // 工具签名：取最后一个带 thoughtSignature 的工具作为缓存源（更接近"最新"）
    if (hasTools) {
      for (let i = toolCalls.length - 1; i >= 0; i--) {
        const sig = toolCalls[i]?.thoughtSignature;
        if (sig) {
          finalSignature = sig;
          break;
        }
      }
    }
    
    if (finalSignature) {
      const cachedContent = reasoningContent || ' ';
      setSignature(sessionId, model, finalSignature, cachedContent, { hasTools, isImageModel: isImage });
    }
  }

  // 生图模型：转换为 markdown 格式
  if (imageUrls.length > 0) {
    let markdown = content ? content + '\n\n' : '';
    markdown += imageUrls.map(url => `![image](${url})`).join('\n\n');
    return { content: markdown, reasoningContent: reasoningContent || null, reasoningSignature, toolCalls, usage: usageData };
  }
  
  return { content, reasoningContent: reasoningContent || null, reasoningSignature, toolCalls, usage: usageData };
}

export async function generateImageForSD(requestBody, token) {
  const headers = buildHeaders(token);
  let data;
  //console.log(JSON.stringify(requestBody,null,2));
  
  try {
    if (useAxios) {
      data = (await httpRequest({
        method: 'POST',
        url: config.api.noStreamUrl,
        headers,
        data: requestBody
      })).data;
    } else {
      const response = await requester.antigravity_fetch(config.api.noStreamUrl, buildRequesterConfig(headers, requestBody));
      if (response.status !== 200) {
        const errorBody = await response.text();
        throw { status: response.status, message: errorBody };
      }
      data = await response.json();
    }
  } catch (error) {
    await handleApiError(error, token);
  }
  
  const parts = data.response?.candidates?.[0]?.content?.parts || [];
  const images = parts.filter(p => p.inlineData).map(p => p.inlineData.data);
  
  return images;
}

export function closeRequester() {
  if (requester) requester.close();
}

// 导出内存清理注册函数（供外部调用）
export { registerMemoryCleanup };
