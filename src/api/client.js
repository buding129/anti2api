import crypto from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';
import tokenManager from '../auth/token_manager.js';
import config from '../config/config.js';
import { MODEL_LIST_CACHE_TTL } from '../constants/index.js';
import fingerprintRequester from '../requester.js';
import { createApiError } from '../utils/errors.js';
import { httpRequest } from '../utils/httpClient.js';
import { saveBase64Image } from '../utils/imageStorage.js';
import logger from '../utils/logger.js';
import memoryManager from '../utils/memoryManager.js';
import { isImageModel, setSignature, shouldCacheSignature } from '../utils/thoughtSignatureCache.js';
import {
  collectStreamChunk,
  createDumpId,
  createStreamCollector,
  dumpFinalRawResponse,
  dumpFinalRequest,
  dumpStreamResponse,
  isDebugDumpEnabled,
} from './debugDump.js';
import { parseGeminiCandidateParts, toOpenAIUsage } from './geminiResponseParser.js';
import { postJsonAndParse, runAxiosSseStream, runNativeSseStream } from './geminiTransport.js';
import { convertToToolCall, registerStreamMemoryCleanup } from './stream_parser.js';
import { createStreamLineProcessor } from './streamLineProcessor.js';
import { getUpstreamStatus, isCallerDoesNotHavePermission, readUpstreamErrorBody } from './upstreamError.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 请求客户端：优先使用 FingerprintRequester，失败则自动降级到 axios
let requester = null;
let useAxios = false;

// 初始化请求客户端
if (config.useNativeAxios === true) {
  useAxios = true;
  logger.info('使用原生 axios 请求');
} else {
  try {
    // 使用 src/bin/config.json 作为 TLS 指纹配置文件
    // 检测是否在 pkg 环境中
    const isPkg = typeof process.pkg !== 'undefined';

    // 根据环境选择配置文件路径
    const configPath = isPkg
      ? path.join(path.dirname(process.execPath), 'bin', 'tls_config.json') // pkg 打包环境
      : path.join(__dirname, '..', 'bin', 'tls_config.json'); // 开发环境
    requester = fingerprintRequester.create({
      configPath,
      timeout: config.timeout ? Math.ceil(config.timeout / 1000) : 30,
      proxy: config.proxy || null,
    });
    logger.info('使用 FingerprintRequester 请求');
  } catch (error) {
    logger.warn('FingerprintRequester 初始化失败，自动降级使用 axios:', error.message);
    useAxios = true;
  }
}

// ==================== 调试：最终请求/原始响应完整输出（单文件追加模式） ====================

// ==================== 模型列表缓存（智能管理） ====================
const getModelCacheTTL = () => {
  return config.cache?.modelListTTL || MODEL_LIST_CACHE_TTL;
};

let modelListCache = null;
let modelListCacheTime = 0;

// 默认模型列表（当 API 请求失败时使用）
// 使用 Object.freeze 防止意外修改，并帮助 V8 优化
const DEFAULT_MODELS = Object.freeze([
  'claude-opus-4-6',
  'claude-opus-4-5',
  'claude-opus-4-6-thinking',
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
  'chat_23310',
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
      owned_by: 'google',
    })),
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
    if (modelListCache && now - modelListCacheTime > ttl) {
      modelListCache = null;
      modelListCacheTime = 0;
    }
  });
}

// 初始化时注册清理回调
registerMemoryCleanup();

// ==================== 辅助函数 ====================

function buildHeaders(token, requestBody = null) {
  const uuid =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  const headers = {
    Host: config.api.host,
    'User-Agent': config.api.userAgent,
    Authorization: `Bearer ${token.access_token}`,
    'Content-Type': 'application/json',
    'Accept-Encoding': 'gzip',
    requestId: `req-${uuid}`,
  };

  // Antigravity 会通过 requestType 区分 agent / image_gen
  // 对齐 gcli2api：优先使用请求体里的 requestType，不存在时按 model 兜底推断
  if (requestBody) {
    const requestType =
      requestBody.requestType ||
      (String(requestBody.model || '')
        .toLowerCase()
        .includes('image')
        ? 'image_gen'
        : 'agent');
    headers['requestType'] = requestType;
  }

  return headers;
}

function buildRequesterConfig(headers, body = null) {
  const reqConfig = {
    method: 'POST',
    headers,
    timeout_ms: config.timeout,
    proxy: config.proxy,
  };
  if (body !== null) reqConfig.body = JSON.stringify(body);
  return reqConfig;
}

// 统一错误处理
async function handleApiError(error, token, dumpId = null) {
  const status = getUpstreamStatus(error);
  const errorBody = await readUpstreamErrorBody(error);

  if (dumpId) {
    await dumpFinalRawResponse(dumpId, String(errorBody ?? ''));
  }

  if (status === 403) {
    if (isCallerDoesNotHavePermission(errorBody)) {
      throw createApiError(`超出模型最大上下文。错误详情: ${errorBody}`, status, errorBody);
    }
    tokenManager.disableCurrentToken(token);
    throw createApiError(`该账号没有使用权限，已自动禁用。错误详情: ${errorBody}`, status, errorBody);
  }

  throw createApiError(`API请求失败 (${status}): ${errorBody}`, status, errorBody);
}

// ==================== 导出函数 ====================

export async function generateAssistantResponse(requestBody, token, callback) {
  const headers = buildHeaders(token, requestBody);
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
    model: requestBody.model,
  };
  const processor = createStreamLineProcessor({
    state,
    onEvent: callback,
    onRawChunk: chunk => collectStreamChunk(streamCollector, chunk),
  });

  try {
    if (useAxios) {
      await runAxiosSseStream({
        url: config.api.url,
        headers,
        data: requestBody,
        timeout: config.timeout,
        processor,
      });
    } else {
      const streamResponse = requester.antigravity_fetchStream(
        config.api.url,
        buildRequesterConfig(headers, requestBody),
      );
      await runNativeSseStream({
        streamResponse,
        processor,
        onErrorChunk: chunk => collectStreamChunk(streamCollector, chunk),
      });
    }

    // 流式响应结束后，以 JSON 格式写入日志
    if (dumpId) {
      await dumpStreamResponse(dumpId, streamCollector);
    }
  } catch (error) {
    try {
      processor.close();
    } catch {}
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
        data: {},
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
  if (modelListCache && now - modelListCacheTime < ttl) {
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
    owned_by: 'google',
  }));

  // 添加默认模型（如果 API 返回的列表中没有）
  const existingIds = new Set(modelList.map(m => m.id));
  for (const defaultModel of DEFAULT_MODELS) {
    if (!existingIds.has(defaultModel)) {
      modelList.push({
        id: defaultModel,
        object: 'model',
        created,
        owned_by: 'google',
      });
    }
  }

  const result = {
    object: 'list',
    data: modelList,
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
        t: modelData.quotaInfo.resetTime,
      };
    }
  });

  return quotas;
}

export async function generateAssistantResponseNoStream(requestBody, token) {
  const headers = buildHeaders(token, requestBody);
  const dumpId = isDebugDumpEnabled() ? createDumpId('no_stream') : null;
  if (dumpId) await dumpFinalRequest(dumpId, requestBody);
  let data;
  try {
    data = await postJsonAndParse({
      useAxios,
      requester,
      url: config.api.noStreamUrl,
      headers,
      body: requestBody,
      timeout: config.timeout,
      requesterConfig: buildRequesterConfig(headers, requestBody),
      dumpId,
      dumpFinalRawResponse,
      rawFormat: 'json',
    });
  } catch (error) {
    await handleApiError(error, token, dumpId);
  }
  //console.log(JSON.stringify(data));
  const parts = data.response?.candidates?.[0]?.content?.parts || [];
  const parsed = parseGeminiCandidateParts({
    parts,
    sessionId: requestBody.request?.sessionId,
    model: requestBody.model,
    convertToToolCall,
    saveBase64Image,
  });

  const usageData = toOpenAIUsage(data.response?.usageMetadata);

  // 将新的签名和思考内容写入全局缓存（按 model），供后续请求兜底使用
  const sessionId = requestBody.request?.sessionId;
  const model = requestBody.model;
  const hasTools = parsed.toolCalls.length > 0;
  const isImage = isImageModel(model);

  // 判断是否应该缓存签名
  if (sessionId && model && shouldCacheSignature({ hasTools, isImageModel: isImage })) {
    // 获取最终使用的签名（优先使用工具签名，回退到思维签名）
    let finalSignature = parsed.reasoningSignature;

    // 工具签名：取最后一个带 thoughtSignature 的工具作为缓存源（更接近"最新"）
    if (hasTools) {
      for (let i = parsed.toolCalls.length - 1; i >= 0; i--) {
        const sig = parsed.toolCalls[i]?.thoughtSignature;
        if (sig) {
          finalSignature = sig;
          break;
        }
      }
    }

    if (finalSignature) {
      const cachedContent = parsed.reasoningContent || ' ';
      setSignature(sessionId, model, finalSignature, cachedContent, { hasTools, isImageModel: isImage });
    }
  }

  // 生图模型：按配置返回 base64 或 markdown
  if (parsed.imageUrls.length > 0) {
    const prefixText = parsed.content ? parsed.content + '\n\n' : '';

    // 开关：直接返回 data:image/...;base64,...（不落盘；用 Markdown 图片语法便于界面直接渲染）
    if (config.imageReturnBase64 === true) {
      return {
        content: prefixText + parsed.imageUrls.map(url => `![image](${url})`).join('\n\n'),
        reasoningContent: parsed.reasoningContent,
        reasoningSignature: parsed.reasoningSignature,
        toolCalls: parsed.toolCalls,
        usage: usageData,
      };
    }

    let markdown = prefixText;
    markdown += parsed.imageUrls.map(url => `![image](${url})`).join('\n\n');
    return {
      content: markdown,
      reasoningContent: parsed.reasoningContent,
      reasoningSignature: parsed.reasoningSignature,
      toolCalls: parsed.toolCalls,
      usage: usageData,
    };
  }

  return {
    content: parsed.content,
    reasoningContent: parsed.reasoningContent,
    reasoningSignature: parsed.reasoningSignature,
    toolCalls: parsed.toolCalls,
    usage: usageData,
  };
}

export async function generateImageForSD(requestBody, token) {
  const headers = buildHeaders(token, requestBody);
  let data;
  //console.log(JSON.stringify(requestBody,null,2));

  try {
    if (useAxios) {
      data = (
        await httpRequest({
          method: 'POST',
          url: config.api.noStreamUrl,
          headers,
          data: requestBody,
        })
      ).data;
    } else {
      const response = await requester.antigravity_fetch(
        config.api.noStreamUrl,
        buildRequesterConfig(headers, requestBody),
      );
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
