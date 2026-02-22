/**
 * SSE 流式响应和心跳机制工具模块
 * 提供统一的流式响应处理、心跳保活、429/503重试等功能
 */

import quotaManager from '../auth/quota_manager.js';
import tokenCooldownManager from '../auth/token_cooldown_manager.js';
import config from '../config/config.js';
import { DEFAULT_HEARTBEAT_INTERVAL, LONG_COOLDOWN_THRESHOLD } from '../constants/index.js';
import logger from '../utils/logger.js';
import memoryManager, { registerMemoryPoolCleanup } from '../utils/memoryManager.js';
import { getGroupKey } from '../utils/modelGroups.js';

// ==================== 心跳机制（防止 CF 超时） ====================
const HEARTBEAT_INTERVAL = config.server.heartbeatInterval || DEFAULT_HEARTBEAT_INTERVAL;
const SSE_HEARTBEAT = Buffer.from(': heartbeat\n\n');

/**
 * 创建心跳定时器
 * @param {Response} res - Express响应对象
 * @returns {NodeJS.Timeout} 定时器
 */
export const createHeartbeat = res => {
  const timer = setInterval(() => {
    if (!res.writableEnded) {
      res.write(SSE_HEARTBEAT);
    } else {
      clearInterval(timer);
    }
  }, HEARTBEAT_INTERVAL);

  // 响应结束时清理
  res.on('close', () => clearInterval(timer));
  res.on('finish', () => clearInterval(timer));

  return timer;
};

// ==================== 预编译的常量字符串（避免重复创建） ====================
const SSE_PREFIX = Buffer.from('data: ');
const SSE_SUFFIX = Buffer.from('\n\n');
const SSE_DONE = Buffer.from('data: [DONE]\n\n');

/**
 * 生成响应元数据
 * @returns {{id: string, created: number}}
 */
export const createResponseMeta = () => ({
  id: `chatcmpl-${Date.now()}`,
  created: Math.floor(Date.now() / 1000),
});

/**
 * 设置流式响应头
 * @param {Response} res - Express响应对象
 */
export const setStreamHeaders = res => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // 禁用 nginx 缓冲
  // 立即发送响应头，确保客户端尽快建立连接
  res.flushHeaders();
};

// ==================== 对象池（减少 GC） ====================
const chunkPool = [];

/**
 * 从对象池获取 chunk 对象
 * @returns {Object}
 */
export const getChunkObject = () => chunkPool.pop() || { choices: [{ index: 0, delta: {}, finish_reason: null }] };

/**
 * 释放 chunk 对象回对象池
 * @param {Object} obj
 */
export const releaseChunkObject = obj => {
  const maxSize = memoryManager.getPoolSizes().chunk;
  if (chunkPool.length < maxSize) chunkPool.push(obj);
};

// 注册内存清理回调
registerMemoryPoolCleanup(chunkPool, () => memoryManager.getPoolSizes().chunk);

/**
 * 获取当前对象池大小（用于监控）
 * @returns {number}
 */
export const getChunkPoolSize = () => chunkPool.length;

/**
 * 清空对象池
 */
export const clearChunkPool = () => {
  chunkPool.length = 0;
};

/**
 * 零拷贝写入流式数据
 * @param {Response} res - Express响应对象
 * @param {Object} data - 要发送的数据
 */
export const writeStreamData = (res, data) => {
  const json = JSON.stringify(data);
  res.write(SSE_PREFIX);
  res.write(json);
  res.write(SSE_SUFFIX);
  // 立即刷新缓冲区，确保数据实时发送给客户端
  if (typeof res.flush === 'function') {
    res.flush();
  }
};

/**
 * 结束流式响应
 * @param {Response} res - Express响应对象
 */
export const endStream = (res, isWriteDone = true) => {
  if (res.writableEnded) return;
  if (isWriteDone) res.write(SSE_DONE);
  res.end();
};

// ==================== 通用重试工具（处理 429/503） ====================

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function parseDurationToMs(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.floor(value));
  if (typeof value !== 'string') return null;

  const s = value.trim();
  if (!s) return null;

  // e.g. "295.285334ms"
  const msMatch = s.match(/^(\d+(\.\d+)?)\s*ms$/i);
  if (msMatch) return Math.max(0, Math.floor(Number(msMatch[1])));

  // e.g. "0.295285334s"
  const secMatch = s.match(/^(\d+(\.\d+)?)\s*s$/i);
  if (secMatch) return Math.max(0, Math.floor(Number(secMatch[1]) * 1000));

  // plain number in string: treat as ms
  const num = Number(s);
  if (Number.isFinite(num)) return Math.max(0, Math.floor(num));
  return null;
}

function tryParseJson(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  if (typeof value !== 'string') return null;
  try {
    return JSON.parse(value);
  } catch {
    // Some messages embed JSON inside a string; try to salvage a JSON object substring.
    const first = value.indexOf('{');
    const last = value.lastIndexOf('}');
    if (first !== -1 && last !== -1 && last > first) {
      const sliced = value.slice(first, last + 1);
      try {
        return JSON.parse(sliced);
      } catch {}
    }
    return null;
  }
}

function extractUpstreamErrorBody(error) {
  // UpstreamApiError created by createApiError(...) stores rawBody
  if (error?.isUpstreamApiError && error.rawBody) {
    return tryParseJson(error.rawBody) || error.rawBody;
  }
  // axios-like error
  if (error?.response?.data) {
    return tryParseJson(error.response.data) || error.response.data;
  }
  // fallback: try parse message
  return tryParseJson(error?.message);
}

function getUpstreamRetryDelayMs(error) {
  // Prefer explicit hints from upstream payload (RetryInfo/quotaResetDelay/quotaResetTimeStamp)
  const body = extractUpstreamErrorBody(error);
  const root = body && typeof body === 'object' ? body : null;
  const inner = root?.error || root;
  const details = Array.isArray(inner?.details) ? inner.details : [];

  let bestMs = null;
  for (const d of details) {
    if (!d || typeof d !== 'object') continue;

    // google.rpc.RetryInfo: { retryDelay: "0.295285334s" }
    const retryDelayMs = parseDurationToMs(d.retryDelay);
    if (retryDelayMs !== null) bestMs = bestMs === null ? retryDelayMs : Math.max(bestMs, retryDelayMs);

    // google.rpc.ErrorInfo metadata: { quotaResetDelay: "295.285334ms", quotaResetTimeStamp: "..." }
    const meta = d.metadata && typeof d.metadata === 'object' ? d.metadata : null;
    const quotaResetDelayMs = parseDurationToMs(meta?.quotaResetDelay);
    if (quotaResetDelayMs !== null) bestMs = bestMs === null ? quotaResetDelayMs : Math.max(bestMs, quotaResetDelayMs);

    const ts = meta?.quotaResetTimeStamp;
    if (typeof ts === 'string') {
      const t = Date.parse(ts);
      if (Number.isFinite(t)) {
        const deltaMs = Math.max(0, t - Date.now());
        bestMs = bestMs === null ? deltaMs : Math.max(bestMs, deltaMs);
      }
    }
  }

  // If it's the capacity exhausted case, still retry but avoid hammering.
  const reason = details.find(d => d?.reason)?.reason;
  if (reason === 'MODEL_CAPACITY_EXHAUSTED') {
    bestMs = bestMs === null ? 1000 : Math.max(bestMs, 1000);
  }

  return bestMs;
}

function computeBackoffMs(attempt, explicitDelayMs) {
  // attempt starts from 0 for first call; on first retry attempt=1
  const maxMs = 2_000; // 最多等待 2s，避免单次生成时间过长
  const hasExplicit = Number.isFinite(explicitDelayMs) && explicitDelayMs !== null;
  const baseMs = hasExplicit ? Math.max(0, Math.floor(explicitDelayMs)) : 500;
  const exp = Math.min(maxMs, Math.floor(baseMs * Math.pow(2, Math.max(0, attempt - 1))));

  // Add small jitter to spread bursts (±20%)
  const jitterFactor = 0.8 + Math.random() * 0.4;
  const expJittered = Math.max(0, Math.floor(exp * jitterFactor));

  if (hasExplicit) {
    // Add a small safety buffer to avoid retrying slightly too early
    const buffered = Math.max(0, Math.floor(explicitDelayMs + 50));
    return Math.min(maxMs, Math.max(expJittered, buffered));
  }

  // Fallback: at least 0.5s for the first retry
  return Math.min(maxMs, Math.max(500, expJittered));
}

/**
 * 从 429 错误中提取恢复时间戳（毫秒）
 * @param {Error} error - 错误对象
 * @returns {number|null} 恢复时间戳，如果无法解析返回 null
 */
function getUpstreamResetTimestamp(error) {
  const body = extractUpstreamErrorBody(error);
  const root = body && typeof body === 'object' ? body : null;
  const inner = root?.error || root;
  const details = Array.isArray(inner?.details) ? inner.details : [];

  for (const d of details) {
    if (!d || typeof d !== 'object') continue;
    const meta = d.metadata && typeof d.metadata === 'object' ? d.metadata : null;
    const ts = meta?.quotaResetTimeStamp;
    if (typeof ts === 'string') {
      const t = Date.parse(ts);
      if (Number.isFinite(t)) {
        return t;
      }
    }
  }
  return null;
}

/**
 * 判断错误是否为可重试的临时性错误（429 或 503 容量不足）
 * @param {number} status - HTTP 状态码
 * @param {Error} error - 错误对象
 * @returns {boolean}
 */
function isRetryableError(status, error) {
  // 429 Rate Limit 总是可重试
  if (status === 429) return true;

  // 503 需要检查是否为容量不足错误
  if (status === 503) {
    const body = extractUpstreamErrorBody(error);
    const root = body && typeof body === 'object' ? body : null;
    const inner = root?.error || root;
    const details = Array.isArray(inner?.details) ? inner.details : [];

    // 检查是否包含 MODEL_CAPACITY_EXHAUSTED
    for (const d of details) {
      if (d?.reason === 'MODEL_CAPACITY_EXHAUSTED') {
        return true;
      }
    }
  }

  return false;
}

/**
 * 带 429/503 重试的执行器
 * @param {Function} fn - 要执行的异步函数，接收 attempt 参数
 * @param {number} maxRetries - 最大重试次数
 * @param {Object} options - 可选参数
 * @param {string} options.loggerPrefix - 日志前缀
 * @param {Function} options.onAttempt - 每次尝试时的回调（用于记录请求次数）
 * @param {string} options.tokenId - Token ID（用于模型系列禁用）
 * @param {string} options.modelId - 模型 ID（用于模型系列禁用）
 * @param {Function} options.refreshQuota - 刷新额度的回调函数（当需要获取准确恢复时间时调用）
 * @returns {Promise<any>}
 */
export async function with429Retry(fn, maxRetries, options = {}, legacyOnAttempt = null) {
  // 兼容旧版调用方式：with429Retry(fn, maxRetries, loggerPrefix, onAttempt)
  let loggerPrefix = '';
  let onAttempt = null;
  let onRetry = null;
  let tokenId = null; // string | (() => string)
  let modelId = null; // string | (() => string)
  let refreshQuota = null;

  if (typeof options === 'string') {
    // 旧版调用方式
    loggerPrefix = options;
    onAttempt = legacyOnAttempt;
  } else if (typeof options === 'object' && options !== null) {
    loggerPrefix = options.loggerPrefix || '';
    onAttempt = options.onAttempt || null;
    onRetry = options.onRetry || null;
    tokenId = options.tokenId || null;
    modelId = options.modelId || null;
    refreshQuota = options.refreshQuota || null;
  }

  const resolveMaybeFn = value => (typeof value === 'function' ? value() : value);

  const retries = Number.isFinite(maxRetries) && maxRetries > 0 ? Math.floor(maxRetries) : 0;
  const cooldownThreshold = config.quota?.longCooldownThreshold || LONG_COOLDOWN_THRESHOLD;
  let attempt = 0;

  // 首次执行 + 最多 retries 次重试
  while (true) {
    try {
      // 每次尝试时调用回调（用于记录请求次数）
      if (typeof onAttempt === 'function') {
        onAttempt(attempt);
      }
      return await fn(attempt);
    } catch (error) {
      // 兼容多种错误格式：error.status, error.statusCode, error.response?.status
      const status = Number(error.status || error.statusCode || error.response?.status);

      if (isRetryableError(status, error)) {
        const explicitDelayMs = getUpstreamRetryDelayMs(error);
        const upstreamResetTimestamp = getUpstreamResetTimestamp(error);
        const errorType = status === 503 ? '503 (容量不足)' : '429';

        // 检查是否是长时间冷却（额度耗尽）- 仅 429 触发模型系列禁用
        const resolvedTokenId = resolveMaybeFn(tokenId);
        const resolvedModelId = resolveMaybeFn(modelId);

        if (
          status === 429 &&
          explicitDelayMs !== null &&
          explicitDelayMs >= cooldownThreshold &&
          resolvedTokenId &&
          resolvedModelId
        ) {
          // 恢复时间超过阈值，触发模型系列禁用
          let finalResetTimestamp = upstreamResetTimestamp;

          // 尝试从 quotas.json 获取更准确的恢复时间
          const { resetTime: quotaResetTime, hasData } = quotaManager.getModelGroupResetTime(
            resolvedTokenId,
            resolvedModelId,
          );

          if (!hasData && typeof refreshQuota === 'function') {
            // 没有额度数据，尝试刷新
            logger.info(`${loggerPrefix}正在获取最新额度数据以确定准确恢复时间...`);
            try {
              await refreshQuota();
              const refreshed = quotaManager.getModelGroupResetTime(resolvedTokenId, resolvedModelId);
              if (refreshed.resetTime) {
                finalResetTimestamp = refreshed.resetTime;
              }
            } catch (e) {
              logger.warn(`${loggerPrefix}获取额度数据失败: ${e.message}`);
            }
          } else if (quotaResetTime) {
            // 使用 quotas.json 中的恢复时间（通常更准确）
            finalResetTimestamp = quotaResetTime;
          }

          if (finalResetTimestamp && finalResetTimestamp > Date.now()) {
            const groupKey = getGroupKey(resolvedModelId);
            const resetDate = new Date(finalResetTimestamp);
            logger.warn(
              `${loggerPrefix}收到 ${errorType}，恢复时间 ${Math.round(explicitDelayMs / 1000 / 60)} 分钟后，` +
                `超过阈值(${Math.round(cooldownThreshold / 1000 / 60)}分钟)，` +
                `禁用 ${groupKey} 系列直到 ${resetDate.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' })}`,
            );
            tokenCooldownManager.setCooldown(resolvedTokenId, resolvedModelId, finalResetTimestamp);
            // 不重试，直接抛出错误
            throw error;
          }
        }

        // 短时间等待，正常重试（重试前可切换 token）
        if (attempt < retries) {
          const nextAttempt = attempt + 1;
          const waitMs = computeBackoffMs(nextAttempt, explicitDelayMs);
          logger.warn(
            `${loggerPrefix}收到 ${errorType}，等待 ${waitMs}ms 后进行第 ${nextAttempt} 次重试（共 ${retries} 次）` +
              (explicitDelayMs !== null ? `（上游提示≈${explicitDelayMs}ms）` : ''),
          );

          const tasks = [sleep(waitMs)];
          if (typeof onRetry === 'function') {
            tasks.push(
              Promise.resolve()
                .then(() => onRetry({ attempt: nextAttempt, status, error, explicitDelayMs }))
                .catch(e => {
                  logger.warn(`${loggerPrefix}重试前切换 token 失败: ${e.message}`);
                }),
            );
          }

          await Promise.all(tasks);
          attempt = nextAttempt;
          continue;
        }
      }
      throw error;
    }
  }
}
