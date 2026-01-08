import dotenv from 'dotenv';
import fs from 'fs';
import crypto from 'crypto';
import log from '../utils/logger.js';
import { deepMerge } from '../utils/deepMerge.js';
import { getConfigPaths } from '../utils/paths.js';
import { parseEnvFile } from '../utils/envParser.js';
import {
  DEFAULT_SERVER_PORT,
  DEFAULT_SERVER_HOST,
  DEFAULT_HEARTBEAT_INTERVAL,
  DEFAULT_TIMEOUT,
  DEFAULT_RETRY_TIMES,
  DEFAULT_MAX_REQUEST_SIZE,
  DEFAULT_MAX_IMAGES,
  MODEL_LIST_CACHE_TTL,
  DEFAULT_GENERATION_PARAMS,
  MEMORY_CLEANUP_INTERVAL
} from '../constants/index.js';

// 生成随机凭据的缓存
let generatedCredentials = null;
// 生成的 API_KEY 缓存
let generatedApiKey = null;

/**
 * 生成或获取 API_KEY
 * 如果用户未配置，自动生成随机密钥
 */
function getApiKey() {
  const apiKey = process.env.API_KEY;
  
  if (apiKey) {
    return apiKey;
  }
  
  // 生成随机 API_KEY（只生成一次）
  if (!generatedApiKey) {
    generatedApiKey = 'sk-' + crypto.randomBytes(24).toString('hex');
  }
  
  return generatedApiKey;
}

// 是否已显示过凭据提示
let credentialsDisplayed = false;

/**
 * 生成或获取管理员凭据
 * 如果用户未配置，自动生成随机凭据
 */
function getAdminCredentials() {
  const username = process.env.ADMIN_USERNAME;
  const password = process.env.ADMIN_PASSWORD;
  const jwtSecret = process.env.JWT_SECRET;
  
  // 如果全部配置了，直接返回
  if (username && password && jwtSecret) {
    return { username, password, jwtSecret };
  }
  
  // 生成随机凭据（只生成一次）
  if (!generatedCredentials) {
    generatedCredentials = {
      username: username || crypto.randomBytes(8).toString('hex'),
      password: password || crypto.randomBytes(16).toString('base64').replace(/[+/=]/g, ''),
      jwtSecret: jwtSecret || crypto.randomBytes(32).toString('hex')
    };
  }
  
  return generatedCredentials;
}

/**
 * 显示生成的凭据提示（只显示一次）
 */
function displayGeneratedCredentials() {
  if (credentialsDisplayed) return;
  credentialsDisplayed = true;
  
  const username = process.env.ADMIN_USERNAME;
  const password = process.env.ADMIN_PASSWORD;
  const apiKey = process.env.API_KEY;
  const jwtSecret = process.env.JWT_SECRET;
  
  const needsUsername = !username;
  const needsPassword = !password;
  const needsApiKey = !apiKey;
  const needsJwtSecret = !jwtSecret;
  
  // 如果有任何凭据需要生成，显示提示
  if (needsUsername || needsPassword || needsApiKey) {
    const credentials = getAdminCredentials();
    log.warn('═══════════════════════════════════════════════════════════');
    log.warn('⚠️  未配置完整凭据，已自动生成随机凭据：');
    if (needsUsername) {
      log.warn(`    用户名: ${credentials.username}`);
    }
    if (needsPassword) {
      log.warn(`    密码:   ${credentials.password}`);
    }
    if (needsApiKey) {
      log.warn(`    API密钥: ${getApiKey()}`);
    }
    log.warn('═══════════════════════════════════════════════════════════');
    log.warn('⚠️  重启后凭据将重新生成！建议在 .env 文件中配置：');
    if (needsUsername) log.warn('    ADMIN_USERNAME=你的用户名');
    if (needsPassword) log.warn('    ADMIN_PASSWORD=你的密码');
    if (needsApiKey) log.warn('    API_KEY=你的密钥');
    log.warn('═══════════════════════════════════════════════════════════');
  } else if (needsJwtSecret) {
    log.warn('⚠️ 未配置 JWT_SECRET，已生成随机密钥（重启后登录会话将失效）');
  }
}

const { envPath, configJsonPath } = getConfigPaths();

// 默认反代系统提示词
const DEFAULT_SYSTEM_INSTRUCTION = '你是聊天机器人，名字叫萌萌，如同名字这般，你的性格是软软糯糯萌萌哒的，专门为用户提供聊天和情绪价值，协助进行小说创作或者角色扮演';

// 默认官方系统提示词（反重力官方要求的）
const DEFAULT_OFFICIAL_SYSTEM_PROMPT = `You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.**Proactiveness**`;

// 确保 .env 存在（如果缺失则创建带默认配置的文件）
if (!fs.existsSync(envPath)) {
  const defaultEnvContent = `# 敏感配置（只在 .env 中配置）
# 如果不配置以下三项，系统会自动生成随机凭据并在启动时显示
# API_KEY=your-api-key
# ADMIN_USERNAME=your-username
# ADMIN_PASSWORD=your-password
# JWT_SECRET=your-jwt-secret

# 可选配置
# PROXY=http://127.0.0.1:7890

# 反代系统提示词
SYSTEM_INSTRUCTION=${DEFAULT_SYSTEM_INSTRUCTION}

# 官方系统提示词（留空则使用内置默认值）
# OFFICIAL_SYSTEM_PROMPT=

# IMAGE_BASE_URL=http://your-domain.com
`;
  fs.writeFileSync(envPath, defaultEnvContent, 'utf8');
  log.info('✓ 已创建 .env 文件，包含默认反代系统提示词');
}

// 加载 config.json
let jsonConfig = {};
if (fs.existsSync(configJsonPath)) {
  jsonConfig = JSON.parse(fs.readFileSync(configJsonPath, 'utf8'));
}

// 加载 .env（指定路径）
dotenv.config({ path: envPath });

// 处理系统提示词中的转义字符
// dotenv 不会自动将 \n 字符串转换为实际换行符，我们需要手动处理
function processEscapeChars(value) {
  if (!value) return value;
  return value
    .replace(/\\\\n/g, '\n')  // 先处理双重转义 \\n -> 换行
    .replace(/\\n/g, '\n');   // 再处理单重转义 \n -> 换行
}

if (process.env.SYSTEM_INSTRUCTION) {
  process.env.SYSTEM_INSTRUCTION = processEscapeChars(process.env.SYSTEM_INSTRUCTION);
}

if (process.env.OFFICIAL_SYSTEM_PROMPT) {
  process.env.OFFICIAL_SYSTEM_PROMPT = processEscapeChars(process.env.OFFICIAL_SYSTEM_PROMPT);
}

// 对于系统提示词，使用自定义解析器重新加载以支持更复杂的多行格式
// dotenv 的解析可能不够完善，我们用自定义解析器补充
try {
  const customEnv = parseEnvFile(envPath);
  if (customEnv.SYSTEM_INSTRUCTION) {
    let customValue = processEscapeChars(customEnv.SYSTEM_INSTRUCTION);
    // 如果自定义解析器得到的值更长，使用它
    if (customValue.length > (process.env.SYSTEM_INSTRUCTION?.length || 0)) {
      process.env.SYSTEM_INSTRUCTION = customValue;
    }
  }
  if (customEnv.OFFICIAL_SYSTEM_PROMPT) {
    let customValue = processEscapeChars(customEnv.OFFICIAL_SYSTEM_PROMPT);
    // 如果自定义解析器得到的值更长，使用它
    if (customValue.length > (process.env.OFFICIAL_SYSTEM_PROMPT?.length || 0)) {
      process.env.OFFICIAL_SYSTEM_PROMPT = customValue;
    }
  }
} catch (e) {
  // 忽略解析错误，使用 dotenv 的结果
}

// 获取代理配置：优先使用 PROXY，其次使用系统代理环境变量
export function getProxyConfig() {
  // 优先使用显式配置的 PROXY
  if (process.env.PROXY) {
    return process.env.PROXY;
  }
  
  // 检查系统代理环境变量（按优先级）
  const systemProxy = process.env.HTTPS_PROXY ||
                      process.env.https_proxy ||
                      process.env.HTTP_PROXY ||
                      process.env.http_proxy ||
                      process.env.ALL_PROXY ||
                      process.env.all_proxy;
  
  if (systemProxy) {
    log.info(`使用系统代理: ${systemProxy}`);
  }
  
  return systemProxy || null;
}

/**
 * 从 JSON 和环境变量构建配置对象
 * @param {Object} jsonConfig - JSON 配置对象
 * @returns {Object} 完整配置对象
 */
export function buildConfig(jsonConfig) {
  return {
    server: {
      port: jsonConfig.server?.port || DEFAULT_SERVER_PORT,
      host: jsonConfig.server?.host || DEFAULT_SERVER_HOST,
      heartbeatInterval: jsonConfig.server?.heartbeatInterval || DEFAULT_HEARTBEAT_INTERVAL,
      // 内存定时清理频率：避免频繁扫描/GC 带来的性能损耗
      memoryCleanupInterval: jsonConfig.server?.memoryCleanupInterval ?? MEMORY_CLEANUP_INTERVAL
    },
    cache: {
      modelListTTL: jsonConfig.cache?.modelListTTL || MODEL_LIST_CACHE_TTL
    },
    rotation: {
      strategy: jsonConfig.rotation?.strategy || 'round_robin',
      requestCount: jsonConfig.rotation?.requestCount || 10
    },
    imageBaseUrl: process.env.IMAGE_BASE_URL || null,
    maxImages: jsonConfig.other?.maxImages || DEFAULT_MAX_IMAGES,
    api: {
      url: jsonConfig.api?.url || 'https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse',
      modelsUrl: jsonConfig.api?.modelsUrl || 'https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:fetchAvailableModels',
      noStreamUrl: jsonConfig.api?.noStreamUrl || 'https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:generateContent',
      host: jsonConfig.api?.host || 'daily-cloudcode-pa.sandbox.googleapis.com',
      userAgent: jsonConfig.api?.userAgent || 'antigravity/1.13.3 windows/amd64'
    },
    defaults: {
      temperature: jsonConfig.defaults?.temperature ?? DEFAULT_GENERATION_PARAMS.temperature,
      top_p: jsonConfig.defaults?.topP ?? DEFAULT_GENERATION_PARAMS.top_p,
      top_k: jsonConfig.defaults?.topK ?? DEFAULT_GENERATION_PARAMS.top_k,
      max_tokens: jsonConfig.defaults?.maxTokens ?? DEFAULT_GENERATION_PARAMS.max_tokens,
      thinking_budget: jsonConfig.defaults?.thinkingBudget ?? DEFAULT_GENERATION_PARAMS.thinking_budget
    },
    security: {
      maxRequestSize: jsonConfig.server?.maxRequestSize || DEFAULT_MAX_REQUEST_SIZE,
      apiKey: getApiKey()
    },
    admin: getAdminCredentials(),
    useNativeAxios: jsonConfig.other?.useNativeAxios !== false,
    timeout: jsonConfig.other?.timeout || DEFAULT_TIMEOUT,
    retryTimes: Number.isFinite(jsonConfig.other?.retryTimes) ? jsonConfig.other.retryTimes : DEFAULT_RETRY_TIMES,
    proxy: getProxyConfig(),
    // 反代系统提示词（从 .env 读取，可在前端修改，空字符串代表不使用）
    systemInstruction: process.env.SYSTEM_INSTRUCTION ?? '',
    // 官方系统提示词（从 .env 读取，可在前端修改，空字符串代表不使用）
    officialSystemPrompt: process.env.OFFICIAL_SYSTEM_PROMPT ?? '',
    // 官方提示词位置配置：'before' = 官方提示词在反代提示词前面，'after' = 官方提示词在反代提示词后面
    officialPromptPosition: jsonConfig.other?.officialPromptPosition || 'before',
    // 是否合并系统提示词为单个 part，false 则保留多 part 结构（需要先开启 useContextSystemPrompt）
    mergeSystemPrompt: jsonConfig.other?.mergeSystemPrompt !== false,
    skipProjectIdFetch: jsonConfig.other?.skipProjectIdFetch === true,
    useContextSystemPrompt: jsonConfig.other?.useContextSystemPrompt === true,
    passSignatureToClient: jsonConfig.other?.passSignatureToClient === true,
    useFallbackSignature: jsonConfig.other?.useFallbackSignature === true,
    // 签名缓存配置（新版）
    cacheAllSignatures: jsonConfig.other?.cacheAllSignatures === true ||
      process.env.CACHE_ALL_SIGNATURES === '1' ||
      process.env.CACHE_ALL_SIGNATURES === 'true',
    cacheToolSignatures: jsonConfig.other?.cacheToolSignatures !== false,
    cacheImageSignatures: jsonConfig.other?.cacheImageSignatures !== false,
    cacheThinking: jsonConfig.other?.cacheThinking !== false,
    // 假非流：非流式请求使用流式获取数据后返回非流式格式（默认启用）
    fakeNonStream: jsonConfig.other?.fakeNonStream !== false,
    // 调试：完整打印最终请求体与原始响应（可能包含敏感内容/大体积数据，只从环境变量读取）
    debugDumpRequestResponse: process.env.DEBUG_DUMP_REQUEST_RESPONSE === '1'
  };
}

const config = buildConfig(jsonConfig);

// 显示生成的凭据提示
displayGeneratedCredentials();

log.info('✓ 配置加载成功');

export default config;

export function getConfigJson() {
  if (fs.existsSync(configJsonPath)) {
    return JSON.parse(fs.readFileSync(configJsonPath, 'utf8'));
  }
  return {};
}

export function saveConfigJson(data) {
  const existing = getConfigJson();
  const merged = deepMerge(existing, data);
  fs.writeFileSync(configJsonPath, JSON.stringify(merged, null, 2), 'utf8');
}
