/**
 * Gemini CLI 格式转换工具
 * 将 OpenAI/Gemini/Claude 格式转换为 Gemini API 原生格式
 *
 * 与 Antigravity 转换器的区别：
 * 1. 不需要 project、requestId、sessionId 等字段（这些在 geminicli_client.js 中添加）
 * 2. 使用标准 Gemini API 格式
 * 3. 复用 thoughtSignature 处理逻辑
 */

import config from '../../config/config.js';
import {
  normalizeClaudeParameters,
  normalizeGeminiParameters,
  normalizeOpenAIParameters,
  toGenerationConfig,
} from '../parameterNormalizer.js';
import { convertGeminiToolsToAntigravity } from '../toolConverter.js';
import {
  cleanParameters,
  getThoughtSignatureForModel,
  getToolSignatureForModel,
  isEnableThinking,
  sanitizeToolName,
} from '../utils.js';
import { createFunctionCallPart, createThoughtPart, getSignatureContext, processToolName } from './common.js';

// ==================== Gemini CLI 模型名称处理 ====================

/**
 * 功能前缀列表
 */
const FEATURE_PREFIXES = ['假流式/', '流式抗截断/'];

/**
 * 检查是否是假流式模型
 * @param {string} modelName - 模型名称
 * @returns {boolean}
 */
export function isFakeStreamingModel(modelName) {
  return modelName.startsWith('假流式/');
}

/**
 * 检查是否是流式抗截断模型
 * @param {string} modelName - 模型名称
 * @returns {boolean}
 */
export function isAntiTruncationModel(modelName) {
  return modelName.startsWith('流式抗截断/');
}

/**
 * 从功能模型名称中提取基础模型名称
 * @param {string} modelName - 模型名称（可能包含功能前缀和后缀）
 * @returns {string} 基础模型名称
 */
export function getBaseModelName(modelName) {
  let baseName = modelName;

  // 移除功能前缀
  for (const prefix of FEATURE_PREFIXES) {
    if (baseName.startsWith(prefix)) {
      baseName = baseName.slice(prefix.length);
      break;
    }
  }

  return baseName;
}

/**
 * 检查模型是否启用最大思考模式
 * @param {string} modelName - 模型名称
 * @returns {boolean}
 */
export function isMaxThinkingModel(modelName) {
  return modelName.includes('-maxthinking');
}

/**
 * 检查模型是否禁用思考模式
 * @param {string} modelName - 模型名称
 * @returns {boolean}
 */
export function isNoThinkingModel(modelName) {
  return modelName.includes('-nothinking');
}

/**
 * 检查模型是否启用搜索功能
 * @param {string} modelName - 模型名称
 * @returns {boolean}
 */
export function isSearchModel(modelName) {
  return modelName.includes('-search');
}

/**
 * 获取实际的 API 模型名称（移除所有功能前缀和后缀）
 * @param {string} modelName - 模型名称
 * @returns {string} 实际的 API 模型名称
 */
export function getActualApiModelName(modelName) {
  let actualName = getBaseModelName(modelName);

  // 移除功能后缀
  actualName = actualName
    .replace(/-maxthinking/g, '')
    .replace(/-nothinking/g, '')
    .replace(/-search/g, '');

  return actualName;
}

/**
 * 提取消息内容（文本和图片）
 * @param {Object|string|Array} content - 消息内容
 * @returns {Object} { text, images }
 */
function extractContent(content) {
  if (typeof content === 'string') {
    return { text: content, images: [] };
  }

  if (Array.isArray(content)) {
    let text = '';
    const images = [];

    for (const part of content) {
      if (part.type === 'text') {
        text += part.text || '';
      } else if (part.type === 'image_url') {
        const imageUrl = part.image_url?.url || '';
        if (imageUrl.startsWith('data:')) {
          // Base64 图片
          const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            images.push({
              inlineData: {
                mimeType: match[1],
                data: match[2],
              },
            });
          }
        } else {
          // URL 图片 - Gemini API 可能不直接支持，转为 fileData
          images.push({
            fileData: {
              mimeType: 'image/jpeg',
              fileUri: imageUrl,
            },
          });
        }
      }
    }

    return { text, images };
  }

  return { text: '', images: [] };
}

// 官方推荐的虚拟签名，用于跳过签名验证（最后的回退）
// 参考: gcli2api/src/converter/gemini_fix.py
const SKIP_THOUGHT_SIGNATURE_VALIDATOR = 'skip_thought_signature_validator';

/**
 * 获取 GeminiCLI 的签名上下文（确保始终有签名）
 * 优先级：缓存签名 > 硬编码签名 > 虚拟签名
 * @param {string} actualModelName - 实际模型名称
 * @param {boolean} hasTools - 是否有工具
 * @returns {Object} 签名上下文
 */
function getGeminiCliSignatureContext(actualModelName, hasTools) {
  // 1. 先尝试从缓存获取（真实签名）
  const cached = getSignatureContext(null, actualModelName, hasTools);

  // 如果有缓存签名，直接返回
  if (cached.reasoningSignature || cached.toolSignature) {
    return cached;
  }

  // 2. 尝试使用硬编码的签名（可能是之前缓存的有效签名）
  const reasoningSignature = getThoughtSignatureForModel(actualModelName);
  const toolSignature = hasTools ? getToolSignatureForModel(actualModelName) : reasoningSignature;

  // 如果硬编码签名存在且不为空，使用它们
  if (reasoningSignature || toolSignature) {
    return {
      reasoningSignature: reasoningSignature || toolSignature,
      reasoningContent: ' ',
      toolSignature: toolSignature || reasoningSignature,
      toolContent: ' ',
    };
  }

  // 3. 最后的回退：使用官方推荐的虚拟签名来跳过验证
  // 这是 gcli2api 使用的方式，参考 gemini_fix.py 第 286 行
  return {
    reasoningSignature: SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    reasoningContent: ' ',
    toolSignature: SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    toolContent: ' ',
  };
}

/**
 * 将 OpenAI 消息转换为 Gemini 格式（支持 thoughtSignature）
 * @param {Array} messages - OpenAI 格式的消息数组
 * @param {boolean} enableThinking - 是否启用思考模式
 * @param {string} actualModelName - 实际模型名称
 * @param {boolean} hasTools - 是否有工具
 * @returns {Object} { contents, systemInstruction }
 */
function convertMessages(messages, enableThinking = false, actualModelName = '', hasTools = false) {
  const contents = [];
  let systemInstruction = null;

  // 获取签名上下文
  // 注意：GeminiCLI 的工具调用始终需要签名，无论是否启用思考模式
  const needSignature = enableThinking || hasTools;
  const signatureContext = needSignature ? getGeminiCliSignatureContext(actualModelName, hasTools) : {};
  const { reasoningSignature, reasoningContent, toolSignature, toolContent } = signatureContext;

  for (const msg of messages) {
    const role = msg.role;

    if (role === 'system') {
      // 系统消息
      const extracted = extractContent(msg.content);
      if (!systemInstruction) {
        systemInstruction = { role: 'user', parts: [] };
      }
      if (extracted.text) {
        systemInstruction.parts.push({ text: extracted.text });
      }
      systemInstruction.parts.push(...extracted.images);
    } else if (role === 'user') {
      // 用户消息
      const extracted = extractContent(msg.content);
      const parts = [];
      if (extracted.text) {
        parts.push({ text: extracted.text });
      }
      parts.push(...extracted.images);
      contents.push({ role: 'user', parts });
    } else if (role === 'assistant') {
      // 助手消息
      const parts = [];

      // 处理 reasoning_content（DeepSeek 格式的思考内容）
      if (enableThinking && msg.reasoning_content) {
        const signature = reasoningSignature || toolSignature;
        if (signature) {
          parts.push(createThoughtPart(msg.reasoning_content, signature));
        }
      } else if (enableThinking) {
        // 没有思考内容但启用了思考模式，添加缓存的签名
        const signature = reasoningSignature || toolSignature;
        const content = signature === reasoningSignature ? reasoningContent : toolContent;
        if (signature) {
          parts.push(createThoughtPart(content || ' ', signature));
        }
      }

      // 处理文本内容
      if (msg.content) {
        const extracted = extractContent(msg.content);
        if (extracted.text) {
          parts.push({ text: extracted.text });
        }
        parts.push(...extracted.images);
      }

      // 处理工具调用
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        for (const toolCall of msg.tool_calls) {
          if (toolCall.type === 'function') {
            const func = toolCall.function;
            let args = {};
            try {
              args = typeof func.arguments === 'string' ? JSON.parse(func.arguments) : func.arguments;
            } catch {
              args = { query: func.arguments };
            }

            const safeName = processToolName(func.name, null, actualModelName);
            // 工具调用始终需要签名（无论是否启用思考模式）
            const signature = toolSignature || reasoningSignature || SKIP_THOUGHT_SIGNATURE_VALIDATOR;
            parts.push(createFunctionCallPart(toolCall.id, safeName, args, signature));
          }
        }
      }

      if (parts.length > 0) {
        contents.push({ role: 'model', parts });
      }
    } else if (role === 'tool') {
      // 工具响应
      const toolCallId = msg.tool_call_id;
      let functionName = msg.name || '';

      // 如果没有提供函数名，尝试从之前的消息中查找
      if (!functionName && toolCallId) {
        for (let i = contents.length - 1; i >= 0; i--) {
          const content = contents[i];
          if (content.role === 'model') {
            for (const part of content.parts) {
              if (part.functionCall && part.functionCall.id === toolCallId) {
                functionName = part.functionCall.name;
                break;
              }
            }
          }
          if (functionName) break;
        }
      }

      const functionResponse = {
        functionResponse: {
          id: toolCallId,
          name: sanitizeToolName(functionName),
          response: { output: msg.content || '' },
        },
      };

      // 合并到最后一个 user 消息（如果存在且包含 functionResponse）
      const lastContent = contents[contents.length - 1];
      if (lastContent?.role === 'user' && lastContent.parts.some(p => p.functionResponse)) {
        lastContent.parts.push(functionResponse);
      } else {
        contents.push({ role: 'user', parts: [functionResponse] });
      }
    }
  }

  return { contents, systemInstruction };
}

/**
 * 将 OpenAI 工具转换为 Gemini 格式
 * @param {Array} tools - OpenAI 格式的工具数组
 * @returns {Array} Gemini 格式的工具数组
 */
function convertTools(tools) {
  if (!tools || tools.length === 0) return [];

  const declarations = tools.map(tool => {
    const func = tool.function || {};
    const rawParams = func.parameters || {};
    const cleanedParams = cleanParameters(rawParams) || {};

    const isRefSchema =
      config.fixGeminiSchemaRef === true &&
      cleanedParams &&
      typeof cleanedParams === 'object' &&
      !Array.isArray(cleanedParams) &&
      typeof cleanedParams.ref === 'string' &&
      cleanedParams.ref.trim();

    if (!isRefSchema) {
      if (cleanedParams.type === undefined) cleanedParams.type = 'OBJECT';
      else if (cleanedParams.type === 'object') cleanedParams.type = 'OBJECT';
      if (
        (cleanedParams.type === 'OBJECT' || cleanedParams.type === 'object') &&
        cleanedParams.properties === undefined
      ) {
        cleanedParams.properties = {};
      }
    }

    return {
      name: sanitizeToolName(func.name),
      description: func.description || '',
      parameters: cleanedParams,
    };
  });

  return [
    {
      functionDeclarations: declarations,
    },
  ];
}

/**
 * 构建 Gemini CLI 系统提示词
 * 注意：GeminiCLI 不添加官方系统提示词，只使用用户提供的系统提示词
 * @param {Object|string} systemInstruction - 从消息中提取的系统指令
 * @returns {Object|null} 系统指令对象
 */
function buildGeminiCliSystemInstruction(systemInstruction) {
  // 提取用户的系统提示词文本
  let userSystemPrompt = null;
  if (systemInstruction && systemInstruction.parts) {
    userSystemPrompt = systemInstruction.parts
      .map(p => p.text || '')
      .filter(t => t.trim())
      .join('\n\n');
  } else if (typeof systemInstruction === 'string') {
    userSystemPrompt = systemInstruction;
  }

  // GeminiCLI 不添加官方系统提示词，只使用用户提供的
  if (!userSystemPrompt || !userSystemPrompt.trim()) {
    return null;
  }

  return {
    role: 'user',
    parts: [{ text: userSystemPrompt.trim() }],
  };
}

/**
 * 将 OpenAI 格式请求转换为 Gemini CLI API 格式
 * @param {Object} openaiRequest - OpenAI 格式的请求体
 * @returns {Object} { geminiRequest, model, features }
 */
export function convertOpenAIToGeminiCli(openaiRequest) {
  const { model, messages, tools, temperature, top_p, max_tokens, stream, ...rest } = openaiRequest;

  // 提取功能特性
  const features = {
    fakeStreaming: isFakeStreamingModel(model),
    antiTruncation: isAntiTruncationModel(model),
    maxThinking: isMaxThinkingModel(model),
    noThinking: isNoThinkingModel(model),
    search: isSearchModel(model),
  };

  // 获取实际的 API 模型名称
  const actualModelName = getActualApiModelName(model);

  // 判断是否启用思考模式
  let enableThinking;
  if (features.noThinking) {
    enableThinking = false;
  } else if (features.maxThinking) {
    enableThinking = true;
  } else {
    enableThinking = isEnableThinking(actualModelName);
  }

  // 转换工具（需要在转换消息前完成，以便判断 hasTools）
  const geminiTools = convertTools(tools);
  const hasTools = geminiTools.length > 0;

  // 转换消息（传入签名相关参数）
  const { contents, systemInstruction } = convertMessages(messages || [], enableThinking, actualModelName, hasTools);

  // 规范化参数
  const normalizedParams = normalizeOpenAIParameters({
    temperature,
    top_p,
    max_tokens,
    ...rest,
  });

  // 生成 generationConfig
  const generationConfig = toGenerationConfig(normalizedParams, enableThinking, actualModelName);

  // 构建 Gemini CLI 请求体
  const geminiRequest = {
    contents,
    generationConfig,
  };

  // 添加系统指令
  const finalSystemInstruction = buildGeminiCliSystemInstruction(systemInstruction);
  if (finalSystemInstruction) {
    geminiRequest.systemInstruction = finalSystemInstruction;
  }

  // 添加工具
  if (hasTools) {
    geminiRequest.tools = geminiTools;
    geminiRequest.toolConfig = {
      functionCallingConfig: {
        mode: 'AUTO',
      },
    };
  }

  // 如果启用搜索功能，添加 Google Search 工具
  if (features.search) {
    if (!geminiRequest.tools) {
      geminiRequest.tools = [];
    }
    geminiRequest.tools.push({
      googleSearch: {},
    });
  }

  return {
    geminiRequest,
    model: actualModelName,
    features,
  };
}

/**
 * 处理 Gemini model 消息中的 thought 和签名
 * @param {Object} content - model 消息内容
 * @param {string} reasoningSignature - 思维签名
 * @param {string} reasoningContent - 思维内容
 * @param {string} toolSignature - 工具签名
 * @param {string} toolContent - 工具内容
 * @param {boolean} enableThinking - 是否启用思考模式
 */
function processGeminiModelThoughts(
  content,
  reasoningSignature,
  reasoningContent,
  toolSignature,
  toolContent,
  enableThinking,
) {
  const parts = content.parts;
  const fallbackSig = reasoningSignature || toolSignature;
  const fallbackContent = fallbackSig === reasoningSignature ? reasoningContent || ' ' : toolContent || ' ';

  // 非思考模型：仅为 inlineData 自动补签名
  if (!enableThinking) {
    if (!fallbackSig) return;
    for (const part of parts) {
      if (part.inlineData && !part.thoughtSignature) {
        part.thoughtSignature = fallbackSig;
      }
    }
    return;
  }

  const isStandaloneSignaturePart = part =>
    part &&
    part.thoughtSignature &&
    !part.thought &&
    !part.functionCall &&
    !part.functionResponse &&
    !part.text &&
    !part.inlineData;

  // 查找 thought 和独立 thoughtSignature 的位置
  let thoughtIndex = -1;
  let signatureIndex = -1;
  let signatureValue = null;

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (part.thought === true && !part.thoughtSignature) {
      thoughtIndex = i;
    }
    if (isStandaloneSignaturePart(part)) {
      signatureIndex = i;
      signatureValue = part.thoughtSignature;
    }
  }

  // 合并或添加 thought 和签名
  if (thoughtIndex !== -1 && signatureIndex !== -1) {
    parts[thoughtIndex].thoughtSignature = signatureValue;
    parts.splice(signatureIndex, 1);
  } else if (thoughtIndex !== -1 && signatureIndex === -1) {
    if (fallbackSig) parts[thoughtIndex].thoughtSignature = fallbackSig;
  } else if (thoughtIndex === -1 && fallbackSig) {
    // 只有在有签名时才添加 thought part
    parts.unshift(createThoughtPart(fallbackContent, fallbackSig));
  }

  // 收集独立的签名 parts（用于 functionCall）
  const standaloneSignatures = [];
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (isStandaloneSignaturePart(part)) {
      standaloneSignatures.unshift({ index: i, signature: part.thoughtSignature });
    }
  }

  // 为 functionCall / inlineData 分配签名
  let sigIndex = 0;
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part.thoughtSignature && (part.functionCall || part.inlineData)) {
      if (sigIndex < standaloneSignatures.length) {
        part.thoughtSignature = standaloneSignatures[sigIndex].signature;
        sigIndex++;
        continue;
      }

      const partFallback = part.functionCall
        ? toolSignature || reasoningSignature
        : reasoningSignature || toolSignature;
      if (partFallback) part.thoughtSignature = partFallback;
    }
  }

  // 移除已使用的独立签名 parts
  for (let i = standaloneSignatures.length - 1; i >= 0; i--) {
    if (i < sigIndex) {
      parts.splice(standaloneSignatures[i].index, 1);
    }
  }
}

/**
 * 将 Gemini 格式请求直接转换为 Gemini CLI API 格式
 * @param {Object} geminiRequest - Gemini 格式的请求体
 * @param {string} modelName - 模型名称
 * @returns {Object} { geminiRequest, model, features }
 */
export function convertGeminiToGeminiCli(geminiRequest, modelName) {
  // 提取功能特性
  const features = {
    fakeStreaming: isFakeStreamingModel(modelName),
    antiTruncation: isAntiTruncationModel(modelName),
    maxThinking: isMaxThinkingModel(modelName),
    noThinking: isNoThinkingModel(modelName),
    search: isSearchModel(modelName),
  };

  const actualModelName = getActualApiModelName(modelName);

  // 判断是否启用思考模式
  let enableThinking;
  if (features.noThinking) {
    enableThinking = false;
  } else if (features.maxThinking) {
    enableThinking = true;
  } else {
    enableThinking = isEnableThinking(actualModelName);
  }
  // 深拷贝请求
  const request = JSON.parse(JSON.stringify(geminiRequest));

  // 处理工具
  const hasTools = request.tools && request.tools.length > 0;
  if (hasTools) {
    // 转换工具格式（如果需要）
    request.tools = convertGeminiToolsToAntigravity(request.tools, null, actualModelName);
  }

  // 获取签名上下文并处理 model 消息中的 thought（GeminiCLI 必须确保有签名）
  if (enableThinking && request.contents && Array.isArray(request.contents)) {
    const { reasoningSignature, reasoningContent, toolSignature, toolContent } = getGeminiCliSignatureContext(
      actualModelName,
      hasTools,
    );

    for (const content of request.contents) {
      if (content.role === 'model' && content.parts && Array.isArray(content.parts)) {
        processGeminiModelThoughts(
          content,
          reasoningSignature,
          reasoningContent,
          toolSignature,
          toolContent,
          enableThinking,
        );
      }
    }
  }
  // 规范化 generationConfig
  if (request.generationConfig) {
    const normalizedParams = normalizeGeminiParameters(request.generationConfig);
    request.generationConfig = toGenerationConfig(normalizedParams, enableThinking, actualModelName);
  } else {
    request.generationConfig = toGenerationConfig({}, enableThinking, actualModelName);
  }
  // 移除不需要的字段
  delete request.safetySettings;

  // 添加工具配置
  if (hasTools && !request.toolConfig) {
    request.toolConfig = {
      functionCallingConfig: {
        mode: 'AUTO',
      },
    };
  }

  // 处理系统指令
  if (request.systemInstruction) {
    request.systemInstruction = buildGeminiCliSystemInstruction(request.systemInstruction);
  }

  // 如果启用搜索功能，添加 Google Search 工具
  if (features.search) {
    if (!request.tools) {
      request.tools = [];
    }
    request.tools.push({
      googleSearch: {},
    });
  }

  // 移除 request 中的 model 字段（model 应该在外层，不在 request 内部）
  // 参考 gcli2api 的实现：request 只包含 contents, generationConfig, tools 等
  delete request.model;

  return {
    geminiRequest: request,
    model: actualModelName,
    features,
  };
}

/**
 * 从 Claude 内容中提取文本和图片
 * @param {string|Array} content - Claude 消息内容
 * @returns {Object} { text, images }
 */
function extractClaudeContent(content) {
  const result = { text: '', images: [] };
  if (typeof content === 'string') {
    result.text = content;
    return result;
  }
  if (Array.isArray(content)) {
    for (const item of content) {
      if (item.type === 'text') {
        result.text += item.text || '';
      } else if (item.type === 'image') {
        const source = item.source;
        if (source && source.type === 'base64' && source.data) {
          result.images.push({
            inlineData: {
              mimeType: source.media_type || 'image/png',
              data: source.data,
            },
          });
        }
      }
    }
  }
  return result;
}

/**
 * 将 Claude 工具转换为 Gemini 格式
 * @param {Array} tools - Claude 格式的工具数组
 * @returns {Array} Gemini 格式的工具数组
 */
function convertClaudeTools(tools) {
  if (!tools || tools.length === 0) return [];

  const declarations = tools.map(tool => {
    const rawParams = tool.input_schema || {};
    const cleanedParams = cleanParameters(rawParams) || {};

    const isRefSchema =
      config.fixGeminiSchemaRef === true &&
      cleanedParams &&
      typeof cleanedParams === 'object' &&
      !Array.isArray(cleanedParams) &&
      typeof cleanedParams.ref === 'string' &&
      cleanedParams.ref.trim();

    if (!isRefSchema) {
      if (cleanedParams.type === undefined) cleanedParams.type = 'OBJECT';
      else if (cleanedParams.type === 'object') cleanedParams.type = 'OBJECT';
      if (
        (cleanedParams.type === 'OBJECT' || cleanedParams.type === 'object') &&
        cleanedParams.properties === undefined
      ) {
        cleanedParams.properties = {};
      }
    }

    return {
      name: sanitizeToolName(tool.name),
      description: tool.description || '',
      parameters: cleanedParams,
    };
  });

  return [
    {
      functionDeclarations: declarations,
    },
  ];
}

/**
 * 将 Claude 消息转换为 Gemini 格式
 * @param {Array} messages - Claude 格式的消息数组
 * @param {boolean} enableThinking - 是否启用思考模式
 * @param {string} actualModelName - 实际模型名称
 * @param {boolean} hasTools - 是否有工具
 * @returns {Array} Gemini 格式的 contents 数组
 */
function convertClaudeMessages(messages, enableThinking = false, actualModelName = '', hasTools = false) {
  const contents = [];

  // 获取签名上下文
  // 注意：GeminiCLI 的工具调用始终需要签名，无论是否启用思考模式
  const needSignature = enableThinking || hasTools;
  const signatureContext = needSignature ? getGeminiCliSignatureContext(actualModelName, hasTools) : {};
  const { reasoningSignature, reasoningContent, toolSignature, toolContent } = signatureContext;

  for (const msg of messages) {
    const role = msg.role;

    if (role === 'user') {
      const content = msg.content;

      // 检查是否包含 tool_result
      if (Array.isArray(content) && content.some(item => item.type === 'tool_result')) {
        // 处理工具结果
        for (const item of content) {
          if (item.type !== 'tool_result') continue;

          const toolUseId = item.tool_use_id;
          let functionName = '';

          // 从之前的消息中查找函数名
          for (let i = contents.length - 1; i >= 0; i--) {
            if (contents[i].role === 'model') {
              for (const part of contents[i].parts) {
                if (part.functionCall && part.functionCall.id === toolUseId) {
                  functionName = part.functionCall.name;
                  break;
                }
              }
            }
            if (functionName) break;
          }

          let resultContent = '';
          if (typeof item.content === 'string') {
            resultContent = item.content;
          } else if (Array.isArray(item.content)) {
            resultContent = item.content
              .filter(c => c.type === 'text')
              .map(c => c.text)
              .join('');
          }

          const functionResponse = {
            functionResponse: {
              id: toolUseId,
              name: functionName,
              response: { output: resultContent },
            },
          };

          const lastContent = contents[contents.length - 1];
          if (lastContent?.role === 'user' && lastContent.parts.some(p => p.functionResponse)) {
            lastContent.parts.push(functionResponse);
          } else {
            contents.push({ role: 'user', parts: [functionResponse] });
          }
        }
      } else {
        // 普通用户消息
        const extracted = extractClaudeContent(content);
        const parts = [];
        if (extracted.text) {
          parts.push({ text: extracted.text });
        }
        parts.push(...extracted.images);
        contents.push({ role: 'user', parts });
      }
    } else if (role === 'assistant') {
      const parts = [];
      let thinkingContent = '';
      let messageSignature = null;
      const toolCalls = [];
      let textContent = '';

      if (typeof msg.content === 'string') {
        textContent = msg.content;
      } else if (Array.isArray(msg.content)) {
        for (const item of msg.content) {
          if (item.type === 'text') {
            textContent += item.text || '';
          } else if (item.type === 'thinking') {
            // Claude thinking block
            if (item.thinking) thinkingContent += item.thinking;
            if (!messageSignature && item.signature) messageSignature = item.signature;
          } else if (item.type === 'tool_use') {
            const safeName = processToolName(item.name, null, actualModelName);
            // 工具调用始终需要签名（无论是否启用思考模式）
            const signature = item.signature || toolSignature || reasoningSignature || SKIP_THOUGHT_SIGNATURE_VALIDATOR;
            toolCalls.push(createFunctionCallPart(item.id, safeName, item.input || {}, signature));
          }
        }
      }

      // 添加思考内容
      if (enableThinking) {
        const signature = messageSignature || reasoningSignature || toolSignature;
        if (signature) {
          let reasoningText = ' ';
          if (thinkingContent.length > 0) {
            reasoningText = thinkingContent;
          } else if (signature === reasoningSignature) {
            reasoningText = reasoningContent || ' ';
          } else if (signature === toolSignature) {
            reasoningText = toolContent || ' ';
          }
          parts.push(createThoughtPart(reasoningText, signature));
        }
      }

      // 添加文本内容
      if (textContent && textContent.trim()) {
        parts.push({ text: textContent.trimEnd() });
      }

      // 添加工具调用
      parts.push(...toolCalls);

      if (parts.length > 0) {
        contents.push({ role: 'model', parts });
      }
    }
  }

  return contents;
}

/**
 * 将 Claude 格式请求转换为 Gemini CLI API 格式
 * @param {Object} claudeRequest - Claude 格式的请求体
 * @returns {Object} { geminiRequest, model, features }
 */
export function convertClaudeToGeminiCli(claudeRequest) {
  const { model, messages, tools, system, max_tokens, temperature, top_p, top_k, ...rest } = claudeRequest;

  // 提取功能特性
  const features = {
    fakeStreaming: isFakeStreamingModel(model),
    antiTruncation: isAntiTruncationModel(model),
    maxThinking: isMaxThinkingModel(model),
    noThinking: isNoThinkingModel(model),
    search: isSearchModel(model),
  };

  const actualModelName = getActualApiModelName(model);

  // 判断是否启用思考模式
  let enableThinking;
  if (features.noThinking) {
    enableThinking = false;
  } else if (features.maxThinking) {
    enableThinking = true;
  } else {
    enableThinking = isEnableThinking(actualModelName);
  }

  // 转换工具
  const geminiTools = convertClaudeTools(tools);
  const hasTools = geminiTools.length > 0;

  // 转换消息
  const contents = convertClaudeMessages(messages || [], enableThinking, actualModelName, hasTools);

  // 规范化参数
  const normalizedParams = normalizeClaudeParameters({
    max_tokens,
    temperature,
    top_p,
    top_k,
    ...rest,
  });

  // 生成 generationConfig
  const generationConfig = toGenerationConfig(normalizedParams, enableThinking, actualModelName);

  // 构建 Gemini CLI 请求体
  const geminiRequest = {
    contents,
    generationConfig,
  };

  // 添加系统指令
  const finalSystemInstruction = buildGeminiCliSystemInstruction(system);
  if (finalSystemInstruction) {
    geminiRequest.systemInstruction = finalSystemInstruction;
  }

  // 添加工具
  if (hasTools) {
    geminiRequest.tools = geminiTools;
    geminiRequest.toolConfig = {
      functionCallingConfig: {
        mode: 'AUTO',
      },
    };
  }

  // 如果启用搜索功能，添加 Google Search 工具
  if (features.search) {
    if (!geminiRequest.tools) {
      geminiRequest.tools = [];
    }
    geminiRequest.tools.push({
      googleSearch: {},
    });
  }

  return {
    geminiRequest,
    model: actualModelName,
    features,
  };
}

/**
 * 检测请求格式类型
 * @param {Object} request - 请求体
 * @returns {string} 'openai' | 'gemini' | 'claude'
 */
export function detectRequestFormat(request) {
  // Claude 格式特征：有 messages 数组，工具使用 input_schema
  if (request.messages && Array.isArray(request.messages)) {
    // 检查是否有 Claude 特有的字段
    if (request.system !== undefined || (request.tools && request.tools[0]?.input_schema)) {
      return 'claude';
    }
    // OpenAI 格式
    return 'openai';
  }

  // Gemini 格式特征：有 contents 数组
  if (request.contents && Array.isArray(request.contents)) {
    return 'gemini';
  }

  // 默认为 OpenAI 格式
  return 'openai';
}

/**
 * 统一转换入口：自动检测格式并转换为 Gemini CLI 格式
 * @param {Object} request - 请求体（OpenAI/Gemini/Claude 格式）
 * @param {string} modelName - 模型名称（可选，用于 Gemini 格式）
 * @returns {Object} { geminiRequest, model, features, sourceFormat }
 */
export function convertToGeminiCli(request, modelName = null) {
  const format = detectRequestFormat(request);

  let result;
  switch (format) {
    case 'claude':
      result = convertClaudeToGeminiCli(request);
      break;
    case 'gemini':
      result = convertGeminiToGeminiCli(request, modelName || request.model || 'gemini-2.5-pro');
      break;
    case 'openai':
    default:
      result = convertOpenAIToGeminiCli(request);
      break;
  }

  return {
    ...result,
    sourceFormat: format,
  };
}
