import fs from 'fs';

/**
 * 解析 .env 文件内容为对象
 * 支持多行字符串（用双引号或单引号包裹）
 */
export function parseEnvFile(filePath) {
  const envData = {};
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  
  let currentKey = null;
  let currentValue = '';
  let inMultiline = false;
  let quoteChar = null;
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    
    if (inMultiline) {
      // 继续收集多行值
      currentValue += '\n' + line;
      // 检查是否结束（以引号结尾）
      if (line.trimEnd().endsWith(quoteChar)) {
        // 移除结尾引号
        currentValue = currentValue.slice(0, -1);
        envData[currentKey] = currentValue;
        inMultiline = false;
        currentKey = null;
        currentValue = '';
        quoteChar = null;
      }
    } else {
      line = line.trim();
      if (!line || line.startsWith('#')) continue;
      
      const eqIndex = line.indexOf('=');
      if (eqIndex === -1) continue;
      
      const key = line.slice(0, eqIndex).trim();
      let value = line.slice(eqIndex + 1);
      
      // 检查是否是引号开头的多行字符串
      const trimmedValue = value.trimStart();
      if ((trimmedValue.startsWith('"') || trimmedValue.startsWith("'")) &&
          !trimmedValue.endsWith(trimmedValue[0])) {
        // 多行字符串开始
        quoteChar = trimmedValue[0];
        currentKey = key;
        currentValue = trimmedValue.slice(1); // 移除开头引号
        inMultiline = true;
      } else {
        // 单行值，移除可能的引号
        value = value.trim();
        if ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        envData[key] = value;
      }
    }
  }
  
  // 处理未闭合的多行字符串
  if (inMultiline && currentKey) {
    envData[currentKey] = currentValue;
  }
  
  return envData;
}

// 需要始终用双引号包裹的字段（避免特殊字符和换行问题）
const QUOTE_REQUIRED_KEYS = [
  'SYSTEM_INSTRUCTION',
  'OFFICIAL_SYSTEM_PROMPT'
];

/**
 * 更新 .env 文件中的键值对
 * 支持多行字符串（自动用双引号包裹）
 */
export function updateEnvFile(filePath, updates) {
  let content = fs.readFileSync(filePath, 'utf8');
  
  Object.entries(updates).forEach(([key, value]) => {
    let formattedValue = value;
    
    if (typeof value === 'string') {
      // 对于需要引号的字段或包含换行符的值，用双引号包裹
      if (QUOTE_REQUIRED_KEYS.includes(key) || value.includes('\n')) {
        // 转义值中已有的双引号
        const escapedValue = value.replace(/"/g, '\\"');
        formattedValue = `"${escapedValue}"`;
      }
    }
    
    // 匹配整个键值对（包括可能的多行值）
    // 1. 先尝试匹配单行格式
    const singleLineRegex = new RegExp(`^${key}=.*$`, 'm');
    // 2. 再尝试匹配多行格式（以引号开头，可能跨多行，以引号结尾）
    const multiLineRegex = new RegExp(`^${key}=["']([\\s\\S]*?)["']$`, 'm');
    
    if (multiLineRegex.test(content)) {
      // 替换多行格式
      content = content.replace(multiLineRegex, `${key}=${formattedValue}`);
    } else if (singleLineRegex.test(content)) {
      // 替换单行格式
      content = content.replace(singleLineRegex, `${key}=${formattedValue}`);
    } else {
      // 键不存在，追加到文件末尾
      content += `\n${key}=${formattedValue}`;
    }
  });
  
  fs.writeFileSync(filePath, content, 'utf8');
}
