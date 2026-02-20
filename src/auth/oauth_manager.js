import axios from 'axios';
import crypto from 'crypto';
import log from '../utils/logger.js';
import config from '../config/config.js';
import { generateProjectId } from '../utils/idGenerator.js'; // TODO: 可移除，已不再使用
import tokenManager from './token_manager.js';
import geminicliTokenManager from './geminicli_token_manager.js';
import { OAUTH_CONFIG, OAUTH_SCOPES, GEMINICLI_OAUTH_CONFIG, GEMINICLI_OAUTH_SCOPES } from '../constants/oauth.js';
import { buildAxiosRequestConfig } from '../utils/httpClient.js';

class OAuthManager {
	constructor() {
		this.state = crypto.randomUUID();
	}

	/**
	 * 生成授权URL
	 * @param {number} port - 回调端口
	 * @param {string} mode - 模式：'antigravity' 或 'geminicli'
	 */
	generateAuthUrl(port, mode = 'antigravity') {
		const oauthConfig = mode === 'geminicli' ? GEMINICLI_OAUTH_CONFIG : OAUTH_CONFIG;
		const scopes = mode === 'geminicli' ? GEMINICLI_OAUTH_SCOPES : OAUTH_SCOPES;

		const params = new URLSearchParams({
			access_type: 'offline',
			client_id: oauthConfig.CLIENT_ID,
			prompt: 'consent',
			redirect_uri: `http://localhost:${port}/oauth-callback`,
			response_type: 'code',
			scope: scopes.join(' '),
			state: `${this.state}_${mode}` // 在 state 中包含 mode 信息
		});
		return `${oauthConfig.AUTH_URL}?${params.toString()}`;
	}

	/**
	 * 交换授权码获取Token
	 * @param {string} code - 授权码
	 * @param {number} port - 回调端口
	 * @param {string} mode - 模式：'antigravity' 或 'geminicli'
	 */
	async exchangeCodeForToken(code, port, mode = 'antigravity') {
		const oauthConfig = mode === 'geminicli' ? GEMINICLI_OAUTH_CONFIG : OAUTH_CONFIG;

		const postData = new URLSearchParams({
			code,
			client_id: oauthConfig.CLIENT_ID,
			client_secret: oauthConfig.CLIENT_SECRET,
			redirect_uri: `http://localhost:${port}/oauth-callback`,
			grant_type: 'authorization_code'
		});

		const response = await axios(buildAxiosRequestConfig({
			method: 'POST',
			url: oauthConfig.TOKEN_URL,
			headers: {
				'Host': 'oauth2.googleapis.com',
				'User-Agent': 'Go-http-client/1.1',
				'Content-Type': 'application/x-www-form-urlencoded',
				'Accept-Encoding': 'gzip'
			},
			data: postData.toString(),
			timeout: config.timeout
		}));

		return response.data;
	}

	/**
	 * 获取用户邮箱
	 */
	async fetchUserEmail(accessToken) {
		try {
			const response = await axios(buildAxiosRequestConfig({
				method: 'GET',
				url: 'https://www.googleapis.com/oauth2/v2/userinfo',
				headers: {
					'Host': 'www.googleapis.com',
					'User-Agent': 'Go-http-client/1.1',
					'Authorization': `Bearer ${accessToken}`,
					'Accept-Encoding': 'gzip'
				},
				timeout: config.timeout
			}));
			return response.data?.email;
		} catch (err) {
			log.warn('获取用户邮箱失败:', err.message);
			return null;
		}
	}

	/**
	 * 资格校验：尝试获取projectId
	 */
	async validateAndGetProjectId(accessToken) {
		try {
			log.info('正在验证账号资格...');
			const projectId = await tokenManager.fetchProjectId({ access_token: accessToken });

			if (projectId === undefined || projectId === null) {
				log.warn('该账号无法获取 projectId，可能无资格或需要稍后重试');
				return { projectId: null, hasQuota: false };
			}

			log.info('账号验证通过，projectId: ' + projectId);
			return { projectId, hasQuota: true };
		} catch (err) {
			log.error('验证账号资格失败: ' + err.message);
			return { projectId: null, hasQuota: false };
		}
	}

	/**
	 * 完整的OAuth认证流程：交换Token -> 获取邮箱 -> 资格校验
	 * @param {string} code - 授权码
	 * @param {number} port - 回调端口
	 * @param {string} mode - 模式：'antigravity' 或 'geminicli'
	 */
	async authenticate(code, port, mode = 'antigravity') {
		// 1. 交换授权码获取Token
		const tokenData = await this.exchangeCodeForToken(code, port, mode);

		if (!tokenData.access_token) {
			throw new Error('Token交换失败：未获取到access_token');
		}

		const account = {
			access_token: tokenData.access_token,
			refresh_token: tokenData.refresh_token,
			expires_in: tokenData.expires_in,
			timestamp: Date.now()
		};

		// 2. 获取用户邮箱
		const email = await this.fetchUserEmail(account.access_token);
		if (email) {
			account.email = email;
			log.info(`[${mode}] 获取到用户邮箱: ${email}`);
		}

		// 3. 资格校验（仅 antigravity 模式需要 projectId）
		if (mode === 'antigravity') {
			const { projectId, hasQuota } = await this.validateAndGetProjectId(account.access_token);
			account.projectId = projectId;
			account.hasQuota = hasQuota;
		}

		account.enable = true;

		return account;
	}

	/**
	 * Gemini CLI 专用认证流程（简化版，不需要 projectId）
	 * @param {string} code - 授权码
	 * @param {number} port - 回调端口
	 */
	async authenticateGeminiCli(code, port) {
		return this.authenticate(code, port, 'GeminiCLI');
	}
}

export default new OAuthManager();
