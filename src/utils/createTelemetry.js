import { randomUUID } from "crypto";
import protobuf from 'protobufjs';
import { QA_PAIRS } from '../constants/index.js';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function createTelemetryBatch(num, trajectoryId) {
  const now = Date.now();
  const seconds = Math.floor(now / 1000);
  const nanos = (now % 1000) * 1000000;

  const sessionId = trajectoryId;
  const conversationId = randomUUID();
  const messageId = randomUUID();
  const traceId1 = randomUUID();
  const traceId2 = randomUUID();
  const sessionId2 = randomUUID();
  const conversationId2 = randomUUID();

  return {
    sessionInfo: { value: 16 },
    sequenceNumber: Math.floor(Math.random() * 10000),
    timestamp: now,
    events: [
      {
        timestampMs: now,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          sessionEvent: {
            sessionId: sessionId,
            conversationId: conversationId,
            turnNumber: 4,
            messageIndex: 1,
            timing: {
              timestamp: { seconds, nanos },
              traceId: traceId1
            },
            version: "1.18.3",
            app: "antigravity",
            platform: "windows"
          }
        },
        eventType: 10
      },
      {
        timestampMs: now + 215,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          fileEvent: {
            operation: {
              timestamp: { seconds, nanos },
              traceId: traceId1,
              details: {
                basePath: "file:///C:/Users/user/.gemini/antigravity/brain/" + conversationId,
                rootPath: "file:///C:/Users/user/.gemini/antigravity/brain"
              }
            }
          }
        },
        eventType: 12
      },
      {
        timestampMs: now + 241,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          fileEvent: {
            operation: {
              timestamp: { seconds, nanos: nanos + 241623200 },
              traceId: traceId2,
              details: {
                rootPath: "file:///C:/Users/user/.gemini/antigravity/brain"
              }
            }
          }
        },
        eventType: 12
      },
      {
        timestampMs: now + 241,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          sessionEvent: {
            sessionId: sessionId2,
            conversationId: conversationId2,
            turnNumber: 1,
            timing: {
              timestamp: { seconds, nanos: nanos + 239604100 },
              traceId: traceId2
            },
            version: "1.18.3",
            app: "antigravity",
            platform: "windows"
          }
        },
        eventType: 10
      },
      {
        timestampMs: now + 241,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 14,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos - 86278600 },
                state: 4,
                messageId: messageId,
                retryCount: 1,
                turnInfo: {
                  sessionId: sessionId,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 238922300 }
                  }
                }]
              },
              userMessage: {
                text: QA_PAIRS[num].question,
                content: { text: QA_PAIRS[num].question },
                inputType: 1,
                config: {
                  details: {
                    mode: { codeMode: 1, chatMode: 1 },
                    features: {
                      settings: { value: { flag: 1 } },
                      experiments: { enabled: 1 }
                    },
                    tokens: { count: 1035 },
                    capabilities: { enabled: 1 },
                    model: { enabled: 1 }
                  },
                  flags: { enabled: 1 }
                }
              }
            },
            turnInfo: {
              sessionId: sessionId,
              turnNumber: 1
            },
            sessionId: sessionId,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 271,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 90,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 265889900 },
                messageId: messageId,
                retryCount: 1,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 3,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 265889900 }
                  }
                }]
              },
              ephemeralContent: {
                message: "The following is an <EPHEMERAL_MESSAGE> not actually sent by the user. It is provided by the system as a set of reminders and general important information to pay attention to. Do NOT respond to this message, just act accordingly.\n\n<EPHEMERAL_MESSAGE>\n<artifact_reminder>\nYou have not yet created any artifacts. Please follow the artifact guidelines and create them as needed based on the task.\nCRITICAL REMINDER: remember that user-facing artifacts should be AS CONCISE AS POSSIBLE. Keep this in mind when editing artifacts.\n</artifact_reminder>\n<no_activeTask_reminder>\nYou are currently not in a task because: a task boundary has never been set yet in this conversation.\nIf there is no obvious task from the user or if you are just conversing, then it is acceptable to not have a task set.\n</no_activeTask_reminder>\n</EPHEMERAL_MESSAGE>",
                tags: ["artifact_reminder", "no_activeTask_reminder"]
              }
            },
            turnInfo: {
              sessionId: sessionId2,
              turnState: 3,
              turnNumber: 1
            },
            sessionId: sessionId2,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 271,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 90,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 265889900 },
                messageId: messageId,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 3,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 265889900 }
                  }
                }]
              },
              ephemeralContent: {
                message: "The following is an <EPHEMERAL_MESSAGE> not actually sent by the user.",
                tags: ["artifact_reminder", "no_active_task_reminder"]
              }
            },
            turnInfo: {
              sessionId: sessionId,
              turnState: 3,
              turnNumber: 4
            },
            sessionId: sessionId,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 310,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 14,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos - 86278600 },
                state: 4,
                messageId: messageId,
                turnInfo: {
                  sessionId: sessionId,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 238922300 }
                  }
                }]
              },
              userMessage: {
                text: QA_PAIRS[num].question,
                content: { text: QA_PAIRS[num].question },
                inputType: 1,
                config: {
                  details: {
                    mode: { codeMode: 1, chatMode: 1 },
                    features: {
                      settings: { value: { flag: 1 } },
                      experiments: { enabled: 1 }
                    },
                    tokens: { count: 1035 },
                    capabilities: { enabled: 1 },
                    model: { enabled: 1 }
                  },
                  flags: { enabled: 1 }
                }
              }
            },
            turnInfo: {
              sessionId: sessionId2,
              turnNumber: 4
            },
            sessionId: sessionId2,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 311,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 98,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 291435600 },
                messageId: messageId,
                retryCount: 1,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 1,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 291435600 }
                  }
                }]
              },
              conversationHistory: {
                content: "# Conversation History\nHere are the conversation IDs, titles, and summaries of your most recent 20 conversations, in reverse chronological order:\n\n<conversation_summaries>\n## Conversation d15427a0-0deb-45ce-8980-442e3c49fbc1: AI Assistant Introduction\n- Created: 2026-02-14T02:26:44Z\n- Last modified: 2026-02-14T02:26:55Z\n\n### USER Objective:\nAI Assistant Introduction\nThe user's main objective is to understand the identity and capabilities of the AI assistant."
              }
            },
            turnInfo: {
              sessionId: sessionId2,
              turnState: 1,
              turnNumber: 1
            },
            sessionId: sessionId2,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 350,
        eventCode: 0,
        metadata: {
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
          messageEvent: {
            messageData: {
              type: 98,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 291435600 },
                messageId: messageId,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 1,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 291435600 }
                  }
                }]
              },
              conversationHistory: {
                content: "# Conversation History\nHere are the conversation IDs, titles, and summaries of your most recent 20 conversations, in reverse chronological order:\n\n<conversation_summaries>\n## Conversation d15427a0-0deb-45ce-8980-442e3c49fbc1: AI Assistant Introduction\n- Created: 2026-02-14T02:26:44Z\n- Last modified: 2026-02-14T02:26:55Z\n\n### USER Objective:\nAI Assistant Introduction\nThe user's main objective is to understand the identity and capabilities of the AI assistant."
              }
            },
            turnInfo: {
              sessionId: sessionId,
              turnState: 1,
              turnNumber: 1
            },
            sessionId: sessionId,
            messageId: messageId
          }
        },
        eventType: 11
      },
      {
        timestampMs: now + 360,
        metadata: {
          messageEvent: {
            messageData: {
              type: 99,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 291435600 },
                messageId: messageId,
                retryCount: 1,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 1,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 291435600 }
                  }
                }]
              },
            },
            turnInfo: {
              sessionId: sessionId2,
              turnState: 1,
              turnNumber: 1
            },
            sessionId: sessionId2,
            messageId: messageId
          },
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
        },
        eventType: 11
      },
      {
        timestampMs: now + 370,
        metadata: {
          messageEvent: {
            messageData: {
              type: 99,
              status: 3,
              details: {
                startTime: { seconds, nanos: nanos + 265889900 },
                state: 5,
                endTime: { seconds, nanos: nanos + 291435600 },
                messageId: messageId,
                turnInfo: {
                  sessionId: sessionId,
                  turnState: 1,
                  conversationId: conversationId
                },
                stateTransitions: [{
                  info: {
                    state: 3,
                    timestamp: { seconds, nanos: nanos + 291435600 }
                  }
                }]
              },
            },
            turnInfo: {
              sessionId: sessionId,
              turnState: 2,
              turnNumber: 4
            },
            sessionId: sessionId,
            messageId: messageId
          },
          appName: "antigravity",
          version: "1.18.3",
          os: "windows",
          arch: "amd64",
          tier: "g1-pro-tier",
          region: "JP",
        },
        eventType: 11
      }
    ]
  };
}

function serializeTelemetryBatch(telemetryData, protoPath = join(__dirname, 'proto', 'telemetry.proto')) {
  try {
    const root = protobuf.loadSync(protoPath);
    const TelemetryBatch = root.lookupType('TelemetryBatch');

    const message = TelemetryBatch.create(telemetryData);
    const buffer = TelemetryBatch.encode(message).finish();

    return { success: true, data: buffer };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

export { createTelemetryBatch, serializeTelemetryBatch };
