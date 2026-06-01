import OpenAI from 'openai';
import {tools,executeToolCall} from "../example/tools"

import type {
  OpenAIStreamingParams,
  OpenAIChatMessage,
  FetchRequestOptions,
  OpenAIChatRole,
} from './types';

export const getOpenAiRequestOptions = (
  { apiKey, model, ...restOfApiParams }: OpenAIStreamingParams,
  messages: OpenAIChatMessage[],
  signal?: AbortSignal
): FetchRequestOptions => ({
  model,
  apiKey,
  messages,
  signal,
  ...restOfApiParams,
});

export const openAiStreamingDataHandler = async (
  requestOpts: FetchRequestOptions,
  onIncomingChunk: (
    contentChunk: string,
    roleChunk: OpenAIChatRole
  ) => void,
  onCloseStream: (beforeTimestamp: number) => void
) => {
  const beforeTimestamp = Date.now();

  const {
    model,
    messages,
    signal: externalSignal,
    apiKey,
    ...rest
  } = requestOpts as any;


  const dynamicClient = new OpenAI({
    apiKey,
    baseURL: 'http://localhost:1234/v1',
    dangerouslyAllowBrowser: true,
  });

  try {
    // =========================
    // 1. FIRST ASSISTANT CALL
    // =========================
    const stream = await dynamicClient.chat.completions.create(
      {
        model,
        messages,
        stream: true,
        tools,
        tool_choice: 'auto',
        ...rest,
      },
      { signal }
    );

    let content = '';
    const toolCalls: Record<
      number,
      { id: string; name: string; arguments: string }
    > = {};

    let lastToolUpdate = Date.now();
    let streamAborted = false;

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;

      // =========================
      // TOOL COLLECTION
      // =========================
      if (delta.tool_calls) {
        for (const toolCall of delta.tool_calls) {
          const index = toolCall.index;

          if (!toolCalls[index]) {
            toolCalls[index] = {
              id: toolCall.id || `call_${Math.random().toString(36).slice(2)}`,
              name: '',
              arguments: '',
            };
          }

          if (toolCall.function?.name) {
            toolCalls[index].name += toolCall.function.name;
            lastToolUpdate = Date.now();
          }

          if (toolCall.function?.arguments) {
            toolCalls[index].arguments += toolCall.function.arguments;
            lastToolUpdate = Date.now();
          }
        }
      }

      const contentChunk = delta.content ?? '';
      content += contentChunk;

      if (contentChunk) {
        onIncomingChunk(contentChunk, 'assistant');
      }

      // =========================
      // 🚨 STOP CONDITION (NEW)
      // =========================
      const toolsReady = Object.values(toolCalls).every(
        (t) => t.name.length > 0 && t.arguments.length > 0
      );

      const idleTime = Date.now() - lastToolUpdate;

      // if (toolsReady && idleTime > 50 && Object.keys(toolCalls).length > 0) {
      //   streamAborted = true;
      //   controller.abort(); // 🔥 STOP STREAM IMMEDIATELY
      //   break;
      // }
    }

    console.log('Tool calls found:', Object.keys(toolCalls).length);

    // =========================
    // 2. NO TOOL → FINISH EARLY
    // =========================
    if (Object.keys(toolCalls).length === 0) {
      onCloseStream(beforeTimestamp);
      return {
        content,
        role: 'assistant' as OpenAIChatRole,
      };
    }

    // =========================
    // 3. FLUSH UI BEFORE TOOL PHASE
    // =========================
    onIncomingChunk('\n', 'assistant');

    const updatedMessages = [...messages];

    updatedMessages.push({
      role: 'assistant',
      content,
      tool_calls: Object.values(toolCalls).map((tc) => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.name,
          arguments: tc.arguments,
        },
      })),
    } as any);

    // =========================
    // 4. TOOL EXECUTION
    // =========================
    const toolNames = Object.values(toolCalls)
      .map((t) => t.name)
      .join(', ');

    onIncomingChunk(
      `⏳ Retrieving data from: ${toolNames}...\n\n`,
      'system' as OpenAIChatRole
    );

    for (const tool of Object.values(toolCalls)) {
      const toolResult = await executeToolCall(
        tool.name,
        tool.arguments
      );

      updatedMessages.push({
        role: 'tool',
        tool_call_id: tool.id,
        content:
          typeof toolResult === 'string'
            ? toolResult
            : JSON.stringify(toolResult),
      } as any);
    }

    // =========================
    // 5. SECOND MODEL CALL
    // =========================
    onIncomingChunk(
      `✨ Generating response...\n\n`,
      'system' as OpenAIChatRole
    );

    const finalStream = await dynamicClient.chat.completions.create(
      {
        model,
        messages: updatedMessages,
        stream: true,
        ...rest,
      },
      { signal }
    );

    let finalContent = '';

    for await (const chunk of finalStream) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;

      const contentChunk = delta.content ?? '';
      finalContent += contentChunk;

      if (contentChunk) {
        onIncomingChunk(contentChunk, 'assistant');
      }
    }

    onCloseStream(beforeTimestamp);

    return {
      content: finalContent,
      role: 'assistant' as OpenAIChatRole,
    };
  } catch (error) {
    console.error('Error in openAiStreamingDataHandler:', error);
    onCloseStream(beforeTimestamp);
    throw error;
  }
};


export default openAiStreamingDataHandler;
