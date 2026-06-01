import React from 'react';
import {
  getOpenAiRequestOptions,
  openAiStreamingDataHandler,
} from './chat-stream-handler';
import type {
  ChatMessage,
  OpenAIChatMessage,
  ChatMessageParams,
  OpenAIStreamingParams,
  OpenAIChatRole,
} from './types';

const MILLISECONDS_PER_SECOND = 1000;

const SYSTEM_PROMPT = `You are an expert Odoo ERP assistant and data analyst at current year is 2026.

        Your job:
        - Identify Main cause of the user
        - Append synonyms of main cause 
and append these words to the Main cause string
		- Change Plural words to single
		- Always try to find a tool to fulfill requirements
		- If you do not find a tool let the user know
    - Use tool results carefully
    - Give the most relevent model based on the data returned by tools
    - NEVER GIVE DIRECT ANSWER EVEN IF YOU THINK YOU KNOW
    - IF you fine multiple relevent models ask user about it`;

const officialOpenAIParams = ({
  content,
  role,
}: ChatMessage): OpenAIChatMessage => ({ content, role });


const createChatMessage = ({
  content,
  role,
  ...restOfParams
}: ChatMessageParams): ChatMessage => ({
  content,
  role,
  timestamp: restOfParams.timestamp ?? Date.now(),
  meta: {
    loading: false,
    responseTime: '',
    chunks: [],
    ...restOfParams.meta,
  },
});


const updateLastItem =
  <T>(msgFn: (message: T) => T) =>
  (currentMessages: T[]) =>
    currentMessages.map((msg, i) => {
      if (currentMessages.length - 1 === i) {
        return msgFn(msg);
      }
      return msg;
    });

export const useChatCompletion = (apiParams: OpenAIStreamingParams) => {
  const [messages, _setMessages] = React.useState<ChatMessage[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [controller, setController] = React.useState<AbortController | null>(
    null
  );

  const abortResponse = () => {
    console.log("Calling abort response")
    if (controller) {
      controller.abort();
      setController(null);
    }
  };

  const resetMessages = () => {
    if (!loading) {
      _setMessages([]);
    }
  };

  const setMessages = (newMessages: ChatMessageParams[]) => {
    if (!loading) {
      _setMessages(newMessages.map(createChatMessage));
    }
  };

  const handleNewData = (chunkContent: string, chunkRole: OpenAIChatRole) => {
  _setMessages(
    updateLastItem((msg) => ({
      content: `${msg.content}${chunkContent}`,

      role: 'assistant' as OpenAIChatRole,

      timestamp: 0,

      meta: {
        ...msg.meta,
        chunks: [
          ...msg.meta.chunks,
          {
            content: chunkContent,
            role: 'assistant' as OpenAIChatRole,
            timestamp: Date.now(),
          },
        ],
      },
    }))
  );
};


  const closeStream = (beforeTimestamp: number) => {
    const afterTimestamp = Date.now();
    const diffInSeconds =
      (afterTimestamp - beforeTimestamp) / MILLISECONDS_PER_SECOND;
    const formattedDiff = diffInSeconds.toFixed(2) + ' sec.';

    _setMessages(
      updateLastItem((msg) => ({
        ...msg,
        timestamp: afterTimestamp,
        meta: {
          ...msg.meta,
          loading: false,
          responseTime: formattedDiff,
        },
      }))
    );
  };

  const submitPrompt = React.useCallback(
    async (newMessages?: ChatMessageParams[]) => {
      if (messages[messages.length - 1]?.meta?.loading) return;

      if (!newMessages || newMessages.length < 1) {
        return;
      }

      setLoading(true);
      const updatedMessages: ChatMessage[] = [
        ...messages,
        ...newMessages.map(createChatMessage),
        createChatMessage({
          content: '',
          role: '',
          timestamp: 0,
          meta: { loading: true },
        }),
      ];

      // Set the updated message list.
      _setMessages(updatedMessages);

      const newController = new AbortController();
      const signal = newController.signal;
      setController(newController);

      const requestOpts = getOpenAiRequestOptions(
        apiParams,
        [
          {
            role: 'system',
            content: SYSTEM_PROMPT,
          },
          ...updatedMessages
            .filter((m, i) => updatedMessages.length - 1 !== i)
            .map(officialOpenAIParams),
        ],
        signal
      );

      try {
        await openAiStreamingDataHandler(
          requestOpts,
          handleNewData,
          closeStream
        );
      } catch (err) {
        if (signal.aborted) {
          console.error(`Request aborted`, err);
        } else {
          console.error(`Error during chat response streaming`, err);
        }
      } finally {
        setController(null);
        setLoading(false);
      }
    },
    [messages]
  );

  return {
    messages,
    loading,
    submitPrompt,
    abortResponse,
    resetMessages,
    setMessages,
  };
};
