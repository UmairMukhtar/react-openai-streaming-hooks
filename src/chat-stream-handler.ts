import type {
  OpenAIStreamingParams,
  OpenAIChatMessage,
  FetchRequestOptions,
  OpenAIChatRole,
  OpenAIChatCompletionChunk,
} from './types';

// Converts the OpenAI API params + chat messages list + an optional AbortSignal into a shape that
// the fetch interface expects.
export const getOpenAiRequestOptions = (
  { apiKey, model, ...restOfApiParams }: OpenAIStreamingParams,
  messages: OpenAIChatMessage[],
  signal?: AbortSignal
): FetchRequestOptions => ({
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  },
  method: 'POST',
  body: JSON.stringify({
    model,
    // Includes all settings related to how the user wants the OpenAI API to execute their request.
    ...restOfApiParams,
    messages,
    stream: true,
  }),
  signal,
});

const CHAT_COMPLETIONS_URL = 'https://api.openai.com/v1/chat/completions';

const textDecoder = new TextDecoder('utf-8');

// Takes a set of fetch request options and calls the onIncomingChunk and onCloseStream functions
// as chunks of a chat completion's data are returned to the client, in real-time.
export const openAiStreamingDataHandler = async (
  requestOpts: FetchRequestOptions,
  onIncomingChunk: (contentChunk: string, roleChunk: OpenAIChatRole) => void,
  onCloseStream: (beforeTimestamp: number) => void
) => {
  // Record the timestamp before the request starts.
  const beforeTimestamp = Date.now();

  // Initiate the completion request
  const response = await fetch(CHAT_COMPLETIONS_URL, requestOpts);

  // If the response isn't OK (non-2XX HTTP code), report the HTTP status and description.
  if (!response.ok) {
    throw new Error(
      `Network response was not ok: ${response.status} - ${response.statusText}`
    );
  }

  // A response body should always exist, if there isn't one, something has gone wrong.
  if (!response.body) {
    throw new Error('No body included in POST response object');
  }

  let content = '';
  let role = '';

  const reader = response.body.getReader();
  let done = false;
  let decodedData = '';

  while (!done) {
    const { value, done: readerDone } = await reader.read();
    done = readerDone;

    if (value) {
      const chunk = new TextDecoder().decode(value);
      decodedData += chunk;

      // Process the chunk and send an update to the registered handler.
      const lines = decodedData.split(/(\n){2}/);
      const numLines = lines.length;

      for (let i = 0; i < numLines - 1; i++) {
        const trimmedLine = lines[i].replace(/(\n)?^data:\s*/, '').trim();

        if (trimmedLine !== '' && trimmedLine !== '[DONE]') {
          const chunk = JSON.parse(trimmedLine);

          const contentChunk: string = (
            chunk.choices[0].delta.content ?? ''
          ).replace(/^`\s*/, '`');

          const roleChunk: OpenAIChatRole = chunk.choices[0].delta.role ?? '';

          content += contentChunk;
          role += roleChunk;

          onIncomingChunk(contentChunk, roleChunk);
        }
      }

      // Keep the last line in `decodedData` in case it's incomplete
      decodedData = lines[numLines - 1];
    }
  }

  onCloseStream(beforeTimestamp);

  return { content, role } as OpenAIChatMessage;
};


export default openAiStreamingDataHandler;
