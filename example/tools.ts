
export const tools = []
export const toolRegistry: Record<
  string,
  (...args: any[]) => Promise<any>
> = {
//   yourtoolfuction
};
export async function executeToolCall(
  toolName: string,
  rawArguments: string
) {
  try {
    console.log("this is tool Registry",toolRegistry) 
    const tool = toolRegistry[toolName];

    if (!tool) {
      throw new Error(
        `Unknown tool: ${toolName}`
      );
    }

    const parsedArgs = JSON.parse(rawArguments);

    console.log('Executing tool:', toolName);

    console.log('Arguments:', parsedArgs);

    const result = await tool(parsedArgs);

    console.log('Tool result:', result);

    return result;
  } catch (error: any) {
    console.error('Tool execution failed:', error);

    return {
      error: error.message,
    };
  }
}

export default tools;