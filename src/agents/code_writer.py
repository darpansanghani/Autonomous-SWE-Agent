import json
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel
import unidiff

from src.llm.router import agent_completion
from src.agents.planner import SubTask
from src.utils.file_utils import safe_read, list_files

class FileOperation(BaseModel):
    action: Literal["create", "modify", "delete"]
    file_path: str
    language: str
    unified_diff: Optional[str] = None
    full_content: Optional[str] = None

class CodeChanges(BaseModel):
    operations: List[FileOperation]
    explanation: str

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full content of a file in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path from repo root"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files in the repository",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_changes",
            "description": "Submit the final code changes as unified diffs when done",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["create", "modify", "delete"]},
                                "file_path": {"type": "string"},
                                "language": {"type": "string"},
                                "unified_diff": {"type": "string", "description": "Required if action is modify"},
                                "full_content": {"type": "string", "description": "Required if action is create"}
                            },
                            "required": ["action", "file_path", "language"]
                        }
                    },
                    "explanation": {"type": "string", "description": "Why these changes solve the issue"}
                },
                "required": ["operations", "explanation"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an expert software engineer making changes to a codebase.

TASK:
{task_description}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

CODE CONTEXT (from codebase search):
{context}

PREVIOUS FEEDBACK:
{feedback}

INSTRUCTIONS:
1. Use tools to explore the codebase if you need more context.
2. Output changes via submit_changes.
3. For modifications, output UNIFIED DIFFS exactly matching existing files.
4. For new files, provide the full content.
5. Match the repo's existing coding style exactly.
"""

async def generate_changes(
    sub_task: SubTask,
    code_context: List[Any],
    repo_path: str,
    feedback_history: List[str]
) -> CodeChanges:
    """Executes a ReAct loop to write code changes."""
    
    # Format context chunks
    context_str = "\n\n".join([f"--- {c.file_path} ---\n{c.body}" for c in code_context])
    feedback_str = "\n".join(f"- {f}" for f in feedback_history) if feedback_history else "None"
    
    messages = [
        {"role": "system", "content": "You are a senior engineer."},
        {"role": "user", "content": SYSTEM_PROMPT.format(
            task_description=sub_task.description,
            acceptance_criteria=sub_task.acceptance_criteria,
            context=context_str,
            feedback=feedback_str
        )}
    ]

    # ReAct loop (max 10 steps to prevent infinite tool calling)
    for _ in range(10):
        response = await agent_completion(
            agent_name="code_writer",
            messages=messages,
            tools=TOOLS
        )
        
        message = response.choices[0].message
        messages.append(message.model_dump())
        
        # Check if model wants to call a tool
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                if func_name == "submit_changes":
                    return CodeChanges.model_validate(args)
                    
                elif func_name == "read_file":
                    try:
                        content = safe_read(repo_path, args["file_path"])
                        result = content
                    except Exception as e:
                        result = f"Error: {e}"
                        
                elif func_name == "list_directory":
                    files = list_files(repo_path)
                    result = "\n".join(files)
                
                # Add tool result to context
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result
                })
        else:
            # Model didn't call tools, meaning it just talked.
            # Force it to submit
            messages.append({"role": "user", "content": "Please use the submit_changes tool to provide your code."})
            
    raise Exception("Code writer loop exceeded maximum steps without submitting changes.")
