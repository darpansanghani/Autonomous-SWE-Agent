from dataclasses import dataclass
from typing import List, Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    signature: str
    body: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    relevance_score: float = 0.0

def parse_and_chunk(file_path: str, content: str, language: str) -> List[CodeChunk]:
    """
    Splits source code into logical chunks (functions/classes) using tree-sitter.
    Currently implements Python. Extensible to others.
    """
    if language != "python" or not content.strip():
        # Fallback to full file if not python or empty
        return [CodeChunk(
            chunk_id=f"{file_path}_full",
            file_path=file_path,
            language=language,
            chunk_type="file",
            name=file_path,
            signature="",
            body=content,
            start_line=1,
            end_line=len(content.splitlines())
        )]
        
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(content, "utf8"))
    
    chunks = []
    lines = content.splitlines()
    
    # We walk the AST to find classes and functions
    # Using a simple cursor approach
    
    def extract_node_text(node) -> str:
        return "\n".join(lines[node.start_point[0]:node.end_point[0]+1])

    cursor = tree.walk()
    reached_root = False
    while not reached_root:
        node = cursor.node
        if node.type in ["function_definition", "class_definition"]:
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            name = extract_node_text(name_node) if name_node else "unknown"
            
            chunk = CodeChunk(
                chunk_id=f"{file_path}_{node.start_point[0]}",
                file_path=file_path,
                language="python",
                chunk_type=node.type.replace("_definition", ""),
                name=name,
                signature=name, # simplify signature for now
                body=extract_node_text(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1
            )
            chunks.append(chunk)

        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            elif cursor.goto_next_sibling():
                retracing = False

    # If no functions/classes found, return the whole file
    if not chunks:
        chunks.append(CodeChunk(
            chunk_id=f"{file_path}_full",
            file_path=file_path,
            language="python",
            chunk_type="file",
            name=file_path,
            signature="",
            body=content,
            start_line=1,
            end_line=len(lines)
        ))
        
    return chunks
