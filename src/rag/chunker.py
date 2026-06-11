import hashlib
from dataclasses import dataclass
from typing import List, Optional
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
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
    Supports Python and JavaScript with clean AST-aware extraction.
    """
    if not content.strip():
        return []

    lines = content.splitlines()

    # Resolve correct language parser
    if language == "python":
        lang = Language(tspython.language())
    elif language in ["javascript", "typescript"]:
        lang = Language(tsjavascript.language())
    else:
        # Fallback to whole file for unsupported languages
        return [CodeChunk(
            chunk_id=f"{file_path}_full",
            file_path=file_path,
            language=language,
            chunk_type="file",
            name=file_path,
            signature="",
            body=content,
            start_line=1,
            end_line=len(lines)
        )]

    parser = Parser(lang)
    tree = parser.parse(bytes(content, "utf8"))
    chunks: List[CodeChunk] = []

    def get_node_text(node) -> str:
        return "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])

    def get_first_line(node) -> str:
        return lines[node.start_point[0]].strip()

    def walk_ast(node, parent_class: Optional[str] = None):
        nonlocal chunks
        
        # Check Python nodes
        if language == "python":
            if node.type == "class_definition":
                name_node = next((c for c in node.children if c.type == "identifier"), None)
                class_name = get_node_text(name_node) if name_node else "unknown_class"
                
                # Add class chunk
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}_class_{node.start_point[0]}",
                    file_path=file_path,
                    language="python",
                    chunk_type="class",
                    name=class_name,
                    signature=get_first_line(node),
                    body=get_node_text(node),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                ))
                
                # Traverse children with parent class info
                for child in node.children:
                    walk_ast(child, class_name)
                return

            elif node.type == "function_definition":
                name_node = next((c for c in node.children if c.type == "identifier"), None)
                func_name = get_node_text(name_node) if name_node else "unknown_func"
                
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}_func_{node.start_point[0]}",
                    file_path=file_path,
                    language="python",
                    chunk_type="function",
                    name=func_name,
                    signature=get_first_line(node),
                    body=get_node_text(node),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_class=parent_class
                ))
                return

        # Check JS nodes
        elif language in ["javascript", "typescript"]:
            if node.type == "class_declaration":
                name_node = next((c for c in node.children if c.type == "identifier"), None)
                class_name = get_node_text(name_node) if name_node else "unknown_class"
                
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}_class_{node.start_point[0]}",
                    file_path=file_path,
                    language=language,
                    chunk_type="class",
                    name=class_name,
                    signature=get_first_line(node),
                    body=get_node_text(node),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                ))
                
                for child in node.children:
                    walk_ast(child, class_name)
                return

            elif node.type in ["function_declaration", "lexical_declaration", "variable_declaration"]:
                # Check for arrow function or standard function inside variable/lexical declarations
                is_func = False
                name = "unknown_func"
                
                if node.type == "function_declaration":
                    is_func = True
                    name_node = next((c for c in node.children if c.type == "identifier"), None)
                    name = get_node_text(name_node) if name_node else "unknown_func"
                else:
                    # e.g., const foo = () => {}
                    declarator = next((c for c in node.children if c.type == "variable_declarator"), None)
                    if declarator:
                        id_node = next((c for c in declarator.children if c.type == "identifier"), None)
                        init_node = next((c for c in declarator.children if c.type in ["arrow_function", "function_expression"]), None)
                        if id_node and init_node:
                            is_func = True
                            name = get_node_text(id_node)

                if is_func:
                    chunks.append(CodeChunk(
                        chunk_id=f"{file_path}_func_{node.start_point[0]}",
                        file_path=file_path,
                        language=language,
                        chunk_type="function",
                        name=name,
                        signature=get_first_line(node),
                        body=get_node_text(node),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent_class=parent_class
                    ))
                    return

            elif node.type == "method_definition":
                name_node = next((c for c in node.children if c.type == "property_identifier"), None)
                method_name = get_node_text(name_node) if name_node else "unknown_method"
                
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}_method_{node.start_point[0]}",
                    file_path=file_path,
                    language=language,
                    chunk_type="method",
                    name=method_name,
                    signature=get_first_line(node),
                    body=get_node_text(node),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_class=parent_class
                ))
                return

        # Recurse down children if we didn't stop
        for child in node.children:
            walk_ast(child, parent_class)

    walk_ast(tree.root_node)

    # Fallback to whole file if no chunks found
    if not chunks:
        chunks.append(CodeChunk(
            chunk_id=f"{file_path}_full",
            file_path=file_path,
            language=language,
            chunk_type="file",
            name=file_path,
            signature="",
            body=content,
            start_line=1,
            end_line=len(lines)
        ))

    return chunks

def build_embedding_text(chunk: CodeChunk, file_imports: List[str]) -> str:
    """Build the text that gets embedded — includes surrounding context."""
    parts = []

    # 1. File path as context
    parts.append(f"# File: {chunk.file_path}")

    # 2. Relevant imports from the file
    if file_imports:
        parts.append("# Imports:\n" + "\n".join(file_imports[:10]))

    # 3. Parent class context (if method)
    if chunk.parent_class:
        parts.append(f"# Part of class: {chunk.parent_class}")

    # 4. The actual code
    parts.append(chunk.body)

    return "\n\n".join(parts)
