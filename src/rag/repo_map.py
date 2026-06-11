import os
import re
from pathlib import Path
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser

from src.utils.language_detect import RepoLanguageProfile

def generate_repo_map(repo_path: str, profile: RepoLanguageProfile) -> dict:
    """
    Creates a highly structured JSON representation of the entire repository.
    Uses tree-sitter AST parsing to extract clean function, class, method, and import signatures
    for both Python and JavaScript, enabling precise static dependency analysis.
    """
    base = Path(repo_path)
    ignore_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    
    # Load parsers
    py_lang = Language(tspython.language())
    py_parser = Parser(py_lang)
    
    js_lang = Language(tsjavascript.language())
    js_parser = Parser(js_lang)
    
    structure = {}
    dependency_graph = {}
    
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            ext = Path(file).suffix
            if ext not in [".py", ".js", ".ts"]:
                continue
                
            full_path = Path(root) / file
            rel_path = full_path.relative_to(base).as_posix()
            
            file_info = {
                "type": "module",
                "imports": [],
                "functions": [],
                "classes": []
            }
            
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    structure[rel_path] = file_info
                    continue
                
                # Determine parser
                if ext == ".py":
                    tree = py_parser.parse(bytes(content, "utf8"))
                    parser_lang = "python"
                else:
                    tree = js_parser.parse(bytes(content, "utf8"))
                    parser_lang = "javascript"
                
                lines = content.splitlines()
                
                def get_node_text(node) -> str:
                    return "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                
                def get_first_line(node) -> str:
                    return lines[node.start_point[0]].strip()

                # Walk AST to extract structural elements
                def walk_file_ast(node):
                    # Python Node types
                    if parser_lang == "python":
                        if node.type == "class_definition":
                            name_node = next((c for c in node.children if c.type == "identifier"), None)
                            name = get_node_text(name_node) if name_node else "unknown_class"
                            
                            # Find bases
                            bases = []
                            argument_list = next((c for c in node.children if c.type == "argument_list"), None)
                            if argument_list:
                                bases = [get_node_text(arg) for arg in argument_list.children if arg.type not in ["(", ")", ","]]
                            
                            # Find methods
                            methods = []
                            block = next((c for c in node.children if c.type == "block"), None)
                            if block:
                                for child in block.children:
                                    if child.type == "function_definition":
                                        m_name_node = next((c for c in child.children if c.type == "identifier"), None)
                                        if m_name_node:
                                            methods.append(get_node_text(m_name_node))
                            
                            file_info["classes"].append({
                                "name": name,
                                "methods": methods,
                                "bases": bases
                            })
                            return
                            
                        elif node.type == "function_definition":
                            name_node = next((c for c in node.children if c.type == "identifier"), None)
                            if name_node:
                                file_info["functions"].append(get_node_text(name_node))
                            return
                            
                        elif node.type in ["import_statement", "import_from_statement"]:
                            file_info["imports"].append(get_first_line(node))
                            return

                    # JS/TS Node types
                    elif parser_lang == "javascript":
                        if node.type == "class_declaration":
                            name_node = next((c for c in node.children if c.type == "identifier"), None)
                            name = get_node_text(name_node) if name_node else "unknown_class"
                            
                            # Find bases (extends)
                            bases = []
                            heritage_node = next((c for c in node.children if c.type == "class_heritage"), None)
                            if heritage_node:
                                base_ident = next((c for c in heritage_node.children if c.type == "identifier"), None)
                                if base_ident:
                                    bases.append(get_node_text(base_ident))
                            
                            # Find methods
                            methods = []
                            body_node = next((c for c in node.children if c.type == "class_body"), None)
                            if body_node:
                                for child in body_node.children:
                                    if child.type == "method_definition":
                                        m_name_node = next((c for c in child.children if c.type == "property_identifier"), None)
                                        if m_name_node:
                                            methods.append(get_node_text(m_name_node))
                                            
                            file_info["classes"].append({
                                "name": name,
                                "methods": methods,
                                "bases": bases
                            })
                            return
                            
                        elif node.type == "function_declaration":
                            name_node = next((c for c in node.children if c.type == "identifier"), None)
                            if name_node:
                                file_info["functions"].append(get_node_text(name_node))
                            return
                            
                        elif node.type in ["lexical_declaration", "variable_declaration"]:
                            declarator = next((c for c in node.children if c.type == "variable_declarator"), None)
                            if declarator:
                                id_node = next((c for c in declarator.children if c.type == "identifier"), None)
                                init_node = next((c for c in declarator.children if c.type in ["arrow_function", "function_expression"]), None)
                                if id_node and init_node:
                                    file_info["functions"].append(get_node_text(id_node))
                            return
                            
                        elif node.type == "import_statement":
                            file_info["imports"].append(get_first_line(node))
                            return

                    for child in node.children:
                        walk_file_ast(child)
                
                walk_file_ast(tree.root_node)
                
                # Resolve dependencies for the dependency graph
                resolved_deps = []
                for imp in file_info["imports"]:
                    # Match standard local import patterns
                    # e.g., from src.utils.logger import log_event or import src.utils.file_utils
                    # or const logger = require('./utils/logger') or import { log } from '../utils/logger'
                    py_from_match = re.search(r"from\s+([\w\.]+)\s+import", imp)
                    py_imp_match = re.search(r"import\s+([\w\.]+)", imp)
                    js_imp_match = re.search(r"from\s+['\"]([\w\./-]+)['\"]", imp)
                    js_req_match = re.search(r"require\(['\"]([\w\./-]+)['\"]\)", imp)
                    
                    target = None
                    if py_from_match:
                        target = py_from_match.group(1)
                    elif py_imp_match:
                        target = py_imp_match.group(1)
                    elif js_imp_match:
                        target = js_imp_match.group(1)
                    elif js_req_match:
                        target = js_req_match.group(1)
                        
                    if target:
                        # Translate imported name to relative file path candidate
                        if parser_lang == "python":
                            parts = target.split(".")
                            # e.g. "src.utils.logger" -> "src/utils/logger.py"
                            py_path = "/".join(parts) + ".py"
                            if (base / py_path).exists():
                                resolved_deps.append(py_path)
                            else:
                                # try module dir with __init__.py
                                init_path = "/".join(parts) + "/__init__.py"
                                if (base / init_path).exists():
                                    resolved_deps.append(init_path)
                        else:
                            # Relative javascript imports e.g. "./utils/logger" or "../models"
                            norm_path = (Path(root) / target).resolve()
                            try:
                                rel_target = norm_path.relative_to(base).as_posix()
                                candidates = [rel_target + ".js", rel_target + ".ts", rel_target + "/index.js"]
                                for cand in candidates:
                                    if (base / cand).exists():
                                        resolved_deps.append(cand)
                                        break
                            except ValueError:
                                pass # outside base or absolute module
                                
                if resolved_deps:
                    dependency_graph[rel_path] = list(set(resolved_deps))
                    
            except Exception:
                pass
                
            structure[rel_path] = file_info

    return {
        "total_files": profile.total_files,
        "languages": profile.languages,
        "structure": structure,
        "dependency_graph": dependency_graph
    }
