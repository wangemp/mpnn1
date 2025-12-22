import re
from dataclasses import dataclass
from typing import List, Tuple

from tree_sitter_languages import get_parser


@dataclass
class CallSite:
    name: str
    args_text: str
    func_name: str
    node_type: str


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//.*?$", "", code, flags=re.M)
    code = re.sub(r"#.*?$", "", code, flags=re.M)
    return code


def extract_scriptlets(text: str) -> str:
    """Extract <% ... %> blocks (classic ASP / JSP style)."""
    parts = re.findall(r"<%[\s\S]*?%>", text)
    if not parts:
        return text
    blocks = [p[2:-2] for p in parts]  # remove <% %>
    return "\n".join(blocks)


def extract_aspx_server_scripts(text: str) -> str:
    """Extract <script runat="server"> ... </script> blocks (ASPX typical)."""
    parts = re.findall(r"<script[^>]*runat\s*=\s*[\"']server[\"'][^>]*>[\s\S]*?</script>", text, flags=re.I)
    if not parts:
        return ""
    # Strip script tags
    blocks = [re.sub(r"^<script[^>]*>|</script>$", "", p, flags=re.I).strip() for p in parts]
    return "\n".join(blocks)


def _node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


# -------------------------
# PHP: Tree-sitter
# -------------------------
def parse_php_calls(code: str) -> List[CallSite]:
    parser = get_parser("php")
    b = code.encode("utf-8", errors="ignore")
    tree = parser.parse(b)
    root = tree.root_node

    calls: List[CallSite] = []
    func_stack = ["<global>"]

    def walk(n):
        t = n.type

        if t == "function_definition":
            fn = "<anon>"
            for ch in n.children:
                if ch.type == "name":
                    fn = _node_text(b, ch)
                    break
            func_stack.append(fn)

        if t in ("function_call_expression", "call_expression"):
            fn = ""
            for ch in n.children:
                if ch.type in ("name", "qualified_name", "identifier"):
                    fn = _node_text(b, ch)
                    break
            if fn:
                calls.append(CallSite(fn, _node_text(b, n), func_stack[-1], t))

        # superglobals
        if t in ("variable_name", "name"):
            txt = _node_text(b, n)
            if any(k in txt for k in ["_GET", "_POST", "_REQUEST", "_COOKIE", "_SERVER"]):
                calls.append(CallSite(txt, txt, func_stack[-1], t))

        for ch in n.children:
            walk(ch)

        if t == "function_definition":
            func_stack.pop()

    walk(root)
    return calls


# -------------------------
# JavaScript (for ASP JScript blocks): Tree-sitter
# -------------------------
def parse_js_calls(code: str) -> List[CallSite]:
    parser = get_parser("javascript")
    b = code.encode("utf-8", errors="ignore")
    tree = parser.parse(b)
    root = tree.root_node

    calls: List[CallSite] = []
    func_stack = ["<global>"]

    def walk(n):
        t = n.type

        if t in ("function_declaration", "method_definition"):
            fn = "<func>"
            # best-effort find identifier
            for ch in n.children:
                if ch.type in ("identifier", "property_identifier"):
                    fn = _node_text(b, ch)
                    break
            func_stack.append(fn)

        if t == "call_expression":
            # callee could be identifier or member_expression
            callee = ""
            # try identifier first
            for ch in n.children:
                if ch.type == "identifier":
                    callee = _node_text(b, ch)
                    break
            if not callee:
                # member_expression like obj.run(...)
                for ch in n.children:
                    if ch.type == "member_expression":
                        callee = _node_text(b, ch)
                        break
            if callee:
                calls.append(CallSite(callee, _node_text(b, n), func_stack[-1], t))

        for ch in n.children:
            walk(ch)

        if t in ("function_declaration", "method_definition"):
            func_stack.pop()

    walk(root)
    return calls


# -------------------------
# VBScript (classic ASP default): regex-based robust callsite extraction
# -------------------------
VB_CALL_RE = re.compile(
    r"(?i)\b([A-Za-z_][A-Za-z0-9_\.]*)\s*\(([^)]*)\)|\b([A-Za-z_][A-Za-z0-9_\.]*)\b\s+([A-Za-z0-9_\"'\.\+\-_/\\\s]+)"
)

VB_FUNC_DECL_RE = re.compile(r"(?i)^\s*(function|sub)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def parse_vbscript_calls(code: str) -> List[CallSite]:
    """
    Best-effort VBScript call extraction for classic ASP:
    - captures CallName(args) patterns
    - captures 'CallName arg1, arg2' patterns
    """
    calls: List[CallSite] = []

    # Track current function/sub name by scanning lines
    current = "<global>"
    for line in code.splitlines():
        m = VB_FUNC_DECL_RE.search(line)
        if m:
            current = m.group(2)

        # Skip empty / comments
        s = line.strip()
        if not s or s.startswith("'"):
            continue

        # Find calls
        for m in VB_CALL_RE.finditer(line):
            if m.group(1) and m.group(2) is not None:
                name = m.group(1)
                args = m.group(2)
                calls.append(CallSite(name=name, args_text=f"{name}({args})", func_name=current, node_type="vb_call_paren"))
            elif m.group(3) and m.group(4):
                name = m.group(3)
                args = m.group(4).strip()
                # avoid grabbing keywords like "If", "Then"
                if name.lower() in {"if", "then", "else", "end", "for", "while", "do", "loop", "set", "dim", "call"}:
                    continue
                calls.append(CallSite(name=name, args_text=f"{name} {args}", func_name=current, node_type="vb_call_space"))

    return calls


# -------------------------
# ASP: detect VBScript vs JScript inside scriptlets
# -------------------------
def guess_asp_script_language(script_text: str) -> str:
    """
    Best-effort:
    - If contains 'function(' or 'var ' or ';' in patterns => js
    - Else => vb
    """
    t = script_text.lower()
    if "function" in t or "var " in t or "let " in t or "const " in t or ";" in script_text:
        return "js"
    # VBScript common tokens
    if "dim " in t or "set " in t or "end if" in t or "createobject" in t:
        return "vb"
    return "vb"


def parse_asp_calls(text: str) -> Tuple[str, List[CallSite]]:
    """
    For .asp/.aspx:
    - extract <% %> blocks
    - [MODIFIED] Apply deobfuscation (Chr restoration)
    - determine js/vb and parse accordingly
    Returns lang key: "asp_vb" or "asp_js"
    """
    # 1. 提取脚本片段
    script = extract_scriptlets(text)

    # If ASPX server script exists, append it (if any)
    server = extract_aspx_server_scripts(text)
    if server:
        script = script + "\n" + server

    # ==============================================================
    # [NEW] 方案一：ASP 去混淆预处理
    # 使用局部导入(Local Import)以避免与 behavior_extractor.py 循环引用
    # ==============================================================
    try:
        from .behavior_extractor import BehaviorExtractor
        # 还原 Chr(99)&Chr(109)&Chr(100) -> "cmd"
        # 这样下面的 parse_vbscript_calls 就能正则匹配到 execute("cmd") 而不是 execute(Chr...)
        script = BehaviorExtractor.deobfuscate_asp(script)
    except ImportError:
        pass # 理论上不应发生，除非文件缺失
    # ==============================================================

    # 2. 判断脚本语言 (VB vs JS)
    lang = guess_asp_script_language(script)
    if lang == "js":
        return "asp_js", parse_js_calls(script)
    return "asp_vb", parse_vbscript_calls(script)


def guess_lang_by_content(text: str) -> str:
    """
    Heuristic language guess for .txt or unknown extensions.
    Returns: "php" or "asp" (classic) or "unknown"
    """
    t = text.lower()

    # --- strong PHP indicators ---
    php_hits = 0
    if "<?php" in t:
        php_hits += 3
    if re.search(r"\$_(get|post|request|cookie|server)\b", t):
        php_hits += 2
    if re.search(r"\b(base64_decode|gzinflate|shell_exec|eval|assert)\b", t):
        php_hits += 2
    if "$" in text:  # PHP vars
        php_hits += 1

    # --- strong classic ASP indicators ---
    asp_hits = 0
    if "<%" in text and "%>" in text:
        asp_hits += 3
    if re.search(r"\b(request|response|server)\b", t):
        asp_hits += 2
    if re.search(r"\bcreateobject\b", t):
        asp_hits += 3
    if re.search(r"\b(adodb\.stream|msxml2\.xmlhttp|serverxmlhttp)\b", t):
        asp_hits += 3
    if re.search(r"\bexecute(global)?\b", t):
        asp_hits += 2

    if php_hits == 0 and asp_hits == 0:
        return "unknown"
    return "php" if php_hits >= asp_hits else "asp"


def parse_file(path: str, text: str) -> Tuple[str, List[CallSite]]:
    lower = path.lower()
    # known extensions first
    if lower.endswith(".php"):
        return "php", parse_php_calls(text)
    if lower.endswith(".asp") or lower.endswith(".aspx"):
        return parse_asp_calls(text)

    # unknown or .txt -> guess by content
    guessed = guess_lang_by_content(text)
    if guessed == "php":
        return "php", parse_php_calls(text)
    if guessed == "asp":
        # 调用 parse_asp_calls 从而也会触发里面的去混淆逻辑
        return parse_asp_calls(text)

    raise ValueError(f"Cannot determine language for file: {path}")