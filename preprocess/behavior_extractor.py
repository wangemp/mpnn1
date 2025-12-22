import re
import math
import os
import sys
import torch
import numpy as np
from dataclasses import dataclass
from typing import List

# 强制依赖 transformers
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("[Error] 未找到 transformers 库，请执行: pip install transformers")
    sys.exit(1)

from .parser import CallSite


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


@dataclass
class BehaviorNode:
    behav: str
    call_name: str
    func_name: str
    feat: List[float]


class BehaviorExtractor:
    def __init__(self, behavior_types: List[str], 
                 local_path: str = None, 
                 remote_name: str = "microsoft/codebert-base", 
                 device: str = "cpu"):
        """
        :param local_path: 本地模型存储路径
        :param remote_name: HuggingFace 远程模型名称
        :param device: 计算设备 (cuda/cpu)
        """
        if "OBFUSCATION" not in behavior_types:
            behavior_types.append("OBFUSCATION")
            
        self.behavior_types = behavior_types
        self.beh2id = {b: i for i, b in enumerate(behavior_types)}
        self.device = device
        
        # =======================================================
        # [核心逻辑] 自动下载并保存到项目文件夹
        # =======================================================
        
        # 简单检查 local_path 下是否有 config.json，以此判断是否已下载
        is_downloaded = local_path and os.path.exists(os.path.join(local_path, "config.json"))

        if is_downloaded:
            print(f"[Info] 发现本地 CodeBERT 模型，正在从 {local_path} 加载...")
            load_path = local_path
        else:
            print(f"[Info] 本地未找到模型，准备从 HuggingFace 下载: {remote_name}")
            print(f"[Info] 下载完成后将自动保存至: {local_path}")
            load_path = remote_name

        try:
            # 1. 加载模型 (如果是远程名，会自动下载到缓存)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModel.from_pretrained(load_path).to(self.device)
            self.model.eval() # 冻结参数，只用于特征提取

            # 2. 如果是从网上下载的，且指定了保存路径，则保存一份到项目里
            if not is_downloaded and local_path:
                print(f"[Info] 正在将模型保存到项目目录: {local_path} ...")
                os.makedirs(local_path, exist_ok=True)
                self.tokenizer.save_pretrained(local_path)
                self.model.save_pretrained(local_path)
                print("[Info] 模型保存成功！下次运行将直接使用本地文件。")

        except Exception as e:
            print(f"[Fatal Error] CodeBERT 加载/下载失败: {e}")
            print("请检查网络连接，或手动下载模型放入 pretrained_models 文件夹。")
            sys.exit(1)

        # =======================================================
        # 行为映射表 (保持不变)
        # =======================================================
        self.php_decode = {"base64_decode", "gzinflate", "gzuncompress", "urldecode", "rawurldecode", "str_rot13"}
        self.php_exec = {"eval", "assert", "system", "exec", "passthru", "shell_exec", "popen", "proc_open"}
        self.php_file = {"fopen", "fwrite", "file_put_contents", "unlink", "chmod", "copy", "move_uploaded_file"}
        self.php_net = {"curl_init", "curl_exec", "fsockopen", "socket_create", "socket_connect"}

        self.asp_input_tokens = {
            "request", "request.querystring", "request.form", "request.cookies", "request.servervariables",
            "server.urldecode", "server.urlencode"
        }
        self.asp_exec = {"execute", "eval", "executeglobal", "shell", "wscript.shell", "run"}
        self.asp_file = {"filesystemobject", "scripting.filesystemobject", "createtextfile", "opentextfile", "deletefile", "adodb.stream", "savetofile"}
        self.asp_net = {"msxml2.serverxmlhttp", "msxml2.xmlhttp", "serverxmlhttp", "xmlhttp", "open", "send", "adodb.connection", "adodb.recordset"}
        self.asp_decode = {"base64", "decode", "urldecode"} 

        self.string_ops = {"concat", "replace", "substring", "split", "join", "format", "sprintf", "mid", "left", "right"}
        self.crypto = {"xor", "aes", "rc4", "des", "md5", "sha"}

    @staticmethod
    def deobfuscate_asp(source_code: str) -> str:
        if not source_code: return ""
        def decode_chr(match):
            try:
                val = int(match.group(1))
                if 32 <= val <= 126: return chr(val)
                return match.group(0) 
            except: return match.group(0)
        pattern_chr = re.compile(r"chr\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
        prev_code = ""
        while source_code != prev_code:
            prev_code = source_code
            source_code = pattern_chr.sub(decode_chr, source_code)
        pattern_concat = re.compile(r"[\"']\s*&\s*[\"']")
        source_code = pattern_concat.sub("", source_code)
        return source_code

    def _get_codebert_embedding(self, text: str) -> List[float]:
        """
        生成 768维 CodeBERT 向量
        """
        if not text or not text.strip():
            return [0.0] * 768
            
        with torch.no_grad():
            # 截断长度设为 512，防止显存溢出
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # 取 [CLS] token 作为句向量
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding.cpu().numpy()[0].tolist()

    def _norm_feat(self, x: float, cap: float) -> float:
        return float(min(x, cap)) / cap

    def map_call_to_behavior(self, lang: str, call: CallSite) -> str:
        n = call.name.strip()
        n_low = n.lower()
        args_text = call.args_text.lower() if call.args_text else ""
        
        if lang == "php":
            if any(k in n for k in ["_GET", "_POST", "_REQUEST", "_COOKIE", "_SERVER"]): return "INPUT"
            if n_low in self.php_decode: return "DECODE"
            if n_low in self.php_exec: return "EXECUTE"
            if n_low in self.php_file: return "FILE_OP"
            if n_low in self.php_net: return "NET_OP"

        if lang in {"asp_vb", "asp_js"}:
            if n_low == "chr" or "chr(" in args_text: return "OBFUSCATION"
            if "&h" in args_text: return "OBFUSCATION"
            if any(tok in n_low for tok in self.asp_input_tokens) or n_low.startswith("request"): return "INPUT"
            if any(k in n_low for k in self.asp_decode) or "urldecode" in n_low: return "DECODE"
            if n_low in self.asp_exec or "wscript.shell" in n_low or "runtime" in n_low: return "EXECUTE"
            if any(k in n_low for k in self.asp_file) or "filesystemobject" in n_low: return "FILE_OP"
            if any(k in n_low for k in self.asp_net) or "xmlhttp" in n_low: return "NET_OP"

        if n_low in self.string_ops: return "STRING_OP"
        if any(k in n_low for k in self.crypto): return "CRYPTO"
        return "OTHER"

    def build_nodes(self, lang: str, calls: List[CallSite]) -> List[BehaviorNode]:
        nodes: List[BehaviorNode] = []
        for c in calls:
            beh = self.map_call_to_behavior(lang, c)
            if beh not in self.beh2id:
                beh = "OTHER"
            
            # 1. 语义特征 (CodeBERT 768维)
            # 构造上下文: "EXECUTE eval $_POST['cmd']"
            context_text = f"{beh} {c.name} {c.args_text}"
            sem_feat = self._get_codebert_embedding(context_text)

            # 2. 统计特征 (人工特征 5维)
            name_len = len(c.name)
            arg_len = len(c.args_text)
            arg_ent = _entropy(c.args_text[:2000])
            has_long_str = 1.0 if re.search(r"['\"][^'\"]{80,}['\"]", c.args_text) else 0.0
            has_base64_like = 1.0 if re.search(r"[A-Za-z0-9+/]{80,}={0,2}", c.args_text) else 0.0

            stat_feat = [
                self._norm_feat(name_len, 50),
                self._norm_feat(arg_len, 800),
                self._norm_feat(min(arg_ent, 8.0), 8.0),
                has_long_str,
                has_base64_like,
            ]
            
            # 最终拼接：773 维
            final_feat = sem_feat + stat_feat
            nodes.append(BehaviorNode(beh, c.name, c.func_name, final_feat))

        if not nodes:
            zero_feat = [0.0] * (768 + 5)
            nodes.append(BehaviorNode("OTHER", "none", "<global>", zero_feat))

        return nodes