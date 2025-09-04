# -*- coding: utf-8 -*-
"""
segmentation.inference.firered_client_impl  (Parse Fix Patch)
- 修复：单段 CLI 模式原来把 --output 当 JSON 读，实际上 speech2text.py 写的是 TSV（uttid \t text）→ 导致 text 为空
- 现在：infer_aed_cli / infer_llm_cli 都解析 TSV；若 TSV 缺失则回退解析 stdout 的 dict 打印
- 仍保留：infer_aed_cli_batch （--wav_scp 批量）
- 继承硬化：环境隔离、faulthandler、CPU 回退、详细日志、超时
"""
import os, json, tempfile, subprocess, shlex, time, datetime, uuid, re
from typing import List, Dict, Any

def _abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p))) if p else p

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_out_path(tmpdir: str, ext: str) -> str:
    os.makedirs(tmpdir, exist_ok=True)
    return os.path.join(tmpdir, f"firered_out_{uuid.uuid4().hex[:8]}.{ext}")

def _preflight(cfg: dict, need_llm: bool=False):
    repo  = _abspath(cfg.get("repo",""))
    model = _abspath(cfg.get("model",""))
    llm   = _abspath(cfg.get("llm","")) if need_llm else None
    errs=[]
    if not os.path.isdir(repo):  errs.append(f"[repo] not found: {repo}")
    if not os.path.isdir(model): errs.append(f"[model] not found: {model}")
    if need_llm and (not llm or not os.path.isdir(llm)):
        errs.append(f"[llm] not found: {llm}")
    if errs:
        raise FileNotFoundError("FireRed CLI preflight failed:\n  - " + "\n  - ".join(errs))
    return repo, model, llm

def _tmpl(fmt: str, **kwargs) -> str:
    return " ".join(str(fmt).format(**kwargs).split())

def _env_base():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return env

def _run_cmd(cmd_str: str, timeout: int=None, cwd: str|None=None, tag: str="") -> str:
    parts = shlex.split(cmd_str, posix=True)
    timeout = timeout or int(os.environ.get("FIRERED_CLI_TIMEOUT", "1800"))
    env = _env_base()
    t0 = time.time()
    start = datetime.datetime.now().strftime("%H:%M:%S")
    preview = " ".join(parts[:12]) + (" ..." if len(parts) > 12 else "")
    print(f"[FireRedCLI][{tag}] ▶ {start}  CWD={cwd}\n  $ {preview}")
    try:
        proc = subprocess.run(parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, text=True, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[FireRedCLI][{tag}] TIMEOUT after {timeout}s\nCMD: {cmd_str}\nCWD:{cwd}")
    dt = time.time() - t0
    end = datetime.datetime.now().strftime("%H:%M:%S")
    if proc.returncode != 0:
        print(f"[FireRedCLI][{tag}] ✖ {end}  took {dt:.1f}s")
        raise RuntimeError(f"CMD failed ({proc.returncode}): {cmd_str}\nCWD:{cwd}\nSTDERR:\n{proc.stderr}")
    print(f"[FireRedCLI][{tag}] ✓ {end}  took {dt:.1f}s")
    return proc.stdout

def _parse_tsv_last_text(path: str) -> str|None:
    try:
        if not os.path.exists(path): return None
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines: return None
        # 单段模式只会有一行，取最后一行更稳健
        last = lines[-1]
        parts = last.split("\t", 1)
        if len(parts) == 2:
            return parts[1]
        # 兜底：整行即文本
        return last
    except Exception:
        return None

def _parse_stdout_text(stdout: str) -> str|None:
    # 解析 speech2text.py 打印的 dict：{'uttid': 'xxx', 'text': '...'} 或 JSON 形式
    m = re.search(r"'text'\s*:\s*'([^']*)'", stdout, re.S)
    if not m:
        m = re.search(r'"text"\s*:\s*"([^"]*)"', stdout, re.S)
    return m.group(1) if m else None

def _maybe_force_cpu(cmd: str) -> str:
    if os.environ.get("FIRERED_FORCE_CPU", "0") == "1":
        return cmd.replace("--use_gpu 1", "--use_gpu 0")
    return cmd

def infer_llm_cli(items: List[Dict[str,Any]], device_id: str, config_path: str) -> List[Dict[str,Any]]:
    cfg = _load_yaml(config_path)["llm"]
    repo, model, llm = _preflight(cfg, need_llm=True)
    outs=[]
    with tempfile.TemporaryDirectory() as td:
        for idx, it in enumerate(items):
            wav = _abspath(it.get("cache_wav") or it["wav"])
            tsv_path = _abspath(_ensure_out_path(td, "tsv"))
            cmd = _tmpl(cfg["cmd_template"], repo=repo, model=model, llm=llm, wav=wav, json=tsv_path, txt=tsv_path)
            cmd = _maybe_force_cpu(cmd)
            stdout = _run_cmd(cmd, cwd=repo, tag=f"LLM s{idx} [{it['start']:.2f}-{it['end']:.2f}s]")
            text = _parse_tsv_last_text(tsv_path)
            if text is None:
                text = _parse_stdout_text(stdout) or ""
            outs.append({"start": it["start"], "end": it["end"], "text": text, "conf": 0.0, "cache_wav": it.get("cache_wav")})
    return outs

def infer_aed_cli(items: List[Dict[str,Any]], device_id: str, config_path: str) -> List[Dict[str,Any]]:
    cfg = _load_yaml(config_path)["aed"]
    repo, model, _ = _preflight(cfg, need_llm=False)
    outs=[]
    with tempfile.TemporaryDirectory() as td:
        for idx, it in enumerate(items):
            wav = _abspath(it.get("cache_wav") or it["wav"])
            tsv_path = _abspath(_ensure_out_path(td, "tsv"))
            cmd = _tmpl(cfg["cmd_template"], repo=repo, model=model, wav=wav, json=tsv_path, txt=tsv_path)
            cmd = _maybe_force_cpu(cmd)
            stdout = _run_cmd(cmd, cwd=repo, tag=f"AED s{idx} [{it['start']:.2f}-{it['end']:.2f}s]")
            text = _parse_tsv_last_text(tsv_path)
            if text is None:
                text = _parse_stdout_text(stdout) or ""
            outs.append({"start": it["start"], "end": it["end"], "text": text, "conf": 0.0, "cache_wav": it.get("cache_wav")})
    return outs

def infer_aed_cli_batch(items: List[Dict[str,Any]], device_id: str, config_path: str) -> List[Dict[str,Any]]:
    cfg_all = _load_yaml(config_path)
    if "aed_batch" not in cfg_all:
        raise KeyError("aed_batch not found in YAML; please add 'aed_batch' section as in the patch README.")
    cfg = cfg_all["aed_batch"]
    repo, model, _ = _preflight(cfg, need_llm=False)
    batch = int(cfg.get("batch_size", 8))

    outs=[]
    with tempfile.TemporaryDirectory() as td:
        scp = os.path.join(td, "batch.scp")
        tsv = os.path.join(td, f"out_{uuid.uuid4().hex[:8]}.tsv")

        id2item = {}
        with open(scp, "w", encoding="utf-8") as f:
            for it in items:
                uttid = f"utt_{uuid.uuid4().hex[:10]}"
                wav = _abspath(it.get("cache_wav") or it["wav"])
                f.write(f"{uttid}\t{wav}\n")
                id2item[uttid] = it

        cmd = _tmpl(cfg["cmd_template"], repo=repo, model=model, scp=scp, batch=batch, txt=tsv)
        cmd = _maybe_force_cpu(cmd)
        _run_cmd(cmd, cwd=repo, tag=f"AED BATCH x{len(items)}")

        if os.path.exists(tsv):
            with open(tsv, "r", encoding="utf-8") as fr:
                for line in fr:
                    line=line.strip()
                    if not line: continue
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        uttid, text = parts
                    else:
                        uttid, text = parts[0], ""
                    it = id2item.get(uttid)
                    if it:
                        outs.append({"start": it["start"], "end": it["end"], "text": text, "conf": 0.0, "cache_wav": it.get("cache_wav")})
    return outs
