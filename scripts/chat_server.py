#!/usr/bin/env python3
"""
Chat server for the king model. Runs on GPU pod, port 8100.
Features: SSE streaming, thinking/answer split, concurrent requests, no token cap.
~8GB VRAM for a 4B model. HF transformers backend (~37 tok/s on B200).
"""
import json
import sys
import time
import re
import torch
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "aceini/q-dist"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

print(f"[chat] Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
vram_gb = round(torch.cuda.memory_allocated() / 1e9, 1)
print(f"[chat] Model loaded. VRAM: {vram_gb}GB", flush=True)

_gen_lock = threading.Lock()


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2048)  # No hard cap
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
            parts.append("assistant:")
            text = "\n".join(parts)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            repetition_penalty=1.1,
        )

        if stream:
            self._stream_response(gen_kwargs, input_len)
        else:
            self._sync_response(gen_kwargs, input_len)

    def _sync_response(self, gen_kwargs, input_len):
        t0 = time.time()
        with _gen_lock:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
        elapsed = time.time() - t0
        new_tokens = output[0][input_len:]
        n_tokens = len(new_tokens)
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        # Strip special tokens but keep <think>/<\/think>
        for st in getattr(tokenizer, 'all_special_tokens', []):
            if st not in ("<think>", "</think>"):
                raw = raw.replace(st, "")
        tps = n_tokens / elapsed if elapsed > 0 else 0

        thinking, answer = _split_thinking(raw)

        result = {
            "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "model": MODEL_NAME,
            "usage": {"completion_tokens": n_tokens, "tokens_per_second": round(tps, 1), "generation_time_s": round(elapsed, 2)},
        }
        if thinking:
            result["thinking"] = thinking

        self._send_json(200, result)

    def _stream_response(self, gen_kwargs, input_len):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False, skip_prompt=True)
        gen_kwargs["streamer"] = streamer

        t0 = time.time()
        n_tokens = [0]
        full_text = []
        # Phase detection: only use <think> tags (reliable).
        # For models that use "Thinking Process:" style, we don't try to split
        # in streaming — the full split happens server-side when stream=false.
        phase = ["answer"]  # default to answer
        think_done = [False]
        has_think_tags = [False]

        def generate():
            with _gen_lock:
                with torch.no_grad():
                    model.generate(**gen_kwargs)

        thread = threading.Thread(target=generate)
        thread.start()

        # Build list of special token strings to strip from output
        _special_strs = set()
        if hasattr(tokenizer, 'all_special_tokens'):
            _special_strs = set(tokenizer.all_special_tokens)
        # Always strip common ones
        _special_strs.update(["<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|end|>"])

        try:
            for chunk in streamer:
                # Strip special tokens from chunk
                clean_chunk = chunk
                for st in _special_strs:
                    clean_chunk = clean_chunk.replace(st, "")
                if not clean_chunk:
                    continue

                full_text.append(clean_chunk)
                joined = "".join(full_text)
                n_tokens[0] += max(1, len(tokenizer.encode(chunk, add_special_tokens=False)))
                elapsed = time.time() - t0
                tps = n_tokens[0] / elapsed if elapsed > 0 else 0

                # Phase detection: only handle explicit <think>/<\/think> tags
                if not think_done[0]:
                    if "<think>" in joined and not has_think_tags[0]:
                        has_think_tags[0] = True
                        phase[0] = "thinking"

                    if has_think_tags[0] and "</think>" in joined:
                        think_done[0] = True
                        phase[0] = "answer"
                        after = joined.split("</think>", 1)[1].strip()
                        self._sse({"choices": [{"delta": {"phase": "answer"}, "finish_reason": None}], "usage": {"tokens_per_second": round(tps, 1)}})
                        if after:
                            self._sse({"choices": [{"delta": {"content": after, "phase": "answer"}, "finish_reason": None}], "usage": {"tokens_per_second": round(tps, 1)}})
                        continue

                # Strip think tags from output
                out = clean_chunk.replace("<think>", "").replace("</think>", "")
                if not out:
                    continue

                self._sse({
                    "choices": [{"delta": {"content": out, "phase": phase[0]}, "finish_reason": None}],
                    "usage": {"tokens_per_second": round(tps, 1)},
                })
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            thread.join()

        elapsed = time.time() - t0
        tps = n_tokens[0] / elapsed if elapsed > 0 else 0
        try:
            # For models without <think> tags, split thinking from answer retroactively
            final_text = "".join(full_text)
            thinking_text, answer_text = _split_thinking(final_text)

            done_event = {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": n_tokens[0], "tokens_per_second": round(tps, 1), "generation_time_s": round(elapsed, 2)},
            }
            if thinking_text:
                done_event["thinking"] = thinking_text
                done_event["answer"] = answer_text
            self._sse(done_event)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok", "model": MODEL_NAME,
                "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
            })
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _split_thinking(text):
    """Split thinking from answer. Handles <think> tags and 'Thinking Process:' headers."""
    # 1. Explicit <think>...</think> tags
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer if answer else "(stopped during thinking)"
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[7:].strip(), "(thinking cut short)"

    # 2. "Thinking Process:" / "Thought:" / "Reasoning:" style headers
    # The model outputs structured thinking then transitions to the actual answer
    # Pattern: thinking block → double newline → answer (often starts differently)
    for header in ["Thinking Process:", "**Thinking Process:**", "Thought:", "Reasoning:", "Let me think"]:
        if text.strip().startswith(header):
            # Find the answer after the thinking block ends
            # Look for patterns like: numbered list ending → double newline → non-list content
            # Or: thinking block → "---" → answer
            # Or: "Draft:" / "Response:" / "Answer:" / "Final" section that's the actual output
            answer_markers = [
                r'\n\n---\n',
                r'\n\n(?:(?:Final )?(?:Answer|Response|Output|Result)[:\s])',
                r'\n\n(?:Here\'s|Here is)',
            ]
            for pattern in answer_markers:
                match = re.search(pattern, text)
                if match:
                    thinking = text[:match.start()].strip()
                    answer = text[match.end():].strip() if text[match.end():].strip() else text[match.start():].strip()
                    return thinking, answer

            # Fallback: find last double-newline followed by short non-list content
            # This catches cases where thinking ends and a clean answer starts
            parts = text.rsplit('\n\n', 1)
            if len(parts) == 2 and not parts[1].strip().startswith(('*', '-', '#', 'Option')):
                last_block = parts[1].strip()
                # If the last block looks like an actual answer (not another thinking step)
                if len(last_block) > 10 and not any(last_block.startswith(m) for m in ['*', '-', '1.', '2.', '3.', '4.']):
                    return parts[0].strip(), last_block

            # If we can't find a clean split, return everything as thinking
            return text.strip(), "(thinking — answer not yet generated)"

    return None, text


class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address), daemon=True)
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


if __name__ == "__main__":
    print(f"[chat] Serving on port {PORT} (threaded, streaming)", flush=True)
    server = ThreadedHTTPServer(("0.0.0.0", PORT), ChatHandler)
    server.serve_forever()
