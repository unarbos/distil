#!/usr/bin/env python3
"""
Chat server for the king model. Runs on GPU pod, port 8100.
Supports streaming, thinking/answer separation, concurrent requests.
~8GB VRAM for a 4B model.
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

# Lock for sequential GPU access (model isn't thread-safe for generate)
_gen_lock = threading.Lock()


def _split_thinking(text):
    """Split response into thinking and answer parts."""
    # <think>...</think> pattern (greedy — grab everything inside)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
        return thinking, answer if answer else "(model stopped during thinking)"
    
    # Model started with <think> but didn't close it (hit max_tokens)
    if text.lstrip().startswith("<think>"):
        content = text.lstrip()[7:].strip()  # Remove <think>
        return content, "(thinking was cut short — try a longer max_tokens)"
    
    # "Thinking Process:" or similar headers
    for header in ["Thinking Process:", "Thought:", "Reasoning:", "Let me think", "**Thinking"]:
        if text.startswith(header):
            parts = re.split(r'\n\n(?=[A-Z]|\*\*)', text, maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    
    # Qwen3.5 sometimes starts with reasoning before </think>
    if "\n</think>\n" in text:
        parts = text.split("\n</think>\n", 1)
        return parts[0].strip(), parts[1].strip()
    
    return None, text


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = min(body.get("max_tokens", 512), 1024)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
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
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        tps = n_tokens / elapsed if elapsed > 0 else 0

        thinking, answer = _split_thinking(response_text)

        result = {
            "choices": [{
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "model": MODEL_NAME,
            "usage": {
                "completion_tokens": n_tokens,
                "tokens_per_second": round(tps, 1),
                "generation_time_s": round(elapsed, 2),
            },
        }
        if thinking:
            result["thinking"] = thinking

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _stream_response(self, gen_kwargs, input_len):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs["streamer"] = streamer

        t0 = time.time()
        n_tokens = 0
        full_text = ""
        in_thinking = None  # None = not determined, True = in thinking, False = in answer

        def generate():
            with _gen_lock:
                with torch.no_grad():
                    model.generate(**gen_kwargs)

        thread = threading.Thread(target=generate)
        thread.start()

        try:
            for chunk in streamer:
                full_text += chunk
                n_tokens += len(tokenizer.encode(chunk, add_special_tokens=False))
                elapsed = time.time() - t0
                tps = n_tokens / elapsed if elapsed > 0 else 0

                # Determine if we're in thinking or answer phase
                if in_thinking is None:
                    for header in ["<think>", "Thinking Process:", "Thought:", "Reasoning:", "Let me think", "**Thinking"]:
                        if full_text.startswith(header):
                            in_thinking = True
                            break
                    if in_thinking is None and len(full_text) > 20:
                        in_thinking = False

                # Check for thinking→answer transition
                phase = "thinking" if in_thinking else "answer"
                if in_thinking:
                    if "</think>" in full_text:
                        in_thinking = False
                        phase = "answer"
                    elif "\n\n" in full_text and len(full_text) > 100:
                        # Heuristic: double newline after substantial thinking = transition
                        remaining = full_text.split("\n\n", 1)[-1]
                        if remaining and remaining[0].isupper():
                            in_thinking = False
                            phase = "answer"

                event = {
                    "choices": [{
                        "delta": {"content": chunk, "phase": phase},
                        "finish_reason": None,
                    }],
                    "usage": {"tokens_per_second": round(tps, 1)},
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            thread.join()

        elapsed = time.time() - t0
        tps = n_tokens / elapsed if elapsed > 0 else 0
        done_event = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "completion_tokens": n_tokens,
                "tokens_per_second": round(tps, 1),
                "generation_time_s": round(elapsed, 2),
            },
        }
        try:
            self.wfile.write(f"data: {json.dumps(done_event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": MODEL_NAME,
                "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
            }).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Silent


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread for concurrent access."""
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._handle, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


if __name__ == "__main__":
    print(f"[chat] Serving on port {PORT} (threaded)", flush=True)
    server = ThreadedHTTPServer(("0.0.0.0", PORT), ChatHandler)
    server.serve_forever()
