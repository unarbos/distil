#!/usr/bin/env python3
"""
Lightweight chat server for the king model.
Runs on the GPU pod, serves OpenAI-compatible chat completions via HF transformers.
~8GB VRAM for a 4B model. Fast enough for interactive chat (~2s/response).
"""
import json
import sys
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "aceini/q-dist"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

print(f"[chat] Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print(f"[chat] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)


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

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concat
            parts = []
            for m in messages:
                role = m.get("role", "user")
                parts.append(f"{role}: {m.get('content', '')}")
            parts.append("assistant:")
            text = "\n".join(parts)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=1.1,
            )

        new_tokens = output[0][input_len:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        result = {
            "choices": [{
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "model": MODEL_NAME,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
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
        # Quiet logging — only errors
        if args and "200" not in str(args[0]):
            print(f"[chat] {args}", flush=True)


if __name__ == "__main__":
    print(f"[chat] Serving on port {PORT}", flush=True)
    server = HTTPServer(("0.0.0.0", PORT), ChatHandler)
    server.serve_forever()
