#!/usr/bin/env python3
"""Stub ComfyUI HTTP server for testing infer.py runner.

Mimics /prompt, /history/{prompt_id}, and /view endpoints, returning valid mp4 video bytes.
Accepts standard ComfyUI CLI flags: --listen, --port, --cache-none.
"""

import argparse
import http.server
import io
import json
import os
import signal
import socketserver
import sys
import urllib.parse
import uuid

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Pre-computed minimal valid MP4 bytes (1 frame, 64x64, h264)
MINIMAL_VALID_MP4 = (
    b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41'
    b'\x00\x00\x00\x08free\x00\x00\x00(mdat\x00\x00\x00\x02\t\x10\x00\x00\x00\x16f\xe0\x00'
    b'\x01\x11`\x00\x00\x03\x00\x01\x00\x00\x03\x002\x8f\x16)\xc0'
)


def get_minimal_mp4() -> bytes:
    return MINIMAL_VALID_MP4


def generate_png_frames(num_frames: int = 5, width: int = 960, height: int = 544) -> list:
    """Generate multiple PNG frames for testing ffmpeg encoding."""
    frames = []
    if HAS_PIL:
        for i in range(num_frames):
            img = Image.new('RGB', (width, height), color=(i * 50 % 256, i * 30 % 256, i * 70 % 256))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            frames.append(buf.getvalue())
    else:
        # Fallback: minimal PNG (1x1 pixel)
        minimal_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
        frames = [minimal_png] * num_frames
    return frames


class StubComfyHandler(http.server.BaseHTTPRequestHandler):
    prompts = {}

    def log_message(self, format, *args):
        # Silence standard request logging in test stdout
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/system_stats"):
            body = json.dumps({"system": {"os": "linux", "status": "ok"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path.startswith("/history/"):
            prompt_id = path[len("/history/"):]
            info = self.prompts.get(prompt_id)
            if info is None:
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if info.get("error"):
                resp = {
                    prompt_id: {
                        "status": {
                            "status_str": "error",
                            "completed": False,
                            "messages": [["Execution Error", info.get("error_msg", "Stub execution error")]],
                        },
                        "outputs": {},
                    }
                }
            else:
                # Return multiple frames as SaveImage output
                num_frames = 5
                for node in (self.prompts.get(prompt_id, {}).get("data", {}).get("prompt", {}) or {}).values():
                    if isinstance(node, dict) and isinstance(node.get("inputs", {}).get("length"), int):
                        num_frames = node["inputs"]["length"]
                frame_entries = []
                for i in range(num_frames):
                    frame_entries.append({
                        "filename": f"{prompt_id}_frame_{i:04d}.png",
                        "subfolder": "",
                        "type": "output",
                    })
                resp = {
                    prompt_id: {
                        "status": {
                            "status_str": "success",
                            "completed": True,
                            "messages": [],
                        },
                        "outputs": {
                            "17": {
                                "images": frame_entries
                            }
                        },
                    }
                }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/view":
            filename = query.get("filename", ["output.mp4"])[0]
            # Check if it's a frame PNG
            if filename.endswith(".png"):
                # Generate PNG frames on demand
                frames = generate_png_frames(5)
                # Extract frame index from filename
                frame_idx = 0
                try:
                    frame_idx = int(filename.split("_frame_")[-1].split(".")[0])
                except:
                    frame_idx = 0
                if frame_idx < len(frames):
                    png_bytes = frames[frame_idx]
                else:
                    png_bytes = frames[0]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png_bytes)))
                self.end_headers()
                self.wfile.write(png_bytes)
                return
            else:
                # Return MP4 for backwards compatibility
                mp4_bytes = get_minimal_mp4()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(mp4_bytes)))
                self.end_headers()
                self.wfile.write(mp4_bytes)
                return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/prompt":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                data = json.loads(body_bytes.decode())
            except Exception:
                data = {}

            prompt_id = str(uuid.uuid4())
            prompt_dict = data.get("prompt", {})
            client_id = data.get("client_id", "")

            # Check if this prompt should trigger an error
            is_error = False
            error_msg = ""
            if os.environ.get("STUB_COMFY_ERROR") == "1":
                is_error = True
                error_msg = "STUB_COMFY_ERROR env active"
            elif client_id == "trigger_error":
                is_error = True
                error_msg = "Client requested error"
            else:
                prompt_str = json.dumps(prompt_dict)
                if "TRIGGER_ERROR" in prompt_str:
                    is_error = True
                    error_msg = "TRIGGER_ERROR found in prompt payload"

            self.prompts[prompt_id] = {
                "data": data,
                "error": is_error,
                "error_msg": error_msg,
            }

            resp_data = json.dumps({"prompt_id": prompt_id, "number": 1, "node_errors": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def run_server(listen: str, port: int):
    # Allow socket reuse
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer((listen, port), StubComfyHandler)

    def handle_signal(sig, frame):
        try:
            server.server_close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        server.serve_forever()
    except Exception:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Stub ComfyUI server")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cache-none", action="store_true", help="ComfyUI cache flag (ignored)")
    args, unknown = parser.parse_known_args()

    run_server(args.listen, args.port)


if __name__ == "__main__":
    main()