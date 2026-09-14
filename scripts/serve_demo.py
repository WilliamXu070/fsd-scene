"""Loopback-only, prediction-only replay server. No target/GT data is loaded."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = PROJECT_ROOT / 'viewer'
SCENE_FIELDS = ('sequence','timestamp','frame_id','boxes','scores','labels','track_ids','road',
                'road_confidence','road_visible','road_ground_z_m','ego_to_world','inference_ms')
IMAGE_EXTENSIONS = {'.png','.jpg','.jpeg','.webp'}


def safe_child(root: Path, candidate: str | Path) -> Path:
    """Resolve symlinks and Windows traversal before opening any served file."""
    root = root.resolve()
    candidate = Path(candidate)
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path outside configured root')
    return path


class ReplayStore:
    """Index only byte offsets; decode a single prediction frame per request."""
    def __init__(self, predictions, data_root, metadata=None, fixture=False):
        self.predictions = Path(predictions).resolve()
        self.data_root = Path(data_root).resolve()
        self.metadata = Path(metadata).resolve() if metadata else None
        self.fixture = fixture
        self.offsets = []
        self.scanned = 0
        self.signature = None
        self.file_identity = None
        self.frame_cache = OrderedDict()
        self.cache_limit = 2
        self.lock = threading.RLock()
        self.image_manifest = {}
        self.manifest_signature = None

    def refresh(self):
        with self.lock:
            if not self.predictions.is_file():
                self.offsets, self.scanned, self.signature = [], 0, None
                self.file_identity = None
                self.frame_cache.clear()
                return
            stat = self.predictions.stat()
            sig = (stat.st_size, stat.st_mtime_ns)
            identity = (stat.st_dev, stat.st_ino)
            if sig == self.signature and identity == self.file_identity:
                return
            if (self.file_identity is not None and identity != self.file_identity) or stat.st_size < self.scanned or (self.signature and stat.st_size == self.signature[0]):
                self.offsets, self.scanned = [], 0
                self.frame_cache.clear()
            with self.predictions.open('rb') as handle:
                handle.seek(self.scanned)
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line or not line.endswith(b'\n'):
                        break  # Do not expose a frame while the writer is appending it.
                    if line.strip():
                        self.offsets.append(offset)
                    self.scanned = handle.tell()
            self.signature = sig
            self.file_identity = identity

    def status(self):
        self.refresh()
        metadata = {}
        if self.metadata and self.metadata.suffix.lower() == '.json' and self.metadata.is_file():
            try:
                metadata = json.loads(self.metadata.read_text(encoding='utf-8-sig'))
            except (ValueError, OSError):
                pass
        provenance = 'fixture' if self.fixture else metadata.get('provenance', 'unverified')
        if provenance not in ('fixture','trained','untrained','unverified'):
            provenance = 'unverified'
        if not self.fixture and provenance == 'fixture':
            provenance = 'unverified'
        return dict(state='ready' if self.offsets else 'waiting',frames=len(self.offsets),
                    source=self.predictions.name,provenance=provenance,
                    checkpoint=str(metadata.get('checkpoint','')),
                    note=str(metadata.get('note','')),
                    bev_min=float(metadata.get('bev_min',-40)),bev_step=float(metadata.get('bev_step',.5)),
                    ground_z=float(metadata.get('ground_z',-1.5)),
                    ground_source=str(metadata.get('ground_source','Provisional mount assumption; not measured from test labels')),
                    camera_names=metadata.get('camera_names',['Front left','Front right','Left fisheye','Right fisheye']))

    def raw_frame(self, index):
        self.refresh()
        with self.lock:
            if index < 0 or index >= len(self.offsets):
                raise IndexError('Frame index outside available replay')
            if index in self.frame_cache:
                self.frame_cache.move_to_end(index)
                return self.frame_cache[index]
            with self.predictions.open('rb') as handle:
                handle.seek(self.offsets[index])
                record = json.loads(handle.readline())
            if not isinstance(record, dict):
                raise ValueError('Prediction frame must be a JSON object')
            self.frame_cache[index] = record
            while len(self.frame_cache) > self.cache_limit:
                self.frame_cache.popitem(last=False)
            return record

    def image_path(self, index, camera):
        names=self.status()['camera_names']
        count=len(names) if isinstance(names,list) and 1<=len(names)<=8 else 4
        if camera not in range(count):
            raise IndexError('Camera index outside configured views')
        record = self.raw_frame(index)
        images = record.get('image_paths')
        candidate = images[camera] if isinstance(images,list) and len(images)>camera else None
        if not candidate:
            manifest = self.data_root / 'manifest.json'
            if manifest.is_file():
                sig = manifest.stat().st_mtime_ns
                if sig != self.manifest_signature:
                    document = json.loads(manifest.read_text(encoding='utf-8-sig'))
                    rows = document.get('samples', []) if isinstance(document, dict) else document
                    self.image_manifest = {(str(row['sequence']), int(row['frame_id'])): row.get('images', []) for row in rows}
                    self.manifest_signature = sig
                images = self.image_manifest.get((str(record.get('sequence', '')), int(record.get('frame_id', -1))), [])
                candidate = images[camera] if len(images) > camera else None
        if not candidate:
            # Support preparation's common image layout without opening targets.
            seq, frame = str(record.get('sequence','')), int(record.get('frame_id',-1))
            possibilities = [f'images/{seq}/image_{camera:02d}/{frame:010d}.png',
                             f'processed/{seq}/image_{camera:02d}/{frame:010d}.jpg',
                             f'processed/{seq}/image_{camera:02d}/{frame:010d}.png',
                             f'data_2d_raw/{seq}/image_{camera:02d}/data_rgb/{frame:010d}.png',
                             f'data_2d_raw/{seq}/image_{camera:02d}/data_rect/{frame:010d}.png']
        else:
            possibilities = [candidate]
        for possibility in possibilities:
            path = safe_child(self.data_root, possibility)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError('Only image files may be served from the dataset')
            if path.is_file():
                return path
        raise FileNotFoundError('Camera image unavailable for this frame')

    def frame(self, index):
        raw = self.raw_frame(index)
        output = {key: raw[key] for key in SCENE_FIELDS if key in raw}
        output['index'] = index
        images = []
        names=self.status()['camera_names']
        count=len(names) if isinstance(names,list) and 1<=len(names)<=8 else 4
        for camera in range(count):
            try:
                self.image_path(index,camera)
                images.append(f'/api/image?index={index}&camera={camera}')
            except (ValueError,FileNotFoundError,IndexError):
                images.append(None)
        output['images'] = images
        return output


class ReplayHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def _send(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # A newer scrub request may cancel this response.

    def _json(self, data, status=200):
        self._send(json.dumps(data,allow_nan=False,separators=(',',':')).encode(),'application/json; charset=utf-8',status)

    def do_GET(self):
        parsed=urlparse(self.path)
        query=parse_qs(parsed.query)
        try:
            if parsed.path == '/api/status':
                self._json(self.store.status())
            elif parsed.path == '/api/frame':
                self._json(self.store.frame(int(query.get('index',['0'])[0])))
            elif parsed.path == '/api/image':
                path=self.store.image_path(int(query.get('index',['0'])[0]),int(query.get('camera',['0'])[0]))
                self._send(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'image/png')
            elif parsed.path.startswith('/api/'):
                self._json({'error':'Unknown API endpoint'},404)
            else:
                relative=unquote(parsed.path).lstrip('/') or 'index.html'
                path=safe_child(VIEWER_ROOT,relative)
                if path.suffix not in {'.html','.css','.js','.txt','.svg'} or not path.is_file():
                    raise FileNotFoundError('Viewer asset unavailable')
                mime = 'text/javascript' if path.suffix == '.js' else mimetypes.guess_type(path.name)[0]
                self._send(path.read_bytes(),mime or 'application/octet-stream')
        except (ValueError,TypeError,KeyError) as exc:
            self._json({'error':str(exc)},400)
        except (IndexError,FileNotFoundError):
            self._json({'error':'Requested frame or image is unavailable'},404)
        except OSError:
            self._json({'error':'Replay file could not be read'},503)


def make_server(predictions, data_root, port=8765, metadata=None, fixture=False):
    store=ReplayStore(predictions,data_root,metadata,fixture)
    server=ThreadingHTTPServer(('127.0.0.1',port),partial(ReplayHandler,store=store))
    server.daemon_threads=True
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--data-root',default='data/kitti360')
    parser.add_argument('--metadata',help='JSON provenance metadata; checkpoint pickle files are never loaded')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--fixture',action='store_true',help='Explicitly label synthetic UI development data')
    args=parser.parse_args()
    server=make_server(args.predictions,args.data_root,args.port,args.metadata,args.fixture)
    print(f'Replay viewer: http://127.0.0.1:{server.server_port} (fixture={args.fixture})',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
