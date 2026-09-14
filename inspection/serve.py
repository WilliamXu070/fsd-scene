"""Loopback paired validation inspector; original prediction API stays unchanged."""
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from collections import OrderedDict
import argparse,importlib.util,json,mimetypes,threading
from fsd.sealing import sha256,source_hash

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_replay',ROOT/'scripts/serve_demo.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
BUNDLE=ROOT/'artifacts/paired-inspection';DATA=ROOT/'data/kitti360'
class ReferenceStore:
    def __init__(self):
        self.manifest=json.loads((BUNDLE/'manifest.json').read_text());self.cache=OrderedDict();self.lock=threading.RLock()
        assert self.manifest['split']=='val' and self.manifest['source_sha256']==source_hash()
        assert sha256(ROOT/'artifacts/final-measurements/reference/scenes.jsonl')==self.manifest['predictions_sha256']
    def frame(self,i):
        with self.lock:return self._frame(i)
    def _frame(self,i):
        if not 0<=i<len(self.manifest['frames']):raise IndexError('Reference frame unavailable')
        if i not in self.cache:
            p=BUNDLE/'frames'/f'{i:04d}.json';assert sha256(p)==self.manifest['frame_sha256'][str(i)]
            self.cache[i]=json.loads(p.read_text())
            while len(self.cache)>3:self.cache.popitem(last=False)
        return self.cache[i]

class Handler(replay.ReplayHandler):
    def __init__(self,*a,references,**kw):self.references=references;super().__init__(*a,**kw)
    def do_GET(self):
        url=urlparse(self.path);query=parse_qs(url.query)
        try:
            if url.path in ('/','/inspect.js','/inspect.css'):
                p=ROOT/'inspection'/({'/':'index.html','/inspect.js':'inspect.js','/inspect.css':'inspect.css'}[url.path]);self._send(p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'text/plain')
            elif url.path=='/prediction':
                self._send((ROOT/'viewer/index.html').read_bytes(),'text/html; charset=utf-8')
            elif url.path=='/reference/manifest':self._json(self.references.manifest)
            elif url.path=='/reference/consistency':self._send((BUNDLE/'consistency.json').read_bytes(),'application/json')
            elif url.path in ('/reference/frame','/reference/targets','/reference/lidar'):
                i=int(query.get('index',['0'])[0]);ref=self.references.frame(i)
                if url.path.endswith('/frame'):
                    # A reference cannot silently drift from the prediction viewport.
                    assert ref['predictions']==self.store.raw_frame(i),'Prediction/reference identity changed'
                    self._json(ref)
                else:
                    kind='cache' if url.path.endswith('/targets') else 'lidar';p=replay.safe_child(DATA,ref['provenance'][kind+'_path']);assert sha256(p)==ref['provenance'][kind+'_sha256']
                    self._send(p.read_bytes(),'application/octet-stream')
            elif url.path.startswith('/reference/'):
                self._json({'error':'Unknown reference route'},404)
            else:super().do_GET()
        except (ValueError,IndexError,KeyError):self._json({'error':'Invalid reference request'},400)
        except (AssertionError,OSError) as e:self._json({'error':str(e)},409)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8773);a=p.parse_args()
    references=ReferenceStore();store=replay.ReplayStore(ROOT/'artifacts/final-measurements/reference/scenes.jsonl',DATA,BUNDLE/'replay-metadata.json')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,store=store,references=references));server.daemon_threads=True
    print(f'Paired validation inspector http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
