"""Auditable selective access to official KITTI-360 ZIP archives.

Only one archive is active at a time. Member payloads pass ZIP CRC verification;
manifest hashes refer to original, decompressed source bytes (not resized PNGs).
"""
from __future__ import annotations
import io
import json
import hashlib
import shutil
import struct
import time
import zipfile
import zlib
from pathlib import Path
import requests

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'
URLS = {
    'calibration': BASE+'384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip',
    'poses': BASE+'89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip',
    'boxes': BASE+'ffa164387078f48a20f0188aa31b0384bb19ce60/data_3d_bboxes.zip',
    'semantics': BASE+'6489aabd632d115c4280b978b2dcf72cb0142ad9/data_3d_semantics.zip',
}
DRIVES = ('0000','0002','0003','0004')

def sequence(drive):
    return f'2013_05_28_drive_{int(drive):04d}_sync'

def ensure_space(path, extra=0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free-extra < 50*1024**3:
        raise RuntimeError(f'Disk safeguard: {free/1024**3:.1f} GiB free; need 50 GiB reserve plus {extra} bytes')

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url=url; self.session=requests.Session(); self.position=0
        r=self.session.get(url, headers={'Range':'bytes=-131072'}, timeout=120)
        r.raise_for_status()
        if r.status_code!=206:
            raise RuntimeError('Official archive does not support selective HTTP ranges')
        self.size=int(r.headers['Content-Range'].split('/')[-1])
        self.etag=r.headers.get('ETag'); self.modified=r.headers.get('Last-Modified')
        self.cache_start=self.size-len(r.content); self.cache=r.content
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def fetch(self,start,length):
        if start>=self.cache_start and start+length<=self.cache_start+len(self.cache):
            return self.cache[start-self.cache_start:start-self.cache_start+length]
        for attempt in range(3):
            try:
                r=self.session.get(self.url,headers={'Range':f'bytes={start}-{start+length-1}'},timeout=(30,180))
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=length:
                    raise IOError(f'Invalid ranged response {r.status_code}, {len(r.content)} expected {length}')
                if r.headers.get('ETag')!=self.etag:
                    raise IOError('Archive changed during acquisition')
                return r.content
            except (requests.RequestException, IOError):
                if attempt==2: raise
                time.sleep(2**attempt)
    def read(self,n=-1):
        if n<0: n=self.size-self.position
        n=min(n,self.size-self.position)
        data=self.fetch(self.position,n) if n else b''
        self.position+=len(data); return data

class RemoteZip:
    def __init__(self,url):
        self.file=RangeFile(url); self.zip=zipfile.ZipFile(self.file)
        self.url=url
    def infos(self): return self.zip.infolist()
    def read(self,info):
        if isinstance(info,str): info=self.zip.getinfo(info)
        # One request normally covers header + payload; tolerate variable ZIP64 extras.
        amount=min(self.file.size-info.header_offset,30+len(info.filename.encode('utf8'))+4096+info.compress_size)
        packed=self.file.fetch(info.header_offset,amount)
        if packed[:4]!=b'PK\x03\x04': raise IOError('Invalid ZIP local header')
        name_len,extra_len=struct.unpack_from('<HH',packed,26)
        start=30+name_len+extra_len
        data=packed[start:start+info.compress_size]
        if len(data)!=info.compress_size:
            data=self.file.fetch(info.header_offset+start,info.compress_size)
        if info.compress_type==zipfile.ZIP_DEFLATED: payload=zlib.decompress(data,-15)
        elif info.compress_type==zipfile.ZIP_STORED: payload=data
        else: raise ValueError(f'Unsupported ZIP method {info.compress_type}')
        if len(payload)!=info.file_size or zlib.crc32(payload)&0xffffffff!=info.CRC:
            raise IOError(f'ZIP CRC mismatch: {info.filename}')
        return payload
    def close(self): self.zip.close(); self.file.session.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

class Acquirer:
    def __init__(self,root):
        self.root=Path(root); ensure_space(self.root)
        self.manifest=self.root/'source_manifest.json'
        self.records=json.loads(self.manifest.read_text()) if self.manifest.exists() else {'archives':{},'members':{}}
    def acquire(self,url,predicate,destination,transform=None):
        with RemoteZip(url) as remote:
            self.records['archives'][url]={'bytes':remote.file.size,'etag':remote.file.etag,'last_modified':remote.file.modified,'verification':'Selected members ZIP CRC32 + original decompressed SHA256; entire archive not downloaded'}
            infos=[x for x in remote.infos() if not x.is_dir() and predicate(x.filename)]
            print(f'ARCHIVE {url} selected={len(infos)} compressed_GiB={sum(x.compress_size for x in infos)/1024**3:.3f}',flush=True)
            for i,info in enumerate(infos):
                out=Path(destination(info.filename)); key=url+'#'+info.filename
                if key in self.records['members'] and out.exists():
                    saved=self.records['members'][key]
                    valid=out.stat().st_size==saved.get('stored_bytes') and saved.get('zip_crc32')==f'{info.CRC:08x}' and saved.get('source_bytes')==info.file_size
                    if valid:
                        digest=hashlib.sha256()
                        with out.open('rb') as stream:
                            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                        valid=digest.hexdigest()==saved.get('stored_sha256')
                    if valid:continue
                    print(f'  REPAIR stored input failed provenance check: {out}',flush=True)
                ensure_space(self.root,info.file_size*2)
                data=remote.read(info)
                source_hash=hashlib.sha256(data).hexdigest()
                out.parent.mkdir(parents=True,exist_ok=True)
                if transform: data=transform(data)
                tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_bytes(data); tmp.replace(out)
                self.records['members'][key]={'path':str(out.relative_to(self.root)).replace('\\','/'),'source_bytes':info.file_size,'zip_crc32':f'{info.CRC:08x}','source_sha256':source_hash,'stored_sha256':hashlib.sha256(data).hexdigest(),'stored_bytes':len(data),'acquired_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                if i%20==0 or i==len(infos)-1:
                    atomic_json(self.manifest,self.records)
                    print(f'  {i+1}/{len(infos)} {info.filename}',flush=True)
            atomic_json(self.manifest,self.records)
        return len(infos)
