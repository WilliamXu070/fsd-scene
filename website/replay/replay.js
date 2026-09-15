import {assetPath, visibleObjects, projectBox, frameDelay, clamp} from './math.mjs';
import {SceneRenderer,prepareVehicleAssets} from './scene-renderer.js?v=20260915b';

const $=id=>document.getElementById(id);
const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
const dataRoot=new URL('./recording/',import.meta.url);
const recordingVersion='20260915-train';
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const names=['Front','Front left','Front right','Back','Back left','Back right'];
const characterPreview=['localhost','127.0.0.1'].includes(location.hostname)&&new URLSearchParams(location.search).get('ego')==='mcqueen';
let metadata=null,frame=null,currentIndex=0,requestedIndex=0,cameraIndex=0,selected=-1;
let renderer=null,playing=false,loading=false,timer=0,scrubTimer=0,serial=0,request=null,embeddedVisible=true;
const cache=new Map(),cameraButtons=[];

function assetURL(path) {const url=new URL(assetPath(path),dataRoot);url.searchParams.set('v',recordingVersion);return url.href;}
function error(message) { $('error').textContent=message;$('error').hidden=!message; }
function playLabel() {
  $('play').textContent=playing?'Pause':reduced.matches?'Step frame':currentIndex===metadata?.frames.length-1?'Replay':'Play replay';
  $('play').setAttribute('aria-pressed',String(playing));
}
function pause(cancel=false) {
  playing=false;clearTimeout(timer);clearTimeout(scrubTimer);playLabel();
  if(cancel) {
    serial++;request?.abort();request=null;loading=false;requestedIndex=currentIndex;
    $('play').disabled=!frame;
    if(frame) {
      $('scrubber').value=String(currentIndex);
      $('load-status').textContent=frame.scene_name+' · source frame '+frame.frame_id+' · paused';
    }
  }
}
function preload(url,signal) {
  return new Promise((resolve,reject)=>{
    const image=new Image();let settled=false;
    const timeout=setTimeout(()=>finish(new Error('Camera image timed out. Retry the frame.')),18000);
    const abort=()=>{finish(new DOMException('Aborted','AbortError'));image.src='';};
    function finish(problem) {
      if(settled)return;settled=true;clearTimeout(timeout);signal.removeEventListener('abort',abort);
      image.onload=null;image.onerror=null;problem?reject(problem):resolve();
    }
    image.onload=async()=>{try{await image.decode();if(signal.aborted)abort();else finish();}catch{finish(new Error('Camera image could not be decoded.'));}};
    image.onerror=()=>finish(new Error('Camera image failed to load. Retry the frame.'));
    signal.addEventListener('abort',abort,{once:true});
    if(signal.aborted)abort();else image.src=url;
  });
}
async function readFrame(index,signal) {
  if(cache.has(index))return cache.get(index);
  const response=await fetch(assetURL(metadata.frames[index].file),{signal});
  if(!response.ok)throw new Error('Recorded frame unavailable. Reload the viewer to retry.');
  const result=await response.json();
  const expected=metadata.frames[index];
  if(result.index!==index || result.sequence!==expected.sequence || result.timestamp!==expected.timestamp || result.images?.length!==metadata.cameras.length)throw new Error('Replay frame identity does not match the manifest.');
  result.images.forEach(assetPath);
  cache.set(index,result);
  while(cache.size>4)cache.delete(cache.keys().next().value);
  return result;
}
async function loadFrame(index) {
  if(!metadata)return false;
  const id=++serial;request?.abort();const controller=new AbortController();request=controller;
  requestedIndex=clamp(index,0,metadata.frames.length-1);loading=true;
  $('play').disabled=true;
  $('load-status').textContent='Loading synchronized frame '+(requestedIndex+1)+'…';error('');
  try {
    const next=await readFrame(requestedIndex,controller.signal);
    await Promise.all(next.images.map(path=>preload(assetURL(path),controller.signal)));
    if(id!==serial || controller.signal.aborted)return false;
    // Commit images, projections and 3D geometry only after all six images decode.
    frame=next;currentIndex=next.index;selected=-1;
    cameraButtons.forEach((button,i)=>{
      const image=button.querySelector('img');
      image.src=assetURL(frame.images[i]);image.alt=names[i]+', source frame '+frame.frame_id;
    });
    $('camera-loading').hidden=true;
    updateObjects();updateCamera();
    $('scrubber').value=String(currentIndex);
    const first=metadata.frames.find(item=>item.sequence===frame.sequence);
    const elapsed=frame.timestamp-first.timestamp;
    $('timestamp').textContent=elapsed.toFixed(2)+' s in scene';
    $('frame-counter').textContent='Frame '+(currentIndex+1)+' / '+metadata.frames.length;
    $('scrubber').setAttribute('aria-valuetext','Frame '+(currentIndex+1)+' of '+metadata.frames.length+', '+frame.scene_name+', '+elapsed.toFixed(2)+' seconds');
    $('load-status').textContent=frame.scene_name+' · source frame '+frame.frame_id+' · six cameras synchronized';
    for(const name of ['play','previous','next','scrubber'])$(name).disabled=false;
    $('previous').disabled=currentIndex===0;$('next').disabled=currentIndex===metadata.frames.length-1;
    playLabel();return true;
  } catch(problem) {
    if(id===serial && problem.name!=='AbortError') {
      controller.abort();pause();error(problem.message);
      $('load-status').textContent=frame?'Previous synchronized frame retained.':'No frame loaded.';
      $('scrubber').value=String(currentIndex);
    }
    return false;
  } finally { if(id===serial){loading=false;request=null;$('play').disabled=!frame;} }
}
function schedule() {
  if(!playing || !frame)return;
  if(currentIndex>=metadata.frames.length-1){pause();return;}
  const delay=frameDelay(frame,metadata.frames[currentIndex+1],Number($('speed').value));
  timer=setTimeout(async()=>{
    if(!playing)return;
    const ok=await loadFrame(currentIndex+1);
    if(playing && ok)schedule();else if(playing)pause();
  },delay);
}
function updateObjects() {
  if(!frame)return;
  const threshold=Number($('confidence').value),visible=visibleObjects(frame,threshold);
  if(!visible.some(o=>o.index===selected))selected=-1;
  if(renderer) {
    renderer.selected=selected;renderer.outlines=$('boxes').checked;
    renderer.update(frame,metadata,threshold);
  }
  const selection=$('object-select');selection.replaceChildren(new Option('Select an object', '-1'));
  for(const o of visible) selection.add(new Option((o.label===0?'Car':'Pedestrian')+' · '+o.score.toFixed(2)+' confidence · track '+o.track,String(o.index)));
  selection.value=String(selected);
  const cars=visible.filter(o=>o.label===0).length;
  $('object-count').textContent=cars+' cars · '+(visible.length-cars)+' pedestrians above threshold';
  updateSelected();updateProjection();
}
function updateSelected() {
  const object=frame && selected>=0?visibleObjects(frame,Number($('confidence').value)).find(o=>o.index===selected):null;
  if(!object) {
    $('object-details').textContent='Choose an object to see its class, confidence, position, dimensions and yaw.';
    return;
  }
  const b=object.box;
  $('object-details').textContent=(object.label===0?'Car':'Pedestrian')+' · confidence '+object.score.toFixed(3)+' · track '+object.track+
    ' | Centre x/y/z: '+b.slice(0,3).map(x=>x.toFixed(2)).join(' / ')+' m | Length/width/height: '+b.slice(3,6).map(x=>x.toFixed(2)).join(' / ')+
    ' m | Yaw: '+(b[6]*180/Math.PI).toFixed(1)+'°. Coordinates: x forward, y left, z up.';
}
function updateCamera() {
  if(!frame)return;
  $('camera-name').textContent=names[cameraIndex];
  $('camera-count').textContent=(cameraIndex+1)+' / '+metadata.cameras.length;
  $('camera-image').src=assetURL(frame.images[cameraIndex]);
  $('camera-image').alt=names[cameraIndex]+', '+frame.scene_name+', source frame '+frame.frame_id;
  cameraButtons.forEach((button,i)=>button.setAttribute('aria-pressed',String(i===cameraIndex)));
  updateProjection();
}
function updateProjection() {
  const svg=$('projection'),hidden=!$('overlay').checked;
  svg.toggleAttribute('hidden',hidden);
  if(!frame || hidden)return;
  const [width,height]=frame.image_size;
  svg.setAttribute('viewBox','0 0 '+width+' '+height);
  let lines='';
  for(const object of visibleObjects(frame,Number($('confidence').value))) {
    const color=object.index===selected?'#fff':object.label===0?'#28d2b1':'#ffb365';
    for(const [a,b] of projectBox(object.box,frame.intrinsics[cameraIndex],frame.camera_to_ego[cameraIndex],width,height)) {
      lines+='<line x1="'+a[0]+'" y1="'+a[1]+'" x2="'+b[0]+'" y2="'+b[1]+'" stroke="'+color+'" stroke-width="'+(object.index===selected?2:1.2)+'" vector-effect="non-scaling-stroke"/>';
    }
  }
  svg.innerHTML='<title>Projected model predictions in '+escape(names[cameraIndex])+'</title>'+lines;
}
function chooseObject(index) {
  selected=index;$('object-select').value=String(index);
  if(renderer){renderer.selected=index;renderer.update(frame,metadata,Number($('confidence').value));}
  updateSelected();updateProjection();
}
function setView(top) {
  renderer?.view(top);$('topdown').setAttribute('aria-pressed',String(top));$('perspective').setAttribute('aria-pressed',String(!top));
}
function provenance() {
  const data=metadata,link=(url,label)=>'<a href="'+escape(url)+'" target="_blank" rel="noopener">'+escape(label)+'</a>';
  $('model-title').textContent=data.model;
  const splitLabel=data.split==='train'?'training-split':'validation';
  $('model-note').textContent=data.frames.length+' recorded '+splitLabel+' keyframes · '+(data.bytes/1048576).toFixed(1)+' MiB full clip · Stage A, refinement off';
  $('replay-notice').textContent='Actual model predictions on '+splitLabel+' inputs · recorded playback · no live inference. This nuScenes run is separate from the earlier KITTI-360 results in the article.';
  $('provenance-body').innerHTML='<p><strong>Checkpoint SHA256</strong><br><code>'+escape(data.checkpoint_sha256)+'</code></p>'+
    '<p>'+escape(data.selection)+' Source: '+escape(data.dataset)+'. The clip buffers before advancing if an image is not ready.</p>'+
    '<ul>'+data.limitations.map(item=>'<li>'+escape(item)+'</li>').join('')+'</ul>'+
    '<p>'+link(data.attribution.url,data.attribution.dataset)+'. '+escape(data.attribution.paper)+'</p>'+
    '<p>'+link(data.attribution.license_url,data.attribution.license)+'; '+link(data.attribution.terms,'dataset terms')+'. '+escape(data.attribution.changes)+'</p>'+
    '<p>'+escape(data.attribution.endorsement)+(characterPreview?' Traffic uses Quaternius CC0 geometry. The ego car is a user-supplied McQueen model with no license supplied, for local preview only.':' Vehicle meshes adapted in Blender from Quaternius and Kenney CC0 assets; original character details and contour surfaces are presentation only.')+' '+link('./scene-style/assets/ASSET-SOURCES.txt','Public vehicle sources and licenses')+'. Three.js and OrbitControls are MIT-licensed. '+link('./vendor/THREE-LICENSE.txt','Three.js license')+
    '. '+link('./recording/asset-integrity.json','Asset hashes')+'.</p>';
}
async function init() {
  try {
    const manifestURL=new URL('manifest.json',dataRoot);manifestURL.searchParams.set('v',recordingVersion);
    const response=await fetch(manifestURL);
    if(!response.ok)throw new Error('Replay manifest unavailable. Reload to retry.');
    metadata=await response.json();
    if(metadata.schema_version!==1 || metadata.cameras.length!==6 || !metadata.frames.length)throw new Error('Unsupported replay manifest.');
    metadata.frames.forEach(item=>assetPath(item.file));
    $('scrubber').max=String(metadata.frames.length-1);
    $('confidence').value=String(metadata.confidence_default);$('confidence-value').textContent=metadata.confidence_default.toFixed(2);
    provenance();
    metadata.cameras.forEach((camera,index)=>{
      const button=document.createElement('button');button.type='button';button.setAttribute('aria-label','Show '+names[index]+' camera');button.setAttribute('aria-pressed',String(index===0));
      const image=document.createElement('img');image.alt=names[index]+' camera';image.decoding='async';
      const name=document.createElement('span');name.textContent=names[index];button.append(image,name);
      button.onclick=()=>{cameraIndex=index;updateCamera();};
      $('camera-tabs').append(button);cameraButtons.push(button);
    });
    try {await prepareVehicleAssets({egoAsset:characterPreview?'ego-mcqueen':'ego-racer'});renderer=new SceneRenderer($('viewport'),chooseObject);}catch(problem){$('webgl-error').hidden=false;console.warn('3D viewer unavailable:',problem.message);}
    if(characterPreview)$('model-note').textContent+=' · User-provided McQueen local preview (license not supplied)';
    if(embeddedVisible && !document.hidden)await loadFrame(0);
  } catch(problem) {error(problem.message);$('load-status').textContent='Replay could not initialize.';}
}
$('play').onclick=async()=>{
  if(playing){pause(true);return;}
  if(reduced.matches){pause(true);await loadFrame(currentIndex>=metadata.frames.length-1?0:currentIndex+1);return;}
  if(currentIndex>=metadata.frames.length-1 && !await loadFrame(0))return;
  playing=true;playLabel();schedule();
};
$('previous').onclick=()=>{const next=requestedIndex-1;pause(true);loadFrame(next);};
$('next').onclick=()=>{const next=requestedIndex+1;pause(true);loadFrame(next);};
$('scrubber').oninput=()=>{
  const target=Number($('scrubber').value);pause(true);$('scrubber').value=String(target);
  scrubTimer=setTimeout(()=>loadFrame(target),75);
};
$('confidence').oninput=()=>{$('confidence-value').textContent=Number($('confidence').value).toFixed(2);updateObjects();};
$('object-select').onchange=()=>chooseObject(Number($('object-select').value));
$('overlay').onchange=updateProjection;
$('boxes').onchange=updateObjects;
$('raw-road').onchange=()=>{
  if(renderer){renderer.rawRoad=$('raw-road').checked;renderer.frame=null;updateObjects();}
  $('surface-note').textContent=$('raw-road').checked?'Original predicted grid: road / sidewalk / other ground.':'Smoothed road / non-road surfaces. Boundaries are display geometry, not predicted lane lines.';
};
$('grid').onchange=()=>{if(renderer){renderer.grid.visible=$('grid').checked;renderer.draw();}};
$('perspective').onclick=()=>setView(false);$('topdown').onclick=()=>setView(true);$('reset-view').onclick=()=>setView(false);
$('speed').onchange=()=>{if(playing){clearTimeout(timer);schedule();}};
document.addEventListener('keydown',event=>{
  if(event.target.closest('input,select,button,a,summary') || !frame)return;
  if(event.code==='Space'){event.preventDefault();$('play').click();}
  else if(event.key==='ArrowRight'){event.preventDefault();$('next').click();}
  else if(event.key==='ArrowLeft'){event.preventDefault();$('previous').click();}
});
document.addEventListener('visibilitychange',()=>{
  if(document.hidden)pause(true);
  else {renderer?.draw();if(metadata && !frame && !loading && embeddedVisible)loadFrame(0);}
});
reduced.addEventListener('change',()=>pause(true));
window.addEventListener('message',event=>{
  if(event.source!==parent || event.origin!==location.origin)return;
  if(event.data?.type==='pause-model-replay')pause(true);
  if(event.data?.type==='model-replay-visibility') {
    embeddedVisible=Boolean(event.data.visible);
    if(!embeddedVisible)pause(true);
    else if(metadata && !frame && !loading && !document.hidden)loadFrame(0);
  }
});
let lastHeight=0,resizeQueued=false;
new ResizeObserver(()=>{
  if(parent===window || resizeQueued)return;
  resizeQueued=true;requestAnimationFrame(()=>{
    resizeQueued=false;const height=Math.ceil(document.body.getBoundingClientRect().height);
    if(height!==lastHeight){lastHeight=height;parent.postMessage({type:'model-replay-height',height},location.origin);}
  });
}).observe(document.body);
if(parent!==window)parent.postMessage({type:'model-replay-ready'},location.origin);
init();
