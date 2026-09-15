import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {LatestFrameRequest, debouncedScrub} from '/replay-requests.js';
import {ReplayTiming} from '/replay-timing.js';
import {prepareVehicleAssets,createVehicle,createPerson,styleScene} from '/scene-style/vehicles.js';
import {RoadSurface} from '/scene-style/road-surface.js';
import {cellsFromRecord} from '/scene-style/road-contours.js';

const $ = id => document.getElementById(id);
const viewport = $('viewport');
try {await prepareVehicleAssets();} catch(problem) {$('error').hidden=false;$('error').textContent=problem.message;throw problem;}
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42,1,.1,240);
camera.up.set(0,0,1);
camera.position.set(-28,-10,30);
const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.75));
renderer.outputColorSpace=THREE.SRGBColorSpace;
viewport.append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
controls.target.set(10,0,0);
controls.minDistance=7;controls.maxDistance=115;
controls.maxPolarAngle=Math.PI/2-.035;
controls.enableDamping=false;
controls.update();
styleScene(scene,renderer);
const ego=createVehicle(true);ego.scale.set(4.5,1.85,1.5);ego.position.set(0,0,.75);scene.add(ego);
const objects=new THREE.Group();scene.add(objects);const pool=[];
const axes=new THREE.Group();
const grid=new THREE.GridHelper(80,16,'#a4b1ba','#d9e0e5');grid.rotation.x=Math.PI/2;grid.position.z=-.06;axes.add(grid);
const xArrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,.1),8,'#708b9e',.8,.5);
const yArrow=new THREE.ArrowHelper(new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,.1),8,'#88a198',.8,.5);axes.add(xArrow,yArrow);axes.visible=false;scene.add(axes);
const roadSurface=new RoadSurface(scene);
const replayTiming=new ReplayTiming();
const sceneRequests=new LatestFrameRequest(undefined,event=>replayTiming.request(event));
const scrubRequest=debouncedScrub(next=>loadFrame(next));
let status={frames:0,bev_min:-40,bev_step:.5,ground_z:-1.5};let frame=null,index=0,playing=false,loading=false,timer=null,requestId=0,firstTimestamp=null;
const cameraCards=[];
function ensureCameraCards(){
  const names=status.camera_names||['Front left','Front right','Left fisheye','Right fisheye'];
  const count=Array.isArray(names)&&names.length>=1&&names.length<=8?names.length:4;
  if(cameraCards.length===count)return;
  cameraCards.length=0;$('cameras').replaceChildren();
  for(let c=0;c<count;c++){
  const figure=document.createElement('figure');figure.className='camera-view';
  figure.innerHTML=`<figcaption><span class="camera-name"></span><span class="camera-id">${String(c).padStart(2,'0')}</span></figcaption><div class="camera-image"><span class="camera-placeholder">No image available</span><img alt="Camera ${c} observation"></div>`;
  $('cameras').append(figure);cameraCards.push(figure);
  }
}
ensureCameraCards();
function error(message){$('error').hidden=!message;$('error').textContent=message||'';}
function applyGround(ground){
  ego.position.z=ground+.75;axes.position.z=ground;
  $('axes').title=`Ground z=${ground.toFixed(2)}m. ${status.ground_source||'Static inference calibration'}`;
}
function updateProvenance(){
  ensureCameraCards();
  applyGround(Number.isFinite(frame?.road_ground_z_m)?frame.road_ground_z_m:(status.ground_z??-1.5));
  const labels={fixture:'UI fixture Â· synthetic geometry',trained:'Trained checkpoint',untrained:'Untrained model output',unverified:'Training status unverified'};
  $('provenance').textContent=labels[status.provenance]||labels.unverified;
  $('provenance').className=`provenance ${status.provenance==='trained'?'trained':''}`;
  $('provenance').title=status.checkpoint||'';
  $('dataset-note').hidden=!status.note;
  $('dataset-note').textContent=status.note||'';
  $('reference-control').hidden=!status.source_reference_available;
  $('data-note').textContent=status.provenance==='fixture'?'Synthetic fixture for interface testing. These shapes are not model detections.':'Scene geometry comes from model predictions. The reference vehicle is a fixed graphic.';
  cameraCards.forEach((card,c)=>card.querySelector('.camera-name').textContent=(status.camera_names||['Front left','Front right','Left fisheye','Right fisheye'])[c]);
}
function updateRoad(record){
  const ground=Number.isFinite(record.road_ground_z_m)?record.road_ground_z_m:(status.ground_z??-1.5);
  applyGround(ground);
  const {cells,width,height}=cellsFromRecord(record);
  if(!width||!height){roadSurface.clear();return;}
  roadSurface.update(cells,width,height,status.bev_min,status.bev_step,ground,$('raw-road').checked);
}
function updateObjects(record){
  const threshold=Number($('confidence').value);let n=0;
  for(let j=0;j<(record.boxes||[]).length;j++){
    const b=record.boxes[j],score=record.scores?.[j]??0,label=record.labels?.[j];
    if(score<threshold||!Array.isArray(b)||b.length!==7||!b.every(Number.isFinite)||b.slice(3,6).some(v=>v<=0)||![0,1].includes(label))continue;
    if(!pool[n]){pool[n]={car:createVehicle(),person:createPerson()};objects.add(pool[n].car,pool[n].person);}
    const slot=pool[n],mesh=label===0?slot.car:slot.person;slot.car.visible=label===0;slot.person.visible=label===1;
    mesh.position.set(b[0],b[1],b[2]);mesh.scale.set(b[3],b[4],b[5]);mesh.rotation.z=b[6];mesh.userData={trackId:record.track_ids?.[j],score,label};n++;
  }
  for(let i=n;i<pool.length;i++){pool[i].car.visible=false;pool[i].person.visible=false;}
  $('object-count').textContent=n;
}
function renderFrame(record){
  updateRoad(record);updateObjects(record);$('empty-state').hidden=true;
  $('scene-title').textContent=status.provenance==='fixture'?'Synthetic scene fixture':'Predicted scene';
  $('sequence').textContent=`${record.sequence??'Sequence unavailable'} / frame ${record.frame_id??record.index}`;
  $('frame-counter').textContent=`Frame ${record.index+1} of ${status.frames}`;
  if(record.index===0&&Number.isFinite(record.timestamp))firstTimestamp=record.timestamp;
  $('timestamp').textContent=Number.isFinite(record.timestamp)&&Number.isFinite(firstTimestamp)?`${(record.timestamp-firstTimestamp).toFixed(2)} s elapsed`:'Timestamp unavailable';
  $('timestamp').title=Number.isFinite(record.timestamp)?`Source timestamp: ${record.timestamp.toFixed(6)} seconds`:'No source timestamp';
  $('inference-time').textContent=Number.isFinite(record.inference_ms)?`${record.inference_ms.toFixed(1)} ms`:'â€”';
  $('scrubber').value=record.index;
  $('source-reference').href=`/inspect?index=${record.index}`;
  let count=0;
  cameraCards.forEach((card,c)=>{
    const img=card.querySelector('img'),placeholder=card.querySelector('.camera-placeholder'),url=record.images?.[c];
    if(url){count++;img.onload=()=>{placeholder.hidden=true;};img.onerror=()=>{img.removeAttribute('src');placeholder.hidden=false;};img.src=url;img.alt=`${status.camera_names?.[c]||'Camera '+c}, frame ${record.frame_id}`;}
    else{img.removeAttribute('src');placeholder.hidden=false;}
  });
  $('camera-count').textContent=`${count} / ${cameraCards.length}`;
}
async function loadFrame(next){
  if(!status.frames)return;
  const request=++requestId,started=performance.now();loading=true;index=Math.max(0,Math.min(status.frames-1,next));
  try{const record=await sceneRequests.load(`/api/frame?index=${index}`);if(!record||request!==requestId)return;frame=record;const applyStarted=performance.now();renderFrame(record);const applied=performance.now();replayTiming.record('scene_update_cpu_ms',applied-applyStarted);replayTiming.record('request_to_scene_apply_ms',applied-started);if(Number.isFinite(record.inference_ms))replayTiming.record('recorded_model_call_ms',record.inference_ms);error('');}
  catch(exc){error(exc.message);setPlaying(false);}
  finally{if(request===requestId)loading=false;}
}
function setPlaying(value){scrubRequest.cancel();playing=value;$('play').textContent=value?'Pause':'Play';$('play').setAttribute('aria-label',value?'Pause replay':'Play replay');clearTimeout(timer);if(value)schedule();}
function schedule(){
  if(!playing)return;
  const current=frame?.timestamp;timer=setTimeout(async()=>{
    if(!playing)return;if(index>=status.frames-1){setPlaying(false);return;}
    const start=performance.now(),prior=current;
    await loadFrame(index+1);
    const dt=frame&&Number.isFinite(prior)?frame.timestamp-prior:.5;
    const duration=Number.isFinite(dt)&&dt>0?Math.min(dt,2):.5;
    if(playing)timer=setTimeout(schedule,Math.max(0,duration*1000/Number($('speed').value)-(performance.now()-start)));
  },0);
}
$('play').onclick=()=>{if(index===status.frames-1&&!playing)loadFrame(0).then(()=>setPlaying(true));else setPlaying(!playing);};
$('previous').onclick=()=>{setPlaying(false);loadFrame(index-1);};$('next').onclick=()=>{setPlaying(false);loadFrame(index+1);};
$('scrubber').oninput=()=>{setPlaying(false);scrubRequest(Number($('scrubber').value));};
$('confidence').oninput=()=>{$('confidence-value').value=Number($('confidence').value).toFixed(2);if(frame)updateObjects(frame);};
$('axes').onchange=()=>{axes.visible=$('axes').checked;};
$('raw-road').onchange=()=>{if(frame)updateRoad(frame);};
function view(top){camera.position.set(top?7.99:-19,top?0:-15,top?58:24);controls.target.set(8,0,0);controls.update();$('topdown').classList.toggle('selected',top);$('perspective').classList.toggle('selected',!top);$('topdown').setAttribute('aria-pressed',top);$('perspective').setAttribute('aria-pressed',!top);}
$('topdown').onclick=()=>view(true);$('perspective').onclick=()=>view(false);
view(false);
window.addEventListener('keydown',event=>{if(event.target instanceof HTMLInputElement||event.target instanceof HTMLSelectElement)return;if(event.code==='Space'&&status.frames){event.preventDefault();setPlaying(!playing);}else if(event.key==='ArrowRight'){setPlaying(false);loadFrame(index+1);}else if(event.key==='ArrowLeft'){setPlaying(false);loadFrame(index-1);}});
async function poll(){try{const response=await fetch('/api/status');if(!response.ok)throw new Error('Replay server unavailable.');status=await response.json();updateProvenance();$('scrubber').max=Math.max(0,status.frames-1);for(const id of ['scrubber','play','previous','next'])$(id).disabled=!status.frames;if(status.frames&&!frame)await loadFrame(0);else if(frame&&!playing)$('frame-counter').textContent=`Frame ${index+1} of ${status.frames}`;if(!status.frames){$('empty-state').hidden=false;}}catch(exc){error(exc.message);}finally{setTimeout(poll,2500);}}
const resize=new ResizeObserver(()=>{const r=viewport.getBoundingClientRect();renderer.setSize(r.width,r.height);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();});resize.observe(viewport);
const timingSnapshot=()=>({...replayTiming.snapshot(),context:{provenance:status.provenance??'unverified',note:status.note??'',
  sequence:frame?.sequence??null,frame_index:frame?.index??null,playing,visibility:document.visibilityState,
  viewport_css:{width:viewport.clientWidth,height:viewport.clientHeight},pixel_ratio:renderer.getPixelRatio(),
  last_render_draw_calls:renderer.info.render.calls,geometries:renderer.info.memory.geometries,textures:renderer.info.memory.textures,
  recorded_model_call_ms:frame?.inference_ms??null}});
Object.defineProperty(window,'sceneReplayTiming',{value:Object.freeze({snapshot:timingSnapshot,reset:()=>{replayTiming.reset();return timingSnapshot();}}),writable:false,configurable:false});
function animate(timestamp){requestAnimationFrame(animate);replayTiming.raf(timestamp);const started=performance.now();renderer.render(scene,camera);replayTiming.record('render_submit_cpu_ms',performance.now()-started);}animate();poll();
