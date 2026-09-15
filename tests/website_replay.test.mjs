import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {assetPath,decodeRoad,visibleObjects,corners,projectBox,frameDelay} from '../website/replay/math.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../website/replay');
const recording=path.join(root,'recording');
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const intrinsic=[[100,0,100],[0,100,100],[0,0,1]];

test('road run-length encoding decodes and rejects malformed runs',()=>{
  assert.deepEqual([...decodeRoad({width:3,height:2,runs:[0,2,1,1,255,2,2,1]})],[0,0,1,255,255,2]);
  for(const runs of [[0,7],[0,5],[0,-1],[4,6],[0,6,1],[]])assert.throws(()=>decodeRoad({width:3,height:2,runs}));
});
test('asset allowlist forbids remote, absolute and traversal requests',()=>{
  assert.equal(assetPath('frames/001.json'),'frames/001.json');
  assert.equal(assetPath('images/frame-000-camera-5.jpg'),'images/frame-000-camera-5.jpg');
  for(const value of ['https://example.com/x','../secret','/api/frame','images/../../secret','C:\\Users\\secret','frames/1.json'])assert.throws(()=>assetPath(value));
});
test('cuboid corners rotate around the predicted centre',()=>{
  const points=corners([1,2,3,4,2,6,Math.PI/2]);
  assert.equal(points.length,8);
  assert.ok(Math.abs(Math.min(...points.map(p=>p[0]))-0)<1e-12);
  assert.ok(Math.abs(Math.max(...points.map(p=>p[1]))-4)<1e-12);
  assert.deepEqual([...new Set(points.map(p=>p[2]))],[0,6]);
});
test('projection preserves camera calibration and clips behind-camera boxes',()=>{
  assert.equal(projectBox([0,0,10,2,2,2,0],intrinsic,identity,200,200).length,12);
  assert.equal(projectBox([0,0,-10,2,2,2,0],intrinsic,identity,200,200).length,0);
  const translated=identity.map(row=>[...row]);translated[0][3]=5;
  assert.deepEqual(projectBox([5,0,10,2,2,2,0],intrinsic,translated,200,200),projectBox([0,0,10,2,2,2,0],intrinsic,identity,200,200));
  for(const edge of projectBox([0,0,.2,2,2,2,.3],intrinsic,identity,200,200))for(const [x,y] of edge) {
    assert.ok(Number.isFinite(x)&&Number.isFinite(y));assert.ok(x>=-1e-6&&x<=200+1e-6&&y>=-1e-6&&y<=200+1e-6);
  }
});
test('source timestamps and speed determine playback, with a scene boundary reset',()=>{
  assert.equal(frameDelay({sequence:'a',timestamp:0},{sequence:'a',timestamp:.5},1),500);
  assert.equal(frameDelay({sequence:'a',timestamp:0},{sequence:'a',timestamp:.5},2),250);
  assert.equal(frameDelay({sequence:'a',timestamp:0},{sequence:'b',timestamp:9000},1),500);
});
test('confidence filtering never invents or changes a box',()=>{
  const frame={boxes:[[0,0,0,4,2,1.5,0],[1,2,3,1,1,1,0]],scores:[.6,.2],labels:[0,1],track_ids:[4,5]};
  assert.equal(visibleObjects(frame,.55).length,1);
  assert.strictEqual(visibleObjects(frame,.55)[0].box,frame.boxes[0]);
});
test('public clip contains 40 current-checkpoint training frames and 240 verified camera assets',()=>{
  const metadata=JSON.parse(fs.readFileSync(path.join(recording,'manifest.json'),'utf8'));
  assert.equal(metadata.checkpoint_sha256,'28d570f3688ce1dd0b2f52c59ae92568abfb1a0ca634147642e5c88139013b50');
  assert.equal(metadata.frames.length,40);assert.equal(metadata.cameras.length,6);
  assert.equal(metadata.refinement_enabled,false);assert.equal(metadata.split,'train');
  assert.deepEqual([...new Set(metadata.frames.map(frame=>frame.scene_name))],['scene-0001']);
  assert.match(metadata.selection,/training keyframes from scene-0001/);
  assert.ok(metadata.limitations.some(item=>/cannot measure generalization/.test(item)));
  const inventory=JSON.parse(fs.readFileSync(path.join(recording,'asset-integrity.json'),'utf8'));
  assert.equal(inventory.assets.filter(a=>a.file.startsWith('images/')).length,240);
  let total=0;
  for(const asset of inventory.assets) {
    assetPath(asset.file);
    const data=fs.readFileSync(path.join(recording,asset.file));
    assert.equal(data.length,asset.bytes);
    assert.equal(crypto.createHash('sha256').update(data).digest('hex'),asset.sha256);
    total+=data.length;
  }
  assert.equal(total,metadata.bytes);
});
test('every real frame decodes, stays synchronized and contains predictions only',()=>{
  const metadata=JSON.parse(fs.readFileSync(path.join(recording,'manifest.json'),'utf8'));
  const allowed=new Set(['index','sequence','scene_name','frame_id','timestamp','boxes','scores','labels','track_ids','road_display','ground_z','images','image_size','intrinsics','camera_to_ego']);
  for(const [index,entry] of metadata.frames.entries()) {
    const frame=JSON.parse(fs.readFileSync(path.join(recording,entry.file),'utf8'));
    assert.deepEqual(Object.keys(frame).filter(key=>!allowed.has(key)),[]);
    assert.equal(frame.index,index);assert.equal(frame.timestamp,entry.timestamp);assert.equal(frame.sequence,entry.sequence);
    assert.equal(frame.images.length,6);assert.deepEqual(frame.image_size,[704,256]);
    assert.equal(decodeRoad(frame.road_display).length,160*160);
    for(let camera=0;camera<6;camera++) {
      assetPath(frame.images[camera]);
      for(const object of visibleObjects(frame,.01))for(const edge of projectBox(object.box,frame.intrinsics[camera],frame.camera_to_ego[camera],704,256))for(const [x,y] of edge) {
        assert.ok(Number.isFinite(x)&&Number.isFinite(y));
        assert.ok(x>=-1e-5&&x<=704+1e-5&&y>=-1e-5&&y<=256+1e-5);
      }
    }
  }
});
test('public JSON has no local paths, target dumps or model serialization',()=>{
  for(const name of ['manifest.json','asset-integrity.json',...fs.readdirSync(path.join(recording,'frames')).map(name=>'frames/'+name)]) {
    const text=fs.readFileSync(path.join(recording,name),'utf8');
    assert.doesNotMatch(text,/[A-Z]:\\\\|Users[\\\/]|"gt_|"annotations"|"depth_target"|"instance_ids"|"detection_valid"|\.pt"|\.pkl"/);
  }
});
