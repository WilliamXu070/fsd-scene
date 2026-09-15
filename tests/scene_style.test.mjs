import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {traceRegions,signedArea,smoothRing,metricRegions,cellsFromRecord} from '../viewer/scene-style/road-contours.js';
import {decodeRoad} from '../website/replay/math.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const area=regions=>regions.reduce((sum,r)=>sum+signedArea(r.outer)+r.holes.reduce((s,h)=>s+signedArea(h),0),0);

test('raw contours exactly conserve occupied area for every 3x3 mask',()=>{
  for(let mask=0;mask<512;mask++) {
    const cells=Uint8Array.from({length:9},(_,i)=>mask>>i&1);
    const regions=traceRegions(cells,3,3,v=>v===1);
    assert.equal(area(regions),cells.reduce((a,b)=>a+b,0));
    for(const r of regions){assert.ok(signedArea(r.outer)>0);r.holes.forEach(h=>assert.ok(signedArea(h)<0));}
  }
});

test('diagonal islands and holes are kept distinct',()=>{
  assert.equal(traceRegions([1,0,0,1],2,2,v=>v===1).length,2);
  const ring=traceRegions([1,1,1,1,0,1,1,1,1],3,3,v=>v===1);
  assert.equal(ring.length,1);assert.equal(ring[0].holes.length,1);assert.equal(area(ring),8);
  assert.ok(signedArea(smoothRing(ring[0].holes[0]))<0);
});

test('surface smoothing is bounded to half a cell per vertex and never mutates source',()=>{
  const ring=[[0,0],[1,0],[2,0],[2,1],[1,1],[0,1]],original=JSON.stringify(ring);
  const smooth=smoothRing(ring);
  smooth.forEach((p,i)=>assert.ok(Math.hypot(p[0]-ring[i][0],p[1]-ring[i][1])<=.5+1e-12));
  assert.equal(JSON.stringify(ring),original);assert.ok(signedArea(smooth)>0);
  const world=metricRegions([0],1,1,v=>v===0,-40,.5,false);
  assert.deepEqual(world[0].outer,[[-40,-40],[-39.5,-40],[-39.5,-39.5],[-40,-39.5]]);
});

test('local viewer retains original confidence and camera coverage policy',()=>{
  const result=cellsFromRecord({road:[[0,1,2,0]],road_confidence:[[.45,.449,.8,.9]],road_visible:[[true,true,false,true]]});
  assert.deepEqual([...result.cells],[0,255,255,0]);assert.equal(result.width,4);assert.equal(result.height,1);
});

test('every real recording frame has valid bounded contour geometry',()=>{
  const recording=path.join(root,'website/replay/recording');
  const manifest=JSON.parse(fs.readFileSync(path.join(recording,'manifest.json')));
  for(const item of manifest.frames) {
    const frame=JSON.parse(fs.readFileSync(path.join(recording,item.file))),before=JSON.stringify(frame);
    const cells=decodeRoad(frame.road_display),{width,height}=frame.road_display;
    for(const accept of [v=>v===0,v=>v!==255]) {
      const raw=traceRegions(cells,width,height,accept);
      assert.equal(area(raw),cells.reduce((sum,v)=>sum+Number(accept(v)),0));
      const smooth=metricRegions(cells,width,height,accept,manifest.bev_min,manifest.bev_step);
      assert.equal(raw.length,smooth.length);
      smooth.forEach((r,i)=>{
        assert.equal(r.holes.length,raw[i].holes.length);
        for(const ring of [r.outer,...r.holes])for(const p of ring)assert.ok(p.every(Number.isFinite));
      });
    }
    assert.equal(JSON.stringify(frame),before);
  }
});

test('Blender vehicle exports are normalized finite geometry with known material names',()=>{
  for(const name of ['traffic-sedan','ego-racer']) {
    const source=fs.readFileSync(path.join(root,'viewer/scene-style/assets',name+'.mesh.js'),'utf8');
    assert.match(source,/^export default /);
    const asset=JSON.parse(source.replace(/^export default /,'').replace(/;\s*$/,''));
    assert.equal(asset.license,'CC0-1.0');assert.match(asset.source,name==='traffic-sedan'?/Quaternius/:/Kenney/);
    let triangles=0;
    for(const part of asset.parts) {
      assert.ok(['body','glass','tire','metal','headlight','taillight'].includes(part.material));
      const positions=Buffer.from(part.position,'base64'),normals=Buffer.from(part.normal,'base64');
      assert.equal(positions.length,normals.length);assert.equal(positions.length%36,0);
      for(let i=0;i<positions.length;i+=4){const value=positions.readFloatLE(i);assert.ok(Number.isFinite(value)&&Math.abs(value)<=.501);assert.ok(Number.isFinite(normals.readFloatLE(i)));}
      triangles+=positions.length/36;
    }
    assert.ok(triangles>1000&&triangles<12000);
  }
});

test('local and Pages render modules/assets are byte-identical',()=>{
  for(const name of ['road-contours.js','road-surface.js','vehicles.js','assets/traffic-sedan.mesh.js','assets/ego-racer.mesh.js','assets/KENNEY-LICENSE.txt','assets/ASSET-SOURCES.txt']) {
    assert.deepEqual(fs.readFileSync(path.join(root,'viewer/scene-style',name)),fs.readFileSync(path.join(root,'website/replay/scene-style',name)),name);
  }
});

test('actual Three.js surface meshes triangulate holes and dispose frame geometry',async()=>{
  const threeURL=pathToFileURL(path.join(root,'website/replay/vendor/three.module.js')).href;
  const contoursURL=pathToFileURL(path.join(root,'viewer/scene-style/road-contours.js')).href;
  const source=fs.readFileSync(path.join(root,'viewer/scene-style/road-surface.js'),'utf8').replace("'three'",JSON.stringify(threeURL)).replace("'./road-contours.js'",JSON.stringify(contoursURL));
  const {RoadSurface}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const THREE=await import(threeURL),scene=new THREE.Scene(),surface=new RoadSurface(scene);
  surface.update([0,0,0,0,255,0,0,0,0],3,3,-1.5,1,0);
  assert.ok(surface.group.children.some(c=>c.isMesh));
  const old=[...surface.group.children];let disposed=0;
  old.forEach(o=>o.geometry.addEventListener('dispose',()=>disposed++));
  surface.update([255],1,1,0,1,0);assert.equal(disposed,old.length);assert.equal(surface.group.children.length,0);
  surface.update([0,1,2,255],2,2,0,.5,0,true);
  assert.deepEqual([...surface.group.children[0].material.map.image.data],[174,182,189,255,213,217,220,255,185,196,186,255,0,0,0,0]);
});
