import * as THREE from 'three';
import {metricRegions} from './road-contours.js';

export class RoadSurface {
  constructor(scene) {
    this.scene=scene;this.group=new THREE.Group();scene.add(this.group);this.raw=false;
    this.roadMaterial=new THREE.MeshStandardMaterial({color:'#ffffff',roughness:1,metalness:0,side:THREE.DoubleSide});
    this.nonroadMaterial=new THREE.MeshStandardMaterial({color:'#dce0e5',roughness:1,metalness:0,side:THREE.DoubleSide});
    this.edgeMaterial=new THREE.LineBasicMaterial({color:'#b5bdc7',transparent:true,opacity:.75});
  }
  clear() {
    for(const object of [...this.group.children]) {
      this.group.remove(object);object.geometry.dispose();
      if(object.userData.temporaryMaterial){object.material.map?.dispose();object.material.dispose();}
    }
  }
  update(cells,width,height,minimum,step,ground,raw=this.raw) {
    this.raw=raw;this.clear();this.group.position.z=ground-.025;
    if(raw) {
      const rgba=new Uint8Array(cells.length*4),colors=[[174,182,189],[213,217,220],[185,196,186]];
      for(let i=0;i<cells.length;i++)if(colors[cells[i]]){rgba.set(colors[cells[i]],i*4);rgba[i*4+3]=255;}
      const map=new THREE.DataTexture(rgba,width,height);map.colorSpace=THREE.SRGBColorSpace;map.needsUpdate=true;
      const mesh=new THREE.Mesh(new THREE.PlaneGeometry(width*step,height*step),new THREE.MeshBasicMaterial({map,transparent:true,side:THREE.DoubleSide}));
      mesh.position.set(minimum+width*step/2,minimum+height*step/2,0);mesh.userData.temporaryMaterial=true;this.group.add(mesh);return;
    }
    // A visible-ground underlay prevents tiny cracks between independently smoothed
    // road/non-road boundaries. Unknown ground remains unfilled.
    this.addRegions(metricRegions(cells,width,height,v=>v!==255,minimum,step),this.nonroadMaterial,false);
    this.addRegions(metricRegions(cells,width,height,v=>v===0,minimum,step),this.roadMaterial,true);
  }
  addRegions(regions,material,boundary) {
    const shapes=[];
    for(const region of regions) {
      const shape=new THREE.Shape(region.outer.map(p=>new THREE.Vector2(...p)));
      shape.holes=region.holes.map(hole=>new THREE.Path(hole.map(p=>new THREE.Vector2(...p))));shapes.push(shape);
      if(boundary)for(const ring of [region.outer,...region.holes]) {
        const line=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ring.map(p=>new THREE.Vector3(p[0],p[1],.014))),this.edgeMaterial);
        this.group.add(line);
      }
    }
    if(!shapes.length)return;
    const mesh=new THREE.Mesh(new THREE.ShapeGeometry(shapes),material);
    mesh.position.z=boundary?.006:0;mesh.receiveShadow=true;this.group.add(mesh);
  }
}
