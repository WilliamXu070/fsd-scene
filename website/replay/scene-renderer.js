// Static replay adapter for the shared post-detection renderer.
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {decodeRoad, visibleObjects} from './math.mjs';
import {createVehicle,createPerson,styleScene,prepareVehicleAssets} from './scene-style/vehicles.js';
import {RoadSurface} from './scene-style/road-surface.js';
export {prepareVehicleAssets};

export class SceneRenderer {
  constructor(viewport, onSelect) {
    this.viewport=viewport;this.pending=0;this.pool=[];this.frame=null;this.selected=-1;this.outlines=false;this.rawRoad=false;
    this.scene=new THREE.Scene();
    this.camera=new THREE.PerspectiveCamera(42,1,.1,240);this.camera.up.set(0,0,1);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    viewport.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.target.set(10,0,0);this.controls.minDistance=7;this.controls.maxDistance=115;
    this.controls.maxPolarAngle=Math.PI/2-.035;this.controls.enableDamping=false;
    this.controls.addEventListener('change',()=>this.draw());
    styleScene(this.scene,this.renderer);
    this.surface=new RoadSurface(this.scene);
    this.cube=new THREE.BoxGeometry(1,1,1);
    this.edges=new THREE.EdgesGeometry(this.cube);
    this.materials={
      outline:new THREE.LineBasicMaterial({color:'#65818e'}),
      selected:new THREE.LineBasicMaterial({color:'#138570'})
    };
    this.ego=createVehicle(true);this.ego.scale.set(4.5,1.85,1.5);this.scene.add(this.ego);
    this.objects=new THREE.Group();this.scene.add(this.objects);
    this.grid=new THREE.GridHelper(80,16,'#a4b1ba','#d9e0e5');this.grid.rotation.x=Math.PI/2;this.grid.visible=false;this.scene.add(this.grid);
    this.resize=new ResizeObserver(()=>{
      const box=viewport.getBoundingClientRect();
      this.renderer.setSize(Math.max(1,box.width),Math.max(1,box.height));
      this.camera.aspect=box.width/Math.max(1,box.height);this.camera.updateProjectionMatrix();this.draw();
    });this.resize.observe(viewport);
    let down=null;
    this.renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
    this.renderer.domElement.addEventListener('pointerup',e=>{
      if(!down || Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;
      const rect=viewport.getBoundingClientRect(),ray=new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(2*(e.clientX-rect.left)/rect.width-1,1-2*(e.clientY-rect.top)/rect.height),this.camera);
      const hit=ray.intersectObjects(this.objects.children,true).find(hit=>hit.object.visible && hit.object.parent.visible && hit.object.type==='Mesh');
      if(hit)onSelect(hit.object.parent.userData.predictionIndex);
    });
    this.renderer.domElement.addEventListener('webglcontextlost',event=>{
      event.preventDefault();document.getElementById('webgl-error').hidden=false;
    });
    this.renderer.domElement.addEventListener('webglcontextrestored',()=>{
      document.getElementById('webgl-error').hidden=true;this.draw();
    });
    this.view(false);
  }
  update(frame,metadata,threshold) {
    if(this.frame!==frame) {
      this.frame=frame;
      const {width,height}=frame.road_display,cells=decodeRoad(frame.road_display);
      this.surface.update(cells,width,height,metadata.bev_min,metadata.bev_step,frame.ground_z,this.rawRoad);
      this.ego.position.z=frame.ground_z+.75;this.grid.position.z=frame.ground_z-.06;
    }
    const visible=visibleObjects(frame,threshold);
    visible.forEach((object,i)=>{
      if(!this.pool[i]) {
        const car=createVehicle(),person=createPerson(),outline=new THREE.LineSegments(this.edges,this.materials.outline);
        this.pool[i]={car,person,outline};this.objects.add(car,person,outline);
      }
      const slot=this.pool[i],mesh=object.label===0?slot.car:slot.person,box=object.box;
      slot.car.visible=object.label===0;slot.person.visible=object.label===1;
      mesh.position.set(box[0],box[1],box[2]);mesh.scale.set(box[3],box[4],box[5]);mesh.rotation.z=box[6];mesh.userData.predictionIndex=object.index;
      slot.outline.position.copy(mesh.position);slot.outline.scale.copy(mesh.scale);slot.outline.rotation.copy(mesh.rotation);
      slot.outline.visible=this.outlines || object.index===this.selected;
      slot.outline.material=object.index===this.selected?this.materials.selected:this.materials.outline;
    });
    for(let i=visible.length;i<this.pool.length;i++)for(const object of Object.values(this.pool[i]))object.visible=false;
    this.draw();
  }
  view(top) {
    this.camera.position.set(top?7.99:-19,top?0:-15,top?58:24);this.controls.target.set(8,0,0);this.controls.update();this.draw();
  }
  draw() {
    if(this.pending || document.hidden)return;
    this.pending=requestAnimationFrame(()=>{this.pending=0;this.renderer.render(this.scene,this.camera);});
  }
}
