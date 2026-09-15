// Adapted from this project's viewer/app.js. Same metric axes and procedural meshes.
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {decodeRoad, visibleObjects} from './math.mjs';

export class SceneRenderer {
  constructor(viewport, onSelect) {
    this.viewport=viewport;this.pending=0;this.pool=[];this.frame=null;this.selected=-1;this.outlines=false;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#f2f4f6');
    this.scene.fog=new THREE.Fog('#f2f4f6',70,150);
    this.camera=new THREE.PerspectiveCamera(42,1,.1,240);this.camera.up.set(0,0,1);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.75));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    viewport.append(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.target.set(10,0,0);this.controls.minDistance=7;this.controls.maxDistance=115;
    this.controls.maxPolarAngle=Math.PI/2-.035;this.controls.enableDamping=false;
    this.controls.addEventListener('change',()=>this.draw());
    this.scene.add(new THREE.HemisphereLight('#ffffff','#b0bbc3',2.8));
    const sun=new THREE.DirectionalLight('#ffffff',2.1);sun.position.set(-10,-25,45);this.scene.add(sun);
    this.cube=new THREE.BoxGeometry(1,1,1);
    this.edges=new THREE.EdgesGeometry(this.cube);
    this.materials={
      body:new THREE.MeshStandardMaterial({color:'#879096',roughness:.55,metalness:.2}),
      dark:new THREE.MeshStandardMaterial({color:'#232e35',roughness:.45,metalness:.3}),
      glass:new THREE.MeshStandardMaterial({color:'#52636e',roughness:.28,metalness:.35}),
      tire:new THREE.MeshStandardMaterial({color:'#43484d',roughness:1}),
      light:new THREE.MeshBasicMaterial({color:'#edf1f0'}),
      rear:new THREE.MeshBasicMaterial({color:'#a96865'}),
      person:new THREE.MeshStandardMaterial({color:'#688ca7',roughness:.8}),
      outline:new THREE.LineBasicMaterial({color:'#65818e'}),
      selected:new THREE.LineBasicMaterial({color:'#138570'})
    };
    this.ego=this.car(true);this.ego.scale.set(4.5,1.85,1.5);this.scene.add(this.ego);
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
  piece(group,material,scale,position) {
    const mesh=new THREE.Mesh(this.cube,material);mesh.scale.set(...scale);mesh.position.set(...position);group.add(mesh);return mesh;
  }
  car(ego=false) {
    const group=new THREE.Group(),m=this.materials,body=ego?m.dark:m.body;
    this.piece(group,body,[.98,.95,.34],[0,0,-.22]);
    this.piece(group,body,[.7,.89,.19],[-.04,0,.015]);
    this.piece(group,m.glass,[.54,.78,.27],[-.055,0,.17]);
    this.piece(group,body,[.5,.8,.07],[-.065,0,.34]);
    for(const x of [-.3,.3])for(const y of [-.48,.48])this.piece(group,m.tire,[.17,.065,.24],[x,y,-.36]);
    for(const y of [-.32,.32]){this.piece(group,m.light,[.012,.18,.055],[.495,y,-.13]);this.piece(group,m.rear,[.012,.17,.055],[-.495,y,-.11]);}
    return group;
  }
  person() {
    const group=new THREE.Group(),m=this.materials.person;
    this.piece(group,m,[.52,.66,.43],[0,0,.035]);this.piece(group,m,[.42,.46,.19],[0,0,.36]);
    for(const y of [-.19,.19])this.piece(group,m,[.31,.23,.44],[0,y,-.29]);
    return group;
  }
  update(frame,metadata,threshold) {
    if(this.frame!==frame) {
      this.frame=frame;
      const {width,height}=frame.road_display,cells=decodeRoad(frame.road_display);
      const colors=[[174,182,189],[213,217,220],[185,196,186]],rgba=new Uint8Array(cells.length*4);
      for(let i=0;i<cells.length;i++)if(cells[i]!==255){rgba.set(colors[cells[i]],i*4);rgba[i*4+3]=255;}
      if(this.road){this.scene.remove(this.road);this.road.geometry.dispose();this.road.material.map.dispose();this.road.material.dispose();}
      const texture=new THREE.DataTexture(rgba,width,height,THREE.RGBAFormat);
      texture.colorSpace=THREE.SRGBColorSpace;texture.magFilter=THREE.NearestFilter;texture.minFilter=THREE.NearestFilter;texture.needsUpdate=true;
      this.road=new THREE.Mesh(new THREE.PlaneGeometry(width*metadata.bev_step,height*metadata.bev_step),new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:THREE.DoubleSide}));
      this.road.position.set(metadata.bev_min+width*metadata.bev_step/2,metadata.bev_min+height*metadata.bev_step/2,frame.ground_z-.025);
      this.road.renderOrder=-1;this.scene.add(this.road);
      this.ego.position.z=frame.ground_z+.75;this.grid.position.z=frame.ground_z-.06;
    }
    const visible=visibleObjects(frame,threshold);
    visible.forEach((object,i)=>{
      if(!this.pool[i]) {
        const car=this.car(),person=this.person(),outline=new THREE.LineSegments(this.edges,this.materials.outline);
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
    this.camera.position.set(top?9.99:-28,top?0:-10,top?72:30);this.controls.target.set(10,0,0);this.controls.update();this.draw();
  }
  draw() {
    if(this.pending || document.hidden)return;
    this.pending=requestAnimationFrame(()=>{this.pending=0;this.renderer.render(this.scene,this.camera);});
  }
}
