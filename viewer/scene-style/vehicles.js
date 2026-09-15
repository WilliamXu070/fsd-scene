import * as THREE from 'three';

const templates=new Map();
const materials={
  body:new THREE.MeshPhysicalMaterial({color:'#969ca5',roughness:.34,metalness:.24,clearcoat:.3}),
  red:new THREE.MeshPhysicalMaterial({color:'#d91f32',roughness:.26,metalness:.15,clearcoat:.7}),
  glass:new THREE.MeshPhysicalMaterial({color:'#34404c',roughness:.2,metalness:.32,clearcoat:1}),
  tire:new THREE.MeshStandardMaterial({color:'#252b32',roughness:.85}),
  metal:new THREE.MeshStandardMaterial({color:'#6d7783',roughness:.37,metalness:.6}),
  headlight:new THREE.MeshBasicMaterial({color:'#eef5fc'}),
  taillight:new THREE.MeshBasicMaterial({color:'#a91725'}),
  eye:new THREE.MeshStandardMaterial({color:'#fafafa',roughness:.6}),
  pupil:new THREE.MeshStandardMaterial({color:'#20323d',roughness:.4}),
  gold:new THREE.MeshStandardMaterial({color:'#ffcf43',roughness:.38,metalness:.1}),
  person:new THREE.MeshStandardMaterial({color:'#668395',roughness:.65})
};
const cylinder=new THREE.CylinderGeometry(1,1,1,32),sphere=new THREE.SphereGeometry(1,20,12);
let ready;
export function prepareVehicleAssets() {
  return ready??=Promise.all(['traffic-sedan','ego-racer'].map(async name=>{
    const {default:asset}=await import('./assets/'+name+'.mesh.js');
    if(asset.schema!==1||!Array.isArray(asset.parts))throw new Error('Unsupported vehicle mesh');
    const parts=asset.parts.map(part=>{
      const decode=value=>{const bytes=Uint8Array.from(atob(value),c=>c.charCodeAt(0));return new Float32Array(bytes.buffer);};
      const position=decode(part.position),normal=decode(part.normal);
      if(position.length!==normal.length||position.length%9||!position.every(Number.isFinite)||!normal.every(Number.isFinite)||!materials[part.material])throw new Error('Invalid vehicle geometry');
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(position,3));geometry.setAttribute('normal',new THREE.BufferAttribute(normal,3));geometry.computeBoundingSphere();
      return {geometry,material:part.material};
    });templates.set(name,{parts,wheelsIncluded:asset.wheels_included===true});
  }));
}

function piece(group,geometry,material,scale,position) {
  const mesh=new THREE.Mesh(geometry,material);mesh.scale.set(...scale);mesh.position.set(...position);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);return mesh;
}

export function createVehicle(ego=false) {
  const group=new THREE.Group(),asset=templates.get(ego?'ego-racer':'traffic-sedan'),parts=asset?.parts;
  if(!parts)throw new Error('Load the vehicle assets before constructing the scene');
  for(const part of parts)piece(group,part.geometry,ego&&part.material==='body'?materials.red:materials[part.material],[1,1,1],[0,0,0]);
  // Round wheels replace the angular source wheels. The detection box, not the
  // display mesh, remains authoritative for projected geometry and inspection.
  if(!asset.wheelsIncluded)for(const x of [-.28,.255])for(const y of [-.43,.43]) {
    piece(group,cylinder,materials.tire,[.087,.11,.22],[x,y,-.28]);
    piece(group,cylinder,materials.metal,[.057,.114,.15],[x,y,-.28]);
  }
  if(ego) {
    // Original character treatment, not a redistributed Disney/Pixar asset.
    // Windshield eyes face +X and are readable when orbiting to the front.
    for(const y of [-.14,.14]) {
      const white=piece(group,sphere,materials.eye,[.018,.126,.092],[.151,y,.323]);white.rotation.y=.53;
      const pupil=piece(group,sphere,materials.pupil,[.013,.037,.044],[.171,y-.017,.329]);pupil.rotation.y=.53;
    }
    // A small gold hood stripe and side flashes, authored as geometry.
    const stripe=new THREE.BoxGeometry(.23,.028,.004);
    piece(group,stripe,materials.gold,[1,1,1],[.33,-.08,.02]);
    piece(group,stripe,materials.gold,[1,1,1],[.33,.08,.02]);
    const flash=new THREE.Shape();flash.moveTo(-.24,-.11);flash.lineTo(-.04,.07);flash.lineTo(-.035,-.025);flash.lineTo(.2,.08);flash.lineTo(.045,-.12);flash.lineTo(.04,-.03);flash.closePath();
    const geometry=new THREE.ShapeGeometry(flash);
    for(const side of [-1,1]) {const mesh=piece(group,geometry,new THREE.MeshBasicMaterial({color:'#ffd24c',side:THREE.DoubleSide}),[1,1,1],[0,.501*side,-.05]);mesh.rotation.x=Math.PI/2;}
  }
  return group;
}

export function createPerson() {
  const group=new THREE.Group();
  piece(group,sphere,materials.person,[.25,.32,.25],[0,0,.03]);
  piece(group,sphere,materials.person,[.2,.21,.105],[0,0,.385]);
  for(const y of [-.17,.17]) {const leg=piece(group,cylinder,materials.person,[.115,.4,.105],[0,y,-.28]);leg.rotation.x=Math.PI/2;}
  return group;
}

export function styleScene(scene,renderer) {
  scene.background=new THREE.Color('#f6f7f9');scene.fog=new THREE.Fog('#f6f7f9',42,100);
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  scene.add(new THREE.HemisphereLight('#ffffff','#b9c0cb',2.5));
  const sun=new THREE.DirectionalLight('#ffffff',2.7);sun.position.set(-15,-18,32);sun.castShadow=true;
  sun.shadow.mapSize.set(1024,1024);Object.assign(sun.shadow.camera,{left:-38,right:38,top:38,bottom:-38,near:.5,far:110});sun.shadow.bias=-.0003;sun.shadow.normalBias=.04;sun.shadow.radius=3;scene.add(sun);
  return sun;
}
