// Ego-centred orbit, shared by the local and recorded viewers.
export function centerOrbit(camera,controls,ego,top=false) {
  const {x,y,z}=ego.position;
  controls.target.set(x,y,z);
  camera.position.set(x+(top?-.01:-23),y+(top?0:-18),z+(top?52:25));
  controls.update();
}

export function followEgoHeight(camera,controls,ego,nextZ) {
  // Move the camera and pivot together, retaining any deliberate user pan.
  const delta=nextZ-ego.position.z;
  ego.position.z=nextZ;
  camera.position.z+=delta;
  controls.target.z+=delta;
  controls.update();
}
