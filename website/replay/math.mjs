// Pure, tested geometry/decoding shared by the recorded replay UI.
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function assetPath(value) {
  if (typeof value !== 'string' || !/^(frames\/\d{3}\.json|images\/frame-\d{3}-camera-\d\.(jpg|jpeg|png))$/.test(value)) throw new Error('Invalid replay asset path.');
  return value;
}
export function decodeRoad(road) {
  const {width, height, runs} = road;
  if(!Number.isInteger(width) || !Number.isInteger(height) || width<1 || height<1 || width>512 || height>512 || !Array.isArray(runs) || runs.length%2) throw new Error('Invalid road representation.');
  const result=new Uint8Array(width*height);
  let offset=0;
  for(let i=0;i<runs.length;i+=2) {
    const value=runs[i], count=runs[i+1];
    if(![0,1,2,255].includes(value) || !Number.isInteger(count) || count<1 || offset+count>result.length) throw new Error('Invalid road run.');
    result.fill(value,offset,offset+count);offset+=count;
  }
  if(offset!==result.length) throw new Error('Incomplete road representation.');
  return result;
}
export function visibleObjects(frame, threshold) {
  return frame.boxes.map((box,index)=>({box,index,score:frame.scores[index],label:frame.labels[index],track:frame.track_ids[index]}))
    .filter(o=>o.score>=threshold && [0,1].includes(o.label) && o.box.length===7 && o.box.every(Number.isFinite) && o.box.slice(3,6).every(x=>x>0));
}
export function corners(box) {
  const [x,y,z,l,w,h,yaw]=box,c=Math.cos(yaw),s=Math.sin(yaw),result=[];
  for(const dz of [-1,1]) for(const dy of [-1,1]) for(const dx of [-1,1]) {
    const a=dx*l/2,b=dy*w/2;
    result.push([x+c*a-s*b,y+s*a+c*b,z+dz*h/2]);
  }
  return result;
}
const edges=[[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]];
function clipLine(a,b,width,height) {
  const dx=b[0]-a[0],dy=b[1]-a[1];
  let lo=0,hi=1;
  const p=[-dx,dx,-dy,dy],q=[a[0],width-a[0],a[1],height-a[1]];
  for(let i=0;i<4;i++) {
    if(Math.abs(p[i])<1e-12) { if(q[i]<0)return null;continue; }
    const t=q[i]/p[i];
    if(p[i]<0)lo=Math.max(lo,t);else hi=Math.min(hi,t);
    if(lo>hi)return null;
  }
  return [[a[0]+lo*dx,a[1]+lo*dy],[a[0]+hi*dx,a[1]+hi*dy]];
}
export function projectBox(box, intrinsic, cameraToEgo, width, height) {
  // Camera-to-ego is a rigid transform; transpose its rotation for the inverse.
  const points=corners(box).map(point=>{
    const p=point.map((value,i)=>value-cameraToEgo[i][3]);
    return [0,1,2].map(j=>p.reduce((sum,value,i)=>sum+cameraToEgo[i][j]*value,0));
  });
  const project=p=>{
    const q=intrinsic.map(row=>row.reduce((sum,value,i)=>sum+value*p[i],0));
    return [q[0]/q[2],q[1]/q[2]];
  };
  const result=[],near=.1;
  for(const [i,j] of edges) {
    let a=points[i],b=points[j];
    if(a[2]<near && b[2]<near)continue;
    if(a[2]<near) {const t=(near-a[2])/(b[2]-a[2]);a=a.map((x,k)=>x+(b[k]-x)*t);}
    if(b[2]<near) {const t=(near-b[2])/(a[2]-b[2]);b=b.map((x,k)=>x+(a[k]-x)*t);}
    const line=clipLine(project(a),project(b),width,height);
    if(line)result.push(line);
  }
  return result;
}
export function frameDelay(current, next, speed) {
  const dt=next && current.sequence===next.sequence ? next.timestamp-current.timestamp : .5;
  return clamp(Number.isFinite(dt)&&dt>0?dt:.5,.05,2)*1000/clamp(speed,.25,4);
}
