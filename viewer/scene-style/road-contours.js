// Presentation geometry only. Input classes, confidence, and detections stay untouched.
export function cellsFromRecord(record) {
  const height=record.road?.length||0,width=record.road?.[0]?.length||0;
  const cells=new Uint8Array(width*height).fill(255);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const value=record.road[y][x],confidence=record.road_confidence?.[y]?.[x]??1;
    if((record.road_visible?.[y]?.[x]??true)&&Number.isFinite(confidence)&&confidence>=.45&&[0,1,2].includes(value))cells[y*width+x]=value;
  }
  return {cells,width,height};
}

export function signedArea(points) {
  return points.reduce((sum,p,i)=>{const q=points[(i+1)%points.length];return sum+p[0]*q[1]-q[0]*p[1];},0)/2;
}

export function containsPoint(ring,p) {
  let inside=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++) {
    const a=ring[i],b=ring[j];
    if((a[1]>p[1])!==(b[1]>p[1]) && p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;
  }
  return inside;
}

export function traceRegions(cells,width,height,accept) {
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>512||height>512||cells.length!==width*height)throw new Error('Invalid surface grid');
  const edges=[],outgoing=new Map(),key=p=>p[1]*(width+1)+p[0];
  const has=(x,y)=>x>=0&&y>=0&&x<width&&y<height&&accept(cells[y*width+x]);
  const edge=(a,b)=>{const id=edges.length;edges.push({a,b,used:false});const k=key(a);if(!outgoing.has(k))outgoing.set(k,[]);outgoing.get(k).push(id);};
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(has(x,y)) {
    if(!has(x,y-1))edge([x,y],[x+1,y]);
    if(!has(x+1,y))edge([x+1,y],[x+1,y+1]);
    if(!has(x,y+1))edge([x+1,y+1],[x,y+1]);
    if(!has(x-1,y))edge([x,y+1],[x,y]);
  }
  const rings=[];
  for(const first of edges) {
    if(first.used)continue;
    const ring=[];let current=first;
    for(let count=0;count<=edges.length;count++) {
      current.used=true;ring.push(current.a);
      if(key(current.b)===key(first.a))break;
      const next=(outgoing.get(key(current.b))||[]).map(i=>edges[i]).filter(e=>!e.used);
      if(!next.length)throw new Error('Open road contour');
      const dx=current.b[0]-current.a[0],dy=current.b[1]-current.a[1];
      next.sort((a,b)=>{
        const turn=e=>Math.atan2(dx*(e.b[1]-e.a[1])-dy*(e.b[0]-e.a[0]),dx*(e.b[0]-e.a[0])+dy*(e.b[1]-e.a[1]));
        return turn(b)-turn(a); // Keep diagonally touching islands separate.
      });current=next[0];
    }
    if(ring.length>=4)rings.push(ring);
  }
  const regions=rings.filter(r=>signedArea(r)>0).map(outer=>({outer,holes:[]}));
  for(const hole of rings.filter(r=>signedArea(r)<0)) {
    const owners=regions.filter(r=>containsPoint(r.outer,hole[0])).sort((a,b)=>signedArea(a.outer)-signedArea(b.outer));
    if(owners.length)owners[0].holes.push(hole);else throw new Error('Orphan road hole');
  }
  return regions;
}

export function smoothRing(ring) {
  // Two local averaging passes, each capped relative to its original vertex.
  // No dilation, component deletion, hallucinated lane lines or temporal lag.
  let points=ring.map(p=>[...p]);
  for(let pass=0;pass<2;pass++)points=points.map((p,i)=>{
    const a=points[(i+points.length-1)%points.length],b=points[(i+1)%points.length];
    const q=[(a[0]+2*p[0]+b[0])/4,(a[1]+2*p[1]+b[1])/4],origin=ring[i];
    const d=Math.hypot(q[0]-origin[0],q[1]-origin[1]),t=d>.5?.5/d:1;
    return [origin[0]+(q[0]-origin[0])*t,origin[1]+(q[1]-origin[1])*t];
  });
  return points;
}

export function metricRegions(cells,width,height,accept,minimum,step,smooth=true) {
  if(!Number.isFinite(minimum)||!Number.isFinite(step)||step<=0)throw new Error('Invalid metric calibration');
  const convert=ring=>(smooth?smoothRing(ring):ring).map(([x,y])=>[minimum+x*step,minimum+y*step]);
  return traceRegions(cells,width,height,accept).map(r=>({outer:convert(r.outer),holes:r.holes.map(convert)}));
}
