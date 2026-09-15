/* Original, deterministic teaching diagrams. No inference, telemetry or remote assets.
 * Model shapes: src/fsd/model.py + configs/baseline.yaml.
 * Timing provenance: artifacts/final-measurements/reference/latency.json (30 updates,
 * eager CUDA events, three history frames). FPN is nested, never counted twice.
 */
(() => {
  'use strict';
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const depthAt = t => 5 + 45 * clamp(t);
  const projectedPixel = depth => 352 + 320 * (.25 * depth) / depth;
  const featureCount = (mode, layers) => mode === 'voxels' ? 160 * 160 * layers * 64 : mode === 'bev' ? 160 * 160 * 64 : 200 * 128;
  const toyUpdate = iteration => {
    const weight = 12 - 8 * Math.pow(.65, iteration);
    const gradient = 2 * (weight - 12);
    return { weight, gradient, loss: (weight - 12) ** 2, next: weight - .175 * gradient };
  };
  const TIMINGS = [
    ['Images + FPN', 6.103154102961223, '#376e8f'],
    ['Depth head', .15799892942110697, '#aa804a'],
    ['Context head', .04677973389625549, '#686c92'],
    ['Depth lifting', 4.835451745986939, '#458a83'],
    ['BEV encoder', 1.9096522688865663, '#8f9c61'],
    ['Road head', .30513493220011395, '#78794b'],
    ['Centre head', .37268266876538597, '#b5764c'],
    ['Geometry head', .31361493468284607, '#966761'],
    ['Refinement', 5.203506135940552, '#6d729a']
  ];
  const MODEL_TOTAL = 19.659810129801432;
  // Public pure functions let the test file check invariants without a browser dependency.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { clamp, depthAt, projectedPixel, featureCount, toyUpdate, TIMINGS, MODEL_TOTAL };
  }
  if (typeof document === 'undefined') return;

  const palette = ['#2d7975', '#ac623e', '#536d9a'];
  const ink = '#343b38', muted = '#7b8077', rule = '#d8dacd';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const txt = (x, y, value, attrs = '') => '<text x="' + x + '" y="' + y + '" ' + attrs + '>' + escape(value) + '</text>';
  const line = (x1, y1, x2, y2, color = rule, attrs = '') => '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" stroke="' + color + '" ' + attrs + '/>';
  const rect = (x, y, w, h, fill = 'none', attrs = '') => '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" fill="' + fill + '" ' + attrs + '/>';
  const circle = (x, y, r, fill, attrs = '') => '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + fill + '" ' + attrs + '/>';
  const poly = (points, fill, attrs = '') => '<polygon points="' + points.map(p => p.join(',')).join(' ') + '" fill="' + fill + '" ' + attrs + '/>';
  const group = (x, y, body) => '<g transform="translate(' + x + ' ' + y + ')">' + body + '</g>';
  const bold = 'class="viz-large"';
  function arrow(x1, y1, x2, y2, color = '#266c67', dashed = false) {
    const a = Math.atan2(y2 - y1, x2 - x1), s = 7;
    return line(x1, y1, x2, y2, color, 'stroke-width="2" ' + (dashed ? 'stroke-dasharray="5 4"' : '')) +
      poly([[x2,y2], [x2-s*Math.cos(a-.5),y2-s*Math.sin(a-.5)], [x2-s*Math.cos(a+.5),y2-s*Math.sin(a+.5)]], color);
  }
  function box(x, y, width, height, color, yaw = 0, dashed = false, opacity = 1) {
    return '<g transform="translate(' + x + ' ' + y + ') rotate(' + yaw + ')" opacity="' + opacity + '">' +
      rect(-width/2, -height/2, width, height, dashed ? 'none' : color, 'fill-opacity=".15" stroke="' + color + '" stroke-width="2" ' + (dashed ? 'stroke-dasharray="4 3"' : '')) +
      line(0, height/4, 0, -height/2 - 4, color, 'stroke-width="2"') + '</g>';
  }
  const scene = t => [
    { id:'A', x:-2.1 + .35*Math.sin(t*5), z:17 + 6*t, yaw:4*Math.sin(t*4), color:palette[0], length:4.3, width:1.9 },
    { id:'B', x:2.4, z:28 - 9*t, yaw:-5, color:palette[1], length:4.7, width:2 },
    { id:'P', x:5.2 - .8*t, z:21, yaw:0, color:palette[2], length:.8, width:.7 }
  ];
  function camera(w, h, t, id, label = 'Camera schematic', offset = 0) {
    const objects = scene(t);
    let body = rect(0,0,w,h,'#e5ece7') + rect(0,h*.4,w,h*.6,'#eeeadd') +
      poly([[w*.43,h*.4],[w*.57,h*.4],[w*.99,h],[w*.01,h]], '#cdd0c7') +
      line(w*.5,h*.41,w*.5,h,'#fffdf4','stroke-width="2" stroke-dasharray="8 9"');
    for (const o of [...objects].sort((a,b)=>b.z-a.z)) {
      const f = w*.75, cx = w*.5 + f*(o.x-offset)/o.z;
      const bottom = h*.41 + f*1.6/o.z;
      const bw = Math.max(4,f*o.width/o.z), bh = f*(o.id==='P'?1.7:1.5)/o.z;
      body += rect(cx-bw/2,bottom-bh,bw,bh,o.color,'rx="2"');
      if (o.id !== 'P') body += rect(cx-bw*.32,bottom-bh*.84,bw*.64,bh*.35,'#d9ebe7','fill-opacity=".8"');
      body += txt(cx,bottom-bh-6,o.id,'text-anchor="middle"');
    }
    body += rect(0,0,w,26,'#fcfbf6','fill-opacity=".88"') + txt(10,18,label);
    return '<defs><clipPath id="' + id + '"><rect width="' + w + '" height="' + h + '" rx="3"/></clipPath></defs><g clip-path="url(#' + id + ')">' + body + '</g>';
  }
  function bev(w, h, t, options = {}) {
    const scale = Math.min((w-42)/14,(h-45)/36);
    const to = (x,z) => [w/2+x*scale,h-25-z*scale];
    let s = rect(0,0,w,h,'#eff1e7');
    for(let x=-6;x<=6;x+=2) { const a=to(x,0),b=to(x,34); s+=line(...a,...b,rule); }
    for(let z=0;z<=34;z+=4) { const a=to(-6,z),b=to(6,z); s+=line(...a,...b,rule); }
    for(const x of [-4,0,4]) { const a=to(x,0),b=to(x,34); s+=line(...a,...b,'#a8af9f','stroke-dasharray="5 5"'); }
    const ego=to(0,1); s += box(...ego,1.9*scale,3*scale,muted) + txt(ego[0]+16,ego[1]+5,'ego');
    if (options.objects !== false) for(const o of scene(t)) {
      const p=to(o.x,o.z); s+=box(...p,o.width*scale,o.length*scale,o.color,o.yaw) + txt(p[0]+9,p[1]-8,o.id);
    }
    s += txt(10,18,options.label || 'Bird’s-eye view · metres');
    return { body:s, to, scale };
  }
  function featureTiles(w,h,t) {
    let s = '';
    for(let level=0;level<3;level++) {
      const size = Math.min(w*.44,(h-28)/3), left=w*.22+level*11, top=25+level*(h-35)/3;
      for(let y=0;y<4;y++) for(let x=0;x<8;x++) {
        const a=.12+.72*(.5+.5*Math.sin(x*.9+y*1.3+t*6+level));
        s+=rect(left+x*size/5,top+y*size/6,size/5-2,size/6-2,palette[level],'fill-opacity="'+a+'"');
      }
    }
    return txt(10,18,'Learned features (schematic)') + s;
  }
  const stages = [
    { name:'Images', shape:'4 × 3 × 256 × 704', coordinates:'Camera pixels', keeps:'Colour and image position; depth is unknown.', cost:'Read and normalize four RGB views.' },
    { name:'ResNet + FPN', shape:'Per camera: 128 × {32×88, 16×44, 8×22}', coordinates:'Image feature grids', keeps:'Multi-scale appearance; the car is now feature vectors.', cost:'Shared convolutional backbone and feature pyramid.' },
    { name:'Depth lifting', shape:'64 depth bins; pooled BEV: 64 × 160 × 160', coordinates:'Calibrated rays → ego-frame ground plane', keeps:'Depth-weighted evidence across cameras; explicit height is pooled away.', cost:'Distribute and accumulate features into BEV cells.' },
    { name:'BEV encoder', shape:'128 × 160 × 160', coordinates:'Ego-frame BEV; 0.5 m cells', keeps:'Local spatial context around the vehicle.', cost:'Three residual BEV blocks.' },
    { name:'Prediction heads', shape:'Road: 3; centre: 2; geometry: 8 channels', coordinates:'BEV grid → oriented 3D boxes', keeps:'Road classes plus up to 200 object proposals.', cost:'Dense heads, proposal decoding and selection.' },
    { name:'Refinement', shape:'128-D proposals; at most 100 final boxes', coordinates:'Ego boxes ↔ current / earlier image features', keeps:'Centre, size and yaw corrections from sampled evidence.', cost:'One spatial-and-temporal block; up to three earlier observations.' }
  ];
  const definitions = {
    opening: {
      title:'Camera observations to scene', duration:10000, seek:'Scene time',
      source:'Original animated schematic. The colours track the same authored objects, not recorded model predictions.',
      render(v,w) {
        const small=w<600, gap=20, pw=small?w:(w-2*gap)/3, ph=small?156:242;
        const h=small?ph*3+gap*2:ph;
        const at=i=>small?[0,i*(ph+gap)]:[i*(pw+gap),0];
        let s='';
        const a=at(0),b=at(1),c=at(2);
        // Four thumbnails use the same timestamp; offsets illustrate different views, not KITTI calibration.
        for(let i=0;i<4;i++) s+=group(a[0]+(i%2)*(pw/2+2),a[1]+Math.floor(i/2)*(ph/2+2),camera(pw/2-2,ph/2-2,v.t,'opening-camera-'+i,'View '+(i+1),(i-1.5)*1.4));
        s+=group(...b,featureTiles(pw,ph,v.t));
        s+=group(...c,bev(pw,ph,v.t).body);
        if(small) { s+=arrow(w/2,ph+2,w/2,ph+gap-2)+arrow(w/2,2*ph+gap+2,w/2,2*(ph+gap)-2); }
        else { s+=arrow(pw+2,ph/2,pw+gap-2,ph/2)+arrow(2*pw+gap+2,ph/2,2*(pw+gap)-2,ph/2); }
        const step=Math.min(2,Math.floor(v.t*3)), pos=at(step);
        s+=rect(...pos,pw,ph,'none','stroke="'+palette[0]+'" stroke-width="3" rx="3"');
        return { h, s, read:'<div class="viz-key"><span style="--key:'+palette[0]+'">A · car</span><span style="--key:'+palette[1]+'">B · car</span><span style="--key:'+palette[2]+'">P · pedestrian</span></div><p><strong>'+['Observe the images','Learn appearance and geometric evidence','Assemble a compact scene'][step]+'</strong> · '+(v.t*5).toFixed(2)+' s</p><p>All views share a timestamp. Colour identity connects an observation with its schematic ground-plane position.</p>' };
      }
    },
    ray: {
      title:'Pixel and depth ray', duration:8000, seek:'Depth along the ray',
      source:'Ideal pinhole illustration: focal length 320 px, principal point 352 px, X/Z = 0.25. The production model also handles calibrated fisheye rays.',
      render(v,w) {
        const d=depthAt(v.t), x0=32, y0=204, end=w-24, dx=end-x0, xp=x0+dx*.08;
        const along=depth=>[x0+dx*depth/55,y0-145*depth/55], p=along(d), pixel=along(4.4);
        let s=rect(xp-7,30,14,215,'#e8e2d5') + txt(xp+12,48,'image plane') +
          arrow(x0,y0,end,y0,muted) + txt(end,226,'Z →','text-anchor="end"') +
          line(x0,y0,...along(55),palette[0],'stroke-width="2"') +
          circle(x0,y0,6,ink) + txt(12,257,'camera') +
          circle(...p,8,palette[0],'stroke="#fffdf8" stroke-width="2"') +
          circle(...pixel,5,palette[1]) +
          line(p[0],p[1],p[0],y0,muted,'stroke-dasharray="4 4"') +
          txt(w-15,24,'Same ray, any depth','text-anchor="end"') +
          txt(15,292,'Fixed pixel: u = 432 px',bold) +
          txt(15,315,'u = 352 + 320 × (X / Z)');
        return { h:332,s,read:'<p><strong>Z = '+d.toFixed(1)+' m · X = '+(.25*d).toFixed(2)+' m</strong></p><p>The green point moves in space. The orange pixel stays fixed: X/Z = 0.25, so u = '+projectedPixel(d).toFixed(0)+' px. An image location alone cannot choose the distance.</p>', value:d.toFixed(1)+' m' };
      }
    },
    representation: {
      title:'Representation storage comparison', duration:9000, seek:'Explicit height layers',
      options:[{type:'choice',key:'mode',label:'Representation',values:[['voxels','Voxels'],['bev','BEV'],['sparse','Sparse objects']],initial:'voxels'}],
      source:'Analytical feature counts, excluding gradients, activations, image features and object geometry. Displayed grid cells are sampled for legibility; these are not runtime estimates.',
      render(v,w) {
        const layers=4+Math.round(v.t*28), mode=v.state.mode, h=280, count=featureCount(mode,layers);
        let s='';
        if(mode==='sparse') {
          s=bev(w,h,.5).body;
          s+=txt(w-12,40,'200 feature vectors','text-anchor="end"');
        } else {
          const size=Math.min(w/15,28), cx=w/2, base=245;
          const project=(x,y,z)=>[cx+(x-y)*size,base-(x+y)*size*.42-z*8];
          const visibleLayers=mode==='voxels'?Math.max(1,Math.round(layers/4)):1;
          for(let y=7;y>=0;y--) for(let x=0;x<8;x++) {
            const objectIndex=(x===2 && y>=3 && y<=4)?0:(x===5 && y>=4 && y<=5)?1:(x===6 && y===3)?2:-1;
            const occupied=objectIndex>=0;
            for(let z=0;z<(occupied?visibleLayers:1);z++) {
              const p=project(x,y,z), q=project(x+1,y,z), r=project(x+1,y+1,z), a=project(x,y+1,z);
              const fill=occupied?palette[objectIndex]:'#cbd5c5';
              s+=poly([p,q,r,a],fill,'fill-opacity="'+(occupied?.65:.25)+'" stroke="#fcfbf6" stroke-width="1"');
              if(mode==='voxels') {
                s+=poly([p,q,[q[0],q[1]+7],[p[0],p[1]+7]],fill,'fill-opacity=".25"');
                s+=poly([q,r,[r[0],r[1]+7],[q[0],q[1]+7]],fill,'fill-opacity=".4"');
              }
            }
          }
          s+=txt(12,23,mode==='voxels'?'Explicit height: '+layers+' layers':'Height pooled into channels');
          s+=txt(12,45,'Sampled display · 160 × 160 underlying grid');
        }
        const formula=mode==='voxels'?'160 × 160 × '+layers+' × 64':mode==='bev'?'160 × 160 × 64':'200 × 128';
        const trade=mode==='voxels'?'Retains explicit height separation; dense storage scales with the height axis.':mode==='bev'?'Keeps a shared ground plane; vertical separation is no longer an explicit grid axis.':'Keeps a fixed object budget; detailed surfaces and empty space are not represented by these vectors.';
        return {h,s,read:'<p><strong>'+formula+' = '+count.toLocaleString('en-US')+' feature values</strong> · '+(count*2/1048576).toFixed(2)+' MiB at FP16</p><p>'+trade+'</p>',value:layers+' layers (voxel case)'};
      }
    },
    architecture: {
      title:'Step through the baseline architecture',duration:18000,seek:'Architecture stage',steps:6,
      source:'Tensor shapes and operator order follow the four-camera baseline in src/fsd/model.py. Visual feature patterns are illustrative. This is our baseline, not an IPFormer implementation.',
      render(v,w) {
        const i=Math.min(5,Math.floor(v.t*6)), info=stages[i], h=286;
        const cols=w<500?3:6, cell=w/cols;
        let s='';
        for(let k=0;k<6;k++) {
          const x=(k%cols)*cell,y=Math.floor(k/cols)*34;
          s+=rect(x+2,y+3,cell-4,28,k===i?'#d9e8de':'#eeede4','rx="3"')+txt(x+cell/2,y+22,(k+1)+'. '+['Images','FPN','Lift','BEV','Heads','Refine'][k],'text-anchor="middle"');
        }
        const top=cols===3?80:50, panelH=h-top;
        if(i===0) s+=group(0,top,camera(w,panelH,.45,'architecture-camera'));
        else if(i===1) s+=group(0,top,featureTiles(w,panelH,.45));
        else {
          const ground=bev(w,panelH,.45,{objects:i>=4,label:i===2?'Depth probabilities → ground cells':i===3?'Residual BEV context':'Proposals in ego coordinates'});
          s+=group(0,top,ground.body);
          const car=ground.to(scene(.45)[0].x,scene(.45)[0].z);
          if(i===2) for(let r=0;r<5;r++) s+=arrow(w/2,top+panelH-18,car[0]+(r-2)*19,car[1]+top+r*8,'#458a83',true);
          if(i===3) for(let r=1;r<4;r++) s+=circle(car[0],car[1]+top,13+r*15,'none','stroke="#458a83" stroke-opacity="'+(1-r*.2)+'"');
          if(i===5) {
            for(let r=0;r<9;r++) { const px=car[0]+((r%3)-1)*14,py=top+car[1]+(Math.floor(r/3)-1)*18; s+=circle(px,py,3,palette[1]); }
            s+=txt(12,top+45,'Centre + 8 box corners');
            s+=txt(12,top+63,'sample current image features');
          }
        }
        return {h,s,read:'<p><strong>'+escape(info.name)+'</strong></p><dl><dt>Tensor</dt><dd>'+escape(info.shape)+'</dd><dt>Coordinates</dt><dd>'+escape(info.coordinates)+'</dd><dt>Retained</dt><dd>'+escape(info.keeps)+'</dd><dt>Work</dt><dd>'+escape(info.cost)+'</dd></dl>',value:(i+1)+' / 6'};
      }
    },
    training: {
      title:'Training and gradient update',duration:16000,seek:'Training demonstration',
      options:[{type:'select',key:'loss',label:'Supervision',values:[['depth','Depth'],['road','Road'],['centre','Object centre'],['geometry','Box geometry'],['yaw','Yaw']],initial:'depth'}],
      source:'Selected paths are a simplified dependency diagram, not a full autograd trace. Scalar example uses squared error; the project combines task-specific losses. No checkpoint is trained by this page.',
      render(v,w) {
        const progress=v.t*8, iteration=Math.min(7,Math.floor(progress)), phase=Math.min(3,Math.floor((progress-iteration)*4));
        const toy=toyUpdate(iteration), selected=v.state.loss, isDepth=selected==='depth';
        const labels=['Image features',isDepth?'Depth head':'Lift + BEV',isDepth?'Depth prediction':selected==='road'?'Road logits':selected==='centre'?'Centre heatmap':'Box + refinement','Loss: '+selected];
        const small=w<500, cols=small?2:4, cell=w/cols, hh=small?104:110;
        let s='';
        for(let k=0;k<4;k++) {
          const column=small && k>=2?3-k:k%cols;
          const x=column*cell+7,y=Math.floor(k/cols)*hh+23;
          s+=rect(x,y,cell-14,48,phase===2?'#eee1d4':'#dde9df','rx="4" stroke="'+(phase===2?palette[1]:palette[0])+'"')+
            txt(x+(cell-14)/2,y+28,labels[k],'text-anchor="middle"');
          if(k<3) {
            const nextColumn=small && k+1>=2?2-k:(k+1)%cols;
            const nx=nextColumn*cell+7,ny=Math.floor((k+1)/cols)*hh+23;
            if(small && k===1) s+=phase===2?arrow(x+(cell-14)/2,ny-6,x+(cell-14)/2,y+54,palette[1]):arrow(x+(cell-14)/2,y+54,x+(cell-14)/2,ny-6);
            else if(small && k===2) s+=phase===2?arrow(nx+cell-12,ny+24,x-2,y+24,palette[1]):arrow(x-2,y+24,nx+cell-12,ny+24);
            else s+=phase===2?arrow(nx-2,ny+24,x+cell-12,y+24,palette[1]):arrow(x+cell-12,y+24,nx-2,ny+24);
          }
        }
        const base=small?222:132;
        const current=phase===3?toy.next:toy.weight;
        s+=txt(12,base,['1. Forward pass','2. Measure error','3. Backpropagate','4. Update weight'][phase],bold);
        s+=line(22,base+45,w-24,base+45,rule,'stroke-width="6"')+
          circle(22+(w-46)*current/14,base+45,7,palette[0])+
          line(22+(w-46)*12/14,base+31,22+(w-46)*12/14,base+59,palette[1],'stroke-width="2"')+
          txt(12,base+83,'Toy output w: '+current.toFixed(3)+'   target: 12');
        const route=isDepth?'Depth supervision trains the depth head and image features.':'This loss reaches its prediction head, BEV features and the differentiable lifting paths.'+(['geometry','yaw'].includes(selected)?' Refined box supervision also reaches sampled image features.':'');
        return {h:base+99,s,read:'<p><strong>Scalar update '+(iteration+1)+' / 8</strong> · L = (w − 12)² = '+toy.loss.toFixed(3)+'</p><p>∂L/∂w = '+toy.gradient.toFixed(3)+'; w′ = '+toy.weight.toFixed(3)+' − 0.175 × ('+toy.gradient.toFixed(3)+') = '+toy.next.toFixed(3)+'.</p><p>'+route+'</p>',value:['Forward','Loss','Backward','Update'][phase]};
      }
    },
    evidence: {
      title:'Synthetic diagnostic scene',duration:12000,seek:'Shared scene timestamp',
      options:[
        {type:'check',key:'reference',label:'Show reference boxes',initial:true},
        {type:'range',key:'threshold',label:'Confidence threshold',initial:.3,min:.1,max:.95,step:.05}
      ],
      source:'All geometry, confidence values and errors are authored fixtures. No KITTI-360 / nuScenes images, private recordings or trained predictions are bundled.',
      render(v,w) {
        const small=w<600,pw=small?w:(w-16)/2,ph=240,by=small?ph+16:0,bx=small?0:pw+16;
        let s=camera(pw,ph,v.t,'diagnostic-camera','Synthetic camera · '+(v.t*5).toFixed(2)+' s');
        const ground=bev(pw,ph,v.t,{objects:false,label:'Illustrative detections · same timestamp'});
        let b=ground.body;
        const originals=scene(v.t);
        if(v.state.reference) for(const o of originals) {
          const p=ground.to(o.x,o.z);
          b+=box(...p,o.width*ground.scale,o.length*ground.scale,muted,o.yaw,true);
          if(o.id==='P') b+=line(...p,pw-76,ph-58,muted)+txt(pw-12,ph-52,'P · missed','text-anchor="end"');
        }
        const predictions=[
          {...originals[0],x:originals[0].x+.35*Math.sin(v.t*12),confidence:.88,label:'A'},
          {...originals[1],x:originals[1].x+.6,confidence:.69,label:'B'},
          {...originals[1],x:originals[1].x-1,z:originals[1].z+1.5,confidence:.42,label:'duplicate'},
          {...originals[0],x:-4.6,z:30,confidence:.36,label:'false +'}
        ];
        for(const [index,o] of predictions.entries()) {
          if(o.confidence<v.state.threshold) continue;
          const p=ground.to(o.x,o.z),left=index===0||index===3;
          const labelY=[ph-62,116,54,77][index];
          b+=box(...p,o.width*ground.scale,o.length*ground.scale,o.color,o.yaw)+
            line(...p,left?73:pw-87,labelY-4,o.color)+
            txt(left?12:pw-12,labelY,o.label+' '+o.confidence.toFixed(2),'text-anchor="'+(left?'start':'end')+'"');
        }
        s+=group(bx,by,b);
        const kept=predictions.filter(o=>o.confidence>=v.state.threshold).length;
        return {h:small?ph*2+16:ph,s,read:'<p><strong>'+kept+' boxes at threshold '+Number(v.state.threshold).toFixed(2)+'</strong> · dashed = authored reference; solid = illustrative detection.</p><p>At 0.30, inspect the duplicate and false positive. Raise the threshold to remove low-confidence boxes; above 0.69, car B is missed too. Pedestrian P has no detection at any threshold.</p>',value:(v.t*5).toFixed(2)+' s'};
      }
    },
    queries: {
      title:'Sparse object query refinement',duration:12000,seek:'Refinement progress',steps:6,
      options:[{type:'check',key:'history',label:'Show starting hypotheses',initial:true}],
      source:'Authored illustration of repeated sparse refinement, not a Sparse4Dv3 execution trace. Our earlier baseline uses one refinement block. Grey dashed boxes mark target geometry.',
      render(v,w) {
        const h=300,ground=bev(w,h,.45,{objects:false,label:'Sample evidence → refine query'});
        let s=ground.body;
        const targets=scene(.45).slice(0,2), speed=1-Math.pow(1-v.t,2);
        for(const [i,o] of targets.entries()) {
          const start={x:o.x+(i?2:-2),z:o.z+(i?-5:6),yaw:i?28:-32};
          const p=ground.to(lerp(start.x,o.x,speed),lerp(start.z,o.z,speed));
          const dst=ground.to(o.x,o.z),beg=ground.to(start.x,start.z);
          s+=box(...dst,o.width*ground.scale,o.length*ground.scale,muted,o.yaw,true);
          if(v.state.history) s+=box(...beg,2.6*ground.scale,5.5*ground.scale,o.color,start.yaw,true,.35);
          s+=line(...p,...dst,o.color,'stroke-dasharray="3 3"')+
            box(...p,lerp(2.6,o.width,speed)*ground.scale,lerp(5.5,o.length,speed)*ground.scale,o.color,lerp(start.yaw,o.yaw,speed));
          for(let r=0;r<5;r++) {
            const angle=r*Math.PI/2;
            s+=circle(p[0]+(r===4?0:Math.cos(angle)*12),p[1]+(r===4?0:Math.sin(angle)*18),2.5,o.color);
          }
          const labelY=i===0?174:225;
          s+=line(...p,i===0?68:w-68,labelY-4,o.color)+
            txt(i===0?12:w-12,labelY,'Q'+(i+1)+' '+lerp(.28,.93,speed).toFixed(2),'text-anchor="'+(i===0?'start':'end')+'"');
        }
        const falseP=ground.to(-4.5,31);
        s+=box(...falseP,2*ground.scale,4*ground.scale,palette[2],18,false,lerp(.8,.05,speed))+
          line(...falseP,66,48,palette[2])+txt(12,44,'Q3 '+lerp(.48,.04,speed).toFixed(2));
        const o=targets[0],x=lerp(o.x-2,o.x,speed),z=lerp(o.z+6,o.z,speed);
        return {h,s,read:'<p><strong>Pass '+Math.min(6,1+Math.floor(v.t*6))+' / 6</strong> · Q1 centre ('+x.toFixed(2)+', '+z.toFixed(2)+') m · yaw '+lerp(-32,o.yaw,speed).toFixed(1)+'°</p><p>Centres, dimensions and yaw change as sample locations move. Useful hypotheses align with the target; unsupported Q3 loses confidence. Dots indicate representative feature-sampling locations.</p>',value:Math.round(v.t*100)+'%'};
      }
    },
    latency: {
      title:'Measured latency and pipeline boundaries',duration:12000,seek:'Slowed timing cursor',
      options:[{type:'choice',key:'mode',label:'Timing view',values:[['model','Model profile'],['pipeline','Pipeline boundary']],initial:'model'}],
      source:'Saved eager CUDA-event profile: 30 updates, three history frames. Mean module times below are diagnostic, not the published uninstrumented total. Cached processing medians come from separate repeat runs.',
      render(v,w) {
        let s='',read='',h;
        if(v.state.mode==='model') {
          const total=TIMINGS.reduce((sum,item)=>sum+item[1],0), start=14, span=w-28;
          let x=start;
          for(const [,ms,color] of TIMINGS) { const size=ms/MODEL_TOTAL*span;s+=rect(x,38,size,42,color);x+=size; }
          s+=rect(x,38,Math.max(0,start+span-x),42,'#d6d3c6');
          const cursor=start+v.t*span;
          s+=line(cursor,28,cursor,90,ink,'stroke-width="2"')+txt(14,22,'Instrumented model timeline (mean)')+
            txt(14,105,'0 ms')+txt(w-14,105,MODEL_TOTAL.toFixed(2)+' ms','text-anchor="end"');
          let active='Other model work',sum=0;
          TIMINGS.forEach(([name,ms,color],i)=>{
            const y=132+i*25;
            if(v.t*MODEL_TOTAL>=sum && v.t*MODEL_TOTAL<sum+ms) active=name;
            sum+=ms;
            s+=rect(15,y-10,9,9,color)+txt(32,y,name)+txt(w-14,y,ms.toFixed(3)+' ms','text-anchor="end"');
          });
          h=386;
          s+=txt(14,h-12,'Grey remainder: '+(MODEL_TOTAL-total).toFixed(3)+' ms');
          read='<p><strong>'+active+' · cursor '+(v.t*MODEL_TOTAL).toFixed(2)+' ms</strong></p><p>FPN’s 0.456 ms is already inside Images + FPN. The grey remainder is the difference from the instrumented total, not another measured module. Playback is slowed for inspection.</p>';
        } else {
          h=310;
          const regions=[[18,49],[90,121],[234,49]];
          s+=rect(13,18,w-26,49,'#f1eee5','stroke="'+rule+'" stroke-dasharray="4 3"')+
            txt(24,38,'Capture / resize / pose estimation')+txt(24,56,'Excluded from the cached benchmark','class="viz-small"')+
            arrow(w/2,70,w/2,86)+
            rect(13,90,w-26,121,'#dbe7dc','stroke="'+palette[0]+'"')+
            txt(24,111,'Cached processing')+txt(24,134,'34.19–34.28 ms median',bold)+
            txt(24,163,'Includes H2D · model · road decoding')+
            txt(24,184,'pose/history · NMS · tracking')+
            arrow(w/2,215,w/2,230)+
            rect(13,234,w-26,49,'#f1eee5','stroke="'+rule+'" stroke-dasharray="4 3"')+
            txt(24,254,'Road mesh / rendering / display')+txt(24,272,'Excluded; UI refresh is a separate clock','class="viz-small"');
          const active=Math.min(2,Math.floor(v.t*3));
          s+=rect(13,regions[active][0],w-26,regions[active][1],'none','stroke="'+palette[0]+'" stroke-width="2"');
          read='<p><strong>Cached processing ≠ camera-to-display latency.</strong></p><p>Boxes show scope, not proportional durations. NMS and tracking are already inside cached processing: do not add them again. Native capture, pose estimation, queues, road mesh construction and display are excluded.</p>';
        }
        return {h,s,read,value:v.state.mode==='model'?(v.t*MODEL_TOTAL).toFixed(2)+' ms':'Boundary '+Math.min(3,1+Math.floor(v.t*3))+' / 3'};
      }
    }
  };

  const controllers = [];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0, previous = 0;
  function schedule() {
    if(!frame && !document.hidden && controllers.some(v=>v.playing && v.visible)) frame=requestAnimationFrame(tick);
  }
  function tick(now) {
    frame=0;
    if(document.hidden) { previous=0;return; }
    const dt=previous?Math.min(now-previous,80):0;
    if(previous && dt<32) { schedule();return; }
    previous=now;
    let active=false;
    for(const v of controllers) {
      if(!v.playing || !v.visible) continue;
      active=true;
      v.t=clamp(v.t+dt/v.config.duration);
      v.render();
      if(v.t>=1) v.pause();
    }
    if(active) schedule(); else previous=0;
  }
  function makeControl(tag,attrs,textContent) {
    const el=document.createElement(tag);
    for(const [key,value] of Object.entries(attrs||{})) el.setAttribute(key,value);
    if(textContent) el.textContent=textContent;
    return el;
  }
  function createFigure(figure) {
    const id=figure.dataset.animation,config=definitions[id];
    if(!config) return;
    const root=makeControl('div',{class:'viz-root'});
    const options=makeControl('div',{class:'viz-options',role:'group','aria-label':config.title+' options'});
    const stage=makeControl('div',{class:'viz-stage'});
    const readout=makeControl('div',{class:'viz-readout'});
    const transport=makeControl('div',{class:'viz-transport',role:'group','aria-label':config.title+' playback'});
    const play=makeControl('button',{type:'button',class:'viz-play','aria-label':'Play '+config.title},'Play');
    const restart=makeControl('button',{type:'button','aria-label':'Restart '+config.title},'Restart');
    const seekId='viz-'+id+'-seek',seekGroup=makeControl('div',{class:'viz-seek'});
    const seekLabel=makeControl('label',{for:seekId},config.seek);
    const output=makeControl('output',{for:seekId});
    const seek=makeControl('input',{id:seekId,type:'range',min:0,max:1000,step:1,value:0});
    const source=makeControl('p',{class:'viz-provenance'},config.source+' Starts paused. Play or scrub to explore.');
    const v={id,config,t:0,state:{},playing:false,visible:false,width:0,
      pause() {
        this.playing=false;
        play.textContent=this.t>=1?'Replay':'Play';
        play.setAttribute('aria-label',(this.t>=1?'Replay ':'Play ')+config.title);
        play.setAttribute('aria-pressed','false');
      },
      render() {
        const w=Math.max(260,Math.round(stage.getBoundingClientRect().width)-2);
        this.width=w;
        const result=config.render(this,w);
        stage.innerHTML='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+w+' '+result.h+'" role="img" aria-labelledby="viz-'+id+'-title viz-'+id+'-desc"><title id="viz-'+id+'-title">'+escape(config.title)+'</title><desc id="viz-'+id+'-desc">'+escape(readText(result.read))+'</desc>'+result.s+'</svg>';
        if(readout.innerHTML!==result.read) readout.innerHTML=result.read;
        seek.value=String(Math.round(this.t*1000));
        output.textContent=result.value || (this.t*100).toFixed(0)+'%';
        seek.setAttribute('aria-valuetext',output.textContent);
        if(this.previousButton) {
          this.previousButton.disabled=Math.floor(this.t*config.steps)===0;
          this.nextButton.disabled=this.t>=1;
        }
      }
    };
    const changed=()=>{v.pause();v.render();};
    for(const option of config.options || []) {
      v.state[option.key]=option.initial;
      if(option.type==='choice') {
        const choices=makeControl('div',{class:'viz-options',role:'group','aria-label':option.label});
        for(const [value,label] of option.values) {
          const button=makeControl('button',{type:'button','aria-pressed':String(value===option.initial)},label);
          button.addEventListener('click',()=>{
            v.state[option.key]=value;
            choices.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
            changed();
          });
          choices.append(button);
        }
        options.append(choices);
      } else {
        const label=makeControl('label',{},option.label);
        let input;
        if(option.type==='select') {
          input=makeControl('select',{'aria-label':option.label});
          for(const [value,name] of option.values) input.append(makeControl('option',{value},name));
          input.value=option.initial;
        } else if(option.type==='check') {
          input=makeControl('input',{type:'checkbox','aria-label':option.label});
          input.checked=option.initial;
        } else input=makeControl('input',{type:'range',min:option.min,max:option.max,step:option.step,value:option.initial,'aria-label':option.label});
        input.addEventListener('input',()=>{
          v.state[option.key]=option.type==='check'?input.checked:option.type==='range'?Number(input.value):input.value;
          changed();
        });
        label.append(input);options.append(label);
      }
    }
    play.addEventListener('click',()=>{
      if(v.playing) { v.pause();return; }
      if(reduced.matches) {
        v.t=v.t>=1?0:clamp(v.t+1/(config.steps||8));
        changed();return;
      }
      if(v.t>=1) v.t=0;
      v.playing=true;
      play.textContent='Pause';
      play.setAttribute('aria-label','Pause '+config.title);
      play.setAttribute('aria-pressed','true');
      previous=0;schedule();
    });
    restart.addEventListener('click',()=>{v.t=0;changed();});
    seek.addEventListener('input',()=>{v.t=Number(seek.value)/1000;changed();});
    transport.append(play,restart);
    if(config.steps) {
      v.previousButton=makeControl('button',{type:'button','aria-label':'Previous '+config.title+' stage'},'← Previous');
      v.nextButton=makeControl('button',{type:'button','aria-label':'Next '+config.title+' stage'},'Next →');
      v.previousButton.addEventListener('click',()=>{const index=Math.max(0,Math.min(config.steps-1,Math.floor(v.t*config.steps))-1);v.t=index===0?0:index/config.steps+.001;changed();});
      v.nextButton.addEventListener('click',()=>{v.t=clamp((Math.floor(v.t*config.steps)+1)/config.steps+.001);changed();});
      transport.append(v.previousButton,v.nextButton);
    }
    seekGroup.append(seekLabel,output,seek);transport.append(seekGroup);
    root.append(options,stage,readout,transport,source);
    figure.insertBefore(root,figure.querySelector('figcaption'));
    controllers.push(v);
    v.render();v.pause();
    const observer=new IntersectionObserver(entries=>{
      v.visible=entries[0].isIntersecting;
      if(!v.visible) v.pause();
      schedule();
    });
    observer.observe(figure);
    const resize=new ResizeObserver(()=> {
      if(Math.round(stage.getBoundingClientRect().width)-2!==v.width) v.render();
    });
    resize.observe(stage);
    v.motionLabel=()=>{
      v.pause();
      if(reduced.matches) {
        play.textContent='Step';
        play.setAttribute('aria-label','Step '+config.title);
        source.textContent=config.source+' Reduced motion: Step advances a still frame; the slider remains available.';
      } else source.textContent=config.source+' Starts paused. Play or scrub to explore.';
    };
    // Keep reduced-motion labeling consistent after any pause, scrub or mode change.
    const pause=v.pause.bind(v);
    v.pause=()=>{pause();if(reduced.matches) {play.textContent='Step';play.setAttribute('aria-label','Step '+config.title);}};
    v.motionLabel();
  }
  function readText(html) {
    // All markup is trusted local constants; parsing only strips markup for SVG descriptions.
    const template=document.createElement('template');
    template.innerHTML=html;
    return template.content.textContent.replace(/\s+/g,' ').trim();
  }
  document.querySelectorAll('[data-animation]').forEach(figure=>{
    try { createFigure(figure); }
    catch(error) {
      const fallback=makeControl('p',{class:'viz-fallback'},'This interactive figure could not load. The caption below describes the intended mechanism. Please reload to try again.');
      figure.insertBefore(fallback,figure.querySelector('figcaption'));
      console.error('Figure initialization failed:',figure.dataset.animation,error);
    }
  });
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden) { controllers.forEach(v=>v.pause());if(frame) cancelAnimationFrame(frame);frame=0;previous=0; }
  });
  reduced.addEventListener('change',()=>controllers.forEach(v=>v.motionLabel()));
})();
