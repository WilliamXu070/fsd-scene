// Bounded, CPU-side QA telemetry. No model, WebGL fence, or display-completion timing.
export const TIMING_DEFINITIONS = Object.freeze({
  scene_update_cpu_ms: 'Synchronous road/object/DOM update, including assigning camera URLs; excludes image decode, layout/paint completion, and GPU completion.',
  render_submit_cpu_ms: 'CPU wall duration of renderer.render(), including any driver blocking; not GPU-complete rendering latency.',
  raf_interval_ms: 'Interval between requestAnimationFrame callback timestamps; browser scheduling, not presentation/compositor completion or fresh perception FPS.',
  request_headers_ms: 'Accepted scene request start to response headers; server/network scheduling included.',
  response_body_json_ms: 'Accepted response headers through response.json completion; combines remaining body transfer and JSON decode, not isolated JSON CPU time.',
  request_to_decoded_ms: 'Accepted scene request start through response.json completion.',
  request_to_scene_apply_ms: 'Scene load start through synchronous scene/DOM application; camera image decode and GPU/display completion excluded.',
  recorded_model_call_ms: 'Copied from the exported inference_ms field; historical model call, potentially cold, not newly measured perception.'
});

class Ring {
  constructor(capacity) {this.values=new Float64Array(capacity);this.cursor=0;this.retained=0;this.observed=0;this.rejected=0;}
  add(value) {
    if(!Number.isFinite(value)||value<0){this.rejected++;return;}
    this.values[this.cursor]=value;this.cursor=(this.cursor+1)%this.values.length;
    this.retained=Math.min(this.retained+1,this.values.length);this.observed++;
  }
  summary() {
    const values=Array.from(this.values.subarray(0,this.retained)).sort((a,b)=>a-b);
    const percentile=q=>{if(!values.length)return null;const at=(values.length-1)*q,lo=Math.floor(at),hi=Math.ceil(at);return values[lo]+(values[hi]-values[lo])*(at-lo);};
    return {observed:this.observed,retained:this.retained,rejected:this.rejected,
      mean_ms:values.length?values.reduce((a,b)=>a+b,0)/values.length:null,
      median_ms:percentile(.5),p95_ms:percentile(.95),min_ms:values[0]??null,max_ms:values.at(-1)??null};
  }
}

export class ReplayTiming {
  constructor({capacity=600,now=()=>performance.now()}={}) {
    if(!Number.isInteger(capacity)||capacity<1||capacity>10000)throw new Error('Timing capacity must be 1..10000');
    this.capacity=capacity;this.now=now;this.reset();
  }
  reset() {
    this.metrics=Object.fromEntries(Object.keys(TIMING_DEFINITIONS).map(key=>[key,new Ring(this.capacity)]));
    this.requests={accepted:0,superseded:0,failed:0};this.lastRaf=null;this.started=this.now();
  }
  record(key,value) {if(this.metrics[key])this.metrics[key].add(value);}
  raf(timestamp) {
    if(!Number.isFinite(timestamp))return;
    if(this.lastRaf!==null)this.record('raf_interval_ms',timestamp-this.lastRaf);
    this.lastRaf=timestamp;
  }
  request(event) {
    if(Object.hasOwn(this.requests,event.outcome))this.requests[event.outcome]++;
    if(event.outcome!=='accepted')return;
    for(const key of ['request_headers_ms','response_body_json_ms','request_to_decoded_ms'])this.record(key,event[key]);
  }
  snapshot() {
    return {schema_version:1,elapsed_since_reset_ms:this.now()-this.started,capacity_per_metric:this.capacity,
      retained_policy:'Last 600 values per metric by default; observed counts include earlier discarded samples.',
      percentile_method:'Linear interpolation over the retained sample values.',requests:{...this.requests},
      metrics:Object.fromEntries(Object.entries(this.metrics).map(([key,ring])=>[key,{definition:TIMING_DEFINITIONS[key],...ring.summary()}])),
      interpretation:'Instrumented browser CPU/scheduling measurements only. No GPU fence, compositor presentation, physical display, image-capture or new model-inference measurement.'};
  }
}
