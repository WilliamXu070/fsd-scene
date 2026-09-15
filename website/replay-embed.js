// Isolate the model viewer from article controls; pause it when it leaves view.
(() => {
  const frame=document.querySelector('iframe[data-model-replay]');
  if(!frame)return;
  let visible=false;
  const pause=()=>frame.contentWindow?.postMessage({type:'pause-model-replay'},location.origin);
  const visibility=()=>frame.contentWindow?.postMessage({type:'model-replay-visibility',visible},location.origin);
  new IntersectionObserver(entries=>{
    visible=entries[0].isIntersecting;
    visibility();
  }).observe(frame);
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin || event.source!==frame.contentWindow)return;
    if(event.data?.type==='model-replay-height' && Number.isFinite(event.data.height))frame.style.height=Math.max(550,Math.min(2400,event.data.height))+'px';
    if(event.data?.type==='model-replay-ready')visibility();
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
})();
