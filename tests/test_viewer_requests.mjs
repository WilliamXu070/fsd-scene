import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../viewer/replay-requests.js', import.meta.url), 'utf8');
const {LatestFrameRequest, debouncedScrub} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };

test('a newer scrub cancels the earlier request and never decodes its late response', async () => {
  const pending=[];
  const loader=new LatestFrameRequest((url, options) => { const job=deferred();pending.push({...job,url,options});return job.promise; });
  const first=loader.load('/api/frame?index=0');
  const second=loader.load('/api/frame?index=1621');
  assert.equal(pending[0].options.signal.aborted,true);
  let staleDecodes=0;
  pending[0].resolve({ok:true,json:async()=>{staleDecodes++;return {index:0};}});
  pending[1].resolve({ok:true,json:async()=>({index:1621})});
  assert.equal(await first,null);
  assert.deepEqual(await second,{index:1621});
  assert.equal(staleDecodes,0);
});

test('a superseded JSON decode or failure cannot replace or interrupt the latest frame', async () => {
  const decode=deferred();let called=0;
  const loader=new LatestFrameRequest(async()=>({ok:true,json:()=>++called===1?decode.promise:Promise.resolve({index:2})}));
  const first=loader.load('first');await Promise.resolve();
  const second=loader.load('second');
  assert.deepEqual(await second,{index:2});
  decode.resolve({index:1});assert.equal(await first,null);
  const failing=new LatestFrameRequest(async()=>({ok:false}));
  await assert.rejects(failing.load('current'),/not ready/);
});

test('one hundred rapid slider inputs issue only the final scrub and pending work can be cancelled', async () => {
  const calls=[];const scrub=debouncedScrub(value=>calls.push(value),10);
  for(let index=0;index<100;index++)scrub(index);
  await new Promise(resolve=>setTimeout(resolve,35));
  assert.deepEqual(calls,[99]);
  scrub(1621);scrub.cancel();
  await new Promise(resolve=>setTimeout(resolve,35));
  assert.deepEqual(calls,[99]);
});
