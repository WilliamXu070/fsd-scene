// Isolated actual-browser UI regression. No model inference or annotation reads.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) => index % 2 === 0 ? [...pairs, [value.slice(2), all[index+1]]] : pairs, []));
const {chromium} = require(args['playwright-core']);
const output = path.resolve(args.output || 'artifacts/viewer-qa/request-regression');
const base = args.url || 'http://127.0.0.1:8768';
fs.mkdirSync(output, {recursive:true});
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:args.browser,
    args:['--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  const context = await browser.newContext({viewport:{width:1440,height:1000}});
  const page = await context.newPage();
  const requests=[],errors=[],consoleErrors=[],failed=[];let loaderStatus=null;
  page.on('request', request => {if(request.url().includes('/api/frame?'))requests.push(request.url());});
  page.on('response', response => {if(response.url().endsWith('/replay-requests.js'))loaderStatus=response.status();});
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {if(message.type()==='error')consoleErrors.push(message.text());});
  page.on('requestfailed', request => failed.push({url:request.url(),reason:request.failure()?.errorText}));
  const setScrub = async index => page.locator('#scrubber').evaluate((element,value)=>{element.value=String(value);element.dispatchEvent(new Event('input',{bubbles:true}));},index);
  const waitFrame = async index => {
    await page.waitForFunction(index=>document.querySelector('#frame-counter').textContent===`Frame ${index+1} of 16`,index);
    await page.waitForFunction(index=>[...document.querySelectorAll('.camera-view img')].every(img=>img.complete&&img.naturalWidth===704&&img.naturalHeight===256&&new URL(img.src).searchParams.get('index')===String(index)),index);
  };
  try {
    await page.goto(base,{waitUntil:'networkidle'});await waitFrame(0);
    assert.equal(loaderStatus,200);
    const initial=await page.evaluate(()=>({
      note:document.querySelector('#dataset-note').textContent,
      noteVisible:!document.querySelector('#dataset-note').hidden,
      noteFits:document.querySelector('#dataset-note').scrollHeight<=document.querySelector('#dataset-note').clientHeight,
      frame:document.querySelector('#frame-counter').textContent,
      images:[...document.querySelectorAll('.camera-view img')].map(img=>({src:img.src,width:img.naturalWidth,height:img.naturalHeight})),
      placeholders:[...document.querySelectorAll('.camera-placeholder')].map(el=>getComputedStyle(el).display),
      renderer:(()=>{const gl=document.querySelector('#viewport canvas').getContext('webgl2');const ext=gl.getExtension('WEBGL_debug_renderer_info');return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):'unavailable';})()
    }));
    assert.match(initial.note,/16 training frames only; no held-out accuracy claim/);
    assert.equal(initial.noteVisible,true);assert.equal(initial.noteFits,true);
    assert.ok(initial.placeholders.every(value=>value==='none'));
    await page.screenshot({path:path.join(output,'01-updated-training-fit.png'),fullPage:true});
    const requestStart=requests.length;
    await page.locator('#scrubber').evaluate(element=>{
      for(let i=0;i<100;i++){element.value=String(i===99?11:(i*7)%16);element.dispatchEvent(new Event('input',{bubbles:true}));}
    });
    await waitFrame(11);await page.waitForTimeout(250);
    const burstRequests=requests.slice(requestStart);
    assert.equal(burstRequests.length,1);
    assert.ok(burstRequests[0].endsWith('index=11'));
    // Delay one real response to exercise cancellation without changing its contents.
    await page.route('**/api/frame?index=4',async route=>{
      const response=await route.fetch();await page.waitForTimeout(350);
      try {await route.fulfill({response});} catch(error) {if(!String(error).match(/closed|cancel|already|Invalid|not found/i))throw error;}
    });
    await setScrub(4);
    await page.waitForRequest(request=>request.url().endsWith('/api/frame?index=4'));
    await setScrub(12);await waitFrame(12);await page.waitForTimeout(500);
    assert.equal(await page.locator('#frame-counter').textContent(),'Frame 13 of 16');
    const staleAfter=await page.locator('#scrubber').inputValue();assert.equal(staleAfter,'12');
    await page.unroute('**/api/frame?index=4');
    await page.screenshot({path:path.join(output,'02-latest-scrub-result.png'),fullPage:true});
    await setScrub(0);await waitFrame(0);
    await page.locator('#play').click();await page.waitForTimeout(1250);
    await page.locator('#play').click();
    const pausedIndex=Number(await page.locator('#scrubber').inputValue());
    assert.ok(pausedIndex>=2&&pausedIndex<=5);
    await waitFrame(pausedIndex);await page.waitForTimeout(700);
    assert.equal(Number(await page.locator('#scrubber').inputValue()),pausedIndex);
    await page.locator('#next').click();await waitFrame(pausedIndex+1);
    await page.locator('#previous').click();await waitFrame(pausedIndex);
    const checks=[];
    for(let index=0;index<16;index++) {await setScrub(index);await waitFrame(index);checks.push({index,images:4,decoded:true});}
    await setScrub(0);await waitFrame(0);
    await page.screenshot({path:path.join(output,'03-final-regression.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    assert.equal(await page.locator('#error').isVisible(),false);
    const report={scope:'UI regression only: existing 16-frame prediction-only training-fit export. No held-out accuracy or perception-speed claim.',
      url:base,browser:await browser.version(),rendering:'Isolated headless Chromium with hardware GPU disabled; SwiftShader software WebGL.',
      browser_connection_fallback:'Standard browser initialization failed with Windows ACL helper error; separate browser MCP profile was already in use. Existing sessions were left untouched.',
      initial,loader_status:loaderStatus,rapid_scrub:{events:100,requests:burstRequests.length,final_index:11},
      stale_response:{delayed_actual_index:4,delay_ms:350,latest_index:12,stable_after_old_response:true},
      playback:{play_ms:1250,advanced_to:pausedIndex,pause_held_ms:700,next_previous_passed:true},
      all_camera_frames:checks,page_errors:errors,console_errors:consoleErrors,failed_requests:failed,
      result:'passed',screenshots:['01-updated-training-fit.png','02-latest-scrub-result.png','03-final-regression.png'],
      limitations:['Recorded training-fit predictions only; UI QA does not assess generalization, object quality, temporal inference, or model FPS.',
      'Headless software WebGL is an actual browser rendering path, but does not establish desktop GPU rendering cadence.']};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
  } finally {await context.close();await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
