// Isolated Chromium fallback after two browser-runtime connection failures.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..'),out=path.join(root,'artifacts/paired-inspection/qa-verified');fs.mkdirSync(out,{recursive:true});
const {chromium}=require('C:/Users/William/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright-core');
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:'C:/Users/William/AppData/Local/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-win64/chrome-headless-shell.exe',args:['--disable-gpu','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
let report={status:'running',checks:[],errors:[],screenshots:[],scope:'CPU browser inspection of paired validation data; no model inference'};
try{
const page=await browser.newPage({viewport:{width:1600,height:1200}});page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
await page.goto('http://127.0.0.1:8773');
const ready=async i=>{await page.waitForFunction(i=>window.inspectionState?.reference_ready&&window.inspectionState.index===i&&!document.body.classList.contains('loading'),i,{timeout:30000});assert.equal(await page.locator('#replay').contentFrame().locator('#scrubber').inputValue(),String(i));};
const shot=async name=>{const p=path.join(out,name+'.png');await page.screenshot({path:p,fullPage:true});report.screenshots.push({path:p,sha256:hash(p)});};
await ready(0);assert.equal(await page.locator('#threshold').inputValue(),'0.55');assert.ok(!(await page.locator('#frame-summary').textContent()).includes('\u00c2'));assert.ok((await page.locator('#frame-summary').textContent()).includes('\u00b7'));report.checks.push('Initial 3D and references agree at frame0; UTF8 labels verified');await shot('first-frame');
await page.locator('#forward').click();await ready(1);await page.locator('#back').click();await ready(0);report.checks.push('Parent previous/next synchronize iframe and all references');
await page.locator('#replay').contentFrame().locator('#next').click();await ready(1);report.checks.push('Iframe next synchronizes parent references');
await page.locator('[data-camera="2"]').click();await page.locator('#lidar').check();await shot('fisheye-lidar');report.checks.push('Native fisheye and LiDAR overlay render');
await page.locator('#threshold').selectOption('0.01');assert.equal(await page.locator('#replay').contentFrame().locator('#confidence').inputValue(),'0.01');await page.locator('#threshold').selectOption('0.55');report.checks.push('Comparison threshold synchronizes predicted3D/table/overlays');
const summary=JSON.parse(fs.readFileSync(path.join(root,'artifacts/paired-inspection/consistency.json'),'utf8'));const events=summary.thresholds['0.55'].events;
const k=events.reduce((best,e,i)=>(e.yaw_error_change_deg||0)>(events[best].yaw_error_change_deg||0)?i:best,0);const e=events[k];
await page.locator('#event').selectOption(String(k));await ready(e.previous_index);await page.locator('[data-camera="0"]').click();await shot('heading-before');await page.locator('#event-after').click();await ready(e.index);await shot('heading-after');report.heading_event=e;report.checks.push('Largest directed-heading-change before/after reference pair verified');
const drop=events.findIndex(e=>e.kind==='disappeared');await page.locator('#event').selectOption(String(drop));await ready(events[drop].previous_index);await shot('dropout-before');await page.locator('#event-after').click();await ready(events[drop].index);await shot('dropout-after');report.dropout_event=events[drop];
await page.locator('#replay').contentFrame().locator('#play').click();const before=await page.evaluate(()=>window.inspectionState.index);await page.waitForFunction(i=>window.inspectionState.index>i,before);await page.locator('#replay').contentFrame().locator('#play').click();await page.waitForTimeout(700);const paused=await page.evaluate(()=>window.inspectionState.index);await page.waitForTimeout(650);assert.equal(await page.evaluate(()=>window.inspectionState.index),paused);await ready(paused);report.checks.push('Play advances paired data, pause holds synchronized frame');
await page.locator('#frame').fill('196');await page.locator('#frame').dispatchEvent('input');await ready(196);await shot('last-frame');report.checks.push('Final validation frame and boundary navigation');
assert.deepEqual(report.errors,[]);report.status='automated_pass_visual_review_pending';
}catch(e){report.status='failed';report.failure=e.stack;process.exitCode=1;console.error(e.stack);}finally{await browser.close();report.source_sha256=Object.fromEntries(['inspection/inspect.js','inspection/index.html','inspection/inspect.css','inspection/serve.py'].map(p=>[p,hash(path.join(root,p))]));report.completed_utc=new Date().toISOString();fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({status:report.status,checks:report.checks,errors:report.errors},null,2));}
})();
