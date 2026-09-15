const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { depthAt, projectedPixel, featureCount, toyUpdate, TIMINGS, MODEL_TOTAL } = require('../website/animations.js');

test('every depth on the ray projects to the same pixel', () => {
  assert.equal(depthAt(0), 5);
  assert.equal(depthAt(1), 50);
  assert.equal(depthAt(-1), 5);
  assert.equal(depthAt(2), 50);
  for (let i = 0; i <= 1000; i++) assert.ok(Math.abs(projectedPixel(depthAt(i/1000)) - 432) < 1e-10);
});

test('feature storage counts retain their dimensional meaning', () => {
  assert.equal(featureCount('bev', 16), 1638400);
  assert.equal(featureCount('voxels', 16), 26214400);
  assert.equal(featureCount('sparse', 16), 25600);
  for (let layers = 4; layers <= 32; layers++) {
    assert.equal(featureCount('voxels', layers), layers * featureCount('bev', layers));
    assert.equal(featureCount('sparse', layers), 200 * 128);
  }
});

test('the toy gradient update actually reduces its displayed loss', () => {
  for (let iteration = 0; iteration < 8; iteration++) {
    const { weight, gradient, next, loss } = toyUpdate(iteration);
    assert.equal(gradient, 2 * (weight - 12));
    assert.equal(loss, (weight - 12) ** 2);
    assert.ok((next - 12) ** 2 < loss);
    assert.ok(Math.abs(next - toyUpdate(iteration + 1).weight) < 1e-10);
  }
});

test('latency stack includes FPN once and reconciles with the instrumented total', () => {
  const sum = TIMINGS.reduce((value, row) => value + row[1], 0);
  assert.equal(TIMINGS.length, 9);
  assert.equal(TIMINGS.filter(row => /FPN/.test(row[0])).length, 1);
  assert.ok(TIMINGS.every(row => Number.isFinite(row[1]) && row[1] > 0));
  assert.ok(sum < MODEL_TOTAL);
  assert.ok(MODEL_TOTAL - sum < .5);
  assert.equal(MODEL_TOTAL, 19.659810129801432);
});

test('all eight surfaces have stable unique hooks and locally served assets', () => {
  const root = path.resolve(__dirname, '../website');
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const hooks = [...html.matchAll(/data-animation="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(hooks, ['opening', 'ray', 'representation', 'architecture', 'training', 'evidence', 'queries', 'latency']);
  assert.equal(new Set(hooks).size, 8);
  assert.doesNotMatch(html, /class="figure-placeholder/);
  for (const match of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
    const target = match[1].split(/[?#]/)[0];
    if (!target || /^[a-z]+:/i.test(target)) continue;
    assert.ok(fs.existsSync(path.resolve(root, target)), target + ' must resolve locally');
  }
});

// Exercise the real controller code with a minimal DOM and deterministic clock.
// This does not replace actual browser layout/keyboard QA.
function controllerHarness(reduce = false) {
  class Element {
    constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.events = {}; this.textContent = ''; this._html = ''; }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    getAttribute(name) { return this.attrs[name]; }
    addEventListener(name, callback) { this.events[name] = callback; }
    append(...nodes) { this.children.push(...nodes); }
    insertBefore(node, before) { this.children.splice(this.children.indexOf(before), 0, node); }
    getBoundingClientRect() { return { width: 700 }; }
    querySelector(selector) { return selector === 'figcaption' ? this.caption : null; }
    set innerHTML(value) { this._html = value; if(this.tag === 'template') this.content = { textContent:value.replace(/<[^>]+>/g, '') }; }
    get innerHTML() { return this._html; }
    dispatch(name) { this.events[name]?.(); }
  }
  const figure = new Element('figure');
  figure.dataset = { animation:'ray' };
  figure.caption = new Element('figcaption');
  figure.children.push(figure.caption);
  const documentEvents = {}, mediaEvents = {}, observers = [], callbacks = new Map();
  let nextId = 0;
  const doc = {
    hidden:false,
    createElement: tag => new Element(tag),
    querySelectorAll: () => [figure],
    addEventListener: (name, cb) => { documentEvents[name] = cb; }
  };
  const media = { matches:reduce, addEventListener:(name, cb) => {mediaEvents[name] = cb;} };
  const context = {
    document:doc, window:{matchMedia:()=>media}, console,
    requestAnimationFrame:callback => { const id=++nextId;callbacks.set(id,callback);return id; },
    cancelAnimationFrame:id => callbacks.delete(id),
    IntersectionObserver:class { constructor(callback) { observers.push(callback); } observe() {} },
    ResizeObserver:class { observe() {} }
  };
  vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,'../website/animations.js'),'utf8'), context);
  observers[0]([{isIntersecting:true}]);
  const all = element => [element,...element.children.flatMap(all)];
  const elements = all(figure), play=elements.find(e=>e.attrs.class==='viz-play');
  const seek=elements.find(e=>e.attrs.id==='viz-ray-seek');
  return {
    play,seek, callbacks,doc,
    tick(time) { const pending=[...callbacks.values()];callbacks.clear();pending.forEach(fn=>fn(time)); },
    hide() { doc.hidden=true;documentEvents.visibilitychange(); },
    leave() { observers[0]([{isIntersecting:false}]); },
    reduce() { media.matches=true;mediaEvents.change(); },
    restart() { elements.find(e=>e.attrs['aria-label']==='Restart Pixel and depth ray').dispatch('click'); }
  };
}

test('play advances, pause freezes, restart resets and the loop finishes', () => {
  const h=controllerHarness();
  assert.equal(h.callbacks.size,0);
  h.play.dispatch('click');
  assert.equal(h.play.textContent,'Pause');
  h.tick(100);h.tick(140);
  assert.ok(Number(h.seek.value)>0);
  h.play.dispatch('click');
  const stopped=h.seek.value;
  h.tick(200);
  assert.equal(h.seek.value,stopped);
  assert.equal(h.callbacks.size,0);
  h.restart();
  assert.equal(h.seek.value,'0');
  h.play.dispatch('click');
  for(let time=250;time<=8500;time+=40) h.tick(time);
  assert.equal(h.seek.value,'1000');
  assert.equal(h.play.textContent,'Replay');
  assert.equal(h.callbacks.size,0);
  h.play.dispatch('click');
  h.tick(9000);h.tick(9040);
  assert.ok(Number(h.seek.value)<1000);
});

test('reduced motion uses still-frame steps and never schedules animation', () => {
  const h=controllerHarness(true);
  assert.equal(h.play.textContent,'Step');
  h.play.dispatch('click');
  assert.equal(h.seek.value,'125');
  assert.equal(h.callbacks.size,0);
  h.seek.value='1000';h.seek.dispatch('input');
  h.play.dispatch('click');
  assert.equal(h.seek.value,'0');
  const running=controllerHarness();
  running.play.dispatch('click');running.tick(100);running.tick(140);
  running.reduce();const stopped=running.seek.value;running.tick(200);
  assert.equal(running.seek.value,stopped);
  assert.equal(running.play.textContent,'Step');
});

test('offscreen and hidden figures pause instead of consuming frames', () => {
  const offscreen=controllerHarness();
  offscreen.play.dispatch('click');offscreen.tick(100);offscreen.tick(140);
  offscreen.leave();const value=offscreen.seek.value;offscreen.tick(180);
  assert.equal(offscreen.seek.value,value);
  assert.equal(offscreen.play.textContent,'Play');
  assert.equal(offscreen.callbacks.size,0);
  const hidden=controllerHarness();
  hidden.play.dispatch('click');hidden.hide();
  assert.equal(hidden.play.textContent,'Play');
  assert.equal(hidden.callbacks.size,0);
});
