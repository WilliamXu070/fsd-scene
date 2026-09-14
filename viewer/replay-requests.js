// Keep rapid scrubbing from decoding or rendering superseded scene responses.
export class LatestFrameRequest {
  constructor(fetcher = (...args) => fetch(...args), observe = null, now = () => performance.now()) { this.fetcher = fetcher; this.observe = observe; this.now = now; this.serial = 0; this.controller = null; }
  async load(url) {
    this.controller?.abort();
    const controller = new AbortController(), serial = ++this.serial;
    this.controller = controller;
    const started = this.now(); let headers = null, decoded = null, outcome = 'superseded';
    try {
      const response = await this.fetcher(url, {signal: controller.signal});
      headers = this.now();
      if (serial !== this.serial) return null;
      if (!response.ok) throw new Error('This prediction frame is not ready.');
      const record = await response.json(); decoded = this.now();
      outcome = serial === this.serial ? 'accepted' : 'superseded';
      return serial === this.serial ? record : null;
    } catch (error) {
      if (serial !== this.serial || error.name === 'AbortError') return null;
      outcome = 'failed'; throw error;
    } finally {
      if (serial === this.serial) this.controller = null;
      this.observe?.({outcome,request_headers_ms:headers===null?null:headers-started,
        response_body_json_ms:decoded===null?null:decoded-headers,request_to_decoded_ms:decoded===null?null:decoded-started});
    }
  }
}

export function debouncedScrub(callback, delay = 70) {
  let timer = null;
  const schedule = value => { clearTimeout(timer); timer = setTimeout(() => { timer = null; callback(value); }, delay); };
  schedule.cancel = () => { clearTimeout(timer); timer = null; };
  return schedule;
}
