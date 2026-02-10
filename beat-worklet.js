class BeatDetectorProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.channelId = (options && options.processorOptions && options.processorOptions.channelId) || 'A';
    this.sampleRate_ = sampleRate;
    // STFT params
    this.fftSize = 1024;
    this.hopSize = 512;
    this.win = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) this.win[i] = 0.5 * (1 - Math.cos((2*Math.PI*i)/(this.fftSize-1)));
    this.buf = new Float32Array(this.fftSize);
    this.bufFill = 0;
    this.prevSpec = new Float32Array(this.fftSize/2);
    // Flux history for threshold
    this.fluxHist = [];
    this.fluxHistLen = 120;
    this.kMad = 3.0;
    // Onset envelope for tempo comb
    this.env = [];
    this.envMaxLen = 8 * 60; // ~8s at 60 Hz equivalent
    this.lastOnsetTime = 0;
    this.minOnsetGapMs = 120; // refractory
    // Tempo smoothing
    this.bpm = null;
    this.bpmAlpha = 0.2;
    this.nextBeatTime = null;
    // PLL params
    this.pllPhase = 0;
    this.pllLastUpdate = 0;
    this.pllKp = 0.12; // small proportional correction
    this.pllKd = 0.02; // small damping
    this.prevErr = 0;
    // Running time
    this.totalSamples = 0;
    // FFT scratch
    this.bitrev = this._bitReverseTable(this.fftSize);
    this.cosTable = new Float32Array(this.fftSize/2);
    this.sinTable = new Float32Array(this.fftSize/2);
    for (let i = 0; i < this.fftSize/2; i++) {
      const ang = -2 * Math.PI * i / this.fftSize;
      this.cosTable[i] = Math.cos(ang);
      this.sinTable[i] = Math.sin(ang);
    }
  }
  static get parameterDescriptors() { return []; }
  _bitReverseTable(n) {
    const bits = Math.log2(n) | 0;
    const rev = new Uint32Array(n);
    for (let i=0;i<n;i++){
      let x=i, r=0;
      for (let j=0;j<bits;j++){ r=(r<<1)|(x&1); x>>=1; }
      rev[i]=r;
    }
    return rev;
  }
  _fftReIm(re, im) {
    const n = re.length;
    // bit-reverse copy
    for (let i=0;i<n;i++){
      const j=this.bitrev[i];
      if (j>i){ const tr=re[i]; re[i]=re[j]; re[j]=tr; const ti=im[i]; im[i]=im[j]; im[j]=ti; }
    }
    for (let size=2; size<=n; size<<=1){
      const half=size>>1, step=n/size|0;
      for (let i=0;i<n;i+=size){
        for (let j=0;j<half;j++){
          const k=j*step|0;
          const c=this.cosTable[k], s=this.sinTable[k];
          const tre=c*re[i+j+half]-s*im[i+j+half];
          const tim=s*re[i+j+half]+c*im[i+j+half];
          re[i+j+half]=re[i+j]-tre;
          im[i+j+half]=im[i+j]-tim;
          re[i+j]+=tre;
          im[i+j]+=tim;
        }
      }
    }
  }
  _median(arr) {
    if (arr.length===0) return 0;
    const a = Array.from(arr).sort((x,y)=>x-y);
    const m = a.length>>1;
    return a.length%2 ? a[m] : 0.5*(a[m-1]+a[m]);
  }
  _mad(arr, med) {
    if (arr.length===0) return 0;
    const dev = arr.map(v => Math.abs(v - med)).sort((x,y)=>x-y);
    const m = dev.length>>1;
    const mad = dev.length%2 ? dev[m] : 0.5*(dev[m-1]+dev[m]);
    // scale factor ~1.4826 to estimate std from MAD
    return 1.4826 * mad;
  }
  _analyzeFrame(frame) {
    // window
    const N = this.fftSize;
    const re = new Float32Array(N), im = new Float32Array(N);
    for (let i=0;i<N;i++){ re[i] = (frame[i] || 0) * this.win[i]; im[i]=0; }
    this._fftReIm(re, im);
    const bins = N/2;
    const mag = new Float32Array(bins);
    for (let i=0;i<bins;i++){
      const a=re[i], b=im[i];
      mag[i] = Math.log1p(Math.hypot(a,b)); // log-mag
    }
    // 3-band positive log-flux
    const sr = this.sampleRate_;
    const binHz = sr / N;
    const lowMax = 180, midMax = 1500;
    const wL=0.7, wM=0.25, wH=0.05;
    let fluxL=0,fluxM=0,fluxH=0, nL=0,nM=0,nH=0;
    for (let i=0;i<bins;i++){
      const d = mag[i] - this.prevSpec[i];
      if (d<=0) continue;
      const f = i * binHz;
      if (f<lowMax){ fluxL+=d; nL++; }
      else if (f<midMax){ fluxM+=d; nM++; }
      else { fluxH+=d; nH++; }
    }
    const norm = (s,n)=> n>0 ? s/n : 0;
    const flux = (wL*norm(fluxL,nL) + wM*norm(fluxM,nM) + wH*norm(fluxH,nH)) / (wL+wM+wH);
    this.prevSpec = mag;
    // Adaptive threshold with median/MAD
    this.fluxHist.push(flux);
    if (this.fluxHist.length > this.fluxHistLen) this.fluxHist.shift();
    const med = this._median(this.fluxHist);
    const mad = this._mad(this.fluxHist, med) || 1e-6;
    const thr = med + this.kMad * mad;
    const tNow = this.totalSamples / this.sampleRate_;
    let onset = false;
    if (flux > thr && (tNow - this.lastOnsetTime) * 1000 > this.minOnsetGapMs) {
      onset = true;
      this.lastOnsetTime = tNow;
      this.port.postMessage({ type:'onset', t: tNow, ch: this.channelId });
    }
    // Envelope for comb
    const envVal = Math.max(0, flux - med);
    this.env.push(envVal);
    if (this.env.length > this.envMaxLen) this.env.shift();
    // Tempo estimate every ~0.25s
    if (!this._lastTempoUpdate || (tNow - this._lastTempoUpdate) > 0.25) {
      this._lastTempoUpdate = tNow;
      const bpm = this._estimateTempoComb();
      if (bpm) {
        if (this.bpm == null) this.bpm = bpm;
        else this.bpm = (1 - this.bpmAlpha) * this.bpm + this.bpmAlpha * bpm;
        // PLL update
        const period = 60 / this.bpm;
        // If we saw an onset recently, phase-correct
        if (onset || (tNow - this.lastOnsetTime) < 0.05) {
          const err = this._phaseError(tNow, period);
          const derr = err - this.prevErr;
          this.prevErr = err;
          const adjust = this.pllKp * err + this.pllKd * derr;
          // Advance next beat to reduce error
          if (!this.nextBeatTime || (this.nextBeatTime - tNow) > 2*period) {
            this.nextBeatTime = this.lastOnsetTime + period;
          } else {
            this.nextBeatTime += adjust;
          }
        } else {
          if (!this.nextBeatTime) this.nextBeatTime = tNow + period;
        }
        this.port.postMessage({ type:'tempo', bpm: this.bpm, period, next: this.nextBeatTime, ch: this.channelId });
      }
    }
  }
  _phaseError(t, period) {
    if (!this.nextBeatTime) return 0;
    const dt = this.nextBeatTime - t;
    // Map to [-period/2, period/2]
    let err = ((dt + period/2) % period) - period/2;
    if (!isFinite(err)) err = 0;
    return err;
  }
  _estimateTempoComb() {
    const env = this.env;
    if (!env || env.length < 90) return null;
    // normalize
    const mean = env.reduce((a,b)=>a+b,0)/env.length;
    const std = Math.sqrt(Math.max(1e-6, env.reduce((a,b)=>a+(b-mean)*(b-mean),0)/env.length));
    const e = env.map(v => (v-mean)/std);
    const minBpm=60, maxBpm=180, fps=60;
    let best=null, bestScore=-Infinity, second=-Infinity;
    for (let bpm=minBpm;bpm<=maxBpm;bpm++){
      const period = fps * 60 / bpm;
      let score=0;
      const beats = Math.max(3, Math.min(8, Math.floor(e.length / period)));
      for (let k=0;k<beats;k++){
        const idx = Math.round(e.length - 1 - k * period);
        const idxH = Math.round(e.length - 1 - (k+0.5) * period);
        if (idx>=0) score += e[idx];
        if (idxH>=0) score += 0.5*e[idxH];
      }
      // penalize obvious sub/over-tempo (harmonic safety)
      if (best && Math.abs(bpm*2 - best) <= 1) score *= 0.85;
      if (best && Math.abs(bpm - 2*best) <= 2) score *= 0.85;
      if (score > bestScore) { second = bestScore; bestScore = score; best = bpm; }
      else if (score > second) { second = score; }
    }
    const conf = bestScore > 0 ? (bestScore - second) / (bestScore + 1e-6) : 0;
    // require some confidence
    if (conf < 0.08) return null;
    return best;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) { this.totalSamples += 128; return true; }
    const ch0 = input[0];
    const ch1 = input[1];
    for (let i=0;i<ch0.length;i++){
      const s = ( (ch0[i]||0) + (ch1 ? (ch1[i]||0) : 0) ) * 0.5;
      this.buf[this.bufFill++] = s;
      this.totalSamples++;
      if (this.bufFill === this.fftSize) {
        this._analyzeFrame(this.buf);
        // shift by hop
        this.buf.copyWithin(0, this.hopSize, this.fftSize);
        this.bufFill = this.fftSize - this.hopSize;
      }
    }
    return true;
  }
}

registerProcessor('beat-detector', BeatDetectorProcessor);

