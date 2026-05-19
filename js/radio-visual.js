/* Extracted from app.js — radio-visual. Uses globals via globalThis (see app.js exposeVsGlobals). */
        class RadioVisualEngine {
            constructor() {
                this.name = 'Radio';
                this.resizeHandler = this.onResize.bind(this);
                this.abortCtrl = null;
                this.root = null;
                this.els = {};
                this.animId = null;
                this.clockTimerId = null;
                this.skin = 'analogue';
                this._lastStationIdx = -1;
                this._vuBuf = null;
                this._tuningDrag = false;
                this._volDrag = false;
                this._rvAutoFadeRaf = null;
            }

            static get AUTOFADE_MS_KEY() { return 'dj.autofade.duration.ms.v1'; }
            static get AUTOFADE_MIN_MS() { return 2000; }
            static get AUTOFADE_MAX_MS() { return 15000; }
            static get AUTOMIX_MAX_KEY() { return 'dj.automix.max.min.v1'; }
            static get AUTOMIX_MIN_MIN() { return 1; }
            static get AUTOMIX_MAX_MIN() { return 20; }

            _stopClick(ev) {
                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                try { window.__suppressNextClick = true; } catch (_) {}
            }

            _stopInteraction(ev) {
                this._stopClick(ev);
            }

            _loadVisualByName(name) {
                try {
                    if (!Array.isArray(modes) || typeof loadMode !== 'function') return;
                    const idx = modes.findIndex((m) => m && m.name === name);
                    if (idx >= 0) loadMode(idx);
                } catch (_) {}
            }

            _withDjDeck(fn) {
                try {
                    const dj = state.activeVisualizer;
                    if (dj && dj.name === 'DJ Decks' && typeof fn === 'function') {
                        fn(dj);
                        return;
                    }
                    const idx = modes.findIndex((m) => m && m.name === 'DJ Decks');
                    if (idx < 0 || typeof loadMode !== 'function') return;
                    loadMode(idx);
                    requestAnimationFrame(() => {
                        const d = state.activeVisualizer;
                        if (d && d.name === 'DJ Decks' && typeof fn === 'function') fn(d);
                    });
                } catch (_) {}
            }

            _stationPrev() {
                if (!Array.isArray(stations) || stations.length === 0) return;
                const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                    ? currentStationIndex : 0;
                if (typeof setStation === 'function') {
                    setStation((idx - 1 + stations.length) % stations.length);
                }
            }

            _stationNext() {
                if (!Array.isArray(stations) || stations.length === 0) return;
                const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                    ? currentStationIndex : 0;
                if (typeof setStation === 'function') {
                    setStation((idx + 1) % stations.length);
                }
            }

            _stationRand() {
                try {
                    if (typeof pickRandomStation === 'function') pickRandomStation();
                } catch (_) {}
            }

            _stationBPrev() {
                if (!Array.isArray(stations) || stations.length === 0) return;
                const idx = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                    ? currentStationBIndex : 0;
                this._setStationB((idx - 1 + stations.length) % stations.length);
            }

            _stationBNext() {
                if (!Array.isArray(stations) || stations.length === 0) return;
                const idx = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                    ? currentStationBIndex : 0;
                this._setStationB((idx + 1) % stations.length);
            }

            _stationBRand() {
                try {
                    if (typeof pickRandomStationB === 'function') pickRandomStationB();
                } catch (_) {}
            }

            _setStationB(index) {
                if (!Array.isArray(stations) || !stations.length) return;
                const idx = Math.max(0, Math.min(stations.length - 1, Number(index) || 0));
                try { currentStationBIndex = idx; } catch (_) {}
                try {
                    if (typeof refreshMixStationB === 'function') refreshMixStationB();
                } catch (_) {}
                try {
                    if (typeof playRadioB === 'function') playRadioB();
                } catch (_) {}
                this._updateStationUi();
            }

            _stationIndexToNorm(idx) {
                if (!Array.isArray(stations) || stations.length <= 1) return 0.5;
                const i = Math.max(0, Math.min(stations.length - 1, Number(idx) || 0));
                return i / Math.max(1, stations.length - 1);
            }

            _normToStationIndex(t) {
                if (!Array.isArray(stations) || !stations.length) return 0;
                return Math.round(Math.max(0, Math.min(1, t)) * (stations.length - 1));
            }

            _getCrossfadeX() {
                const dc = document.getElementById('dj-crossfader');
                const mc = document.getElementById('mix-crossfader');
                return Math.max(0, Math.min(1, Number((dc && dc.value) || (mc && mc.value) || 0)));
            }

            _setCrossfadeX(x) {
                const v = Math.max(0, Math.min(1, Number(x) || 0));
                try {
                    if (typeof applyCrossfade === 'function') applyCrossfade(v);
                } catch (_) {}
                if (this.els.crossDigital) this.els.crossDigital.value = String(v);
                this._syncCrossfadeKnob();
            }

            _syncCrossfadeKnob() {
                const x = this._getCrossfadeX();
                if (this.els.crossDigital) this.els.crossDigital.value = String(x);
            }

            _readAutoFadeDurationMs() {
                let ms = 5000;
                try {
                    const stored = Number(localStorage.getItem(RadioVisualEngine.AUTOFADE_MS_KEY));
                    if (Number.isFinite(stored)) ms = stored;
                } catch (_) {}
                return Math.max(
                    RadioVisualEngine.AUTOFADE_MIN_MS,
                    Math.min(RadioVisualEngine.AUTOFADE_MAX_MS, ms)
                );
            }

            _writeAutoFadeDurationMs(ms) {
                const clamped = Math.max(
                    RadioVisualEngine.AUTOFADE_MIN_MS,
                    Math.min(RadioVisualEngine.AUTOFADE_MAX_MS, Number(ms) || 5000)
                );
                try { localStorage.setItem(RadioVisualEngine.AUTOFADE_MS_KEY, String(clamped)); } catch (_) {}
                if (this.els.autoFadeReadout) {
                    this.els.autoFadeReadout.textContent = `${(clamped / 1000).toFixed(1)}s`;
                }
                return clamped;
            }

            _autoFadeDurationNorm() {
                const ms = this._readAutoFadeDurationMs();
                const span = RadioVisualEngine.AUTOFADE_MAX_MS - RadioVisualEngine.AUTOFADE_MIN_MS;
                return span > 0 ? (ms - RadioVisualEngine.AUTOFADE_MIN_MS) / span : 0.5;
            }

            _setAutoFadeDurationNorm(t) {
                const span = RadioVisualEngine.AUTOFADE_MAX_MS - RadioVisualEngine.AUTOFADE_MIN_MS;
                const ms = RadioVisualEngine.AUTOFADE_MIN_MS + (Math.max(0, Math.min(1, t)) * span);
                this._writeAutoFadeDurationMs(ms);
                this._setKnobRotation(this.els.autoFadeKnob, (this._autoFadeDurationNorm() * 270) - 135);
            }

            _readAutoMixMaxMin() {
                let mins = 20;
                try {
                    const stored = Number(localStorage.getItem(RadioVisualEngine.AUTOMIX_MAX_KEY));
                    if (Number.isFinite(stored)) mins = stored;
                } catch (_) {}
                return Math.max(
                    RadioVisualEngine.AUTOMIX_MIN_MIN,
                    Math.min(RadioVisualEngine.AUTOMIX_MAX_MIN, Math.round(mins))
                );
            }

            _writeAutoMixMaxMin(mins) {
                const clamped = Math.max(
                    RadioVisualEngine.AUTOMIX_MIN_MIN,
                    Math.min(RadioVisualEngine.AUTOMIX_MAX_MIN, Math.round(Number(mins) || 20))
                );
                try { localStorage.setItem(RadioVisualEngine.AUTOMIX_MAX_KEY, String(clamped)); } catch (_) {}
                if (this.els.autoMixReadout) {
                    this.els.autoMixReadout.textContent = `${clamped}m max`;
                }
                return clamped;
            }

            _autoMixMaxNorm() {
                const mins = this._readAutoMixMaxMin();
                const span = RadioVisualEngine.AUTOMIX_MAX_MIN - RadioVisualEngine.AUTOMIX_MIN_MIN;
                return span > 0 ? (mins - RadioVisualEngine.AUTOMIX_MIN_MIN) / span : 1;
            }

            _setAutoMixMaxNorm(t) {
                const span = RadioVisualEngine.AUTOMIX_MAX_MIN - RadioVisualEngine.AUTOMIX_MIN_MIN;
                const mins = RadioVisualEngine.AUTOMIX_MIN_MIN + (Math.max(0, Math.min(1, t)) * span);
                this._writeAutoMixMaxMin(mins);
                this._setKnobRotation(this.els.autoMixKnob, (this._autoMixMaxNorm() * 270) - 135);
            }

            _resumeDecksForCrossfade() {
                try {
                    const x = this._getCrossfadeX();
                    const ga = 1 - x;
                    const gb = x;
                    const thresh = 0.03;
                    if (ga > thresh && typeof audioEl !== 'undefined' && audioEl && audioEl.src && audioEl.paused) {
                        audioEl.play().catch(() => {});
                    }
                    if (gb > thresh && typeof audioElB !== 'undefined' && audioElB && audioElB.src && audioElB.paused) {
                        audioElB.play().catch(() => {});
                    }
                } catch (_) {}
            }

            _runLocalAutoFade() {
                if (this._rvAutoFadeRaf) {
                    try { cancelAnimationFrame(this._rvAutoFadeRaf); } catch (_) {}
                    this._rvAutoFadeRaf = null;
                }
                const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                    ? state.activeVisualizer : null;
                if (eng && typeof eng.triggerAutoFadeFromShortcut === 'function') {
                    try { eng.triggerAutoFadeFromShortcut(); return; } catch (_) {}
                }
                const x = this._getCrossfadeX();
                const targetDeck = x < 0.5 ? 'b' : 'a';
                const endVal = targetDeck === 'b' ? 1 : 0;
                const startVal = x;
                if (Math.abs(endVal - startVal) < 0.001) return;
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                if (targetDeck === 'b') {
                    try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                } else {
                    try {
                        const m = (typeof getDeckAMediaForPlaybackState === 'function')
                            ? getDeckAMediaForPlaybackState()
                            : audioEl;
                        if (!m || !m.src) {
                            if (typeof playRadio === 'function') playRadio();
                        } else if (m.paused) {
                            m.play().catch(() => {});
                        }
                    } catch (_) {
                        try { if (typeof playRadio === 'function') playRadio(); } catch (_) {}
                    }
                }
                const durMs = this._readAutoFadeDurationMs();
                const startTs = performance.now();
                const tick = (ts) => {
                    const t = Math.max(0, Math.min(1, (ts - startTs) / durMs));
                    const eased = 1 - Math.pow(1 - t, 3);
                    this._setCrossfadeX(startVal + ((endVal - startVal) * eased));
                    this._resumeDecksForCrossfade();
                    if (t >= 1) {
                        this._rvAutoFadeRaf = null;
                        return;
                    }
                    this._rvAutoFadeRaf = requestAnimationFrame(tick);
                };
                this._rvAutoFadeRaf = requestAnimationFrame(tick);
            }

            _triggerAutoFade() {
                const btn = document.getElementById('mix-autofade') || document.getElementById('dj-autofade');
                if (btn) {
                    try { btn.click(); return; } catch (_) {}
                }
                this._runLocalAutoFade();
            }

            _toggleAutoMix() {
                const btn = document.getElementById('mix-automix') || document.getElementById('dj-automix');
                if (btn) {
                    try { btn.click(); return; } catch (_) {}
                }
                try {
                    const key = 'dj.automix.enabled.v1';
                    const on = localStorage.getItem(key) === '1';
                    const next = !on;
                    localStorage.setItem(key, next ? '1' : '0');
                    try { state.autoMixEnabled = next; } catch (_) {}
                } catch (_) {}
            }

            _deckBActive() {
                try {
                    return !!(typeof audioElB !== 'undefined' && audioElB && audioElB.src &&
                        audioElB.src !== 'about:blank' && !audioElB.paused && !audioElB.ended);
                } catch (_) {
                    return false;
                }
            }

            _deckAActive() {
                try {
                    const media = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    return !!(media && media.src && media.src !== 'about:blank' && !media.paused && !media.ended);
                } catch (_) {
                    return false;
                }
            }

            _deckHasSource(media) {
                try {
                    if (typeof deckHasSource === 'function') return deckHasSource(media);
                } catch (_) {}
                return !!(media && media.src && media.src !== 'about:blank');
            }

            async _toggleDeckA() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    try {
                        if (eng && typeof eng.cancelAutoFade === 'function') eng.cancelAutoFade();
                    } catch (_) {}
                    const media = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    if (!media || !this._deckHasSource(media)) {
                        try {
                            if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                                eng.clearSuppressEnsureCrossfadeDeckPlayback();
                            }
                        } catch (_) {}
                        if (typeof playRadio === 'function') playRadio();
                        return;
                    }
                    if (media.paused) {
                        try {
                            if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                                eng.clearSuppressEnsureCrossfadeDeckPlayback();
                            }
                        } catch (_) {}
                        await media.play().catch(() => {
                            try { if (typeof playRadio === 'function') playRadio(); } catch (_) {}
                        });
                    } else {
                        try {
                            if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                                eng.clearSuppressEnsureCrossfadeDeckPlayback();
                            }
                        } catch (_) {}
                        media.pause();
                    }
                } catch (_) {}
            }

            async _toggleDeckB() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    try {
                        if (eng && typeof eng.cancelAutoFade === 'function') eng.cancelAutoFade();
                    } catch (_) {}
                    if (!audioElB || !this._deckHasSource(audioElB) || audioElB.paused) {
                        try {
                            if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                                eng.clearSuppressEnsureCrossfadeDeckPlayback();
                            }
                        } catch (_) {}
                        if (typeof playRadioB === 'function') playRadioB();
                    } else {
                        audioElB.pause();
                    }
                } catch (_) {}
            }

            _syncDeckSwitches() {
                const aOn = this._deckAActive();
                const bOn = this._deckBActive();
                if (this.els.deckAKnob) {
                    this.els.deckAKnob.classList.toggle('is-on', aOn);
                    this.els.deckAKnob.setAttribute('aria-pressed', aOn ? 'true' : 'false');
                }
                if (this.els.deckBKnob) {
                    this.els.deckBKnob.classList.toggle('is-on', bOn);
                    this.els.deckBKnob.setAttribute('aria-pressed', bOn ? 'true' : 'false');
                }
            }

            _wireDeckSwitch(knobEl, deck) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch');
                knobEl.setAttribute('role', 'switch');
                knobEl.tabIndex = 0;
                const run = async () => {
                    if (deck === 'a') await this._toggleDeckA();
                    else await this._toggleDeckB();
                    this._syncDeckSwitches();
                    try { resetIdleTimer(); } catch (_) {}
                };
                knobEl.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    run();
                }, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('keydown', (ev) => {
                    if (ev.key !== 'Enter' && ev.key !== ' ') return;
                    ev.preventDefault();
                    this._stopClick(ev);
                    run();
                }, { signal: this.abortCtrl.signal });
            }

            _wireClickKnob(knobEl, onClick) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch');
                knobEl.setAttribute('role', 'button');
                knobEl.tabIndex = 0;
                const run = () => {
                    try { onClick(); } catch (_) {}
                    try { resetIdleTimer(); } catch (_) {}
                };
                knobEl.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    run();
                }, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('keydown', (ev) => {
                    if (ev.key !== 'Enter' && ev.key !== ' ') return;
                    ev.preventDefault();
                    this._stopClick(ev);
                    run();
                }, { signal: this.abortCtrl.signal });
            }

            _syncVolumeFromGlobal() {
                const vs = document.getElementById('volume-slider');
                const v = vs ? Number(vs.value) : 0.5;
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                if (this.els.volDigital) this.els.volDigital.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
            }

            _applyVolume(val) {
                const v = Math.max(0, Math.min(1, Number(val) || 0));
                try {
                    if (typeof setVolume === 'function') setVolume(v);
                } catch (_) {}
                const vs = document.getElementById('volume-slider');
                if (vs) vs.value = String(v);
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                if (this.els.volDigital) this.els.volDigital.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
            }

            _setKnobRotation(knobEl, deg) {
                if (!knobEl) return;
                try { knobEl.style.setProperty('--radio-knob-deg', `${deg}deg`); } catch (_) {}
            }

            _needlePercentA() {
                return this._stationIndexToNorm(
                    (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                        ? currentStationIndex : 0
                ) * 100;
            }

            _needlePercentB() {
                return this._stationIndexToNorm(
                    (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                        ? currentStationBIndex : 0
                ) * 100;
            }

            _updateStationUi() {
                let nameA = '—';
                let nameB = '—';
                try {
                    const idxA = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                        ? currentStationIndex : -1;
                    if (idxA >= 0 && stations[idxA]) nameA = stations[idxA].name || nameA;
                    else if (stationBannerNameEl && stationBannerNameEl.textContent) {
                        nameA = stationBannerNameEl.textContent;
                    }
                    const idxB = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                        ? currentStationBIndex : -1;
                    if (idxB >= 0 && stations[idxB]) nameB = stations[idxB].name || nameB;
                } catch (_) {}
                if (this.els.stationNameA) this.els.stationNameA.textContent = nameA;
                if (this.els.stationNameB) this.els.stationNameB.textContent = nameB;
                if (this.els.stationDigital) {
                    this.els.stationDigital.textContent = `A: ${nameA}  |  B: ${nameB}`;
                }
                const pctA = this._needlePercentA();
                const pctB = this._needlePercentB();
                if (this.els.needle) this.els.needle.style.left = `${pctA}%`;
                if (this.els.tunerGlow) this.els.tunerGlow.style.left = `${pctA}%`;
                if (this.els.needleB) {
                    this.els.needleB.style.left = `${pctB}%`;
                    const bOn = this._deckBActive();
                    this.els.needleB.classList.toggle('is-active', bOn);
                    this.els.needleB.classList.toggle('is-idle', !bOn);
                }
                if (this.els.tunerGlowB) {
                    this.els.tunerGlowB.style.left = `${pctB}%`;
                    this.els.tunerGlowB.classList.toggle('is-active', this._deckBActive());
                }
                this._syncCrossfadeKnob();
                this._syncDeckSwitches();
                let meta = '';
                try {
                    if (typeof currentNowPlayingICY !== 'undefined' && currentNowPlayingICY) {
                        meta = String(currentNowPlayingICY);
                    } else if (stationBannerNowplayingEl && stationBannerNowplayingEl.textContent) {
                        meta = stationBannerNowplayingEl.textContent;
                    }
                } catch (_) {}
                if (this.els.digitalMeta) {
                    this.els.digitalMeta.textContent = meta ? meta : (state.isPlaying ? 'STREAMING' : 'STANDBY');
                }
                const live = !!(state.isPlaying || (typeof audioEl !== 'undefined' && audioEl && !audioEl.paused && audioEl.src));
                if (this.els.onAir) this.els.onAir.classList.toggle('is-live', live);
                if (this.els.digitalStatus) {
                    this.els.digitalStatus.textContent = live ? 'ON AIR · RECEIVING' : 'OFF AIR';
                }
                this._lastStationIdx = (typeof currentStationIndex === 'number') ? currentStationIndex : -1;
            }

            _tickClock() {
                if (!this.els.digitalClock) return;
                try {
                    const d = new Date();
                    this.els.digitalClock.textContent = d.toLocaleString(undefined, {
                        weekday: 'short',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                    });
                } catch (_) {}
            }

            _setSkin(skin) {
                const next = (skin === 'digital') ? 'digital' : 'analogue';
                this.skin = next;
                try { localStorage.setItem('radioVisual.skin.v1', next); } catch (_) {}
                if (this.els.stageAnalog) this.els.stageAnalog.classList.toggle('is-active', next === 'analogue');
                if (this.els.stageDigital) this.els.stageDigital.classList.toggle('is-active', next === 'digital');
                if (this.els.btnSkinAnalog) this.els.btnSkinAnalog.classList.toggle('is-active', next === 'analogue');
                if (this.els.btnSkinDigital) this.els.btnSkinDigital.classList.toggle('is-active', next === 'digital');
            }

            _drawMeters() {
                const canvas = this.els.vuCanvas;
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 4 || h < 4) return;
                ctx.clearRect(0, 0, w, h);
                const bars = this.skin === 'digital' ? 24 : 18;
                const gap = 2;
                const barW = Math.max(2, (w - gap * (bars + 1)) / bars);
                let levels = [];
                try {
                    if (state.analyserNode && state.audioCtx) {
                        const fft = state.analyserNode.fftSize || 256;
                        if (!this._vuBuf || this._vuBuf.length !== fft) {
                            this._vuBuf = new Uint8Array(fft);
                        }
                        state.analyserNode.getByteFrequencyData(this._vuBuf);
                        const step = Math.floor(this._vuBuf.length / bars) || 1;
                        for (let i = 0; i < bars; i++) {
                            let sum = 0;
                            const start = i * step;
                            const end = Math.min(this._vuBuf.length, start + step);
                            for (let j = start; j < end; j++) sum += this._vuBuf[j];
                            levels.push(sum / Math.max(1, end - start) / 255);
                        }
                    }
                } catch (_) {}
                if (!levels.length) {
                    const t = performance.now() * 0.003;
                    for (let i = 0; i < bars; i++) {
                        levels.push(0.15 + 0.12 * Math.sin(t + i * 0.55));
                    }
                }
                const warm = this.skin === 'analogue';
                for (let i = 0; i < bars; i++) {
                    const lv = Math.max(0.04, Math.min(1, levels[i] || 0));
                    const bh = Math.max(3, lv * h * 0.92);
                    const x = gap + i * (barW + gap);
                    const y = h - bh;
                    const grd = ctx.createLinearGradient(0, h, 0, y);
                    if (warm) {
                        grd.addColorStop(0, '#5a3010');
                        grd.addColorStop(0.5, '#e87830');
                        grd.addColorStop(1, '#ffe8a0');
                    } else {
                        grd.addColorStop(0, '#003848');
                        grd.addColorStop(0.5, '#00c8b8');
                        grd.addColorStop(1, '#e0ffff');
                    }
                    ctx.fillStyle = grd;
                    ctx.fillRect(x, y, barW, bh);
                }
            }

            _resizeCanvases() {
                const fit = (canvas) => {
                    if (!canvas || !canvas.parentElement) return;
                    const r = canvas.parentElement.getBoundingClientRect();
                    const dpr = Math.min(window.devicePixelRatio || 1, 2);
                    const w = Math.max(1, Math.floor(r.width * dpr));
                    const h = Math.max(1, Math.floor(r.height * dpr));
                    if (canvas.width !== w || canvas.height !== h) {
                        canvas.width = w;
                        canvas.height = h;
                    }
                };
                fit(this.els.vuCanvas);
                fit(this.els.digitalEqCanvas);
            }

            _bindAction(btn, fn) {
                if (!btn) return;
                btn.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    try { fn(); } catch (_) {}
                    try { resetIdleTimer(); } catch (_) {}
                }, { signal: this.abortCtrl.signal });
            }


            _buildFeatureButtons(gridEl) {
                if (!gridEl) return;
                const items = [
                    { label: 'Mixer', fn: () => { try { if (typeof toggleMixPanel === 'function') toggleMixPanel(); } catch (_) {} } },
                    { label: 'Avatar', fn: () => {
                        try {
                            if (typeof uiLocked !== 'undefined' && uiLocked) return;
                            if (typeof webmOn !== 'undefined' && webmOn && typeof hideWebm === 'function') hideWebm();
                            else if (typeof showWebm === 'function') {
                                if (!webmList.length && typeof loadWebmList === 'function') loadWebmList().finally(() => { if (webmList.length) showWebm(); });
                                else showWebm();
                            }
                        } catch (_) {}
                    }},
                    { label: 'Text-In', fn: () => {
                        try {
                            if (typeof openTextInForTarget === 'function') openTextInForTarget('global');
                            else if (typeof toggleTextInPanel === 'function') toggleTextInPanel();
                        } catch (_) {}
                    }},
                    { label: 'Video', fn: () => {
                        this._withDjDeck((dj) => {
                            if (dj.deckBVizMode === 'video') { dj.tearDownDeckBViz(); dj.syncDeckBVisualButtons(); }
                            else dj.startDeckBVideoVisual();
                        });
                    }},
                    { label: 'ProjectM', fn: () => this._loadVisualByName('ProjectM v2') },
                    { label: 'Audio:Bar', fn: () => this._loadVisualByName('Audio Bars') },
                    { label: 'Queue', fn: () => { this._withDjDeck((dj) => dj.toggleDeckBQueuePanel()); } }
                ];
                items.forEach((it) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.textContent = it.label;
                    this._bindAction(b, it.fn);
                    gridEl.appendChild(b);
                });
            }

            _mkKnobBlock(label, knob, readoutEl) {
                const b = document.createElement('div');
                b.className = 'radio-visual-knob-block';
                const l = document.createElement('div');
                l.className = 'radio-visual-knob-label';
                l.textContent = label;
                b.appendChild(l);
                b.appendChild(knob);
                if (readoutEl) b.appendChild(readoutEl);
                return b;
            }

            _wirePointerKnob(knobEl, onValue, opts) {
                if (!knobEl) return;
                let active = false;
                let moved = false;
                let startY = 0;
                let startVal = 0;
                const tapSlop = 4;
                const onDown = (ev) => {
                    this._stopClick(ev);
                    active = true;
                    moved = false;
                    startY = ev.clientY;
                    startVal = onValue.get();
                    if (opts && opts.onDragStart) {
                        try { opts.onDragStart(); } catch (_) {}
                    }
                    try { knobEl.setPointerCapture(ev.pointerId); } catch (_) {}
                };
                const onMove = (ev) => {
                    if (!active) return;
                    if (Math.abs(ev.clientY - startY) > tapSlop) moved = true;
                    if (!moved) return;
                    onValue.set(Math.max(0, Math.min(1, startVal + (startY - ev.clientY) * 0.004)));
                };
                const onUp = (ev) => {
                    if (!active) return;
                    active = false;
                    this._stopClick(ev);
                    try { knobEl.releasePointerCapture(ev.pointerId); } catch (_) {}
                    if (!moved && opts && opts.onTap) {
                        try { opts.onTap(); } catch (_) {}
                    }
                };
                knobEl.addEventListener('pointerdown', onDown, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointermove', onMove, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointerup', onUp, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointercancel', onUp, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('click', (ev) => this._stopClick(ev), { signal: this.abortCtrl.signal });
            }

            init() {
                container.innerHTML = '';
                try { initAudio(); } catch (_) {}
                try { if (!state.isPlaying && typeof playRadio === 'function') playRadio(); } catch (_) {}
                this.abortCtrl = new AbortController();
                const sig = { signal: this.abortCtrl.signal };
                try {
                    const stored = localStorage.getItem('radioVisual.skin.v1');
                    if (stored === 'digital' || stored === 'analogue') this.skin = stored;
                } catch (_) {}

                const root = document.createElement('div');
                root.id = 'radio-visual-root';
                root.className = 'radio-visual-root';
                root.setAttribute('role', 'application');
                root.setAttribute('aria-label', 'Radio');
                container.appendChild(root);
                this.root = root;

                const skinToggle = document.createElement('div');
                skinToggle.className = 'radio-visual-skin-toggle';
                const btnA = document.createElement('button');
                btnA.type = 'button';
                btnA.className = 'radio-visual-skin-btn';
                btnA.dataset.skin = 'analogue';
                btnA.textContent = 'Analogue';
                const btnD = document.createElement('button');
                btnD.type = 'button';
                btnD.className = 'radio-visual-skin-btn';
                btnD.dataset.skin = 'digital';
                btnD.textContent = 'Digital';
                skinToggle.appendChild(btnA);
                skinToggle.appendChild(btnD);
                root.appendChild(skinToggle);

                const stageA = document.createElement('section');
                stageA.className = 'radio-visual-stage radio-visual-skin--analogue is-active';
                stageA.setAttribute('aria-label', 'Analogue radio');
                const onAir = document.createElement('div');
                onAir.className = 'radio-visual-on-air';
                onAir.id = 'radio-visual-on-air';
                onAir.textContent = 'ON AIR';
                const stationsLine = document.createElement('div');
                stationsLine.className = 'radio-visual-stations-line';
                stationsLine.id = 'radio-visual-stations-line';
                const stPartA = document.createElement('span');
                stPartA.className = 'radio-visual-station-part radio-visual-station-part--a';
                const stLabA = document.createElement('span');
                stLabA.className = 'radio-visual-station-tag';
                stLabA.textContent = 'A';
                const stNameA = document.createElement('span');
                stNameA.className = 'radio-visual-station-name';
                stNameA.id = 'radio-visual-station-name-a';
                stNameA.textContent = '—';
                stPartA.appendChild(stLabA);
                stPartA.appendChild(stNameA);
                const stSep = document.createElement('span');
                stSep.className = 'radio-visual-station-sep';
                stSep.textContent = '·';
                const stPartB = document.createElement('span');
                stPartB.className = 'radio-visual-station-part radio-visual-station-part--b';
                const stLabB = document.createElement('span');
                stLabB.className = 'radio-visual-station-tag';
                stLabB.textContent = 'B';
                const stNameB = document.createElement('span');
                stNameB.className = 'radio-visual-station-name';
                stNameB.id = 'radio-visual-station-name-b';
                stNameB.textContent = '—';
                stPartB.appendChild(stLabB);
                stPartB.appendChild(stNameB);
                stationsLine.appendChild(stPartA);
                stationsLine.appendChild(stSep);
                stationsLine.appendChild(stPartB);
                const tunerShell = document.createElement('div');
                tunerShell.className = 'radio-visual-tuner-shell';
                const tunerRail = document.createElement('div');
                tunerRail.className = 'radio-visual-tuner-rail';
                tunerRail.id = 'radio-visual-tuner-rail';
                const ticks = document.createElement('div');
                ticks.className = 'radio-visual-tuner-ticks';
                ticks.id = 'radio-visual-ticks';
                if (Array.isArray(stations) && stations.length > 1) {
                    const n = Math.min(7, stations.length);
                    for (let i = 0; i < n; i++) {
                        const si = Math.round((i / Math.max(1, n - 1)) * (stations.length - 1));
                        const lab = stations[si] ? String(stations[si].name || '').slice(0, 6) : String(si + 1);
                        const sp = document.createElement('span');
                        sp.textContent = lab;
                        ticks.appendChild(sp);
                    }
                } else {
                    const sp = document.createElement('span');
                    sp.textContent = '—';
                    ticks.appendChild(sp);
                }
                const glow = document.createElement('div');
                glow.className = 'radio-visual-tuner-glow';
                glow.id = 'radio-visual-tuner-glow';
                const glowB = document.createElement('div');
                glowB.className = 'radio-visual-tuner-glow radio-visual-tuner-glow--deck-b';
                glowB.id = 'radio-visual-tuner-glow-b';
                const needle = document.createElement('div');
                needle.className = 'radio-visual-tuner-needle radio-visual-tuner-needle--deck-a';
                needle.id = 'radio-visual-needle';
                const needleB = document.createElement('div');
                needleB.className = 'radio-visual-tuner-needle radio-visual-tuner-needle--deck-b';
                needleB.id = 'radio-visual-needle-b';
                const vuWrap = document.createElement('div');
                vuWrap.className = 'radio-visual-vu-wrap';
                const vuCanvas = document.createElement('canvas');
                vuCanvas.className = 'radio-visual-vu-canvas';
                vuCanvas.id = 'radio-visual-vu';
                vuWrap.appendChild(vuCanvas);
                tunerRail.appendChild(ticks);
                tunerRail.appendChild(glow);
                tunerRail.appendChild(glowB);
                tunerRail.appendChild(needle);
                tunerRail.appendChild(needleB);
                tunerShell.appendChild(tunerRail);
                tunerShell.appendChild(vuWrap);                const knobs = document.createElement('div');
                knobs.className = 'radio-visual-knobs-row radio-visual-knobs-row--all';
                const mkControlKnob = (id, aria) => {
                    const k = document.createElement('div');
                    k.className = 'radio-visual-knob';
                    k.id = id;
                    k.setAttribute('aria-label', aria);
                    return k;
                };
                const mkReadout = (text) => {
                    const r = document.createElement('div');
                    r.className = 'radio-visual-knob-readout';
                    r.textContent = text;
                    return r;
                };
                const volKnob = mkControlKnob('radio-visual-vol-knob', 'Volume');
                volKnob.setAttribute('role', 'slider');
                const deckAKnob = mkControlKnob('radio-visual-deck-a-knob', 'Deck A play / pause');
                deckAKnob.classList.add('radio-visual-knob--deck-a');
                const deckBKnob = mkControlKnob('radio-visual-deck-b-knob', 'Deck B play / pause');
                deckBKnob.classList.add('radio-visual-knob--deck-b');
                const crossKnob = mkControlKnob('radio-visual-cross-knob', 'Crossfade between decks');
                const autoFadeKnob = mkControlKnob('radio-visual-autofade-knob', 'Auto-fade duration; click to fade');
                autoFadeKnob.setAttribute('role', 'slider');
                const autoMixKnob = mkControlKnob('radio-visual-automix-knob', 'Auto-mix max interval; click to toggle');
                autoMixKnob.setAttribute('role', 'slider');
                const autoFadeReadout = mkReadout(`${(this._readAutoFadeDurationMs() / 1000).toFixed(1)}s`);
                const autoMixReadout = mkReadout(`${this._readAutoMixMaxMin()}m max`);
                knobs.appendChild(this._mkKnobBlock('Volume', volKnob));
                knobs.appendChild(this._mkKnobBlock('Deck A', deckAKnob));
                knobs.appendChild(this._mkKnobBlock('Deck B', deckBKnob));
                knobs.appendChild(this._mkKnobBlock('Crossfade', crossKnob));
                knobs.appendChild(this._mkKnobBlock('Auto-Fade', autoFadeKnob, autoFadeReadout));
                knobs.appendChild(this._mkKnobBlock('Auto-Mix', autoMixKnob, autoMixReadout));
                const analogBtns = document.createElement('div');
                analogBtns.className = 'radio-visual-btn-grid';
                analogBtns.id = 'radio-visual-analog-btns';
                stageA.appendChild(onAir);
                stageA.appendChild(stationsLine);
                stageA.appendChild(knobs);
                stageA.appendChild(tunerShell);
                stageA.appendChild(analogBtns);

                const stageD = document.createElement('section');
                stageD.className = 'radio-visual-stage radio-visual-skin--digital';
                stageD.setAttribute('aria-label', 'Digital radio');
                const dPanel = document.createElement('div');
                dPanel.className = 'radio-visual-digital-panel';
                const dDisp = document.createElement('div');
                dDisp.className = 'radio-visual-digital-display';
                const mkLine = (cls, id, txt) => {
                    const el = document.createElement('div');
                    el.className = 'radio-visual-digital-line' + (cls ? ' ' + cls : '');
                    if (id) el.id = id;
                    el.textContent = txt;
                    return el;
                };
                const stD = mkLine('radio-visual-digital-line--title', 'radio-visual-station-d', '—');
                const dStat = mkLine('', 'radio-visual-digital-status', 'STANDBY');
                const dMeta = mkLine('', 'radio-visual-digital-meta', '—');
                const dClk = mkLine('', 'radio-visual-digital-clock', '—');
                dDisp.appendChild(stD);
                dDisp.appendChild(dStat);
                dDisp.appendChild(dMeta);
                dDisp.appendChild(dClk);
                const volRow = document.createElement('div');
                volRow.className = 'radio-visual-digital-vol-row';
                const volLbl = document.createElement('label');
                volLbl.htmlFor = 'radio-visual-vol-digital';
                volLbl.textContent = 'VOL';
                const volDig = document.createElement('input');
                volDig.type = 'range';
                volDig.className = 'radio-visual-digital-vol';
                volDig.id = 'radio-visual-vol-digital';
                volDig.min = '0';
                volDig.max = '1';
                volDig.step = '0.01';
                volDig.value = '0.5';
                volRow.appendChild(volLbl);
                volRow.appendChild(volDig);
                const stRow = document.createElement('div');
                stRow.className = 'radio-visual-digital-station-row';
                stRow.setAttribute('aria-label', 'Deck A');
                [['prev', 'A◀'], ['next', 'A▶'], ['rand', 'A Rand']].forEach(([act, lab]) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvAction = act;
                    b.dataset.rvDeck = 'a';
                    b.textContent = lab;
                    stRow.appendChild(b);
                });
                const stRowB = document.createElement('div');
                stRowB.className = 'radio-visual-digital-station-row';
                stRowB.setAttribute('aria-label', 'Deck B');
                [['prev', 'B◀'], ['next', 'B▶'], ['rand', 'B Rand']].forEach(([act, lab]) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvAction = act;
                    b.dataset.rvDeck = 'b';
                    b.textContent = lab;
                    stRowB.appendChild(b);
                });
                const crossRow = document.createElement('div');
                crossRow.className = 'radio-visual-digital-vol-row';
                const crossLbl = document.createElement('label');
                crossLbl.htmlFor = 'radio-visual-cross-digital';
                crossLbl.textContent = 'XFADE';
                const crossDig = document.createElement('input');
                crossDig.type = 'range';
                crossDig.className = 'radio-visual-digital-vol';
                crossDig.id = 'radio-visual-cross-digital';
                crossDig.min = '0';
                crossDig.max = '1';
                crossDig.step = '0.01';
                crossDig.value = String(this._getCrossfadeX());
                crossRow.appendChild(crossLbl);
                crossRow.appendChild(crossDig);
                const eqWrap = document.createElement('div');
                eqWrap.className = 'radio-visual-digital-eq';
                const eqCanvas = document.createElement('canvas');
                eqCanvas.className = 'radio-visual-vu-canvas';
                eqCanvas.id = 'radio-visual-digital-eq';
                eqWrap.appendChild(eqCanvas);
                const digBtns = document.createElement('div');
                digBtns.className = 'radio-visual-btn-grid';
                digBtns.id = 'radio-visual-digital-btns';
                dPanel.appendChild(dDisp);
                dPanel.appendChild(volRow);
                dPanel.appendChild(stRow);
                dPanel.appendChild(stRowB);
                dPanel.appendChild(crossRow);
                dPanel.appendChild(eqWrap);
                dPanel.appendChild(digBtns);
                stageD.appendChild(dPanel);
                root.appendChild(stageA);
                root.appendChild(stageD);

                this.els = {
                    btnSkinAnalog: btnA,
                    btnSkinDigital: btnD,
                    stageAnalog: stageA,
                    stageDigital: stageD,
                    onAir,
                    stationNameA: stNameA,
                    stationNameB: stNameB,
                    stationDigital: stD,
                    ticks,
                    needle,
                    needleB,
                    tunerGlow: glow,
                    tunerGlowB: glowB,
                    tunerRail,
                    vuCanvas,
                    digitalEqCanvas: eqCanvas,
                    volKnob,
                    deckAKnob,
                    deckBKnob,
                    crossKnob,
                    autoFadeKnob,
                    autoMixKnob,
                    autoFadeReadout,
                    autoMixReadout,
                    volDigital: volDig,
                    crossDigital: crossDig,
                    digitalStatus: dStat,
                    digitalMeta: dMeta,
                    digitalClock: dClk,
                    analogBtns,
                    digitalBtns: digBtns
                };

                this._buildFeatureButtons(analogBtns);
                this._buildFeatureButtons(digBtns);
                this._setSkin(this.skin);
                this._syncVolumeFromGlobal();
                this._setAutoFadeDurationNorm(this._autoFadeDurationNorm());
                this._setAutoMixMaxNorm(this._autoMixMaxNorm());
                this._syncDeckSwitches();
                this._updateStationUi();
                this._tickClock();

                btnA.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('analogue'); }, sig);
                btnD.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('digital'); }, sig);
                volDig.addEventListener('input', () => this._applyVolume(volDig.value), sig);
                crossDig.addEventListener('input', () => this._setCrossfadeX(crossDig.value), sig);
                const bindDeckRow = (rowEl) => {
                    rowEl.querySelectorAll('[data-rv-action]').forEach((b) => {
                        b.addEventListener('click', (ev) => {
                            this._stopClick(ev);
                            const deck = b.dataset.rvDeck || 'a';
                            const a = b.dataset.rvAction;
                            if (deck === 'b') {
                                if (a === 'prev') this._stationBPrev();
                                else if (a === 'next') this._stationBNext();
                                else if (a === 'rand') this._stationBRand();
                            } else if (a === 'prev') this._stationPrev();
                            else if (a === 'next') this._stationNext();
                            else if (a === 'rand') this._stationRand();
                        }, sig);
                    });
                };
                bindDeckRow(stRow);
                bindDeckRow(stRowB);
                tunerRail.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    if (!Array.isArray(stations) || stations.length < 2) return;
                    const r = tunerRail.getBoundingClientRect();
                    const t = Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
                    const idx = Math.round(t * (stations.length - 1));
                    if (typeof setStation === 'function') setStation(idx);
                }, sig);
                this._wirePointerKnob(volKnob, {
                    get: () => Number(document.getElementById('volume-slider')?.value || 0.5),
                    set: (v) => this._applyVolume(v)
                });
                this._wireDeckSwitch(deckAKnob, 'a');
                this._wireDeckSwitch(deckBKnob, 'b');
                this._wireClickKnob(crossKnob, () => this._triggerAutoFade());
                this._wirePointerKnob(autoFadeKnob, {
                    get: () => this._autoFadeDurationNorm(),
                    set: (t) => this._setAutoFadeDurationNorm(t)
                }, { onTap: () => this._triggerAutoFade() });
                this._wirePointerKnob(autoMixKnob, {
                    get: () => this._autoMixMaxNorm(),
                    set: (t) => this._setAutoMixMaxNorm(t)
                }, { onTap: () => this._toggleAutoMix() });

                const stopRv = (ev) => this._stopInteraction(ev);
                root.addEventListener('click', stopRv, sig);
                root.addEventListener('pointerdown', stopRv, sig);
                root.addEventListener('pointerup', stopRv, sig);

                window.addEventListener('resize', this.resizeHandler, sig);
                this.onResize();
                this.clockTimerId = setInterval(() => { try { this._tickClock(); } catch (_) {} }, 1000);
                this.animateFrame();
                document.getElementById('mode-sub').innerText = 'Radio';
            }

            animateFrame() {
                this.animId = requestAnimationFrame(() => this.animateFrame());
                try {
                    this._updateStationUi();
                    this._drawMeters();
                } catch (_) {}
            }

            onResize() {
                try { this._resizeCanvases(); } catch (_) {}
            }

            destroy() {
                if (this._rvAutoFadeRaf) {
                    try { cancelAnimationFrame(this._rvAutoFadeRaf); } catch (_) {}
                    this._rvAutoFadeRaf = null;
                }
                try { if (this.animId) cancelAnimationFrame(this.animId); } catch (_) {}
                this.animId = null;
                if (this.clockTimerId) {
                    try { clearInterval(this.clockTimerId); } catch (_) {}
                    this.clockTimerId = null;
                }
                try { if (this.abortCtrl) this.abortCtrl.abort(); } catch (_) {}
                this.abortCtrl = null;
                try {
                    window.removeEventListener('resize', this.resizeHandler);
                } catch (_) {}
                this.root = null;
                this.els = {};
                try { container.innerHTML = ''; } catch (_) {}
            }
        }
