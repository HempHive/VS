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
                this.digitalCenterMode = 'spectrum';
                this._digitalVolStep = 0.05;
            }

            static get AUTOFADE_MS_KEY() { return 'dj.autofade.duration.ms.v1'; }
            static get AUTOFADE_MIN_MS() { return 2000; }
            static get AUTOFADE_MAX_MS() { return 15000; }
            static get AUTOMIX_MAX_KEY() { return 'dj.automix.max.min.v1'; }
            static get AUTOMIX_ENABLED_KEY() { return 'dj.automix.enabled.v1'; }
            static get AUTOMIX_MIN_MIN() { return 1; }
            static get AUTOMIX_MAX_MIN() { return 20; }
            static get AUTOFADE_CHANGE_STATION_KEY() { return 'dj.autofade.changeStation.enabled.v1'; }

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
                if (this.els.crossKnob) {
                    this._setKnobRotation(this.els.crossKnob, (x * 270) - 135);
                }
                if (this.els.crossDigital) this.els.crossDigital.value = String(x);
            }

            _isRadioVisualActive() {
                const vis = state && state.activeVisualizer;
                return !!(vis && (vis === this || vis.name === 'Radio' || vis.name === 'Radio Visual'));
            }

            _syncFadeKnobs() {
                const active = !!(this._rvFadeActive || this._rvAutoFadeRaf);
                if (this.els.crossKnob) {
                    this.els.crossKnob.classList.toggle('is-on', active);
                    this.els.crossKnob.setAttribute('aria-pressed', active ? 'true' : 'false');
                }
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
                    this.els.autoMixReadout.textContent = String(clamped);
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
                this._rvFadeActive = true;
                this._syncFadeKnobs();
                const endFadeLed = () => {
                    if (!this._rvAutoFadeRaf) {
                        this._rvFadeActive = false;
                        this._syncFadeKnobs();
                    }
                };
                const scheduleFadeLedOff = () => {
                    const dur = this._readAutoFadeDurationMs();
                    if (this._rvFadeLedTimer) {
                        try { clearTimeout(this._rvFadeLedTimer); } catch (_) {}
                    }
                    this._rvFadeLedTimer = setTimeout(() => {
                        endFadeLed();
                        this._rvFadeLedTimer = null;
                    }, dur + 120);
                };
                const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                    ? state.activeVisualizer : null;
                if (eng && typeof eng.triggerAutoFadeFromShortcut === 'function') {
                    try { eng.triggerAutoFadeFromShortcut(); } catch (_) {}
                    scheduleFadeLedOff();
                    return;
                }
                const x = this._getCrossfadeX();
                let targetDeck = x < 0.5 ? 'b' : 'a';
                if (this._rvAutoFadeRaf && this._rvFadeTargetDeck) {
                    const destX = this._rvFadeTargetDeck === 'b' ? 1 : 0;
                    if (Math.abs(x - destX) > 0.001) {
                        targetDeck = this._rvFadeTargetDeck === 'a' ? 'b' : 'a';
                    }
                }
                const endVal = targetDeck === 'b' ? 1 : 0;
                const startVal = x;
                this._rvFadeTargetDeck = targetDeck;
                if (Math.abs(endVal - startVal) < 0.001) {
                    this._rvFadeTargetDeck = null;
                    endFadeLed();
                    return;
                }
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
                        this._rvFadeTargetDeck = null;
                        this._rvFadeActive = false;
                        this._syncFadeKnobs();
                        return;
                    }
                    this._rvAutoFadeRaf = requestAnimationFrame(tick);
                };
                this._rvAutoFadeRaf = requestAnimationFrame(tick);
            }

            _triggerAutoFade() {
                this._rvFadeActive = true;
                this._syncFadeKnobs();
                if (this._isRadioVisualActive()) {
                    this._runLocalAutoFade();
                    return;
                }
                const btn = document.getElementById('mix-autofade') || document.getElementById('dj-autofade');
                if (btn) {
                    try { btn.click(); } catch (_) {}
                    const dur = this._readAutoFadeDurationMs();
                    if (this._rvFadeLedTimer) {
                        try { clearTimeout(this._rvFadeLedTimer); } catch (_) {}
                    }
                    this._rvFadeLedTimer = setTimeout(() => {
                        if (!this._rvAutoFadeRaf) {
                            this._rvFadeActive = false;
                            this._syncFadeKnobs();
                        }
                        this._rvFadeLedTimer = null;
                    }, dur + 120);
                    return;
                }
                this._runLocalAutoFade();
            }

            _isAutoMixEnabled() {
                try {
                    if (typeof isAutoMixEnabledGlobal === 'function') return isAutoMixEnabledGlobal();
                } catch (_) {}
                try {
                    return localStorage.getItem(RadioVisualEngine.AUTOMIX_ENABLED_KEY) === '1';
                } catch (_) {
                    return false;
                }
            }

            _isAutoFadeChangeStationEnabled() {
                try {
                    if (typeof isAutoFadeChangeStationEnabledGlobal === 'function') {
                        return isAutoFadeChangeStationEnabledGlobal();
                    }
                } catch (_) {}
                try {
                    const raw = localStorage.getItem(RadioVisualEngine.AUTOFADE_CHANGE_STATION_KEY);
                    if (raw === '0') return false;
                    if (raw === '1') return true;
                } catch (_) {}
                return true;
            }

            _toggleAutoMix() {
                if (this._isRadioVisualActive()) {
                    const next = !this._isAutoMixEnabled();
                    try { localStorage.setItem(RadioVisualEngine.AUTOMIX_ENABLED_KEY, next ? '1' : '0'); } catch (_) {}
                    try { state.autoMixEnabled = next; } catch (_) {}
                    this._syncAutoMixKnob();
                    return;
                }
                const btn = document.getElementById('mix-automix') || document.getElementById('dj-automix');
                if (btn) {
                    try { btn.click(); } catch (_) {}
                    this._syncAutoMixKnob();
                    return;
                }
                try {
                    const next = !this._isAutoMixEnabled();
                    localStorage.setItem(RadioVisualEngine.AUTOMIX_ENABLED_KEY, next ? '1' : '0');
                    try { state.autoMixEnabled = next; } catch (_) {}
                } catch (_) {}
                this._syncAutoMixKnob();
            }

            _toggleAutoFadeChangeStation() {
                const cb = document.getElementById('dj-autofade-change-station');
                if (cb) {
                    cb.checked = !cb.checked;
                    try { cb.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
                    this._syncAutoFadeChangeStationKnob();
                    return;
                }
                const next = !this._isAutoFadeChangeStationEnabled();
                try {
                    localStorage.setItem(RadioVisualEngine.AUTOFADE_CHANGE_STATION_KEY, next ? '1' : '0');
                } catch (_) {}
                this._syncAutoFadeChangeStationKnob();
            }

            _syncAutoMixKnob() {
                if (!this.els.autoMixKnob) return;
                const on = this._isAutoMixEnabled();
                this.els.autoMixKnob.classList.toggle('is-on', on);
                this.els.autoMixKnob.setAttribute('aria-pressed', on ? 'true' : 'false');
            }

            _syncAutoFadeChangeStationKnob() {
                if (!this.els.autoFadeKnob) return;
                const on = this._isAutoFadeChangeStationEnabled();
                this.els.autoFadeKnob.classList.toggle('is-on', on);
                this.els.autoFadeKnob.setAttribute('aria-pressed', on ? 'true' : 'false');
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
                    this._setKnobRotation(this.els.deckAKnob, (this._needlePercentA() / 100 * 270) - 135);
                }
                if (this.els.deckBKnob) {
                    this.els.deckBKnob.classList.toggle('is-on', bOn);
                    this.els.deckBKnob.setAttribute('aria-pressed', bOn ? 'true' : 'false');
                    this._setKnobRotation(this.els.deckBKnob, (this._needlePercentB() / 100 * 270) - 135);
                }
            }

            _stepDigitalVolume(delta) {
                const vs = document.getElementById('volume-slider');
                const cur = vs ? Number(vs.value) : 0.5;
                const step = Number(this._digitalVolStep) || 0.05;
                this._applyVolume(cur + (Number(delta) || 0) * step);
                this._syncDigitalVolumeUi();
            }

            _syncDigitalVolumeUi() {
                const vs = document.getElementById('volume-slider');
                const v = vs ? Number(vs.value) : 0.5;
                if (this.els.volDigitalReadout) {
                    this.els.volDigitalReadout.textContent = `${Math.round(v * 100)}%`;
                }
            }

            _setDigitalCenterMode(mode) {
                const next = (mode === 'deckB') ? 'deckB' : 'spectrum';
                this.digitalCenterMode = next;
                try { localStorage.setItem('radioVisual.digitalCenter.v1', next); } catch (_) {}
                if (this.els.digitalCenterSpectrum) {
                    this.els.digitalCenterSpectrum.classList.toggle('is-active', next === 'spectrum');
                }
                if (this.els.digitalCenterDeckB) {
                    this.els.digitalCenterDeckB.classList.toggle('is-active', next === 'deckB');
                }
                if (this.els.btnDigitalSpectrum) {
                    this.els.btnDigitalSpectrum.classList.toggle('is-active', next === 'spectrum');
                }
                if (this.els.btnDigitalDeckB) {
                    this.els.btnDigitalDeckB.classList.toggle('is-active', next === 'deckB');
                }
                if (next === 'deckB') this._syncDigitalDeckBVideo();
            }

            _tearDownDigitalDeckBPlayer() {
                try {
                    if (this.els.digitalDeckBVideo) {
                        this.els.digitalDeckBVideo.pause();
                        this.els.digitalDeckBVideo.removeAttribute('src');
                    }
                } catch (_) {}
            }

            _syncDigitalDeckBVideo() {
                const vid = this.els.digitalDeckBVideo;
                if (!vid || this.digitalCenterMode !== 'deckB') return;
                try {
                    let layer = null;
                    if (typeof getDeckBVideoPlaybackSources === 'function') {
                        const sources = getDeckBVideoPlaybackSources();
                        if (sources && sources.length) layer = sources[0];
                    }
                    if (!layer && typeof deckVideoFeeds !== 'undefined' && deckVideoFeeds.b && deckVideoFeeds.b.url) {
                        layer = { url: deckVideoFeeds.b.url, label: deckVideoFeeds.b.label || 'Deck B' };
                    }
                    if (layer && typeof applyDeckBVideoPayloadToElement === 'function') {
                        applyDeckBVideoPayloadToElement(vid, layer, null);
                        return;
                    }
                    if (layer && layer.url) {
                        vid.src = layer.url;
                        vid.play().catch(() => {});
                    }
                } catch (_) {}
            }

            _drawDigitalSpectrum() {
                const canvas = this.els.digitalSpectrumCanvas;
                if (!canvas || this.skin !== 'digital' || this.digitalCenterMode !== 'spectrum') return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 8 || h < 8) return;
                const cx = w * 0.5;
                const cy = h * 0.52;
                const t = performance.now() * 0.001;
                ctx.clearRect(0, 0, w, h);
                const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.55);
                bg.addColorStop(0, '#0a1e28');
                bg.addColorStop(0.55, '#061018');
                bg.addColorStop(1, '#020608');
                ctx.fillStyle = bg;
                ctx.fillRect(0, 0, w, h);
                let levels = [];
                try {
                    if (state.analyserNode && state.audioCtx) {
                        const fft = state.analyserNode.fftSize || 256;
                        if (!this._vuBuf || this._vuBuf.length !== fft) {
                            this._vuBuf = new Uint8Array(fft);
                        }
                        state.analyserNode.getByteFrequencyData(this._vuBuf);
                        const bars = 72;
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
                    for (let i = 0; i < 72; i++) {
                        levels.push(0.12 + 0.1 * Math.sin(t * 2.2 + i * 0.42));
                    }
                }
                const innerR = Math.min(w, h) * 0.14;
                const outerR = Math.min(w, h) * 0.44;
                ctx.strokeStyle = 'rgba(0, 255, 220, 0.08)';
                ctx.lineWidth = 1;
                for (let ring = 1; ring <= 4; ring++) {
                    ctx.beginPath();
                    ctx.arc(cx, cy, innerR + ((outerR - innerR) * ring) / 4, 0, Math.PI * 2);
                    ctx.stroke();
                }
                const n = levels.length;
                for (let i = 0; i < n; i++) {
                    const lv = Math.min(1, Math.max(0.06, Math.pow((levels[i] || 0) * 3.1, 0.68)));
                    const a0 = (i / n) * Math.PI * 2 - Math.PI / 2;
                    const a1 = ((i + 1) / n) * Math.PI * 2 - Math.PI / 2;
                    const r1 = innerR + (outerR - innerR) * lv;
                    ctx.beginPath();
                    ctx.moveTo(cx + Math.cos(a0) * innerR, cy + Math.sin(a0) * innerR);
                    ctx.lineTo(cx + Math.cos(a0) * r1, cy + Math.sin(a0) * r1);
                    ctx.lineTo(cx + Math.cos(a1) * r1, cy + Math.sin(a1) * r1);
                    ctx.lineTo(cx + Math.cos(a1) * innerR, cy + Math.sin(a1) * innerR);
                    ctx.closePath();
                    const hue = 170 + (i / n) * 80;
                    ctx.fillStyle = `hsla(${hue}, 90%, 58%, ${0.35 + lv * 0.55})`;
                    ctx.fill();
                }
                ctx.beginPath();
                ctx.arc(cx, cy, innerR * 0.92, 0, Math.PI * 2);
                const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, innerR);
                core.addColorStop(0, 'rgba(0, 255, 220, 0.35)');
                core.addColorStop(1, 'rgba(0, 40, 60, 0.05)');
                ctx.fillStyle = core;
                ctx.fill();
            }

            _wireCrossfadeKnob(knobEl) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch', 'radio-visual-knob--fade-btn');
                knobEl.setAttribute('role', 'slider');
                knobEl.tabIndex = 0;
                knobEl.setAttribute('aria-label', 'Crossfade between decks; drag to mix, tap to auto-fade');
                this._wirePointerKnob(knobEl, {
                    get: () => this._getCrossfadeX(),
                    set: (t) => this._setCrossfadeX(t)
                }, { onTap: () => this._triggerAutoFade() });
                this._syncCrossfadeKnob();
            }

            _wireAutoFadeKnob(knobEl) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch', 'radio-visual-knob--fade-btn');
                knobEl.setAttribute('role', 'slider');
                knobEl.tabIndex = 0;
                knobEl.setAttribute('aria-label', 'Auto-fade duration; drag to adjust, tap to toggle change station before fading');
                this._wirePointerKnob(knobEl, {
                    get: () => this._autoFadeDurationNorm(),
                    set: (t) => this._setAutoFadeDurationNorm(t)
                }, { onTap: () => this._toggleAutoFadeChangeStation() });
                this._setKnobRotation(knobEl, (this._autoFadeDurationNorm() * 270) - 135);
                this._syncAutoFadeChangeStationKnob();
            }

            _wireAutoMixKnob(knobEl) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch', 'radio-visual-knob--fade-btn');
                knobEl.setAttribute('role', 'slider');
                knobEl.tabIndex = 0;
                knobEl.setAttribute('aria-label', 'Auto-mix interval; drag to adjust, tap to toggle auto-mix');
                this._wirePointerKnob(knobEl, {
                    get: () => this._autoMixMaxNorm(),
                    set: (t) => this._setAutoMixMaxNorm(t)
                }, { onTap: () => this._toggleAutoMix() });
                this._setKnobRotation(knobEl, (this._autoMixMaxNorm() * 270) - 135);
                this._syncAutoMixKnob();
            }

            _wireDeckKnob(knobEl, deck) {
                if (!knobEl) return;
                knobEl.classList.add('radio-visual-knob--switch');
                knobEl.classList.add(deck === 'a' ? 'radio-visual-knob--deck-a' : 'radio-visual-knob--deck-b');
                knobEl.setAttribute('role', 'slider');
                knobEl.tabIndex = 0;
                this._wirePointerKnob(knobEl, {
                    get: () => (deck === 'a' ? this._needlePercentA() : this._needlePercentB()) / 100,
                    set: (t) => {
                        const idx = this._normToStationIndex(t);
                        if (deck === 'a') {
                            if (typeof setStation === 'function') setStation(idx);
                        } else {
                            this._setStationB(idx);
                        }
                    }
                }, {
                    onTap: async () => {
                        if (deck === 'a') await this._toggleDeckA();
                        else await this._toggleDeckB();
                        this._syncDeckSwitches();
                    }
                });
            }

            _syncVolumeFromGlobal() {
                const vs = document.getElementById('volume-slider');
                const v = vs ? Number(vs.value) : 0.5;
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
                this._syncDigitalVolumeUi();
            }

            _applyVolume(val) {
                const v = Math.max(0, Math.min(1, Number(val) || 0));
                try {
                    if (typeof setVolume === 'function') setVolume(v);
                } catch (_) {}
                const vs = document.getElementById('volume-slider');
                if (vs) vs.value = String(v);
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
                this._syncDigitalVolumeUi();
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
                this._syncFadeKnobs();
                this._syncAutoMixKnob();
                this._syncAutoFadeChangeStationKnob();
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
                try { this.onResize(); } catch (_) {}
                if (next === 'digital' && this.digitalCenterMode === 'deckB') {
                    this._syncDigitalDeckBVideo();
                }
            }

            _drawBarMeter(canvas, { bars = 18, warm = false } = {}) {
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 4 || h < 4) return;
                ctx.clearRect(0, 0, w, h);
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

            _drawMeters() {
                if (this.skin === 'analogue') {
                    this._drawBarMeter(this.els.vuCanvas, { bars: 18, warm: true });
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
                if (this.skin === 'analogue') {
                    fit(this.els.vuCanvas);
                } else {
                    fit(this.els.digitalSpectrumCanvas);
                }
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
                    { label: 'Queue', fn: () => { this._withDjDeck((dj) => dj.toggleDeckBQueuePanel()); } },
                    { label: 'Stations', fn: () => {
                        try {
                            if (typeof uiLocked !== 'undefined' && uiLocked) return;
                            if (typeof toggleTopMenuPanel === 'function') toggleTopMenuPanel();
                        } catch (_) {}
                    } }
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

            _mkKnobBlock(label, knob, readoutEl, readoutOnKnob = false) {
                const b = document.createElement('div');
                b.className = 'radio-visual-knob-block';
                const l = document.createElement('div');
                l.className = 'radio-visual-knob-label';
                l.textContent = label;
                b.appendChild(l);
                b.appendChild(knob);
                if (readoutEl) {
                    if (readoutOnKnob) {
                        readoutEl.classList.add('radio-visual-knob-readout--on-knob');
                        knob.appendChild(readoutEl);
                    } else {
                        b.appendChild(readoutEl);
                    }
                }
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
                try {
                    const centerStored = localStorage.getItem('radioVisual.digitalCenter.v1');
                    if (centerStored === 'deckB' || centerStored === 'spectrum') {
                        this.digitalCenterMode = centerStored;
                    }
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
                tunerShell.appendChild(vuWrap);
                const knobs = document.createElement('div');
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
                const crossKnob = mkControlKnob('radio-visual-cross-knob', 'Cross-fade between decks');
                const autoFadeKnob = mkControlKnob('radio-visual-autofade-knob', 'Auto-fade');
                const autoMixKnob = mkControlKnob('radio-visual-automix-knob', 'Auto-mix max interval; click to toggle');
                autoMixKnob.setAttribute('role', 'slider');
                const autoFadeReadout = mkReadout(`${(this._readAutoFadeDurationMs() / 1000).toFixed(1)}s`);
                const autoMixReadout = mkReadout(String(this._readAutoMixMaxMin()));
                knobs.appendChild(this._mkKnobBlock('Volume', volKnob));
                knobs.appendChild(this._mkKnobBlock('Deck A', deckAKnob));
                knobs.appendChild(this._mkKnobBlock('Deck B', deckBKnob));
                knobs.appendChild(this._mkKnobBlock('Crossfade', crossKnob));
                knobs.appendChild(this._mkKnobBlock('Auto-Fade', autoFadeKnob, autoFadeReadout, true));
                knobs.appendChild(this._mkKnobBlock('Auto-Mix', autoMixKnob, autoMixReadout, true));
                const analogBtns = document.createElement('div');
                analogBtns.className = 'radio-visual-btn-grid radio-visual-analog-actions';
                analogBtns.id = 'radio-visual-analog-btns';
                tunerShell.appendChild(analogBtns);
                stageA.appendChild(knobs);
                stageA.appendChild(tunerShell);

                const stageD = document.createElement('section');
                stageD.className = 'radio-visual-stage radio-visual-skin--digital';
                stageD.setAttribute('aria-label', 'Digital radio');
                const dPanel = document.createElement('div');
                dPanel.className = 'radio-visual-digital-panel';
                const mkLine = (cls, id, txt) => {
                    const el = document.createElement('div');
                    el.className = 'radio-visual-digital-line' + (cls ? ' ' + cls : '');
                    if (id) el.id = id;
                    el.textContent = txt;
                    return el;
                };
                const stD = mkLine('radio-visual-digital-line--title', 'radio-visual-station-d', '—');
                const dClk = mkLine('radio-visual-digital-line--clock', 'radio-visual-digital-clock', '—');
                const digBtns = document.createElement('div');
                digBtns.className = 'radio-visual-btn-grid radio-visual-digital-feature-btns';
                digBtns.id = 'radio-visual-digital-btns';
                const digitalCenter = document.createElement('div');
                digitalCenter.className = 'radio-visual-digital-center';
                const digitalCenterSpectrum = document.createElement('div');
                digitalCenterSpectrum.className = 'radio-visual-digital-center-pane is-active';
                const digitalSpectrumCanvas = document.createElement('canvas');
                digitalSpectrumCanvas.className = 'radio-visual-digital-spectrum-canvas';
                digitalSpectrumCanvas.id = 'radio-visual-digital-spectrum';
                digitalCenterSpectrum.appendChild(digitalSpectrumCanvas);
                const spectrumOverlay = document.createElement('div');
                spectrumOverlay.className = 'radio-visual-digital-spectrum-overlay';
                spectrumOverlay.setAttribute('aria-live', 'polite');
                spectrumOverlay.appendChild(stD);
                spectrumOverlay.appendChild(dClk);
                digitalCenterSpectrum.appendChild(spectrumOverlay);
                const digitalCenterDeckB = document.createElement('div');
                digitalCenterDeckB.className = 'radio-visual-digital-center-pane';
                const digitalDeckBMount = document.createElement('div');
                digitalDeckBMount.className = 'radio-visual-digital-deck-b-mount';
                const digitalDeckBVideo = document.createElement('video');
                digitalDeckBVideo.className = 'radio-visual-digital-deck-b-video';
                digitalDeckBVideo.playsInline = true;
                digitalDeckBVideo.muted = true;
                digitalDeckBMount.appendChild(digitalDeckBVideo);
                digitalCenterDeckB.appendChild(digitalDeckBMount);
                digitalCenter.appendChild(digitalCenterSpectrum);
                digitalCenter.appendChild(digitalCenterDeckB);
                const digitalToolbar = document.createElement('div');
                digitalToolbar.className = 'radio-visual-digital-toolbar';
                digitalToolbar.id = 'radio-visual-digital-toolbar';
                const btnDigitalSpectrum = document.createElement('button');
                btnDigitalSpectrum.type = 'button';
                btnDigitalSpectrum.className = 'radio-visual-btn';
                btnDigitalSpectrum.textContent = 'Spectrum';
                const btnDigitalDeckB = document.createElement('button');
                btnDigitalDeckB.type = 'button';
                btnDigitalDeckB.className = 'radio-visual-btn';
                btnDigitalDeckB.textContent = 'Deck B';
                digitalToolbar.appendChild(btnDigitalSpectrum);
                digitalToolbar.appendChild(btnDigitalDeckB);
                const volGroup = document.createElement('div');
                volGroup.className = 'radio-visual-digital-toolbar-vol';
                const volLbl = document.createElement('span');
                volLbl.className = 'radio-visual-digital-vol-step-label';
                volLbl.textContent = 'VOL';
                const volDown = document.createElement('button');
                volDown.type = 'button';
                volDown.className = 'radio-visual-btn radio-visual-digital-step-btn';
                volDown.textContent = '−';
                volDown.setAttribute('aria-label', 'Volume down');
                const volDigitalReadout = document.createElement('span');
                volDigitalReadout.className = 'radio-visual-digital-vol-readout';
                volDigitalReadout.id = 'radio-visual-vol-readout';
                volDigitalReadout.textContent = '50%';
                const volUp = document.createElement('button');
                volUp.type = 'button';
                volUp.className = 'radio-visual-btn radio-visual-digital-step-btn';
                volUp.textContent = '+';
                volUp.setAttribute('aria-label', 'Volume up');
                volGroup.appendChild(volLbl);
                volGroup.appendChild(volDown);
                volGroup.appendChild(volDigitalReadout);
                volGroup.appendChild(volUp);
                digitalToolbar.appendChild(volGroup);
                [['a', 'A Play'], ['b', 'B Play'], ['fade', 'Fade'], ['mix', 'Mix']].forEach(([act, lab]) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvDigital = act;
                    b.textContent = lab;
                    digitalToolbar.appendChild(b);
                });
                [['prev', 'A◀', 'a'], ['next', 'A▶', 'a'], ['rand', 'A Rand', 'a'],
                 ['prev', 'B◀', 'b'], ['next', 'B▶', 'b'], ['rand', 'B Rand', 'b']].forEach(([act, lab, deck]) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvAction = act;
                    b.dataset.rvDeck = deck;
                    b.textContent = lab;
                    digitalToolbar.appendChild(b);
                });
                const crossGroup = document.createElement('div');
                crossGroup.className = 'radio-visual-digital-toolbar-xfade';
                const crossLbl = document.createElement('span');
                crossLbl.className = 'radio-visual-digital-xfade-label';
                crossLbl.textContent = 'XF';
                const crossDig = document.createElement('input');
                crossDig.type = 'range';
                crossDig.className = 'radio-visual-digital-vol';
                crossDig.id = 'radio-visual-cross-digital';
                crossDig.min = '0';
                crossDig.max = '1';
                crossDig.step = '0.01';
                crossDig.value = String(this._getCrossfadeX());
                crossGroup.appendChild(crossLbl);
                crossGroup.appendChild(crossDig);
                digitalToolbar.appendChild(crossGroup);
                dPanel.appendChild(digBtns);
                dPanel.appendChild(digitalCenter);
                dPanel.appendChild(digitalToolbar);
                stageD.appendChild(dPanel);

                root.appendChild(stageA);
                root.appendChild(stageD);

                this.els = {
                    btnSkinAnalog: btnA,
                    btnSkinDigital: btnD,
                    stageAnalog: stageA,
                    stageDigital: stageD,
                    stationDigital: stD,
                    ticks,
                    needle,
                    needleB,
                    tunerGlow: glow,
                    tunerGlowB: glowB,
                    tunerRail,
                    vuCanvas,
                    volKnob,
                    deckAKnob,
                    deckBKnob,
                    crossKnob,
                    autoFadeKnob,
                    autoMixKnob,
                    autoFadeReadout,
                    autoMixReadout,
                    crossDigital: crossDig,
                    btnDigitalSpectrum,
                    btnDigitalDeckB,
                    digitalCenterSpectrum,
                    digitalCenterDeckB,
                    digitalSpectrumCanvas,
                    digitalDeckBVideo,
                    volDigitalReadout,
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
                this._setDigitalCenterMode(this.digitalCenterMode);
                this._syncDeckSwitches();
                this._updateStationUi();
                this._tickClock();

                btnA.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('analogue'); }, sig);
                btnD.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('digital'); }, sig);
                volDown.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    this._stepDigitalVolume(-1);
                }, sig);
                volUp.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    this._stepDigitalVolume(1);
                }, sig);
                btnDigitalSpectrum.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    this._setDigitalCenterMode('spectrum');
                }, sig);
                btnDigitalDeckB.addEventListener('click', (ev) => {
                    this._stopClick(ev);
                    this._setDigitalCenterMode('deckB');
                }, sig);
                crossDig.addEventListener('input', () => this._setCrossfadeX(crossDig.value), sig);
                digitalToolbar.querySelectorAll('[data-rv-digital]').forEach((b) => {
                    b.addEventListener('click', (ev) => {
                        this._stopClick(ev);
                        const act = b.dataset.rvDigital;
                        if (act === 'a') this._toggleDeckA();
                        else if (act === 'b') this._toggleDeckB();
                        else if (act === 'fade') this._triggerAutoFade();
                        else if (act === 'mix') this._toggleAutoMix();
                        this._syncDeckSwitches();
                    }, sig);
                });
                digitalToolbar.querySelectorAll('[data-rv-action]').forEach((b) => {
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
                this._wireDeckKnob(deckAKnob, 'a');
                this._wireDeckKnob(deckBKnob, 'b');
                this._wireCrossfadeKnob(crossKnob);
                this._wireAutoFadeKnob(autoFadeKnob);
                this._wireAutoMixKnob(autoMixKnob);

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
                    if (this.skin === 'digital') {
                        this._drawDigitalSpectrum();
                        if (this.digitalCenterMode === 'deckB') {
                            const now = performance.now();
                            if (!this._deckBVideoSyncAt || (now - this._deckBVideoSyncAt) > 800) {
                                this._deckBVideoSyncAt = now;
                                this._syncDigitalDeckBVideo();
                            }
                        }
                    }
                } catch (_) {}
            }

            onResize() {
                try { this._resizeCanvases(); } catch (_) {}
            }

            destroy() {
                try { this._tearDownDigitalDeckBPlayer(); } catch (_) {}
                if (this._rvAutoFadeRaf) {
                    try { cancelAnimationFrame(this._rvAutoFadeRaf); } catch (_) {}
                    this._rvAutoFadeRaf = null;
                }
                if (this._rvFadeLedTimer) {
                    try { clearTimeout(this._rvFadeLedTimer); } catch (_) {}
                    this._rvFadeLedTimer = null;
                }
                this._rvFadeActive = false;
                this._rvFadeTargetDeck = null;
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
