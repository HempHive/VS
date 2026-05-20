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
                this._lastStationIdxA = -2;
                this._lastStationIdxB = -2;
                this._donutCoreHueA = 175;
                this._donutCoreHueB = 285;
                this._vuBuf = null;
                this._tuningDrag = false;
                this._volDrag = false;
                this._rvAutoFadeRaf = null;
                this.digitalCenterMode = 'spectrum';
                this._digitalDeckBView = 'video';
                this._digitalVolStep = 0.05;
                this._volMuted = false;
                this._volUnmuteNorm = 0.5;
                this._digitalStageClickTimer = null;
                this._digitalBgGifIdx = 0;
                this._digitalBgGifEnabled = true;
                /** Smoothed ring radii per band: low (base), mid, high (top). */
                this._spectrumRingSmooth = { low: null, mid: null, high: null };
                this._digitalBgGifFilesList = null;
                this._digitalBgGifManifestPromise = null;
            }

            static get AUTOFADE_MS_KEY() { return 'dj.autofade.duration.ms.v1'; }
            static get AUTOFADE_MIN_MS() { return 2000; }
            static get AUTOFADE_MAX_MS() { return 15000; }
            static get AUTOMIX_MAX_KEY() { return 'dj.automix.max.min.v1'; }
            static get AUTOMIX_ENABLED_KEY() { return 'dj.automix.enabled.v1'; }
            static get AUTOMIX_MIN_MIN() { return 1; }
            static get AUTOMIX_MAX_MIN() { return 20; }
            static get AUTOFADE_CHANGE_STATION_KEY() { return 'dj.autofade.changeStation.enabled.v1'; }
            static get DIGITAL_BG_GIF_DIR() { return 'assets/gifs/digital/'; }
            /** Fallback if assets/gifs/digital/manifest.json is missing. */
            static get DIGITAL_BG_GIF_LIST() {
                return ['dig.gif', 'diga.gif', 'digb.gif', 'digc.gif', 'digd.gif', 'dige.gif', 'digf.gif', 'digg.gif', 'digh.gif', 'digi.gif'];
            }
            static get DIGITAL_BG_GIF_MANIFEST() { return 'assets/gifs/digital/manifest.json'; }
            static get DIGITAL_BG_GIF_STORAGE_KEY() { return 'radioVisual.digitalBgGif.v1'; }
            static get DIGITAL_BG_GIF_ENABLED_KEY() { return 'radioVisual.digitalBgGif.enabled.v1'; }

            _stopClick(ev) {
                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                try { window.__suppressNextClick = true; } catch (_) {}
            }

            _stopInteraction(ev) {
                this._stopClick(ev);
            }

            /** Do not swallow pointer/click on real controls — preventDefault on pointerdown would block their click. */
            _shouldBypassRootGestureSuppression(ev) {
                try {
                    const t = ev.target;
                    if (!t || typeof t.closest !== 'function') return false;
                    if (t.closest('button')) return true;
                    if (t.closest('input')) return true;
                    if (t.closest('select')) return true;
                    if (t.closest('textarea')) return true;
                } catch (_) {}
                return false;
            }

            _sortDigitalBgGifNames(names) {
                const list = (names || [])
                    .map((f) => String(f || '').trim())
                    .filter((f) => /\.gif$/i.test(f));
                const first = list.filter((f) => f.toLowerCase() === 'dig.gif');
                const rest = list.filter((f) => f.toLowerCase() !== 'dig.gif').sort((a, b) =>
                    a.localeCompare(b, undefined, { sensitivity: 'base' })
                );
                return [...first, ...rest];
            }

            _digitalBgGifFilesFromStatic() {
                return this._sortDigitalBgGifNames(RadioVisualEngine.DIGITAL_BG_GIF_LIST);
            }

            _digitalBgGifFiles() {
                if (this._digitalBgGifFilesList && this._digitalBgGifFilesList.length) {
                    return this._digitalBgGifFilesList;
                }
                return this._digitalBgGifFilesFromStatic();
            }

            _refreshDigitalBgGifList() {
                if (this._digitalBgGifManifestPromise) return this._digitalBgGifManifestPromise;
                this._digitalBgGifManifestPromise = (async () => {
                    try {
                        const url = `${RadioVisualEngine.DIGITAL_BG_GIF_MANIFEST}?t=${Date.now()}`;
                        const res = await fetch(url, { cache: 'no-store' });
                        if (res.ok) {
                            const data = await res.json();
                            const raw = Array.isArray(data)
                                ? data
                                : (data && (data.gifs || data.files)) || [];
                            const sorted = this._sortDigitalBgGifNames(raw);
                            if (sorted.length) {
                                this._digitalBgGifFilesList = sorted;
                                if (this._digitalBgGifIdx >= sorted.length) {
                                    this._digitalBgGifIdx = 0;
                                }
                                return sorted;
                            }
                        }
                    } catch (_) {}
                    this._digitalBgGifFilesList = this._digitalBgGifFilesFromStatic();
                    return this._digitalBgGifFilesList;
                })().finally(() => {
                    this._digitalBgGifManifestPromise = null;
                });
                return this._digitalBgGifManifestPromise;
            }

            _resolveDigitalBgGifStartIndex(files) {
                if (!files.length) return 0;
                try {
                    const stored = localStorage.getItem(RadioVisualEngine.DIGITAL_BG_GIF_STORAGE_KEY);
                    const idx = files.indexOf(stored);
                    if (idx >= 0) return idx;
                } catch (_) {}
                return 0;
            }

            _applyDigitalSpectrumBgFile(filename, idx, trySkipOnError = true) {
                const bgEl = this.els.spectrumBg;
                const files = this._digitalBgGifFiles();
                if (!bgEl || !files.length) return;
                const i = (typeof idx === 'number' && idx >= 0 && idx < files.length) ? idx : 0;
                const file = filename || files[i];
                const url = RadioVisualEngine.DIGITAL_BG_GIF_DIR + file;
                const prev = bgEl.querySelector('img');
                if (prev) {
                    try { prev.remove(); } catch (_) {}
                }
                const img = document.createElement('img');
                img.className = 'radio-visual-digital-spectrum-bg-img';
                img.alt = '';
                img.decoding = 'async';
                img.onload = () => {
                    try {
                        bgEl.appendChild(img);
                        bgEl.classList.add('is-visible');
                        this._digitalBgGifIdx = i;
                        try { localStorage.setItem(RadioVisualEngine.DIGITAL_BG_GIF_STORAGE_KEY, file); } catch (_) {}
                    } catch (_) {}
                };
                img.onerror = () => {
                    if (!trySkipOnError || files.length < 2) {
                        bgEl.classList.remove('is-visible');
                        return;
                    }
                    const next = (i + 1) % files.length;
                    if (next === i) {
                        bgEl.classList.remove('is-visible');
                        return;
                    }
                    this._applyDigitalSpectrumBgFile(files[next], next, false);
                };
                img.src = url;
            }

            _isDigitalBgGifEnabled() {
                if (this._digitalBgGifEnabled === false) return false;
                try {
                    const raw = localStorage.getItem(RadioVisualEngine.DIGITAL_BG_GIF_ENABLED_KEY);
                    if (raw === '0') return false;
                    if (raw === '1') return true;
                } catch (_) {}
                return this._digitalBgGifEnabled !== false;
            }

            _setDigitalBgGifEnabled(enabled) {
                this._digitalBgGifEnabled = !!enabled;
                try {
                    localStorage.setItem(
                        RadioVisualEngine.DIGITAL_BG_GIF_ENABLED_KEY,
                        enabled ? '1' : '0'
                    );
                } catch (_) {}
                const bgEl = this.els.spectrumBg;
                if (!bgEl) return;
                if (!enabled) {
                    bgEl.classList.remove('is-visible');
                    const img = bgEl.querySelector('img');
                    if (img) {
                        try { img.remove(); } catch (_) {}
                    }
                    return;
                }
                this._refreshDigitalBgGifList().then(() => {
                    this._applyCurrentDigitalBgGif();
                }).catch(() => {});
            }

            _toggleDigitalBgGifEnabled() {
                this._setDigitalBgGifEnabled(!this._isDigitalBgGifEnabled());
            }

            _syncDigitalVisBgButton() {
                const btn = this.els.btnVis;
                if (!btn) return;
                const on = this._isDigitalBgGifEnabled();
                btn.classList.toggle('is-active', on);
                btn.setAttribute('aria-pressed', on ? 'true' : 'false');
                btn.title = on
                    ? 'Tap: next background · Hold: turn off background'
                    : 'Background off · Tap to turn on';
            }

            _applyCurrentDigitalBgGif() {
                const files = this._digitalBgGifFiles();
                if (!this.els.spectrumBg || !files.length) return;
                const idx = Math.max(0, Math.min(files.length - 1, this._digitalBgGifIdx || 0));
                this._applyDigitalSpectrumBgFile(files[idx], idx);
            }

            _initDigitalSpectrumBg() {
                try {
                    const raw = localStorage.getItem(RadioVisualEngine.DIGITAL_BG_GIF_ENABLED_KEY);
                    if (raw === '0') this._digitalBgGifEnabled = false;
                    else if (raw === '1') this._digitalBgGifEnabled = true;
                } catch (_) {}
                this._syncDigitalVisBgButton();
                if (!this._isDigitalBgGifEnabled()) return;
                this._refreshDigitalBgGifList().then(() => {
                    if (!this._isDigitalBgGifEnabled()) return;
                    const files = this._digitalBgGifFiles();
                    if (!files.length) return;
                    const start = this._resolveDigitalBgGifStartIndex(files);
                    this._applyDigitalSpectrumBgFile(files[start], start);
                }).catch(() => {});
            }

            _cycleDigitalVisBg() {
                this._refreshDigitalBgGifList().then(() => {
                    if (!this._isDigitalBgGifEnabled()) return;
                    const files = this._digitalBgGifFiles();
                    if (!this.els.spectrumBg || !files.length) return;
                    const next = (this._digitalBgGifIdx + 1) % files.length;
                    this._applyDigitalSpectrumBgFile(files[next], next);
                }).catch(() => {});
            }

            _onDigitalVisBgTap() {
                if (!this._isDigitalBgGifEnabled()) {
                    this._setDigitalBgGifEnabled(true);
                    this._syncDigitalVisBgButton();
                    return;
                }
                this._cycleDigitalVisBg();
            }

            _wireDigitalVisBgButton(btn, sig) {
                if (!btn) return;
                let longPressTimer = null;
                let longPressHandled = false;
                const longPressMs = 500;
                const clearLongPress = () => {
                    if (longPressTimer) {
                        clearTimeout(longPressTimer);
                        longPressTimer = null;
                    }
                };
                btn.addEventListener('pointerdown', (ev) => {
                    this._stopClick(ev);
                    longPressHandled = false;
                    clearLongPress();
                    longPressTimer = setTimeout(() => {
                        longPressTimer = null;
                        longPressHandled = true;
                        this._toggleDigitalBgGifEnabled();
                        this._syncDigitalVisBgButton();
                    }, longPressMs);
                }, sig);
                btn.addEventListener('pointerup', (ev) => {
                    this._stopClick(ev);
                    clearLongPress();
                    if (!longPressHandled) this._onDigitalVisBgTap();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('pointercancel', () => {
                    clearLongPress();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _loadVisualByName(name) {
                try {
                    if (!Array.isArray(modes) || typeof loadMode !== 'function') return;
                    const idx = modes.findIndex((m) => m && m.name === name);
                    if (idx >= 0) {
                        try { globalThis.__vizLaunchedFromRadioVisual = true; } catch (_) {}
                        loadMode(idx);
                    }
                } catch (_) {}
            }

            _isDigitalDeckBPanelOpen() {
                return this.skin === 'digital' && this.digitalCenterMode === 'deckB';
            }

            _deckBFeature(kind, fallback) {
                if (this._isDigitalDeckBPanelOpen()) {
                    this._setDigitalDeckBView(kind);
                    return;
                }
                try { fallback(); } catch (_) {}
            }

            _tearDownDigitalDeckBView() {
                try {
                    if (this._rvDigitalPmAnimId) cancelAnimationFrame(this._rvDigitalPmAnimId);
                } catch (_) {}
                this._rvDigitalPmAnimId = null;
                try {
                    if (this._rvDigitalPmCycleTimeout) clearTimeout(this._rvDigitalPmCycleTimeout);
                } catch (_) {}
                this._rvDigitalPmCycleTimeout = null;
                try {
                    if (this._rvDigitalBarsAnimId) cancelAnimationFrame(this._rvDigitalBarsAnimId);
                } catch (_) {}
                this._rvDigitalBarsAnimId = null;
                this._rvDigitalPmVisualizer = null;
                this._rvDigitalPmCanvas = null;
                this._rvDigitalPmResize = null;
                this._rvDigitalBarsRenderer = null;
                this._rvDigitalBarsScene = null;
                if (this.els.digitalDeckBContent) {
                    this.els.digitalDeckBContent.innerHTML = '';
                    this.els.digitalDeckBContent.classList.remove('is-active');
                }
                if (this.els.digitalDeckBVideo) {
                    this.els.digitalDeckBVideo.classList.remove('is-hidden');
                }
            }

            _setDigitalDeckBView(view) {
                const next = (view === 'projectm' || view === 'bars' || view === 'queue') ? view : 'video';
                this._digitalDeckBView = next;
                if (next === 'video') {
                    this._tearDownDigitalDeckBView();
                    if (this.els.digitalDeckBVideo) this.els.digitalDeckBVideo.classList.remove('is-hidden');
                    this._syncDigitalDeckBVideo();
                    return;
                }
                if (this.els.digitalDeckBVideo) this.els.digitalDeckBVideo.classList.add('is-hidden');
                if (next === 'projectm') this._showDigitalDeckBProjectM();
                else if (next === 'bars') this._showDigitalDeckBAudioBars();
                else if (next === 'queue') this._showDigitalDeckBQueue();
            }

            _showDigitalDeckBProjectM() {
                const mount = this.els.digitalDeckBContent;
                if (!mount) return;
                this._tearDownDigitalDeckBView();
                mount.classList.add('is-active');
                try { initAudio(); } catch (_) {}
                if (!state || !state.audioCtx || typeof butterchurn === 'undefined') return;
                const canvas = document.createElement('canvas');
                canvas.className = 'radio-visual-digital-deck-b-canvas';
                mount.appendChild(canvas);
                const presetMap = (typeof butterchurnPresets !== 'undefined' && butterchurnPresets.getPresets)
                    ? butterchurnPresets.getPresets() : {};
                const presetKeys = Object.keys(presetMap);
                let currentPresetIdx = Math.floor(Math.random() * Math.max(1, presetKeys.length));
                const resizePm = () => {
                    try {
                        const rect = mount.getBoundingClientRect();
                        const w = Math.max(64, Math.floor(rect.width));
                        const h = Math.max(64, Math.floor(rect.height));
                        const dpr = Math.min(window.devicePixelRatio || 1, 2);
                        canvas.width = Math.floor(w * dpr);
                        canvas.height = Math.floor(h * dpr);
                        if (this._rvDigitalPmVisualizer && this._rvDigitalPmVisualizer.setRendererSize) {
                            this._rvDigitalPmVisualizer.setRendererSize(w, h);
                        }
                    } catch (_) {}
                };
                const rect0 = mount.getBoundingClientRect();
                const w0 = Math.max(64, Math.floor(rect0.width));
                const h0 = Math.max(64, Math.floor(rect0.height));
                const dpr0 = Math.min(window.devicePixelRatio || 1, 2);
                canvas.width = Math.floor(w0 * dpr0);
                canvas.height = Math.floor(h0 * dpr0);
                const pxRatio = Number(visualSettings.pixelRatio) || Math.min(window.devicePixelRatio || 1, 2);
                let viz;
                try {
                    viz = butterchurn.createVisualizer(state.audioCtx, canvas, {
                        width: w0, height: h0, pixelRatio: pxRatio
                    });
                } catch (_) {
                    viz = null;
                }
                if (!viz) return;
                this._rvDigitalPmVisualizer = viz;
                this._rvDigitalPmCanvas = canvas;
                try { viz.connectAudio(state.analyserNode); } catch (_) {}
                const loadPmPreset = (idx) => {
                    if (!presetKeys.length || !this._rvDigitalPmVisualizer) return;
                    try {
                        this._rvDigitalPmVisualizer.loadPreset(
                            presetMap[presetKeys[idx]],
                            Number(visualSettings.transitionSec) || 2.7
                        );
                    } catch (_) {}
                };
                loadPmPreset(currentPresetIdx);
                const nextPmPreset = () => {
                    if (presetKeys.length <= 1) return;
                    let next = currentPresetIdx;
                    let guard = 0;
                    while (next === currentPresetIdx && guard++ < 8) {
                        next = Math.floor(Math.random() * presetKeys.length);
                    }
                    currentPresetIdx = next;
                    loadPmPreset(currentPresetIdx);
                };
                const schedule = () => {
                    const minS = Number(visualSettings.shuffleMinSec) || 12;
                    const maxS = Number(visualSettings.shuffleMaxSec) || 25;
                    const lo = Math.min(minS, maxS);
                    const hi = Math.max(minS, maxS);
                    const delay = (lo + Math.random() * (hi - lo)) * 1000;
                    this._rvDigitalPmCycleTimeout = setTimeout(() => {
                        nextPmPreset();
                        schedule();
                    }, delay);
                };
                schedule();
                this._rvDigitalPmResize = resizePm;
                resizePm();
                const loop = () => {
                    this._rvDigitalPmAnimId = requestAnimationFrame(loop);
                    try {
                        if (this._rvDigitalPmVisualizer) this._rvDigitalPmVisualizer.render();
                    } catch (_) {}
                };
                loop();
            }

            _showDigitalDeckBAudioBars() {
                const mount = this.els.digitalDeckBContent;
                if (!mount || typeof THREE === 'undefined' || typeof sceneBars !== 'function') return;
                this._tearDownDigitalDeckBView();
                mount.classList.add('is-active');
                try { initAudio(); } catch (_) {}
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
                camera.position.z = 8;
                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                const canvas = renderer.domElement;
                canvas.className = 'radio-visual-digital-deck-b-canvas';
                mount.appendChild(canvas);
                const updateFn = sceneBars(scene, camera);
                let dataArray = null;
                const resizeBars = () => {
                    try {
                        const rect = mount.getBoundingClientRect();
                        const w = Math.max(64, Math.floor(rect.width));
                        const h = Math.max(64, Math.floor(rect.height));
                        camera.aspect = w / h;
                        camera.updateProjectionMatrix();
                        renderer.setSize(w, h, false);
                    } catch (_) {}
                };
                this._rvDigitalBarsRenderer = renderer;
                this._rvDigitalBarsScene = scene;
                this._rvDigitalPmResize = resizeBars;
                resizeBars();
                const loop = () => {
                    this._rvDigitalBarsAnimId = requestAnimationFrame(loop);
                    try {
                        const an = state.analyserNode || state.analyserNodeA;
                        if (!an) return;
                        if (!dataArray || dataArray.length !== an.frequencyBinCount) {
                            dataArray = new Uint8Array(an.frequencyBinCount);
                        }
                        an.getByteFrequencyData(dataArray);
                        if (updateFn) updateFn(dataArray, performance.now());
                        renderer.render(scene, camera);
                    } catch (_) {}
                };
                loop();
            }

            _showDigitalDeckBQueue() {
                const mount = this.els.digitalDeckBContent;
                if (!mount) return;
                this._tearDownDigitalDeckBView();
                mount.classList.add('is-active');
                const wrap = document.createElement('div');
                wrap.className = 'radio-visual-digital-deck-b-queue';
                const title = document.createElement('div');
                title.className = 'radio-visual-digital-deck-b-queue-title';
                title.textContent = 'Deck B queue';
                const list = document.createElement('ul');
                list.className = 'radio-visual-digital-deck-b-queue-list';
                try {
                    const q = (typeof deckFileQueues !== 'undefined' && deckFileQueues.b) ? deckFileQueues.b : [];
                    if (!q.length) {
                        const li = document.createElement('li');
                        li.textContent = 'Queue empty';
                        list.appendChild(li);
                    } else {
                        q.forEach((item, idx) => {
                            const li = document.createElement('li');
                            li.textContent = item && item.name ? String(item.name) : `Track ${idx + 1}`;
                            list.appendChild(li);
                        });
                    }
                } catch (_) {
                    const li = document.createElement('li');
                    li.textContent = 'Queue unavailable';
                    list.appendChild(li);
                }
                wrap.appendChild(title);
                wrap.appendChild(list);
                mount.appendChild(wrap);
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
                if (this._isRadioVisualActive()) this._updateHudModeLines();
            }

            _crossfaderAudibleDeckKey() {
                const x = this._getCrossfadeX();
                return x >= 0.5 ? 'b' : 'a';
            }

            _stationNameForDeck(deck) {
                try {
                    if (!Array.isArray(stations) || !stations.length) return '—';
                    if (deck === 'b') {
                        const idx = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                            ? currentStationBIndex : -1;
                        if (idx >= 0 && stations[idx]) return stations[idx].name || '—';
                    } else {
                        const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                            ? currentStationIndex : -1;
                        if (idx >= 0 && stations[idx]) return stations[idx].name || '—';
                    }
                } catch (_) {}
                return '—';
            }

            _updateHudModeLines() {
                if (!this._isRadioVisualActive()) return;
                try {
                    if (typeof globalThis.updateModeSubStationLine === 'function') {
                        globalThis.updateModeSubStationLine();
                        return;
                    }
                    const titleEl = document.getElementById('mode-title');
                    const subEl = document.getElementById('mode-sub');
                    const visName = (this.name === 'Radio') ? 'Radio Visual' : (this.name || 'Radio');
                    if (titleEl) titleEl.textContent = visName;
                    if (subEl) {
                        const deck = this._crossfaderAudibleDeckKey();
                        subEl.textContent = this._stationNameForDeck(deck);
                    }
                } catch (_) {}
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
                this._retuneIncomingDeckForAutoFade(targetDeck);
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
                const next = !this._isAutoFadeChangeStationEnabled();
                try {
                    localStorage.setItem(RadioVisualEngine.AUTOFADE_CHANGE_STATION_KEY, next ? '1' : '0');
                } catch (_) {}
                const cb = document.getElementById('dj-autofade-change-station');
                if (cb) cb.checked = next;
                this._syncAutoFadeChangeStationKnob();
            }

            _retuneIncomingDeckForAutoFade(targetDeck) {
                if (!this._isAutoFadeChangeStationEnabled()) return;
                try {
                    if (targetDeck === 'b') {
                        if (typeof pickRandomStationB === 'function') pickRandomStationB();
                    } else if (typeof pickRandomStation === 'function') pickRandomStation();
                } catch (_) {}
            }

            _syncAutoMixKnob() {
                if (!this.els.autoMixKnob) return;
                const on = this._isAutoMixEnabled();
                this.els.autoMixKnob.classList.toggle('is-on', on);
                this.els.autoMixKnob.setAttribute('aria-pressed', on ? 'true' : 'false');
            }

            _syncAutoFadeChangeStationKnob() {
                const on = this._isAutoFadeChangeStationEnabled();
                if (this.els.autoFadeKnob) {
                    this.els.autoFadeKnob.classList.toggle('is-on', on);
                    this.els.autoFadeKnob.setAttribute('aria-pressed', on ? 'true' : 'false');
                }
                if (this.els.btnXfadeStation) {
                    this.els.btnXfadeStation.classList.toggle('is-active', on);
                    this.els.btnXfadeStation.setAttribute('aria-pressed', on ? 'true' : 'false');
                }
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

            _deckEngCancelAutoFade() {
                try {
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    if (eng && typeof eng.cancelAutoFade === 'function') eng.cancelAutoFade();
                } catch (_) {}
            }

            _deckEngClearSuppress() {
                try {
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                        eng.clearSuppressEnsureCrossfadeDeckPlayback();
                    }
                } catch (_) {}
            }

            async _startDeckA() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    this._deckEngCancelAutoFade();
                    const media = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    if (!media || !this._deckHasSource(media)) {
                        this._deckEngClearSuppress();
                        if (typeof playRadio === 'function') playRadio();
                        return;
                    }
                    if (media.paused) {
                        this._deckEngClearSuppress();
                        await media.play().catch(() => {
                            try { if (typeof playRadio === 'function') playRadio(); } catch (_) {}
                        });
                    }
                } catch (_) {}
            }

            async _stopDeckA() {
                try {
                    this._deckEngCancelAutoFade();
                    this._deckEngClearSuppress();
                    const media = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    if (media && !media.paused) media.pause();
                } catch (_) {}
            }

            async _startDeckB() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    this._deckEngCancelAutoFade();
                    if (!audioElB || !this._deckHasSource(audioElB) || audioElB.paused) {
                        this._deckEngClearSuppress();
                        if (typeof playRadioB === 'function') playRadioB();
                    }
                } catch (_) {}
            }

            async _stopDeckB() {
                try {
                    this._deckEngCancelAutoFade();
                    this._deckEngClearSuppress();
                    if (audioElB && !audioElB.paused) audioElB.pause();
                } catch (_) {}
            }

            async _toggleDeckA() {
                if (this._deckAActive()) await this._stopDeckA();
                else await this._startDeckA();
            }

            async _toggleDeckB() {
                if (this._deckBActive()) await this._stopDeckB();
                else await this._startDeckB();
            }

            async _deckKnobTap(deck) {
                const on = deck === 'a' ? this._deckAActive() : this._deckBActive();
                if (!on) {
                    if (deck === 'a') await this._startDeckA();
                    else await this._startDeckB();
                } else if (deck === 'a') {
                    this._stationRand();
                } else {
                    this._stationBRand();
                }
            }

            async _deckKnobLongPress(deck) {
                if (deck === 'a') {
                    if (this._deckAActive()) await this._stopDeckA();
                } else if (this._deckBActive()) {
                    await this._stopDeckB();
                }
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
                if (next === 'deckB') {
                    this._setDigitalDeckBView(this._digitalDeckBView || 'video');
                } else {
                    this._tearDownDigitalDeckBView();
                }
            }

            _tearDownDigitalDeckBPlayer() {
                try { this._tearDownDigitalDeckBView(); } catch (_) {}
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

            /** Blend first/last bins so the closed ribbon has no cliff at the seam. */
            _blendSpectrumCircularEnds(values, width = 7) {
                const n = values.length;
                if (n < 4) return values;
                const w = Math.min(width, Math.floor(n / 4));
                const out = values.slice();
                let acc = 0;
                for (let k = 0; k < w; k++) acc += out[k] + out[n - 1 - k];
                const seam = acc / (2 * w);
                out[0] = seam;
                out[n - 1] = seam;
                for (let k = 1; k < w; k++) {
                    const blend = 1 - (k / w) * (k / w);
                    out[k] = out[k] * (1 - blend) + seam * blend;
                    out[n - 1 - k] = out[n - 1 - k] * (1 - blend) + seam * blend;
                }
                return out;
            }

            _unifySpectrumRibbonSeam(radii) {
                return this._blendSpectrumCircularEnds(radii, 8);
            }

            static get SPECTRUM_ANGULAR_BINS() { return 72; }

            static get SPECTRUM_FLOWER_LAYERS() {
                return [
                    { key: 'low', layerIndex: 0, phaseBins: 0, hueOff: -42, alpha: 0.5, maxRing: 0.62 },
                    { key: 'mid', layerIndex: 1, phaseBins: 2.5, hueOff: 58, alpha: 0.58, maxRing: 0.64 },
                    { key: 'high', layerIndex: 2, phaseBins: 5, hueOff: 128, alpha: 0.72, maxRing: 0.66 }
                ];
            }

            _sampleDigitalSpectrumBandLevels(t) {
                const n = RadioVisualEngine.SPECTRUM_ANGULAR_BINS;
                const bands = 3;
                const out = { low: [], mid: [], high: [], fromAnalyser: false };
                const keys = ['low', 'mid', 'high'];
                try {
                    if (state.analyserNode && state.audioCtx) {
                        const fft = state.analyserNode.fftSize || 256;
                        if (!this._vuBuf || this._vuBuf.length !== fft) {
                            this._vuBuf = new Uint8Array(fft);
                        }
                        state.analyserNode.getByteFrequencyData(this._vuBuf);
                        for (let b = 0; b < bands; b++) {
                            const start = Math.floor(b * this._vuBuf.length / bands);
                            const end = Math.floor((b + 1) * this._vuBuf.length / bands);
                            const span = Math.max(1, end - start);
                            const levels = [];
                            for (let i = 0; i < n; i++) {
                                const f = start + (i / Math.max(1, n - 1)) * span;
                                const i0 = Math.floor(f);
                                const i1 = Math.min(this._vuBuf.length - 1, i0 + 1);
                                const tt = f - i0;
                                const v = (this._vuBuf[i0] || 0) * (1 - tt) + (this._vuBuf[i1] || 0) * tt;
                                levels.push(v / 255);
                            }
                            out[keys[b]] = levels;
                        }
                        out.fromAnalyser = true;
                        return out;
                    }
                } catch (_) {}
                const phase = { low: 0, mid: 1.15, high: 2.35 };
                for (let b = 0; b < bands; b++) {
                    const levels = [];
                    const ph = phase[keys[b]];
                    for (let i = 0; i < n; i++) {
                        levels.push(
                            0.1 + 0.09 * Math.sin(t * (2.1 + b * 0.35) + i * 0.42 + ph)
                            + 0.04 * Math.sin(t * 4.2 + i * 0.17 + ph * 2)
                        );
                    }
                    out[keys[b]] = levels;
                }
                return out;
            }

            _smoothSpectrumBandLevels(levels, fromAnalyser) {
                const n = levels.length;
                let lo = 0;
                let hi = n - 1;
                if (fromAnalyser) {
                    let peak = 0;
                    for (let i = 0; i < n; i++) peak = Math.max(peak, levels[i] || 0);
                    const thresh = Math.max(0.02, peak * 0.12);
                    while (lo < n && (levels[lo] || 0) < thresh) lo++;
                    while (hi > lo && (levels[hi] || 0) < thresh) hi--;
                    const minSpan = Math.max(10, Math.floor(n * 0.22));
                    if (hi - lo < minSpan) {
                        const mid = Math.floor((lo + hi) * 0.5);
                        lo = Math.max(0, mid - Math.floor(minSpan / 2));
                        hi = Math.min(n - 1, lo + minSpan - 1);
                    }
                    const margin = Math.max(1, Math.floor((hi - lo + 1) * 0.06));
                    lo = Math.max(0, lo - margin);
                    hi = Math.min(n - 1, hi + margin);
                }
                const span = Math.max(1, hi - lo);
                const band = [];
                for (let i = 0; i < n; i++) {
                    const f = lo + (i / Math.max(1, n - 1)) * span;
                    const i0 = Math.floor(f);
                    const i1 = Math.min(n - 1, i0 + 1);
                    const tt = f - i0;
                    const v = (levels[i0] || 0) * (1 - tt) + (levels[i1] || 0) * tt;
                    band.push(v);
                }
                const bandSeam = this._blendSpectrumCircularEnds(band, 6);
                const smooth = [];
                for (let i = 0; i < n; i++) {
                    const im1 = (i - 1 + n) % n;
                    const im2 = (i - 2 + n) % n;
                    const ip1 = (i + 1) % n;
                    const ip2 = (i + 2) % n;
                    smooth.push(
                        (bandSeam[im2] + 2 * bandSeam[im1] + 3 * bandSeam[i] + 2 * bandSeam[ip1] + bandSeam[ip2]) / 9
                    );
                }
                return smooth;
            }

            _radiiFromSpectrumSmooth(smooth, t, bandKey, layerOpts) {
                const n = smooth.length;
                const sorted = smooth.slice().sort((a, b) => a - b);
                const p78 = sorted[Math.min(n - 1, Math.floor(n * 0.78))] || 0.001;
                const norm = 1 / Math.max(0.05, p78);
                const gain = 0.76;
                const gamma = 1.26;
                const maxRing = layerOpts.maxRing ?? 0.64;
                const floor = 0.04;
                const target = [];
                for (let i = 0; i < n; i++) {
                    const raw = Math.min(1, (smooth[i] || 0) * norm * gain);
                    let shaped = Math.pow(raw, gamma);
                    shaped += 0.005 * Math.sin(t * 1.85 + i * 0.11 + layerOpts.layerIndex * 0.9);
                    target.push(Math.min(maxRing, Math.max(floor, shaped)));
                }
                if (!this._spectrumRingSmooth[bandKey] || this._spectrumRingSmooth[bandKey].length !== n) {
                    this._spectrumRingSmooth[bandKey] = target.slice();
                }
                const radii = [];
                const attack = 0.28;
                const release = 0.12;
                const smoothState = this._spectrumRingSmooth[bandKey];
                for (let i = 0; i < n; i++) {
                    const tgt = target[i];
                    const prev = smoothState[i] ?? floor;
                    const k = tgt > prev ? attack : release;
                    const next = prev + (tgt - prev) * k;
                    smoothState[i] = next;
                    radii.push(next);
                }
                return this._unifySpectrumRibbonSeam(radii);
            }

            _mirrorSpectrumRadii(radii) {
                const n = radii.length;
                return radii.map((_, i) => radii[(n - 1 - i) % n]);
            }

            _computeDigitalSpectrumRadiiAndEq() {
                const t = performance.now() * 0.001;
                const sampled = this._sampleDigitalSpectrumBandLevels(t);
                const n = RadioVisualEngine.SPECTRUM_ANGULAR_BINS;
                const layersL = [];
                const layersR = [];
                const smoothMix = [];
                for (const layer of RadioVisualEngine.SPECTRUM_FLOWER_LAYERS) {
                    const levels = sampled[layer.key] || [];
                    const smooth = this._smoothSpectrumBandLevels(levels, sampled.fromAnalyser);
                    const radii = this._radiiFromSpectrumSmooth(smooth, t, layer.key, layer);
                    layersL.push({ ...layer, radii });
                    layersR.push({ ...layer, radii: this._mirrorSpectrumRadii(radii) });
                    for (let i = 0; i < n; i++) smoothMix[i] = (smoothMix[i] || 0) + (smooth[i] || 0);
                }
                for (let i = 0; i < n; i++) smoothMix[i] /= 3;
                const eqBars = 32;
                const eqHeights = [];
                for (let b = 0; b < eqBars; b++) {
                    const u = (b + 0.5) / eqBars;
                    const idx = u * (n - 1);
                    const i0 = Math.floor(idx);
                    const tt = idx - i0;
                    const sm = smoothMix[i0] * (1 - tt) + smoothMix[Math.min(n - 1, i0 + 1)] * tt;
                    eqHeights.push(Math.min(0.9, Math.max(0.04, Math.pow(sm * 2.05, 0.72))));
                }
                return { n, layersL, layersR, eqHeights, t };
            }

            _donutHueForStationIndex(idx) {
                if (!Array.isArray(stations) || idx < 0 || !stations[idx]) {
                    return Math.floor(Math.random() * 360);
                }
                let h = 0;
                const s = String(stations[idx].name || '') + '|' + String(stations[idx].url || '') + '|' + idx;
                for (let i = 0; i < s.length; i++) {
                    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
                }
                return ((h % 360) + 360) % 360;
            }

            _syncDonutCoreHues() {
                const idxA = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                    ? currentStationIndex : -1;
                const idxB = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                    ? currentStationBIndex : -1;
                if (idxA !== this._lastStationIdxA) {
                    this._lastStationIdxA = idxA;
                    this._donutCoreHueA = this._donutHueForStationIndex(idxA);
                }
                if (idxB !== this._lastStationIdxB) {
                    this._lastStationIdxB = idxB;
                    this._donutCoreHueB = this._donutHueForStationIndex(idxB);
                }
            }

            _spectrumAngle(ii, n, phaseBins = 0) {
                const phase = (phaseBins / n) * Math.PI * 2;
                return ((ii + 0.5) / n) * Math.PI * 2 - Math.PI / 2 + phase;
            }

            _fillDigitalSpectrumPetal(ctx, cx, cy, innerR, outerR, radii, n, layer, coreHue) {
                const li = layer.layerIndex;
                const layerCount = 3;
                const span = (outerR - innerR) / layerCount;
                const zoneInner = innerR + li * span;
                const zoneOuter = innerR + (li + 1) * span;
                const petalFloor = 0.1;
                const hue = (((Number(coreHue) || 0) + layer.hueOff) % 360 + 360) % 360;
                ctx.beginPath();
                for (let i = 0; i <= n; i++) {
                    const ii = i % n;
                    const a = this._spectrumAngle(ii, n, layer.phaseBins);
                    const norm = Math.min(1, Math.max(petalFloor, radii[ii] / (layer.maxRing || 0.64)));
                    const r = zoneInner + (zoneOuter - zoneInner) * norm;
                    const x = cx + Math.cos(a) * r;
                    const y = cy + Math.sin(a) * r;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                const innerPad = zoneInner + span * 0.06;
                for (let i = n - 1; i >= 0; i--) {
                    const a = this._spectrumAngle(i, n, layer.phaseBins);
                    ctx.lineTo(cx + Math.cos(a) * innerPad, cy + Math.sin(a) * innerPad);
                }
                ctx.closePath();
                const petal = ctx.createRadialGradient(cx, cy, zoneInner, cx, cy, zoneOuter);
                petal.addColorStop(0, `hsla(${hue}, 88%, 58%, ${layer.alpha * 0.55})`);
                petal.addColorStop(0.55, `hsla(${(hue + 18) % 360}, 92%, 52%, ${layer.alpha})`);
                petal.addColorStop(1, `hsla(${(hue + 36) % 360}, 85%, 42%, ${layer.alpha * 0.85})`);
                ctx.fillStyle = petal;
                ctx.fill();
                ctx.strokeStyle = `hsla(${hue}, 90%, 70%, ${layer.alpha * 0.35})`;
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            _drawDigitalSpectrumFlower(canvas, layers, n, coreHue = 175) {
                if (!canvas || !layers || !layers.length) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 8 || h < 8) return;
                const cx = w * 0.5;
                const cy = h * 0.5;
                ctx.clearRect(0, 0, w, h);
                const innerR = Math.min(w, h) * 0.14;
                const outerR = Math.min(w, h) * 0.44;
                const coreR = innerR * 1.08;
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
                ctx.lineWidth = 1;
                for (let ring = 1; ring <= 3; ring++) {
                    ctx.beginPath();
                    ctx.arc(cx, cy, innerR + ((outerR - innerR) * ring) / 3, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                const ordered = layers.slice().sort((a, b) => a.layerIndex - b.layerIndex);
                for (const layer of ordered) {
                    if (!layer.radii || !layer.radii.length) continue;
                    this._fillDigitalSpectrumPetal(ctx, cx, cy, innerR, outerR, layer.radii, n, layer, coreHue);
                }
                const hue = ((Number(coreHue) || 0) % 360 + 360) % 360;
                ctx.beginPath();
                ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
                const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
                core.addColorStop(0, `hsla(${hue}, 88%, 62%, 0.88)`);
                core.addColorStop(0.45, `hsla(${hue}, 76%, 40%, 0.78)`);
                core.addColorStop(1, `hsla(${(hue + 28) % 360}, 68%, 24%, 0.62)`);
                ctx.fillStyle = core;
                ctx.fill();
            }

            _drawDigitalCarDash(eqHeights, t) {
                const canvas = this.els.digitalCarDashCanvas;
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 16 || h < 24) return;
                const roundRectPath = (x0, y0, rw, rh, rad) => {
                    const rr = Math.min(rad, rw * 0.5, rh * 0.5);
                    ctx.beginPath();
                    ctx.moveTo(x0 + rr, y0);
                    ctx.lineTo(x0 + rw - rr, y0);
                    ctx.quadraticCurveTo(x0 + rw, y0, x0 + rw, y0 + rr);
                    ctx.lineTo(x0 + rw, y0 + rh - rr);
                    ctx.quadraticCurveTo(x0 + rw, y0 + rh, x0 + rw - rr, y0 + rh);
                    ctx.lineTo(x0 + rr, y0 + rh);
                    ctx.quadraticCurveTo(x0, y0 + rh, x0, y0 + rh - rr);
                    ctx.lineTo(x0, y0 + rr);
                    ctx.quadraticCurveTo(x0, y0, x0 + rr, y0);
                    ctx.closePath();
                };
                const dashDim = 0.5;
                ctx.clearRect(0, 0, w, h);
                const m = Math.max(2, Math.min(w, h) * 0.04);
                const bw = w - m * 2;
                const bh = h - m * 2;
                const bx = m;
                const by = m;
                const bezelR = Math.min(14, bh * 0.12);
                const inset = Math.max(3, m * 0.55);
                const ix = bx + inset;
                const iy = by + inset;
                const iw = bw - inset * 2;
                const ih = bh - inset * 2;
                const ir = bezelR * 0.65;
                const lcdH = ih * 0.34;
                const lcdY = iy + ih * 0.06;
                const lcdPad = iw * 0.06;
                const eqTop = lcdY + lcdH + ih * 0.07;
                ctx.save();
                ctx.globalAlpha = dashDim;
                const chassis = ctx.createLinearGradient(bx, by, bx + bw, by + bh);
                chassis.addColorStop(0, '#1c1a22');
                chassis.addColorStop(0.45, '#0a090e');
                chassis.addColorStop(1, '#252230');
                ctx.fillStyle = chassis;
                roundRectPath(bx, by, bw, bh, bezelR);
                ctx.fill();
                ctx.strokeStyle = 'rgba(200, 200, 220, 0.22)';
                ctx.lineWidth = Math.max(1, w / 200);
                ctx.stroke();
                const recess = ctx.createLinearGradient(ix, iy, ix, iy + ih);
                recess.addColorStop(0, '#08070a');
                recess.addColorStop(1, '#121018');
                ctx.fillStyle = recess;
                roundRectPath(ix, iy, iw, ih, ir);
                ctx.fill();
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)';
                ctx.lineWidth = Math.max(1, w / 240);
                ctx.stroke();
                const lcdGrad = ctx.createLinearGradient(ix + lcdPad, lcdY, ix + iw - lcdPad, lcdY + lcdH);
                lcdGrad.addColorStop(0, '#061a14');
                lcdGrad.addColorStop(0.5, '#0a2818');
                lcdGrad.addColorStop(1, '#051210');
                ctx.fillStyle = lcdGrad;
                roundRectPath(ix + lcdPad, lcdY, iw - lcdPad * 2, lcdH, ir * 0.5);
                ctx.fill();
                ctx.strokeStyle = 'rgba(0, 255, 160, 0.12)';
                ctx.stroke();
                ctx.save();
                ctx.beginPath();
                roundRectPath(ix + lcdPad, lcdY, iw - lcdPad * 2, lcdH, ir * 0.5);
                ctx.clip();
                for (let gx = 0; gx < iw; gx += 3) {
                    const a = 0.04 + 0.03 * Math.sin(t * 1.2 + gx * 0.08);
                    ctx.fillStyle = `rgba(0, 255, 180, ${a})`;
                    ctx.fillRect(ix + lcdPad + gx, lcdY, 2, lcdH);
                }
                ctx.restore();
                ctx.font = `700 ${Math.max(8, Math.min(w, h) * 0.075)}px ui-monospace, Menlo, Consolas, monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = 'rgba(0, 255, 200, 0.55)';
                ctx.shadowColor = 'rgba(0, 255, 200, 0.35)';
                ctx.shadowBlur = 6;
                ctx.fillText('DIGITAL TUNER', ix + iw * 0.5, lcdY + lcdH * 0.38);
                ctx.shadowBlur = 0;
                ctx.font = `600 ${Math.max(7, Math.min(w, h) * 0.055)}px ui-monospace, Menlo, Consolas, monospace`;
                ctx.fillStyle = 'rgba(120, 255, 200, 0.35)';
                ctx.fillText('STEREO · EQ', ix + iw * 0.5, lcdY + lcdH * 0.72);
                ctx.restore();
                const eqH = Math.max(ih * 0.38, h * 0.22);
                const eqW = iw - lcdPad * 2;
                const eqX = ix + lcdPad;
                const eqR = ir * 0.45;
                const eqBg = ctx.createLinearGradient(eqX, eqTop, eqX, eqTop + eqH);
                eqBg.addColorStop(0, '#040508');
                eqBg.addColorStop(1, '#0a1018');
                ctx.fillStyle = eqBg;
                roundRectPath(eqX, eqTop, eqW, eqH, eqR);
                ctx.fill();
                ctx.strokeStyle = 'rgba(0, 200, 255, 0.15)';
                ctx.stroke();
                const nb = eqHeights.length;
                const gap = Math.max(1, eqW * 0.012);
                const barW = (eqW - gap * (nb + 1)) / nb;
                const baseY = eqTop + eqH - gap * 2;
                const maxBar = eqH - gap * 3;
                for (let i = 0; i < nb; i++) {
                    const x = eqX + gap + i * (barW + gap);
                    const ht = maxBar * eqHeights[i];
                    const y = baseY - ht;
                    const hue = (i / nb) * 280;
                    const barG = ctx.createLinearGradient(x, y, x, baseY);
                    barG.addColorStop(0, `hsla(${hue}, 95%, 62%, 0.95)`);
                    barG.addColorStop(0.55, `hsla(${hue + 40}, 88%, 48%, 0.88)`);
                    barG.addColorStop(1, `hsla(${hue + 80}, 75%, 28%, 0.75)`);
                    ctx.fillStyle = barG;
                    const bw2 = Math.max(1, barW);
                    ctx.fillRect(x, y, bw2, ht);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
                    ctx.fillRect(x, y, bw2, Math.max(1, ht * 0.08));
                }
                ctx.strokeStyle = 'rgba(0, 255, 220, 0.2)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(eqX + gap, baseY + 1);
                ctx.lineTo(eqX + eqW - gap, baseY + 1);
                ctx.stroke();
            }

            _drawDigitalSpectrum() {
                if (this.skin !== 'digital' || this.digitalCenterMode !== 'spectrum') return;
                const cL = this.els.digitalSpectrumCanvasL;
                const cR = this.els.digitalSpectrumCanvasR;
                if (!cL || !cR) return;
                const pack = this._computeDigitalSpectrumRadiiAndEq();
                this._syncDonutCoreHues();
                this._drawDigitalSpectrumFlower(cL, pack.layersL, pack.n, this._donutCoreHueA);
                this._drawDigitalSpectrumFlower(cR, pack.layersR, pack.n, this._donutCoreHueB);
                this._drawDigitalCarDash(pack.eqHeights, pack.t);
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
                knobEl.classList.add(
                    'radio-visual-knob--switch',
                    'radio-visual-knob--fade-btn',
                    'radio-visual-knob--autofade-change'
                );
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
                const deckLabel = deck === 'a' ? 'Deck A' : 'Deck B';
                knobEl.setAttribute('aria-label',
                    `${deckLabel}; tap to start or random station when playing, hold to stop`);
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
                        await this._deckKnobTap(deck);
                        this._syncDeckSwitches();
                    },
                    onLongPress: async () => {
                        await this._deckKnobLongPress(deck);
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
                this._syncVolumeMuteLed();
            }

            _toggleVolumeMute() {
                const vs = document.getElementById('volume-slider');
                const cur = vs ? Number(vs.value) : 0;
                if (this._volMuted) {
                    this._volMuted = false;
                    const restore = Math.max(0.02, Math.min(1, this._volUnmuteNorm || 0.5));
                    this._applyVolume(restore);
                } else {
                    if (cur > 0.001) this._volUnmuteNorm = cur;
                    this._volMuted = true;
                    this._applyVolume(0);
                }
            }

            _syncVolumeMuteLed() {
                if (!this.els.volKnob) return;
                this.els.volKnob.classList.toggle('is-on', !!this._volMuted);
                this.els.volKnob.setAttribute('aria-pressed', this._volMuted ? 'true' : 'false');
            }

            _applyVolume(val) {
                const v = Math.max(0, Math.min(1, Number(val) || 0));
                if (v > 0.001) this._volMuted = false;
                try {
                    if (typeof setVolume === 'function') setVolume(v);
                } catch (_) {}
                const vs = document.getElementById('volume-slider');
                if (vs) vs.value = String(v);
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
                this._syncDigitalVolumeUi();
                this._syncVolumeMuteLed();
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
                    const idxB = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                        ? currentStationBIndex : -1;
                    if (idxB >= 0 && stations[idxB]) nameB = stations[idxB].name || nameB;
                } catch (_) {}
                if (this.els.stationDigitalA) this.els.stationDigitalA.textContent = nameA;
                if (this.els.stationDigitalB) this.els.stationDigitalB.textContent = nameB;
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
                this._updateHudModeLines();
                this._syncDonutCoreHues();
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

            _isDigitalStageUiTarget(el) {
                if (!el || !el.closest) return false;
                return !!el.closest(
                    'button, a, input, textarea, select, label, [role="slider"], video[controls]'
                );
            }

            _bindDigitalStageInteractions(digitalCenterEl) {
                if (!digitalCenterEl || !this.abortCtrl) return;
                const sig = this.abortCtrl.signal;
                digitalCenterEl.addEventListener('click', (ev) => {
                    if (this.skin !== 'digital') return;
                    if (this._isDigitalStageUiTarget(ev.target)) return;
                    clearTimeout(this._digitalStageClickTimer);
                    this._digitalStageClickTimer = setTimeout(() => {
                        this._digitalStageClickTimer = null;
                        if (this.digitalCenterMode === 'deckB') {
                            this._setDigitalCenterMode('spectrum');
                        }
                    }, 280);
                }, { signal: sig });
                digitalCenterEl.addEventListener('dblclick', (ev) => {
                    if (this.skin !== 'digital') return;
                    if (this._isDigitalStageUiTarget(ev.target)) return;
                    try { ev.preventDefault(); } catch (_) {}
                    clearTimeout(this._digitalStageClickTimer);
                    this._digitalStageClickTimer = null;
                    try {
                        const fs = globalThis.toggleFullscreen;
                        if (typeof fs === 'function') fs();
                    } catch (_) {}
                }, { signal: sig });
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
                    fit(this.els.digitalSpectrumCanvasL);
                    fit(this.els.digitalSpectrumCanvasR);
                    fit(this.els.digitalCarDashCanvas);
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


            _buildFeatureButtons(gridEl, { deckBInPanel = false } = {}) {
                if (!gridEl) return;
                const g = globalThis;
                const items = [
                    { label: 'Mixer', fn: () => { try { g.toggleMixPanel?.(); } catch (_) {} } },
                    { label: 'Avatar', fn: () => {
                        try {
                            if (typeof g.toggleWebmOverlay === 'function') g.toggleWebmOverlay();
                        } catch (_) {}
                    }},
                    { label: 'Text-In', fn: () => {
                        try {
                            if (typeof g.openTextInForTarget === 'function') g.openTextInForTarget('global');
                            else if (typeof g.toggleTextInPanel === 'function') g.toggleTextInPanel();
                        } catch (_) {}
                    }},
                    { label: 'Video', fn: () => {
                        if (deckBInPanel) {
                            this._deckBFeature('video', () => {
                                this._withDjDeck((dj) => {
                                    if (dj.deckBVizMode === 'video') { dj.tearDownDeckBViz(); dj.syncDeckBVisualButtons(); }
                                    else dj.startDeckBVideoVisual();
                                });
                            });
                            return;
                        }
                        this._withDjDeck((dj) => {
                            if (dj.deckBVizMode === 'video') { dj.tearDownDeckBViz(); dj.syncDeckBVisualButtons(); }
                            else dj.startDeckBVideoVisual();
                        });
                    }},
                    { label: 'ProjectM', fn: () => {
                        if (deckBInPanel) {
                            this._deckBFeature('projectm', () => this._loadVisualByName('ProjectM v2'));
                            return;
                        }
                        this._loadVisualByName('ProjectM v2');
                    }},
                    { label: 'Audio:Bar', fn: () => {
                        if (deckBInPanel) {
                            this._deckBFeature('bars', () => this._loadVisualByName('Audio Bars'));
                            return;
                        }
                        this._loadVisualByName('Audio Bars');
                    }},
                    { label: 'Queue', fn: () => {
                        if (deckBInPanel) {
                            this._deckBFeature('queue', () => {
                                this._withDjDeck((dj) => dj.toggleDeckBQueuePanel());
                            });
                            return;
                        }
                        this._withDjDeck((dj) => dj.toggleDeckBQueuePanel());
                    }},
                    { label: 'Stations', fn: () => {
                        try {
                            if (g.uiLocked) return;
                            if (typeof g.toggleRadioPanel === 'function') g.toggleRadioPanel();
                            else if (typeof g.toggleTopMenuPanel === 'function') g.toggleTopMenuPanel();
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
                let longPressHandled = false;
                let longPressTimer = null;
                let startY = 0;
                let startVal = 0;
                const tapSlop = 4;
                const longPressMs = (opts && opts.longPressMs) || 500;
                const clearLongPress = () => {
                    if (longPressTimer) {
                        clearTimeout(longPressTimer);
                        longPressTimer = null;
                    }
                };
                const onDown = (ev) => {
                    this._stopClick(ev);
                    active = true;
                    moved = false;
                    longPressHandled = false;
                    clearLongPress();
                    startY = ev.clientY;
                    startVal = onValue.get();
                    if (opts && opts.onDragStart) {
                        try { opts.onDragStart(); } catch (_) {}
                    }
                    if (opts && opts.onLongPress) {
                        longPressTimer = setTimeout(() => {
                            longPressTimer = null;
                            longPressHandled = true;
                            moved = true;
                            try { opts.onLongPress(); } catch (_) {}
                        }, longPressMs);
                    }
                    try { knobEl.setPointerCapture(ev.pointerId); } catch (_) {}
                };
                const onMove = (ev) => {
                    if (!active) return;
                    if (Math.abs(ev.clientY - startY) > tapSlop) {
                        moved = true;
                        clearLongPress();
                    }
                    if (!moved) return;
                    onValue.set(Math.max(0, Math.min(1, startVal + (startY - ev.clientY) * 0.004)));
                };
                const onUp = (ev) => {
                    if (!active) return;
                    active = false;
                    this._stopClick(ev);
                    clearLongPress();
                    try { knobEl.releasePointerCapture(ev.pointerId); } catch (_) {}
                    if (!moved && !longPressHandled && opts && opts.onTap) {
                        try { opts.onTap(); } catch (_) {}
                    }
                    longPressHandled = false;
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
                volKnob.classList.add('radio-visual-knob--switch', 'radio-visual-knob--vol-mute');
                volKnob.setAttribute('aria-label', 'Volume; drag to adjust, tap to mute or unmute');
                const deckAKnob = mkControlKnob('radio-visual-deck-a-knob', 'Deck A');
                deckAKnob.classList.add('radio-visual-knob--deck-a');
                const deckBKnob = mkControlKnob('radio-visual-deck-b-knob', 'Deck B');
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
                analogBtns.className = 'radio-visual-analog-actions';
                analogBtns.id = 'radio-visual-analog-btns';
                stageA.appendChild(knobs);
                stageA.appendChild(tunerShell);
                stageA.appendChild(analogBtns);

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
                const stNameA = mkLine('radio-visual-digital-line--station-a', 'radio-visual-station-name-a', '—');
                const dClk = mkLine('radio-visual-digital-line--clock', 'radio-visual-digital-clock', '—');
                const stNameB = mkLine('radio-visual-digital-line--station-b', 'radio-visual-station-name-b', '—');
                const digBtns = document.createElement('div');
                digBtns.className = 'radio-visual-btn-grid radio-visual-digital-feature-btns';
                digBtns.id = 'radio-visual-digital-btns';
                const digitalCenter = document.createElement('div');
                digitalCenter.className = 'radio-visual-digital-center';
                const digitalCenterSpectrum = document.createElement('div');
                digitalCenterSpectrum.className = 'radio-visual-digital-center-pane is-active';
                const spectrumBg = document.createElement('div');
                spectrumBg.className = 'radio-visual-digital-spectrum-bg';
                spectrumBg.setAttribute('aria-hidden', 'true');
                const spectrumRow = document.createElement('div');
                spectrumRow.className = 'radio-visual-digital-spectrum-row';
                const spectrumSideL = document.createElement('div');
                spectrumSideL.className = 'radio-visual-digital-spectrum-side radio-visual-digital-spectrum-side--left';
                const digitalSpectrumCanvasL = document.createElement('canvas');
                digitalSpectrumCanvasL.className = 'radio-visual-digital-spectrum-canvas';
                digitalSpectrumCanvasL.id = 'radio-visual-digital-spectrum-l';
                spectrumSideL.appendChild(digitalSpectrumCanvasL);
                const dashStack = document.createElement('div');
                dashStack.className = 'radio-visual-digital-dash-stack';
                const centerInfo = document.createElement('div');
                centerInfo.className = 'radio-visual-digital-center-info';
                centerInfo.setAttribute('aria-live', 'polite');
                centerInfo.appendChild(stNameA);
                centerInfo.appendChild(stNameB);
                centerInfo.appendChild(dClk);
                const carDisplay = document.createElement('div');
                carDisplay.className = 'radio-visual-digital-car-display';
                const digitalCarDashCanvas = document.createElement('canvas');
                digitalCarDashCanvas.className = 'radio-visual-digital-car-dash-canvas';
                digitalCarDashCanvas.id = 'radio-visual-digital-car-dash';
                const dashXfade = document.createElement('div');
                dashXfade.className = 'radio-visual-digital-dash-xfade';
                const xfLblA = document.createElement('span');
                xfLblA.className = 'radio-visual-digital-dash-xfade-end';
                xfLblA.textContent = 'A';
                const crossDig = document.createElement('input');
                crossDig.type = 'range';
                crossDig.className = 'radio-visual-digital-vol radio-visual-digital-dash-xfade-range';
                crossDig.id = 'radio-visual-cross-digital';
                crossDig.min = '0';
                crossDig.max = '1';
                crossDig.step = '0.01';
                crossDig.value = String(this._getCrossfadeX());
                crossDig.setAttribute('aria-label', 'Crossfade between deck A and deck B');
                const xfLblB = document.createElement('span');
                xfLblB.className = 'radio-visual-digital-dash-xfade-end';
                xfLblB.textContent = 'B';
                dashXfade.appendChild(xfLblA);
                dashXfade.appendChild(crossDig);
                dashXfade.appendChild(xfLblB);
                carDisplay.appendChild(digitalCarDashCanvas);
                carDisplay.appendChild(dashXfade);
                dashStack.appendChild(centerInfo);
                dashStack.appendChild(carDisplay);
                const spectrumSideR = document.createElement('div');
                spectrumSideR.className = 'radio-visual-digital-spectrum-side radio-visual-digital-spectrum-side--right';
                const digitalSpectrumCanvasR = document.createElement('canvas');
                digitalSpectrumCanvasR.className = 'radio-visual-digital-spectrum-canvas';
                digitalSpectrumCanvasR.id = 'radio-visual-digital-spectrum-r';
                spectrumSideR.appendChild(digitalSpectrumCanvasR);
                spectrumRow.appendChild(spectrumSideL);
                spectrumRow.appendChild(dashStack);
                spectrumRow.appendChild(spectrumSideR);
                digitalCenterSpectrum.appendChild(spectrumBg);
                digitalCenterSpectrum.appendChild(spectrumRow);
                const digitalCenterDeckB = document.createElement('div');
                digitalCenterDeckB.className = 'radio-visual-digital-center-pane';
                const digitalDeckBMount = document.createElement('div');
                digitalDeckBMount.className = 'radio-visual-digital-deck-b-mount';
                const digitalDeckBContent = document.createElement('div');
                digitalDeckBContent.className = 'radio-visual-digital-deck-b-content';
                const digitalDeckBVideo = document.createElement('video');
                digitalDeckBVideo.className = 'radio-visual-digital-deck-b-video';
                digitalDeckBVideo.playsInline = true;
                digitalDeckBVideo.muted = true;
                digitalDeckBMount.appendChild(digitalDeckBContent);
                digitalDeckBMount.appendChild(digitalDeckBVideo);
                digitalCenterDeckB.appendChild(digitalDeckBMount);
                digitalCenter.appendChild(digitalCenterSpectrum);
                digitalCenter.appendChild(digitalCenterDeckB);
                digitalCenter.title = 'Tap: return to Spectrum when Deck B view is open. Double-click: fullscreen.';
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
                const btnVis = document.createElement('button');
                btnVis.type = 'button';
                btnVis.className = 'radio-visual-btn radio-visual-digital-step-btn radio-visual-digital-vis-btn';
                btnVis.textContent = '🔆';
                btnVis.title = 'Tap: next background · Hold: turn off background';
                btnVis.setAttribute('aria-label', 'Digital background visual');
                volGroup.appendChild(volLbl);
                volGroup.appendChild(volDown);
                volGroup.appendChild(volDigitalReadout);
                volGroup.appendChild(volUp);
                volGroup.appendChild(btnVis);
                digitalToolbar.appendChild(volGroup);
                const mkRvDigitalBtn = (act, lab) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvDigital = act;
                    b.textContent = lab;
                    digitalToolbar.appendChild(b);
                };
                const mkRvStationBtn = (act, lab, deck) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvAction = act;
                    b.dataset.rvDeck = deck;
                    b.textContent = lab;
                    digitalToolbar.appendChild(b);
                };
                mkRvDigitalBtn('a', 'A Play');
                mkRvStationBtn('prev', 'A◀', 'a');
                mkRvStationBtn('next', 'A▶', 'a');
                mkRvStationBtn('rand', 'A Rand', 'a');
                mkRvDigitalBtn('fade', 'Fade');
                mkRvDigitalBtn('mix', 'Mix');
                mkRvDigitalBtn('b', 'B Play');
                mkRvStationBtn('prev', 'B◀', 'b');
                mkRvStationBtn('next', 'B▶', 'b');
                mkRvStationBtn('rand', 'B Rand', 'b');
                const btnXfadeStation = document.createElement('button');
                btnXfadeStation.type = 'button';
                btnXfadeStation.className = 'radio-visual-btn radio-visual-digital-xfade-station-btn';
                btnXfadeStation.dataset.rvDigital = 'xfade-station';
                btnXfadeStation.textContent = '🔀';
                btnXfadeStation.title = 'Change station when auto-fading (toggle)';
                btnXfadeStation.setAttribute('aria-label', 'Change station when auto-fading');
                digitalToolbar.appendChild(btnXfadeStation);
                dPanel.appendChild(digitalCenter);
                dPanel.appendChild(digitalToolbar);
                dPanel.appendChild(digBtns);
                stageD.appendChild(dPanel);

                root.appendChild(stageA);
                root.appendChild(stageD);

                this.els = {
                    btnSkinAnalog: btnA,
                    btnSkinDigital: btnD,
                    stageAnalog: stageA,
                    stageDigital: stageD,
                    stationDigitalA: stNameA,
                    stationDigitalB: stNameB,
                    digitalDeckBMount,
                    digitalDeckBContent,
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
                    btnVis,
                    btnXfadeStation,
                    spectrumBg,
                    digitalCenterSpectrum,
                    digitalCenterDeckB,
                    digitalSpectrumCanvasL,
                    digitalSpectrumCanvasR,
                    digitalCarDashCanvas,
                    digitalDeckBVideo,
                    volDigitalReadout,
                    digitalClock: dClk,
                    digitalCenter,
                    analogBtns,
                    digitalBtns: digBtns
                };

                this._buildFeatureButtons(analogBtns);
                this._buildFeatureButtons(digBtns, { deckBInPanel: true });
                this._bindDigitalStageInteractions(digitalCenter);
                this._setSkin(this.skin);
                this._syncVolumeFromGlobal();
                this._setAutoFadeDurationNorm(this._autoFadeDurationNorm());
                this._setAutoMixMaxNorm(this._autoMixMaxNorm());
                this._setDigitalCenterMode(this.digitalCenterMode);
                this._syncDeckSwitches();
                this._syncAutoFadeChangeStationKnob();
                this._syncDonutCoreHues();
                this._initDigitalSpectrumBg();
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
                this._wireDigitalVisBgButton(btnVis, sig);
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
                        else if (act === 'xfade-station') this._toggleAutoFadeChangeStation();
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
                }, { onTap: () => this._toggleVolumeMute() });
                this._wireDeckKnob(deckAKnob, 'a');
                this._wireDeckKnob(deckBKnob, 'b');
                this._wireCrossfadeKnob(crossKnob);
                this._wireAutoFadeKnob(autoFadeKnob);
                this._wireAutoMixKnob(autoMixKnob);

                const stopRv = (ev) => {
                    if (this._shouldBypassRootGestureSuppression(ev)) return;
                    this._stopInteraction(ev);
                };
                root.addEventListener('click', stopRv, sig);
                root.addEventListener('pointerdown', stopRv, sig);
                root.addEventListener('pointerup', stopRv, sig);

                window.addEventListener('resize', this.resizeHandler, sig);
                this.onResize();
                this.clockTimerId = setInterval(() => { try { this._tickClock(); } catch (_) {} }, 1000);
                this.animateFrame();
                this._updateHudModeLines();
            }

            animateFrame() {
                this.animId = requestAnimationFrame(() => this.animateFrame());
                try {
                    this._updateStationUi();
                    this._drawMeters();
                    if (this.skin === 'digital') {
                        this._drawDigitalSpectrum();
                        if (this.digitalCenterMode === 'deckB' && this._digitalDeckBView === 'video') {
                            const now = performance.now();
                            if (!this._deckBVideoSyncAt || (now - this._deckBVideoSyncAt) > 800) {
                                this._deckBVideoSyncAt = now;
                                this._syncDigitalDeckBVideo();
                            }
                        }
                        if (this.digitalCenterMode === 'deckB' && this._rvDigitalPmResize) {
                            try { this._rvDigitalPmResize(); } catch (_) {}
                        }
                    }
                } catch (_) {}
            }

            onResize() {
                try { this._resizeCanvases(); } catch (_) {}
                if (this._rvDigitalPmResize) {
                    try { this._rvDigitalPmResize(); } catch (_) {}
                }
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
                try { if (this._digitalStageClickTimer) clearTimeout(this._digitalStageClickTimer); } catch (_) {}
                this._digitalStageClickTimer = null;
                this._spectrumRingSmooth = { low: null, mid: null, high: null };
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
