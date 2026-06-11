/* Extracted from app.js — radio-visual. Uses globals via globalThis (see app.js exposeVsGlobals). */
        class RadioVisualEngine {
            static isRadioModeName(name) {
                return name === 'Analogue radio' || name === 'Digital Radio'
                    || name === 'Radio' || name === 'Radio Visual';
            }

            /** Fixed layout width for Digital Radio (container-query scaling). */
            static get DIGITAL_STAGE_DESIGN_W() { return 960; }
            static get DIGITAL_HUB_CYCLE() {
                return ['equaliser', 'volume', 'effects', 'live', 'video', 'ai-off', 'spectrum'];
            }
            /** EFFECTS hub centre: two 4×1 columns (mixer FX + sample pads). */
            static get DIGITAL_HUB_MIX_FX_COLS() {
                return [
                    [
                        { label: 'Low-pass', mixId: 'fx-lowpass' },
                        { label: 'Bass', mixId: 'fx-bass' },
                        { label: 'High-pass', mixId: 'fx-highpass' },
                        { label: 'TK-FILTER', mixId: 'fx-tk' }
                    ],
                    [
                        { label: 'AIR📢', sample: 'assets/audio/wav6.mp3' },
                        { label: 'WAA', sample: 'assets/audio/wav4.mp3' },
                        { label: 'WAAA', sample: 'assets/audio/wav5.mp3' },
                        { label: 'FX1', sample: 'assets/audio/wav1.mp3' }
                    ]
                ];
            }
            static get SPECTRUM_STAGING_SCALE_MIN() { return 0.35; }
            static get SPECTRUM_STAGING_SCALE_MAX() { return 2.8; }
            static get SPECTRUM_SETTINGS_KEY() { return 'radioVisual.digitalSpectrum.v1'; }
            static get DEFAULT_SPECTRUM_RIBBON_ROTATE() {
                return {
                    rotateLHigh: 0,
                    rotateLMid: 0,
                    rotateLLow: 0,
                    rotateRHigh: 0,
                    rotateRMid: 0,
                    rotateRLow: 0
                };
            }
            static get DEFAULT_SPECTRUM_SETTINGS() {
                return {
                    colorStreamId: 'aurora',
                    scale: 1,
                    opacity: 1,
                    audioStrength: 1,
                    colorFlow: 1,
                    eqColorStreamId: 'aurora',
                    eqScale: 1,
                    eqOpacity: 1,
                    eqAudioStrength: 1,
                    eqColorFlow: 1,
                    ...RadioVisualEngine.DEFAULT_SPECTRUM_RIBBON_ROTATE
                };
            }
            static get SPECTRUM_COLOR_STREAM_PRESETS() {
                return {
                    aurora: {
                        label: 'Aurora',
                        palette: [18, 35, 55, 95, 140, 185, 220, 265, 310, 350]
                    },
                    sunset: {
                        label: 'Sunset',
                        palette: [0, 12, 24, 38, 52, 18, 350, 330, 305, 280]
                    },
                    ocean: {
                        label: 'Ocean',
                        palette: [185, 195, 205, 215, 225, 200, 190, 210, 220, 230]
                    },
                    neon: {
                        label: 'Neon Pulse',
                        palette: [300, 320, 340, 0, 60, 120, 180, 240, 280, 310]
                    },
                    forest: {
                        label: 'Forest',
                        palette: [85, 95, 105, 115, 125, 100, 90, 110, 120, 130]
                    },
                    fire: {
                        label: 'Fire',
                        palette: [0, 8, 18, 28, 38, 45, 35, 25, 15, 5]
                    },
                    ice: {
                        label: 'Ice',
                        palette: [195, 200, 205, 210, 215, 190, 220, 200, 210, 205]
                    },
                    candy: {
                        label: 'Candy',
                        palette: [330, 340, 350, 0, 10, 300, 315, 325, 335, 345]
                    },
                    vaporwave: {
                        label: 'Vaporwave',
                        palette: [280, 300, 320, 180, 200, 260, 290, 310, 190, 270]
                    },
                    mono: {
                        label: 'Mono Blue',
                        palette: [210, 215, 220, 205, 225, 200, 230, 210, 218, 212]
                    }
                };
            }
            static clampSpectrumRibbonRotate(raw, fallback) {
                const d = fallback || RadioVisualEngine.DEFAULT_SPECTRUM_RIBBON_ROTATE;
                const src = (raw && typeof raw === 'object') ? raw : {};
                const clampDeg = (key) => {
                    const v = Number(src[key]);
                    if (!Number.isFinite(v)) return d[key];
                    return Math.max(0, Math.min(360, Math.round(v)));
                };
                return {
                    rotateLHigh: clampDeg('rotateLHigh'),
                    rotateLMid: clampDeg('rotateLMid'),
                    rotateLLow: clampDeg('rotateLLow'),
                    rotateRHigh: clampDeg('rotateRHigh'),
                    rotateRMid: clampDeg('rotateRMid'),
                    rotateRLow: clampDeg('rotateRLow')
                };
            }
            static clampSpectrumSettings(raw) {
                const d = RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                const presets = RadioVisualEngine.SPECTRUM_COLOR_STREAM_PRESETS;
                const pickStream = (id, fb) => ((raw && presets[id]) ? id : fb);
                const colorStreamId = pickStream(raw && raw.colorStreamId, d.colorStreamId);
                const eqColorStreamId = pickStream(raw && raw.eqColorStreamId, d.eqColorStreamId);
                const scaleMin = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MIN;
                const scaleMax = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MAX;
                const clampScale = (val, fb) => {
                    let scale = Number(val);
                    if (!Number.isFinite(scale)) scale = fb;
                    return Math.max(scaleMin, Math.min(scaleMax, scale));
                };
                const clampOpacity = (val, fb) => {
                    let opacity = Number(val);
                    if (!Number.isFinite(opacity)) opacity = fb;
                    return Math.max(0.15, Math.min(1, opacity));
                };
                const clampStrength = (val, fb) => {
                    let strength = Number(val);
                    if (!Number.isFinite(strength)) strength = fb;
                    return Math.max(0.25, Math.min(3, strength));
                };
                const clampFlow = (val, fb) => {
                    let flow = Number(val);
                    if (!Number.isFinite(flow)) flow = fb;
                    return Math.max(0.25, Math.min(3, flow));
                };
                return {
                    colorStreamId,
                    scale: clampScale(raw && raw.scale, d.scale),
                    opacity: clampOpacity(raw && raw.opacity, d.opacity),
                    audioStrength: clampStrength(raw && raw.audioStrength, d.audioStrength),
                    colorFlow: clampFlow(raw && raw.colorFlow, d.colorFlow),
                    eqColorStreamId,
                    eqScale: clampScale(raw && raw.eqScale, d.eqScale),
                    eqOpacity: clampOpacity(raw && raw.eqOpacity, d.eqOpacity),
                    eqAudioStrength: clampStrength(raw && raw.eqAudioStrength, d.eqAudioStrength),
                    eqColorFlow: clampFlow(raw && raw.eqColorFlow, d.eqColorFlow),
                    ...RadioVisualEngine.clampSpectrumRibbonRotate(raw, d)
                };
            }
            static loadSpectrumSettingsFromStorage() {
                try {
                    const raw = localStorage.getItem(RadioVisualEngine.SPECTRUM_SETTINGS_KEY);
                    const parsed = raw ? JSON.parse(raw) : null;
                    if (!parsed || typeof parsed !== 'object') {
                        return { ...RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS };
                    }
                    return RadioVisualEngine.clampSpectrumSettings(parsed);
                } catch (_) {
                    return { ...RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS };
                }
            }
            static saveSpectrumSettingsToStorage(settings) {
                const next = RadioVisualEngine.clampSpectrumSettings(settings);
                try { localStorage.setItem(RadioVisualEngine.SPECTRUM_SETTINGS_KEY, JSON.stringify(next)); } catch (_) {}
                return next;
            }
            static get DIGITAL_HUB_LABELS() {
                return {
                    equaliser: 'EQUALISER',
                    spectrum: 'SPECTRUM',
                    volume: 'VOLUME',
                    effects: 'EFFECTS',
                    live: 'LIVE',
                    video: 'VIDEO',
                    'ai-off': 'OFF'
                };
            }
            static get DIGITAL_AI_VIDEO_DIR() { return 'assets/ai/'; }
            static get DIGITAL_AI_VIDEO_MANIFEST() { return 'assets/ai/manifest.json'; }
            /** Hub AI / LIVE overlay clips (random pick, then next random on end). */
            static get DIGITAL_AI_VIDEO_LIST() {
                return ['ai.mp4', 'ai1.mp4', 'ai2.mp4', 'ai3.mp4', 'ai4.mp4', 'ai5.mp4', 'ai6.mp4'];
            }
            /** Toolbar centre strip: clock → volume → crossfade → auto-mix countdown. */
            static get DIGITAL_TOOLBAR_CENTER_CYCLE() {
                return ['clock', 'volume', 'crossfade', 'automix', 'remaining'];
            }
            static get DIGITAL_TOOLBAR_XFADE_STEP() { return 0.05; }
            static get DIGITAL_AI_VIDEO_SRC() { return 'assets/ai/ai.mp4'; }
            /** Centre hub modes that show and play {@link DIGITAL_AI_VIDEO_LIST}. */
            static digitalHubShowsAiVideo(mode) {
                return mode === 'ai' || mode === 'live';
            }
            /** Centre hub modes with a full-panel video surface (AI clips or deck / logo). */
            static digitalHubShowsCenterVideo(mode) {
                return mode === 'ai' || mode === 'live' || mode === 'video';
            }
            static configureEmbeddableIframe(iframe, title) {
                if (!iframe) return;
                iframe.setAttribute('title', title || 'Embedded content');
                iframe.setAttribute(
                    'sandbox',
                    'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads allow-modals allow-fullscreen'
                );
                iframe.setAttribute('allow', 'fullscreen; autoplay; encrypted-media; picture-in-picture');
                iframe.setAttribute('allowfullscreen', '');
                iframe.setAttribute('referrerpolicy', 'no-referrer-when-downgrade');
            }

            static digitalStagingViewSupportsMountFullscreen(view) {
                return view === 'projectm';
            }
            static digitalHubShowsDeckVideo(mode) {
                return mode === 'video';
            }
            /** Deck A: TK / Loop / Arp · Deck B: Reverb / Flanger / CUT (mirrors DJ Deck beat FX). */
            static get DIGITAL_HUB_FX_KNOBS() {
                return {
                    a: [
                        { suffix: 'tk', label: 'TK', color: '#b8980c' },
                        { suffix: 'loop', label: 'Loop', color: '#2a8fa0' },
                        { suffix: 'arp', label: 'Arp', color: '#8647a8' }
                    ],
                    b: [
                        { suffix: 'reverb', label: 'Reverb', color: '#ff4040' },
                        { suffix: 'flanger', label: 'Flanger', color: '#3fff00' },
                        { suffix: 'cut', label: 'CUT', color: '#bada55' }
                    ]
                };
            }

            constructor(options = {}) {
                const lockedSkin = options.skin === 'digital' ? 'digital'
                    : (options.skin === 'analogue' ? 'analogue' : null);
                this._skinLocked = options.skinLocked !== false && !!lockedSkin;
                if (this._skinLocked) {
                    this.skin = lockedSkin;
                    this.name = options.name || (lockedSkin === 'digital' ? 'Digital Radio' : 'Analogue radio');
                } else {
                    this.skin = 'analogue';
                    this.name = options.name || 'Radio';
                }
                this.resizeHandler = this.onResize.bind(this);
                this.abortCtrl = null;
                this.root = null;
                this.els = {};
                this.animId = null;
                this.clockTimerId = null;
                this._lastStationIdxA = -2;
                this._lastStationIdxB = -2;
                this._donutCoreHueA = 175;
                this._donutCoreHueB = 285;
                this._vuBuf = null;
                this._tuningDrag = false;
                this._volDrag = false;
                this._rvAutoFadeRaf = null;
                this.digitalCenterMode = 'spectrum';
                /** Centre hub: equaliser → volume → effects → live → video → off → spectrum */
                this._digitalHubMode = 'equaliser';
                this._digitalHubAiReturnMode = 'equaliser';
                /** VIDEO toolbar: 0 off · 1 equaliser panel · 2 deck B centre */
                this._digitalVideoToolbarStep = 0;
                this._digitalHubDeckVideoSrc = '';
                this._digitalHubDeckVideoMode = 'idle';
                this._digitalDeckBVideoSrc = '';
                this._digitalDeckBVideoMode = 'idle';
                /** Current hub AI clip path (relative), or null when overlay hidden. */
                this._digitalHubAiVideoSrc = null;
                /** Active ping-pong slot: 0 = video A, 1 = video B. */
                this._digitalHubAiVideoSlot = 0;
                this._digitalHubAiAdvancing = false;
                this._digitalHubAiVideoLoadGen = 0;
                this._digitalHubAiEndGuard = false;
                this._digitalAiVideoFilesList = null;
                this._digitalAiVideoManifestPromise = null;
                /** Derived from hub mode for spectrum flower / HUD layout. */
                this._digitalSpectrumLayout = 'full';
                /** Spectrum hub staging scale (1 = default flower size). */
                this._spectrumStagingScale = 1;
                this._spectrumStagingPanNorm = 0;
                this._spectrumSliderHideTimer = null;
                this._digitalDeckBView = 'video';
                /** Staging overlay in spectrum pane: null | video | projectm | bars | queue | karaoke */
                this._digitalStagingView = null;
                this._digitalKaraokeUrl = (typeof globalThis.KARAOKE_NERDS_EMBED_URL === 'string')
                    ? globalThis.KARAOKE_NERDS_EMBED_URL
                    : 'https://www.karaokenerds.com/#query';
                /** Local playlists panel (Deck A & B) over the spectrum area */
                this._digitalLocalQueueVisible = false;
                /** Radio stations panel (Deck A & B) over the spectrum area */
                this._digitalStationsVisible = false;
                this._rvAutoMixTimerId = null;
                this._rvAutoMixNextFadeAt = 0;
                this._rvAutoMixNextFadeIntervalId = null;
                this._rvAutoMixCyclePending = false;
                this._digitalVolStep = 0.05;
                /** Toolbar centre readout mode: clock | volume | crossfade | automix | remaining */
                this._digitalToolbarCenterMode = 'clock';
                this._digitalToolbarVolumePeekTimer = null;
                this._digitalToolbarVolumePeekActive = false;
                this._rvLastGlobalVolumeNorm = undefined;
                this._volMuted = false;
                this._volUnmuteNorm = 0.5;
                this._digitalStageClickTimer = null;
                this._digitalBgGifIdx = 0;
                this._digitalBgGifEnabled = false;
                /** Smoothed ring radii per band: low (base), mid, high (top). */
                this._spectrumRingSmooth = { low: null, mid: null, high: null };
                this._digitalBgGifFilesList = null;
                this._digitalBgGifManifestPromise = null;
                this._outerHuePhase = 0;
                this._outerHueLastT = 0;
                this._spectrumSettings = RadioVisualEngine.loadSpectrumSettingsFromStorage();
                this._spectrumStagingScale = this._spectrumSettings.scale;
                this._rvStagingPmFsToggleAt = 0;
                /** After long Space-hold pause, block auto-fade resume ticks until play again. */
                this._suppressCrossfadeResume = false;
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
            /** Set after the one-time AI toolbar intro blink on first Digital Radio load. */
            static get DIGITAL_AI_INTRO_SEEN_KEY() { return 'radioVisual.digitalAiIntroSeen.v1'; }

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
                    if (t.closest('input, [role="slider"]')) return true;
                    if (t.closest('select')) return true;
                    if (t.closest('textarea')) return true;
                    if (t.closest('.radio-visual-digital-automix-panel')) return true;
                    if (t.closest('.radio-visual-digital-spectrum-staging-slider')) return true;
                    if (t.closest('.radio-visual-digital-spectrum-staging-scale-wrap')) return true;
                    if (t.closest('.radio-visual-digital-spectrum-staging-pan-wrap')) return true;
                    if (t.closest('.radio-visual-digital-hub-panel')) return true;
                    if (t.closest('.radio-visual-digital-hub-ai')) return true;
                    if (t.closest('.radio-visual-digital-hub-ai-video')) return true;
                    if (t.closest('.radio-visual-digital-hub-deck-video')) return true;
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
                this._digitalBgGifSwapGen = (this._digitalBgGifSwapGen || 0) + 1;
                const swapGen = this._digitalBgGifSwapGen;
                const fadeMs = 450;

                const fail = () => {
                    if (swapGen !== this._digitalBgGifSwapGen) return;
                    if (!trySkipOnError || files.length < 2) {
                        bgEl.classList.remove('is-visible');
                        bgEl.querySelectorAll('img').forEach((el) => {
                            try { el.remove(); } catch (_) {}
                        });
                        return;
                    }
                    const next = (i + 1) % files.length;
                    if (next === i) {
                        bgEl.classList.remove('is-visible');
                        return;
                    }
                    this._applyDigitalSpectrumBgFile(files[next], next, false);
                };

                const revealLoadedImg = (img) => {
                    if (swapGen !== this._digitalBgGifSwapGen) {
                        try { img.remove(); } catch (_) {}
                        return;
                    }
                    const prevImgs = Array.from(bgEl.querySelectorAll('img')).filter((el) => el !== img);
                    try {
                        img.classList.remove('is-bg-loading');
                        bgEl.classList.add('is-visible');
                        this._digitalBgGifIdx = i;
                        try { localStorage.setItem(RadioVisualEngine.DIGITAL_BG_GIF_STORAGE_KEY, file); } catch (_) {}
                    } catch (_) {}
                    prevImgs.forEach((el) => el.classList.add('is-bg-fade-out'));
                    window.setTimeout(() => {
                        if (swapGen !== this._digitalBgGifSwapGen) return;
                        prevImgs.forEach((el) => {
                            try { el.remove(); } catch (_) {}
                        });
                    }, fadeMs + 40);
                };

                const whenReady = (img) => {
                    const run = () => revealLoadedImg(img);
                    try {
                        if (typeof img.decode === 'function') {
                            img.decode().then(run).catch(run);
                        } else {
                            run();
                        }
                    } catch (_) {
                        run();
                    }
                };

                const img = document.createElement('img');
                img.className = 'radio-visual-digital-spectrum-bg-img is-bg-loading';
                img.alt = '';
                img.decoding = 'async';
                img.onload = () => whenReady(img);
                img.onerror = () => {
                    try { img.remove(); } catch (_) {}
                    fail();
                };
                try { bgEl.appendChild(img); } catch (_) {}
                try { bgEl.classList.add('is-visible'); } catch (_) {}
                img.src = url;
                try {
                    if (img.complete && img.naturalWidth > 0) whenReady(img);
                } catch (_) {}
            }

            _isDigitalBgGifEnabled() {
                try {
                    const raw = localStorage.getItem(RadioVisualEngine.DIGITAL_BG_GIF_ENABLED_KEY);
                    if (raw === '0') return false;
                    if (raw === '1') return true;
                } catch (_) {}
                return !!this._digitalBgGifEnabled;
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
                try { btn.removeAttribute('title'); } catch (_) {}
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
                    if (raw === '1') {
                        this._digitalBgGifEnabled = true;
                    } else if (raw === '0') {
                        this._digitalBgGifEnabled = false;
                    } else {
                        this._digitalBgGifEnabled = false;
                    }
                } catch (_) {
                    this._digitalBgGifEnabled = false;
                }
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

            _resetDigitalVisBgToFirst() {
                this._refreshDigitalBgGifList().then(() => {
                    const files = this._digitalBgGifFiles();
                    if (!this.els.spectrumBg || !files.length) return;
                    if (!this._isDigitalBgGifEnabled()) {
                        this._setDigitalBgGifEnabled(true);
                    }
                    this._applyDigitalSpectrumBgFile(files[0], 0);
                    this._syncDigitalVisBgButton();
                }).catch(() => {});
            }

            setDigitalBgGifFromOptions(filename) {
                const name = String(filename || '').trim();
                if (!name) {
                    this._setDigitalBgGifEnabled(false);
                    this._syncDigitalVisBgButton();
                    return;
                }
                this._refreshDigitalBgGifList().then(() => {
                    const files = this._digitalBgGifFiles();
                    const idx = files.indexOf(name);
                    if (idx < 0) return;
                    this._setDigitalBgGifEnabled(true);
                    this._applyDigitalSpectrumBgFile(name, idx);
                    this._syncDigitalVisBgButton();
                }).catch(() => {});
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
                btn.addEventListener('contextmenu', (ev) => {
                    this._stopClick(ev);
                    try { ev.preventDefault(); } catch (_) {}
                    clearLongPress();
                    longPressHandled = true;
                    this._resetDigitalVisBgToFirst();
                }, sig);
                btn.addEventListener('pointerdown', (ev) => {
                    this._stopClick(ev);
                    if (ev.button === 2) {
                        clearLongPress();
                        longPressHandled = true;
                        return;
                    }
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
                    if (ev.button === 2) {
                        longPressHandled = false;
                        return;
                    }
                    if (!longPressHandled) this._onDigitalVisBgTap();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('pointercancel', () => {
                    clearLongPress();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _ingestLocalFilesToDeckAndPlay(deckKey, files) {
                try {
                    if (typeof addDeckLocalFilesToDeck === 'function') {
                        addDeckLocalFilesToDeck(deckKey, files, { forceImmediate: true, preserveCrossfade: true });
                    } else if (typeof ingestLocalFilesToDeckAndPlay === 'function') {
                        ingestLocalFilesToDeckAndPlay(deckKey, files);
                    }
                } catch (_) {}
            }

            _afterDeckLocalFileDrop() {
                try {
                    if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi();
                } catch (_) {}
                try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                try { this._syncDeckSwitches(); } catch (_) {}
                try { this._updateStationUi(); } catch (_) {}
            }

            _buildDigitalLocalQueuePanel() {
                const panel = document.createElement('div');
                panel.id = 'radio-visual-digital-local-queue-panel';
                panel.className = 'radio-visual-digital-local-queue-panel dj-deck-b-queue-panel display-none';
                panel.setAttribute('aria-label', 'Local file queues');
                const header = document.createElement('div');
                header.className = 'dj-queue-header';
                header.textContent = 'Local playlists · Deck A & B';
                const cols = document.createElement('div');
                cols.className = 'dj-queue-columns';
                const mkCol = (deck) => {
                    const col = document.createElement('div');
                    col.className = 'dj-queue-col';
                    col.dataset.deck = deck;
                    const head = document.createElement('div');
                    head.className = 'dj-queue-col-head';
                    const title = document.createElement('div');
                    title.className = 'dj-queue-col-title';
                    title.textContent = deck === 'b' ? 'Deck B queue' : 'Deck A queue';
                    const btns = document.createElement('div');
                    btns.className = 'dj-queue-col-head-btns';
                    const mkBtn = (id, label, extraClass) => {
                        const b = document.createElement('button');
                        b.type = 'button';
                        b.id = id;
                        b.textContent = label;
                        if (extraClass) b.className = extraClass;
                        return b;
                    };
                    btns.appendChild(mkBtn(`rv-digital-queue-add-${deck}`, 'Add files…', 'dj-queue-add'));
                    btns.appendChild(mkBtn(
                        `rv-digital-queue-folder-${deck}`,
                        'Folder…',
                        'dj-queue-folder'
                    ));
                    btns.appendChild(mkBtn(
                        `rv-digital-queue-url-${deck}`,
                        'Add URL…',
                        'dj-queue-add-url'
                    ));
                    const shuffleBtn = mkBtn(
                        `rv-digital-queue-shuffle-${deck}`,
                        '🔀',
                        'dj-queue-shuffle'
                    );
                    shuffleBtn.title = deck === 'b' ? 'Shuffle Deck B queue' : 'Shuffle Deck A queue';
                    shuffleBtn.setAttribute('aria-label', shuffleBtn.title);
                    btns.appendChild(shuffleBtn);
                    const clearBtn = mkBtn(
                        `rv-digital-queue-clear-${deck}`,
                        'X',
                        'dj-queue-clear-all'
                    );
                    clearBtn.title = deck === 'b' ? 'Clear Deck B queue' : 'Clear Deck A queue';
                    clearBtn.setAttribute('aria-label', clearBtn.title);
                    btns.appendChild(clearBtn);
                    const ul = document.createElement('ul');
                    ul.id = `rv-digital-queue-list-${deck}`;
                    ul.className = 'dj-queue-list';
                    head.appendChild(title);
                    head.appendChild(btns);
                    col.appendChild(head);
                    col.appendChild(ul);
                    return col;
                };
                cols.appendChild(mkCol('a'));
                cols.appendChild(mkCol('b'));
                panel.appendChild(header);
                panel.appendChild(cols);
                return panel;
            }

            _refreshDigitalLocalQueueUi() {
                if (!this.root || !this._digitalLocalQueueVisible) return;
                try {
                    const ulA = this.root.querySelector('#rv-digital-queue-list-a');
                    const ulB = this.root.querySelector('#rv-digital-queue-list-b');
                    const fill = (ul, deckKey) => {
                        if (!ul) return;
                        ul.innerHTML = '';
                        const q = (typeof deckFileQueues !== 'undefined')
                            ? (deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a)
                            : [];
                        if (!q || !q.length) return;
                        q.forEach((item, idx) => {
                            const li = document.createElement('li');
                            li.className = 'dj-queue-item';
                            const idxSpan = document.createElement('span');
                            idxSpan.className = 'dj-queue-idx';
                            idxSpan.textContent = String(idx + 1);
                            const nameSpan = document.createElement('span');
                            nameSpan.className = 'dj-queue-name';
                            nameSpan.textContent = item.name || 'Track';
                            const play = document.createElement('button');
                            play.type = 'button';
                            play.className = 'dj-queue-play';
                            play.textContent = '▹';
                            play.title = 'Play this track now';
                            play.addEventListener('click', (ev) => {
                                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                                if (typeof playQueuedTrackNow === 'function') playQueuedTrackNow(deckKey, idx);
                                try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                                try { this._syncDeckSwitches(); } catch (_) {}
                                try { this._updateStationUi(); } catch (_) {}
                            });
                            const toOther = document.createElement('button');
                            toOther.type = 'button';
                            toOther.className = 'dj-queue-to-deck ' + (deckKey === 'a' ? 'dj-queue-to-deck--to-b' : 'dj-queue-to-deck--to-a');
                            toOther.textContent = deckKey === 'a' ? 'B' : 'A';
                            toOther.title = deckKey === 'a' ? 'Send to Deck B queue' : 'Send to Deck A queue';
                            toOther.addEventListener('click', (ev) => {
                                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                                if (typeof moveQueuedTrackToOtherDeck === 'function') {
                                    moveQueuedTrackToOtherDeck(deckKey, idx);
                                }
                                try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                            });
                            const rm = document.createElement('button');
                            rm.type = 'button';
                            rm.className = 'dj-queue-remove';
                            rm.textContent = '✕';
                            rm.title = 'Remove from queue';
                            rm.addEventListener('click', (ev) => {
                                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                                if (typeof removeQueuedTrack === 'function') removeQueuedTrack(deckKey, idx);
                                try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                            });
                            li.appendChild(idxSpan);
                            li.appendChild(nameSpan);
                            li.appendChild(play);
                            li.appendChild(toOther);
                            li.appendChild(rm);
                            ul.appendChild(li);
                        });
                    };
                    fill(ulA, 'a');
                    fill(ulB, 'b');
                } catch (_) {}
            }

            _syncDigitalLocalQueueButton() {
                const grid = this.els.digitalBtns;
                if (!grid) return;
                const on = !!this._digitalLocalQueueVisible;
                grid.querySelectorAll('[data-rv-local-queue]').forEach((btn) => {
                    btn.classList.toggle('is-active', on);
                    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
                });
            }

            _buildDigitalStationsPanel() {
                const panel = document.createElement('div');
                panel.id = 'radio-visual-digital-stations-panel';
                panel.className = 'radio-visual-digital-stations-panel display-none';
                panel.setAttribute('aria-label', 'Radio stations');
                const header = document.createElement('div');
                header.className = 'radio-visual-digital-stations-header';
                header.textContent = 'Radio stations · Deck A & B';
                const cols = document.createElement('div');
                cols.className = 'radio-visual-digital-stations-columns';
                const wireNav = (deck, delta) => {
                    const g = globalThis;
                    if (g.uiLocked) return;
                    const list = g.stations;
                    if (!list || !list.length) return;
                    const cur = deck === 'b' ? g.currentStationBIndex : g.currentStationIndex;
                    const idx = (cur + delta + list.length) % list.length;
                    if (deck === 'b') {
                        if (typeof g.setStationB === 'function') g.setStationB(idx);
                    } else if (typeof g.setStation === 'function') {
                        g.setStation(idx);
                    }
                    requestAnimationFrame(() => {
                        try { this._scrollDigitalStationsListToActive(deck); } catch (_) {}
                    });
                };
                const wireRandom = (deck) => {
                    const g = globalThis;
                    if (g.uiLocked) return;
                    if (deck === 'b') {
                        if (typeof g.pickRandomStationB === 'function') g.pickRandomStationB();
                    } else if (typeof g.pickRandomStation === 'function') {
                        g.pickRandomStation();
                    }
                    this._scrollDigitalStationsAfterDeckChange(deck);
                };
                const mkCol = (deck) => {
                    const col = document.createElement('div');
                    col.className = 'radio-visual-digital-stations-col';
                    col.dataset.deck = deck;
                    const head = document.createElement('div');
                    head.className = 'radio-visual-digital-stations-col-head';
                    const deckBadge = document.createElement('span');
                    deckBadge.className = `radio-visual-digital-stations-deck radio-visual-digital-stations-deck--${deck}`;
                    deckBadge.textContent = deck === 'b' ? 'B' : 'A';
                    const title = document.createElement('span');
                    title.className = 'radio-visual-digital-stations-title';
                    title.textContent = deck === 'b' ? 'Deck B' : 'Deck A';
                    const nav = document.createElement('div');
                    nav.className = 'radio-visual-digital-stations-nav';
                    const prev = document.createElement('button');
                    prev.type = 'button';
                    prev.className = 'radio-visual-digital-stations-nav-btn';
                    prev.textContent = '◀';
                    prev.title = deck === 'b' ? 'Previous station on Deck B' : 'Previous station on Deck A';
                    prev.setAttribute('aria-label', prev.title);
                    const next = document.createElement('button');
                    next.type = 'button';
                    next.className = 'radio-visual-digital-stations-nav-btn';
                    next.textContent = '▶';
                    next.title = deck === 'b' ? 'Random station on Deck B' : 'Random station on Deck A';
                    next.setAttribute('aria-label', next.title);
                    prev.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        wireNav(deck, -1);
                    });
                    next.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        wireRandom(deck);
                    });
                    nav.appendChild(prev);
                    nav.appendChild(next);
                    head.appendChild(deckBadge);
                    head.appendChild(title);
                    head.appendChild(nav);
                    const scroll = document.createElement('div');
                    scroll.className = 'radio-visual-digital-stations-scroll';
                    const list = document.createElement('div');
                    list.id = `rv-digital-stations-list-${deck}`;
                    list.className = 'radio-visual-digital-stations-list';
                    scroll.appendChild(list);
                    col.appendChild(head);
                    col.appendChild(scroll);
                    return col;
                };
                cols.appendChild(mkCol('a'));
                cols.appendChild(mkCol('b'));
                panel.appendChild(header);
                panel.appendChild(cols);
                return panel;
            }

            _refreshDigitalStationsUi() {
                if (!this.root || !this._digitalStationsVisible) return;
                try {
                    const g = globalThis;
                    const stationList = g.stations;
                    if (!stationList || !stationList.length) return;
                    const fill = (root, deckKey) => {
                        if (!root) return;
                        root.innerHTML = '';
                        stationList.forEach((s, i) => {
                            const item = document.createElement('div');
                            item.className = 'radio-item';
                            item.dataset.index = String(i);
                            item.dataset.deck = deckKey;
                            const activeIdx = deckKey === 'b' ? g.currentStationBIndex : g.currentStationIndex;
                            if (i === activeIdx) item.classList.add('active');
                            const nameEl = document.createElement('div');
                            nameEl.textContent = s.name;
                            const goEl = document.createElement('div');
                            goEl.textContent = '➤';
                            item.appendChild(nameEl);
                            item.appendChild(goEl);
                            item.addEventListener('click', () => {
                                if (g.uiLocked) return;
                                if (deckKey === 'b') {
                                    if (typeof g.setStationB === 'function') g.setStationB(i);
                                } else if (typeof g.setStation === 'function') {
                                    g.setStation(i);
                                }
                            });
                            root.appendChild(item);
                        });
                    };
                    fill(this.root.querySelector('#rv-digital-stations-list-a'), 'a');
                    fill(this.root.querySelector('#rv-digital-stations-list-b'), 'b');
                    requestAnimationFrame(() => {
                        requestAnimationFrame(() => {
                            try { this._scrollDigitalStationsListsToActive(); } catch (_) {}
                        });
                    });
                } catch (_) {}
            }

            _scrollDigitalStationsListToActive(deckKey) {
                const dk = deckKey === 'b' ? 'b' : 'a';
                const list = this.root?.querySelector(`#rv-digital-stations-list-${dk}`);
                if (!list) return;
                const scroll = list.closest('.radio-visual-digital-stations-scroll');
                if (!scroll) return;
                const active = list.querySelector('.radio-item.active');
                if (!active) return;
                const scrollRect = scroll.getBoundingClientRect();
                const activeRect = active.getBoundingClientRect();
                const delta = (activeRect.top + activeRect.height / 2)
                    - (scrollRect.top + scrollRect.height / 2);
                scroll.scrollTop += delta;
            }

            _scrollDigitalStationsListsToActive() {
                if (!this._digitalStationsVisible) return;
                try { this._scrollDigitalStationsListToActive('a'); } catch (_) {}
                try { this._scrollDigitalStationsListToActive('b'); } catch (_) {}
            }

            _scrollDigitalStationsAfterDeckChange(deckKey) {
                if (!this._digitalStationsVisible) return;
                requestAnimationFrame(() => {
                    try { this._scrollDigitalStationsListToActive(deckKey); } catch (_) {}
                });
            }

            _syncDigitalStationsActiveHighlight() {
                if (!this.root || !this._digitalStationsVisible) return;
                try {
                    const g = globalThis;
                    const apply = (root, activeIdx) => {
                        if (!root) return;
                        Array.from(root.children).forEach((el) => {
                            const idx = Number(el.dataset.index || -1);
                            el.classList.toggle('active', idx === activeIdx);
                        });
                    };
                    apply(this.root.querySelector('#rv-digital-stations-list-a'), g.currentStationIndex);
                    apply(this.root.querySelector('#rv-digital-stations-list-b'), g.currentStationBIndex);
                } catch (_) {}
            }

            _syncDigitalStationsButton() {
                const grid = this.els.digitalBtns;
                if (!grid) return;
                const on = !!this._digitalStationsVisible;
                grid.querySelectorAll('[data-rv-stations]').forEach((btn) => {
                    btn.classList.toggle('is-active', on);
                    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
                });
            }

            _closeDigitalStationsPanel() {
                if (!this._digitalStationsVisible) return;
                this._digitalStationsVisible = false;
                const pane = this.els.digitalCenterSpectrum;
                const panel = this.els.digitalStationsPanel;
                if (pane) pane.classList.remove('is-stations-open');
                if (panel) panel.classList.add('display-none');
                try { this._syncDigitalStationsButton(); } catch (_) {}
            }

            _toggleDigitalStationsPanel() {
                const opening = !this._digitalStationsVisible;
                if (opening) {
                    try { this._closeDigitalLocalQueuePanel(); } catch (_) {}
                    if (this._digitalStagingView) {
                        this._digitalStagingView = null;
                        try { this._tearDownDigitalStagingView(); } catch (_) {}
                        try { this._syncDigitalStagingButtons(); } catch (_) {}
                    }
                }
                this._digitalStationsVisible = !this._digitalStationsVisible;
                const pane = this.els.digitalCenterSpectrum;
                const panel = this.els.digitalStationsPanel;
                if (pane) pane.classList.toggle('is-stations-open', this._digitalStationsVisible);
                if (panel) panel.classList.toggle('display-none', !this._digitalStationsVisible);
                if (this._digitalStationsVisible) {
                    try { this._refreshDigitalStationsUi(); } catch (_) {}
                }
                try { this._syncDigitalStationsButton(); } catch (_) {}
                try { resetIdleTimer(); } catch (_) {}
            }

            _closeDigitalLocalQueuePanel() {
                if (!this._digitalLocalQueueVisible) return;
                this._digitalLocalQueueVisible = false;
                const pane = this.els.digitalCenterSpectrum;
                const panel = this.els.digitalLocalQueuePanel;
                if (pane) pane.classList.remove('is-local-queue-open');
                if (panel) panel.classList.add('display-none');
                try { this._syncDigitalLocalQueueButton(); } catch (_) {}
            }

            _toggleDigitalLocalQueuePanel() {
                const opening = !this._digitalLocalQueueVisible;
                if (opening) {
                    try { this._closeDigitalStationsPanel(); } catch (_) {}
                }
                this._digitalLocalQueueVisible = !this._digitalLocalQueueVisible;
                if (this._digitalLocalQueueVisible && this._digitalStagingView) {
                    this._digitalStagingView = null;
                    try { this._tearDownDigitalStagingView(); } catch (_) {}
                    try { this._syncDigitalStagingButtons(); } catch (_) {}
                }
                const pane = this.els.digitalCenterSpectrum;
                const panel = this.els.digitalLocalQueuePanel;
                if (pane) pane.classList.toggle('is-local-queue-open', this._digitalLocalQueueVisible);
                if (panel) panel.classList.toggle('display-none', !this._digitalLocalQueueVisible);
                if (this._digitalLocalQueueVisible) {
                    try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                }
                try { this._syncDigitalLocalQueueButton(); } catch (_) {}
                try { resetIdleTimer(); } catch (_) {}
            }

            _wireDigitalLocalQueuePanel(sig) {
                if (!this.root) return;
                const fiQA = document.createElement('input');
                fiQA.type = 'file';
                fiQA.accept = 'audio/*,video/*';
                fiQA.multiple = true;
                fiQA.style.display = 'none';
                fiQA.setAttribute('aria-hidden', 'true');
                this.root.appendChild(fiQA);
                const fiQB = document.createElement('input');
                fiQB.type = 'file';
                fiQB.accept = 'audio/*,video/*';
                fiQB.multiple = true;
                fiQB.style.display = 'none';
                fiQB.setAttribute('aria-hidden', 'true');
                this.root.appendChild(fiQB);
                const fiQAFolder = document.createElement('input');
                fiQAFolder.type = 'file';
                fiQAFolder.multiple = true;
                fiQAFolder.style.display = 'none';
                fiQAFolder.setAttribute('aria-hidden', 'true');
                fiQAFolder.setAttribute('webkitdirectory', '');
                this.root.appendChild(fiQAFolder);
                const fiQBFolder = document.createElement('input');
                fiQBFolder.type = 'file';
                fiQBFolder.multiple = true;
                fiQBFolder.style.display = 'none';
                fiQBFolder.setAttribute('aria-hidden', 'true');
                fiQBFolder.setAttribute('webkitdirectory', '');
                this.root.appendChild(fiQBFolder);

                const applyRvDeckLocalFiles = (deckKey, files) => {
                    if (!files || !files.length) return;
                    try {
                        if (typeof addDeckLocalFilesToDeck === 'function') {
                            addDeckLocalFilesToDeck(deckKey, files);
                        }
                    } catch (_) {}
                    this._afterDeckLocalFileDrop();
                };
                const openRvDeckLocalFolderPicker = async (deckKey, fiFolder) => {
                    let files = null;
                    try {
                        if (typeof pickDeckLocalFolderFiles === 'function') {
                            files = await pickDeckLocalFolderFiles();
                        }
                    } catch (_) {}
                    if (files === null) {
                        try { fiFolder.click(); } catch (_) {}
                        return;
                    }
                    applyRvDeckLocalFiles(deckKey, files);
                };
                const wireDeckFiles = (deckKey, fi, fiFolder) => {
                    const dk = deckKey === 'b' ? 'b' : 'a';
                    fi.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        applyRvDeckLocalFiles(dk, files);
                    }, sig);
                    fiFolder.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        applyRvDeckLocalFiles(dk, files);
                    }, sig);
                };
                wireDeckFiles('a', fiQA, fiQAFolder);
                wireDeckFiles('b', fiQB, fiQBFolder);

                const btnAddA = this.root.querySelector('#rv-digital-queue-add-a');
                const btnAddB = this.root.querySelector('#rv-digital-queue-add-b');
                const btnFolderA = this.root.querySelector('#rv-digital-queue-folder-a');
                const btnFolderB = this.root.querySelector('#rv-digital-queue-folder-b');
                const btnUrlA = this.root.querySelector('#rv-digital-queue-url-a');
                const btnUrlB = this.root.querySelector('#rv-digital-queue-url-b');
                if (btnAddA) btnAddA.addEventListener('click', () => { try { fiQA.click(); } catch (_) {} }, sig);
                if (btnAddB) btnAddB.addEventListener('click', () => { try { fiQB.click(); } catch (_) {} }, sig);
                if (btnFolderA) {
                    btnFolderA.addEventListener('click', () => {
                        openRvDeckLocalFolderPicker('a', fiQAFolder).catch(() => {});
                    }, sig);
                }
                if (btnFolderB) {
                    btnFolderB.addEventListener('click', () => {
                        openRvDeckLocalFolderPicker('b', fiQBFolder).catch(() => {});
                    }, sig);
                }
                if (btnUrlA) {
                    btnUrlA.addEventListener('click', () => {
                        try {
                            if (typeof promptAddUrlForDeck === 'function') promptAddUrlForDeck('a');
                            this._refreshDigitalLocalQueueUi();
                            this._afterDeckLocalFileDrop();
                        } catch (_) {}
                    }, sig);
                }
                if (btnUrlB) {
                    btnUrlB.addEventListener('click', () => {
                        try {
                            if (typeof promptAddUrlForDeck === 'function') promptAddUrlForDeck('b');
                            this._refreshDigitalLocalQueueUi();
                            this._afterDeckLocalFileDrop();
                        } catch (_) {}
                    }, sig);
                }
                const btnClearA = this.root.querySelector('#rv-digital-queue-clear-a');
                const btnClearB = this.root.querySelector('#rv-digital-queue-clear-b');
                const btnShuffleA = this.root.querySelector('#rv-digital-queue-shuffle-a');
                const btnShuffleB = this.root.querySelector('#rv-digital-queue-shuffle-b');
                const onShuffleQueue = (deckKey) => {
                    try {
                        if (typeof shuffleDeckFileQueue === 'function') shuffleDeckFileQueue(deckKey);
                    } catch (_) {}
                    try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                };
                const onClearQueue = (deckKey) => {
                    try {
                        if (typeof clearDeckFileQueue === 'function') clearDeckFileQueue(deckKey);
                    } catch (_) {}
                    try { this._refreshDigitalLocalQueueUi(); } catch (_) {}
                };
                if (btnClearA) {
                    btnClearA.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        onClearQueue('a');
                    }, sig);
                }
                if (btnClearB) {
                    btnClearB.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        onClearQueue('b');
                    }, sig);
                }
                if (btnShuffleA) {
                    btnShuffleA.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        onShuffleQueue('a');
                    }, sig);
                }
                if (btnShuffleB) {
                    btnShuffleB.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        onShuffleQueue('b');
                    }, sig);
                }
            }

            /** Left / right spectrum flowers: drop local audio (or video) like DJ jog wheels. */
            _wireDigitalSpectrumLocalDrop(zoneEls, deckKey, sig) {
                const zones = (Array.isArray(zoneEls) ? zoneEls : [zoneEls]).filter(Boolean);
                if (!zones.length) return;
                const dk = deckKey === 'b' ? 'b' : 'a';
                const label = dk === 'b' ? 'Deck B' : 'Deck A';
                const setDropHighlight = (on) => {
                    zones.forEach((el) => el.classList.toggle('is-local-drop-target', !!on));
                };
                const onDragOver = (ev) => {
                    try {
                        const dt = ev.dataTransfer;
                        if (!dt || !Array.from(dt.types || []).includes('Files')) return;
                        ev.preventDefault();
                        dt.dropEffect = 'copy';
                        setDropHighlight(true);
                    } catch (_) {}
                };
                const onDragLeave = (ev) => {
                    try {
                        const rel = ev.relatedTarget;
                        if (rel && zones.some((z) => z.contains(rel))) return;
                        setDropHighlight(false);
                    } catch (_) {}
                };
                const onDrop = (ev) => {
                    try {
                        ev.preventDefault();
                        ev.stopPropagation();
                    } catch (_) {}
                    setDropHighlight(false);
                    const dt = ev.dataTransfer;
                    if (!dt) return;
                    const finish = (files) => {
                        if (!files || !files.length) return;
                        this._ingestLocalFilesToDeckAndPlay(dk, files);
                        this._afterDeckLocalFileDrop();
                    };
                    if (typeof collectMediaFilesFromDataTransfer === 'function') {
                        collectMediaFilesFromDataTransfer(dt).then(finish).catch(() => {
                            if (dt.files && dt.files.length) finish(Array.from(dt.files));
                        });
                    } else if (dt.files && dt.files.length) {
                        finish(Array.from(dt.files));
                    }
                };
                zones.forEach((el) => {
                    el.addEventListener('dragover', onDragOver, sig);
                    el.addEventListener('dragleave', onDragLeave, sig);
                    el.addEventListener('drop', onDrop, sig);
                });
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

            _stagingContentMount() {
                return this.els.digitalStagingMount || null;
            }

            _toggleDigitalStagingFeature(kind) {
                const k = (kind === 'projectm' || kind === 'bars' || kind === 'queue' || kind === 'video' || kind === 'karaoke')
                    ? kind : null;
                if (!k) return;
                if (k === 'video') {
                    this._toggleDigitalVideoToolbarButton();
                    return;
                }
                if (this._digitalLocalQueueVisible) {
                    try { this._closeDigitalLocalQueuePanel(); } catch (_) {}
                }
                if (this._digitalStationsVisible) {
                    try { this._closeDigitalStationsPanel(); } catch (_) {}
                }
                if (this._digitalStagingView === k) {
                    this._digitalStagingView = null;
                    this._tearDownDigitalStagingView();
                    this._syncDigitalStagingButtons();
                    return;
                }
                this._digitalStagingView = k;
                if (this.digitalCenterMode !== 'spectrum') {
                    this._setDigitalCenterMode('spectrum');
                }
                this._setDigitalStagingView(k);
                this._syncDigitalStagingButtons();
            }

            _syncDigitalStagingButtons() {
                const grid = this.els.digitalBtns;
                const on = this._digitalStagingView;
                if (grid) {
                    grid.querySelectorAll('[data-rv-staging]').forEach((btn) => {
                        const active = !!(on && btn.dataset.rvStaging === on);
                        btn.classList.toggle('is-active', active);
                        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
                    });
                }
                if (this.els.btnDigitalVideo) {
                    this._syncDigitalVideoToolbarButton();
                }
                try { this._syncDigitalSpectrumButtonState(); } catch (_) {}
            }

            _failDigitalStagingView() {
                this._digitalStagingView = null;
                this._tearDownDigitalStagingView();
                this._syncDigitalStagingButtons();
            }

            _tearDownDigitalStagingView() {
                try { this._exitDigitalStagingPmFullscreen(); } catch (_) {}
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
                this.nextPreset = null;
                try {
                    if (typeof globalThis.updateSkipPresetButtonVisibility === 'function') {
                        globalThis.updateSkipPresetButtonVisibility();
                    }
                } catch (_) {}
                this._rvDigitalBarsRenderer = null;
                this._rvDigitalBarsScene = null;
                const mount = this._stagingContentMount();
                if (mount) {
                    mount.innerHTML = '';
                    mount.classList.remove('is-active');
                    mount.setAttribute('aria-hidden', 'true');
                    try { delete mount.dataset.rvStaging; } catch (_) {}
                }
                const stagingVids = [
                    this.els.digitalStagingVideo,
                    this.els.digitalStagingVideoA,
                    this.els.digitalStagingVideoB
                ].filter(Boolean);
                stagingVids.forEach((vid) => {
                    try {
                        vid.pause();
                        vid.removeAttribute('src');
                    } catch (_) {}
                    try { vid.classList.add('is-hidden'); } catch (_) {}
                });
                if (this.els.digitalStagingVideoStack) {
                    try { this.els.digitalStagingVideoStack.remove(); } catch (_) {}
                    this.els.digitalStagingVideoStack = null;
                }
                this.els.digitalStagingVideoA = null;
                this.els.digitalStagingVideoB = null;
                this._digitalStagingVideoSrc = '';
                this._digitalStagingVideoMode = '';
                if (this.els.digitalDeckBContent) {
                    this.els.digitalDeckBContent.innerHTML = '';
                    this.els.digitalDeckBContent.classList.remove('is-active');
                }
                if (this.els.digitalDeckBVideo) {
                    this.els.digitalDeckBVideo.classList.remove('is-hidden');
                }
                try { this._syncDigitalStagingBgVisibility(); } catch (_) {}
            }

            _setDigitalStagingView(view) {
                const mount = this._stagingContentMount();
                if (!mount) return;
                mount.classList.add('is-active');
                mount.setAttribute('aria-hidden', 'false');
                try {
                    if (view) mount.dataset.rvStaging = view;
                    else delete mount.dataset.rvStaging;
                } catch (_) {}
                try { this._syncDigitalStagingBgVisibility(); } catch (_) {}
                if (view === 'video') {
                    this._showDigitalStagingVideo();
                    return;
                }
                if (this.els.digitalStagingVideo) {
                    this.els.digitalStagingVideo.classList.add('is-hidden');
                }
                if (view === 'projectm') this._showDigitalStagingProjectM();
                else if (view === 'bars') this._showDigitalStagingAudioBars();
                else if (view === 'queue') this._showDigitalStagingQueue();
                else if (view === 'karaoke') this._showDigitalStagingKaraoke();
            }

            _normalizeDigitalEmbedUrl(url) {
                let href = (url && String(url).trim()) || '';
                if (!href) href = this._digitalKaraokeUrl || 'https://www.karaokenerds.com/#query';
                else if (!/^https?:\/\//i.test(href)) href = 'https://' + href.replace(/^\/+/, '');
                try {
                    if (typeof globalThis.normalizeKaraokeNerdsEmbedUrl === 'function') {
                        return globalThis.normalizeKaraokeNerdsEmbedUrl(href);
                    }
                } catch (_) {}
                return href;
            }

            _showDigitalStagingKaraoke(mountEl, url) {
                const mount = mountEl || this._stagingContentMount();
                if (!mount) return;
                this._tearDownDigitalStagingView();
                mount.classList.add('is-active');
                mount.setAttribute('aria-hidden', 'false');
                try { this._syncDigitalStagingBgVisibility(); } catch (_) {}
                const href = this._normalizeDigitalEmbedUrl(url);
                const shell = document.createElement('div');
                shell.className = 'radio-visual-digital-embed-shell';
                const iframe = document.createElement('iframe');
                iframe.className = 'radio-visual-digital-embed-frame';
                RadioVisualEngine.configureEmbeddableIframe(iframe, 'Karaoke Nerds');
                iframe.src = href;
                shell.appendChild(iframe);
                mount.appendChild(shell);
                try {
                    mount.title = 'Double-click for fullscreen · Esc to exit';
                } catch (_) {}
            }

            toggleDigitalStagingKaraoke(url) {
                try {
                    if (globalThis.uiLocked) return;
                } catch (_) {}
                if (url) this._digitalKaraokeUrl = this._normalizeDigitalEmbedUrl(url);
                this._toggleDigitalStagingFeature('karaoke');
            }

            _syncDigitalVideoToolbarButton() {
                const btn = this.els.btnDigitalVideo;
                if (!btn) return;
                const step = this._digitalVideoToolbarStep || 0;
                btn.classList.toggle('is-active', step > 0);
                btn.setAttribute('aria-pressed', step > 0 ? 'true' : 'false');
                const labels = [
                    'Show playing video in equaliser',
                    'Show video in deck view',
                    'Hide video'
                ];
                btn.setAttribute('aria-label', labels[Math.min(step, labels.length - 1)]);
            }

            _resetDigitalVideoToolbarStep() {
                this._digitalVideoToolbarStep = 0;
                this._syncDigitalVideoToolbarButton();
            }

            _toggleDigitalVideoToolbarButton() {
                if (this._digitalLocalQueueVisible) {
                    try { this._closeDigitalLocalQueuePanel(); } catch (_) {}
                }
                if (this._digitalStationsVisible) {
                    try { this._closeDigitalStationsPanel(); } catch (_) {}
                }
                if (this._digitalStagingView) {
                    this._digitalStagingView = null;
                    try { this._tearDownDigitalStagingView(); } catch (_) {}
                    try { this._syncDigitalStagingButtons(); } catch (_) {}
                }
                const step = this._digitalVideoToolbarStep || 0;
                if (step === 0) {
                    this._digitalVideoToolbarStep = 1;
                    if (this.digitalCenterMode !== 'spectrum') this._setDigitalCenterMode('spectrum');
                    this._digitalHubMode = 'video';
                    try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                    try { this._syncDigitalHubDeckVideo(true); } catch (_) {}
                    this._syncDigitalVideoToolbarButton();
                    return;
                }
                if (step === 1 && this.digitalCenterMode === 'spectrum' && this._digitalHubMode === 'video') {
                    this._digitalVideoToolbarStep = 2;
                    this._setDigitalCenterMode('deckB');
                    try { this._syncDigitalDeckBVideo(true); } catch (_) {}
                    this._syncDigitalVideoToolbarButton();
                    return;
                }
                this._digitalVideoToolbarStep = 0;
                this._digitalHubMode = 'equaliser';
                if (this.digitalCenterMode === 'deckB') this._setDigitalCenterMode('spectrum');
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                this._syncDigitalVideoToolbarButton();
            }

            _pauseDigitalHubDeckVideo() {
                const vid = this.els.digitalHubDeckVideo;
                if (!vid) return;
                try { if (!vid.paused) vid.pause(); } catch (_) {}
            }

            _applyDigitalHubDeckVideoPayload(vid, cur) {
                if (!vid || !cur || !cur.url) return;
                this._digitalHubDeckVideoMode = cur.mode || 'idle';
                if (!cur.syncFrom && cur.mode === 'idle') {
                    this._applyDigitalIdleLogoVideo(vid, cur.url);
                } else {
                    this._applyDigitalStagingVideoPayload(vid, cur);
                }
                this._digitalHubDeckVideoSrc = String(cur.url);
            }

            _syncDigitalHubDeckVideo(forceLoad) {
                if (!RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode)) return;
                const vid = this.els.digitalHubDeckVideo;
                if (!vid) return;
                try {
                    const payload = this._resolveDigitalStagingVideoPayload();
                    const want = String(payload.url || '');
                    if (!payload.syncFrom && payload.mode === 'idle') {
                        const same = this._digitalHubDeckVideoSrc && want && (
                            (typeof urlsMediaMatch === 'function')
                                ? urlsMediaMatch(this._digitalHubDeckVideoSrc, want)
                                : this._digitalHubDeckVideoSrc === want
                        );
                        if (!forceLoad && same && !vid.paused) return;
                        this._applyDigitalIdleLogoVideo(vid, want);
                        this._digitalHubDeckVideoSrc = want;
                        this._digitalHubDeckVideoMode = 'idle';
                        return;
                    }
                    if (!forceLoad && this._digitalHubDeckVideoSrc && want && (
                        (typeof urlsMediaMatch === 'function')
                            ? urlsMediaMatch(this._digitalHubDeckVideoSrc, want)
                            : this._digitalHubDeckVideoSrc === want
                    )) {
                        if (payload.syncFrom) {
                            this._applyDigitalHubDeckVideoPayload(vid, payload);
                        } else if (vid.paused) {
                            vid.play().catch(() => {});
                        }
                        return;
                    }
                    this._applyDigitalHubDeckVideoPayload(vid, payload);
                } catch (_) {}
            }

            _setDigitalDeckBView(view) {
                const next = (view === 'projectm' || view === 'bars' || view === 'queue') ? view : 'video';
                this._digitalDeckBView = next;
                if (next === 'video') {
                    this._tearDownDigitalStagingView();
                    if (this.els.digitalDeckBContent) {
                        this.els.digitalDeckBContent.innerHTML = '';
                        this.els.digitalDeckBContent.classList.remove('is-active');
                    }
                    if (this.els.digitalDeckBVideo) this.els.digitalDeckBVideo.classList.remove('is-hidden');
                    this._syncDigitalDeckBVideo();
                    return;
                }
                if (this.els.digitalDeckBVideo) this.els.digitalDeckBVideo.classList.add('is-hidden');
                const mount = this.els.digitalDeckBContent;
                if (!mount) return;
                mount.classList.add('is-active');
                if (next === 'projectm') this._showDigitalStagingProjectM(mount);
                else if (next === 'bars') this._showDigitalStagingAudioBars(mount);
                else if (next === 'queue') this._showDigitalStagingQueue(mount);
            }

            _syncDigitalStagingBgVisibility() {
                const bgEl = this.els.spectrumBg;
                if (!bgEl) return;
                const suppress = !!(
                    this._digitalStagingView &&
                    this.digitalCenterMode === 'spectrum' &&
                    this._digitalStagingView !== 'bars'
                );
                bgEl.classList.toggle('is-staging-suppressed', suppress);
                if (!suppress && this._isDigitalBgGifEnabled()) {
                    try {
                        if (bgEl.querySelector('img')) bgEl.classList.add('is-visible');
                    } catch (_) {}
                }
            }

            _resolveDigitalStagingDefaultVideoUrl() {
                try {
                    if (typeof DECK_B_IDLE_LOGO_URL === 'string' && DECK_B_IDLE_LOGO_URL) {
                        return DECK_B_IDLE_LOGO_URL;
                    }
                } catch (_) {}
                return 'assets/video/logo.mp4';
            }

            /** Idle hub / staging surface: loop logo.mp4 (matches applyDeckVideoMirrorToElement). */
            _applyDigitalIdleLogoVideo(vid, url) {
                if (!vid) return;
                const want = String(url || this._resolveDigitalStagingDefaultVideoUrl());
                const had = String(vid.currentSrc || vid.src || '');
                const same = (typeof urlsMediaMatch === 'function')
                    ? urlsMediaMatch(want, had)
                    : want === had;
                try { vid.loop = true; } catch (_) {}
                try { vid.muted = true; } catch (_) {}
                try { vid.playsInline = true; } catch (_) {}
                this._wireDigitalStagingVideoLoopFallback(vid);
                const play = () => { try { vid.play().catch(() => {}); } catch (_) {} };
                if (same) {
                    play();
                    return;
                }
                try { vid.pause(); } catch (_) {}
                try { vid.removeAttribute('src'); vid.load(); } catch (_) {}
                try { vid.src = want; } catch (_) {}
                play();
                try { vid.addEventListener('loadeddata', play, { once: true }); } catch (_) {}
                try { vid.addEventListener('canplay', play, { once: true }); } catch (_) {}
            }

            _stagingVideoSyncFromIsViable(syncFrom, deckKey) {
                if (!syncFrom || !deckKey) return false;
                try {
                    if (!state || !state.deckSourceMode || state.deckSourceMode[deckKey] !== 'local') {
                        return false;
                    }
                } catch (_) {
                    return false;
                }
                try {
                    if (syncFrom instanceof HTMLVideoElement) return true;
                    if (syncFrom instanceof HTMLAudioElement) {
                        const d = Number(syncFrom.duration);
                        return Number.isFinite(d) && d > 1.5 && d < 43200;
                    }
                } catch (_) {}
                return false;
            }

            _resolveDigitalStagingVideoPayload() {
                try {
                    const metaB = (typeof getDeckActiveVideoMeta === 'function') ? getDeckActiveVideoMeta('b') : null;
                    const metaA = (typeof getDeckActiveVideoMeta === 'function') ? getDeckActiveVideoMeta('a') : null;
                    const meta = metaB || metaA;
                    if (meta && meta.url) {
                        const deckKey = meta.deckKey === 'a' ? 'a' : 'b';
                        const sf = meta.syncFrom || meta.media;
                        if (this._stagingVideoSyncFromIsViable(sf, deckKey)) {
                            return {
                                url: meta.url,
                                label: meta.label,
                                syncFrom: sf,
                                mode: 'deck-sync'
                            };
                        }
                        if (sf instanceof HTMLVideoElement) {
                            return { url: meta.url, label: meta.label, loop: true, mode: 'deck' };
                        }
                    }
                } catch (_) {}
                return {
                    url: this._resolveDigitalStagingDefaultVideoUrl(),
                    label: 'Video',
                    loop: true,
                    mode: 'idle'
                };
            }

            _wireDigitalStagingVideoLoopFallback(vid) {
                if (!vid || vid.dataset.rvStagingLoopWired === '1') return;
                vid.dataset.rvStagingLoopWired = '1';
                vid.addEventListener('ended', () => {
                    try {
                        if (!vid.loop) return;
                        vid.currentTime = 0;
                        vid.play().catch(() => {});
                    } catch (_) {}
                });
            }

            _applyDigitalStagingVideoPayload(vid, cur) {
                if (!vid || !cur || !cur.url) return;
                if (!cur.syncFrom) {
                    this._setDigitalStagingVideoOpacity(vid, 1);
                }
                const want = String(cur.url);
                const had = String(vid.currentSrc || vid.src || '');
                const same = (typeof urlsMediaMatch === 'function')
                    ? urlsMediaMatch(want, had)
                    : want === had;
                const loop = cur.syncFrom ? false : !!cur.loop;
                try { vid.loop = loop; } catch (_) {}
                try { vid.muted = true; } catch (_) {}
                try { vid.playsInline = true; } catch (_) {}
                this._wireDigitalStagingVideoLoopFallback(vid);

                const playVid = () => {
                    try { vid.play().catch(() => {}); } catch (_) {}
                };

                if (!same) {
                    this._digitalStagingVideoSrc = want;
                    this._digitalStagingVideoMode = cur.mode || 'idle';
                    const onReady = () => playVid();
                    try {
                        vid.addEventListener('loadeddata', onReady, { once: true });
                    } catch (_) {}
                    try { vid.src = want; } catch (_) {}
                    try {
                        if (vid.complete && vid.naturalWidth > 0) onReady();
                    } catch (_) {}
                    return;
                }

                if (cur.syncFrom) {
                    const sf = cur.syncFrom;
                    try {
                        if (sf.paused || sf.ended) {
                            if (!vid.paused) vid.pause();
                            return;
                        }
                        if (vid.paused) playVid();
                        const drift = Math.abs(Number(vid.currentTime) - Number(sf.currentTime));
                        if (drift > 0.35) {
                            let t = Number(sf.currentTime) || 0;
                            const md = Number(sf.duration);
                            if (Number.isFinite(md) && md > 0) t = Math.min(Math.max(0, t), md - 0.05);
                            const vdur = Number(vid.duration);
                            if (Number.isFinite(vdur) && vdur > 0) t = Math.min(t, vdur - 0.05);
                            try { vid.currentTime = t; } catch (_) {}
                        }
                    } catch (_) {}
                    return;
                }

                playVid();
            }

            _ensureDigitalStagingVideoStack(mount) {
                if (!mount) return null;
                let stack = mount.querySelector('.radio-visual-digital-staging-video-stack');
                let vidA = this.els.digitalStagingVideoA;
                let vidB = this.els.digitalStagingVideoB;
                if (!stack) {
                    stack = document.createElement('div');
                    stack.className = 'radio-visual-digital-staging-video-stack';
                    vidA = document.createElement('video');
                    vidA.className = 'radio-visual-digital-staging-video radio-visual-digital-staging-video--a';
                    vidB = document.createElement('video');
                    vidB.className = 'radio-visual-digital-staging-video radio-visual-digital-staging-video--b';
                    [vidA, vidB].forEach((v) => {
                        v.playsInline = true;
                        v.muted = true;
                    });
                    stack.appendChild(vidA);
                    stack.appendChild(vidB);
                    mount.appendChild(stack);
                }
                this.els.digitalStagingVideoStack = stack;
                this.els.digitalStagingVideoA = vidA || stack.querySelector('.radio-visual-digital-staging-video--a');
                this.els.digitalStagingVideoB = vidB || stack.querySelector('.radio-visual-digital-staging-video--b');
                this.els.digitalStagingVideo = this.els.digitalStagingVideoStack;
                return { stack, vidA: this.els.digitalStagingVideoA, vidB: this.els.digitalStagingVideoB };
            }

            _showDigitalStagingVideo() {
                const mount = this._stagingContentMount();
                if (!mount) return;
                this._rvDigitalPmAnimId && cancelAnimationFrame(this._rvDigitalPmAnimId);
                this._rvDigitalPmAnimId = null;
                this._rvDigitalBarsAnimId && cancelAnimationFrame(this._rvDigitalBarsAnimId);
                this._rvDigitalBarsAnimId = null;
                mount.innerHTML = '';
                mount.classList.add('is-active');
                const stackEls = this._ensureDigitalStagingVideoStack(mount);
                if (!stackEls) return;
                const { stack, vidA, vidB } = stackEls;
                [vidA, vidB].forEach((v) => { try { v.classList.remove('is-hidden'); } catch (_) {} });
                this._wireDigitalStagingVideoFullscreen(stack, mount);
                this._digitalStagingVideoSrc = '';
                this._digitalStagingVideoMode = 'idle';
                this._syncDigitalStagingBgVisibility();
                this._syncDigitalStagingVideo(null, true);
            }

            _wireDigitalStagingVideoFullscreen(vid, mount) {
                if (!vid || !mount || !this.abortCtrl) return;
                if (vid.dataset.rvVideoFsWired === '1') return;
                vid.dataset.rvVideoFsWired = '1';
                const sig = { signal: this.abortCtrl.signal };
                try { vid.title = 'Double-click for fullscreen · Esc to exit'; } catch (_) {}
                try { mount.title = vid.title; } catch (_) {}
                const onDbl = (ev) => {
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    try {
                        if (typeof toggleVideoSurfaceFullscreen === 'function') {
                            toggleVideoSurfaceFullscreen(vid, mount);
                        }
                    } catch (_) {}
                };
                vid.addEventListener('dblclick', onDbl, sig);
                mount.addEventListener('dblclick', onDbl, sig);
            }

            _wireDigitalDeckBVideoFullscreen(vid, mount, sig) {
                if (!vid || !mount) return;
                if (vid.dataset.rvDeckBVideoFsWired === '1') return;
                vid.dataset.rvDeckBVideoFsWired = '1';
                try { vid.title = 'Double-click for fullscreen · Esc to exit'; } catch (_) {}
                const onDbl = (ev) => {
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    try {
                        if (typeof toggleVideoSurfaceFullscreen === 'function') {
                            toggleVideoSurfaceFullscreen(vid, mount);
                        }
                    } catch (_) {}
                };
                const opts = sig ? sig : undefined;
                vid.addEventListener('dblclick', onDbl, opts);
                mount.addEventListener('dblclick', onDbl, opts);
            }

            _setDigitalStagingVideoOpacity(vid, op) {
                if (!vid) return;
                const o = Math.max(0, Math.min(1, Number(op) || 0));
                try {
                    vid.style.opacity = String(o);
                    vid.style.pointerEvents = o > 0.35 ? 'auto' : 'none';
                    vid.classList.toggle('is-hidden', o <= 0.001);
                } catch (_) {}
                if (o <= 0.001) {
                    try { if (!vid.paused) vid.pause(); } catch (_) {}
                }
            }

            _setDigitalStagingVideoLayer(vid, layer, op) {
                if (!vid) return;
                const o = Math.max(0, Math.min(1, Number(op) || 0));
                this._setDigitalStagingVideoOpacity(vid, o);
                if (o <= 0.001) return;
                if (layer && layer.url && typeof applyDeckBVideoPayloadToElement === 'function') {
                    applyDeckBVideoPayloadToElement(vid, layer, null);
                }
            }

            _applyDigitalStagingVideoCrossfadeOpacities() {
                if (this._digitalStagingView !== 'video') return;
                const vidA = this.els.digitalStagingVideoA;
                const vidB = this.els.digitalStagingVideoB;
                if (!vidA || !vidB || this._digitalStagingVideoMode !== 'deck-dual') return;
                try {
                    const plan = (typeof computeDigitalStagingVideoCrossfadePlan === 'function')
                        ? computeDigitalStagingVideoCrossfadePlan()
                        : null;
                    if (!plan || !plan.dual) return;
                    this._setDigitalStagingVideoOpacity(vidA, plan.opA);
                    this._setDigitalStagingVideoOpacity(vidB, plan.opB);
                } catch (_) {}
            }

            _syncDigitalStagingVideo(vidEl, forceLoad) {
                if (!this._digitalStagingView || this._digitalStagingView !== 'video') return;
                const vidA = this.els.digitalStagingVideoA;
                const vidB = this.els.digitalStagingVideoB;
                if (!vidA || !vidB) {
                    const legacy = vidEl || this.els.digitalStagingVideo;
                    if (!legacy || legacy.classList && legacy.classList.contains('radio-visual-digital-staging-video-stack')) return;
                    try {
                        const payload = this._resolveDigitalStagingVideoPayload();
                        this._digitalStagingVideoMode = payload.mode || 'idle';
                        this._applyDigitalStagingVideoPayload(legacy, payload);
                    } catch (_) {}
                    return;
                }
                try {
                    const plan = (typeof computeDigitalStagingVideoCrossfadePlan === 'function')
                        ? computeDigitalStagingVideoCrossfadePlan()
                        : null;
                    if (plan && (plan.layerA || plan.layerB)) {
                        if (plan.dual) {
                            this._digitalStagingVideoMode = 'deck-dual';
                            this._setDigitalStagingVideoLayer(vidA, plan.layerA, plan.opA);
                            this._setDigitalStagingVideoLayer(vidB, plan.layerB, plan.opB);
                            return;
                        }
                        const single = plan.layerB && plan.opB >= plan.opA ? { layer: plan.layerB, vid: vidB, other: vidA }
                            : { layer: plan.layerA, vid: vidA, other: vidB };
                        this._digitalStagingVideoMode = 'deck-sync';
                        this._setDigitalStagingVideoLayer(single.other, null, 0);
                        this._setDigitalStagingVideoLayer(single.vid, single.layer, 1);
                        return;
                    }
                    if (!forceLoad && this._digitalStagingVideoMode === 'idle') {
                        const payload = this._resolveDigitalStagingVideoPayload();
                        const want = String(payload.url || '');
                        if (this._digitalStagingVideoSrc && want && (
                            (typeof urlsMediaMatch === 'function')
                                ? urlsMediaMatch(this._digitalStagingVideoSrc, want)
                                : this._digitalStagingVideoSrc === want
                        )) {
                            if (vidA.paused) vidA.play().catch(() => {});
                            return;
                        }
                    }
                    this._digitalStagingVideoMode = 'idle';
                    this._setDigitalStagingVideoLayer(vidB, null, 0);
                    const payload = this._resolveDigitalStagingVideoPayload();
                    if (!payload.syncFrom && payload.mode === 'idle') {
                        this._applyDigitalIdleLogoVideo(vidA, payload.url);
                        this._digitalStagingVideoSrc = String(payload.url || '');
                    } else {
                        this._applyDigitalStagingVideoPayload(vidA, payload);
                    }
                } catch (_) {}
            }

            _refreshDigitalDeckVideoMirrors() {
                try {
                    if (RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode)) {
                        this._syncDigitalHubDeckVideo(true);
                    }
                } catch (_) {}
                try {
                    if (this._digitalStagingView === 'video') {
                        this._syncDigitalStagingVideo(null);
                    }
                } catch (_) {}
                try {
                    if (this.digitalCenterMode === 'deckB') this._syncDigitalDeckBVideo();
                } catch (_) {}
            }

            _showDigitalStagingProjectM(mountEl) {
                const mount = mountEl || this._stagingContentMount();
                if (!mount) return;
                this._tearDownDigitalStagingView();
                mount.classList.add('is-active');
                try { initAudio(); } catch (_) {}
                if (!state || !state.audioCtx || typeof butterchurn === 'undefined') {
                    this._failDigitalStagingView();
                    return;
                }
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
                if (!viz) {
                    this._failDigitalStagingView();
                    return;
                }
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
                const restartPmCycle = () => {
                    try {
                        if (this._rvDigitalPmCycleTimeout) clearTimeout(this._rvDigitalPmCycleTimeout);
                    } catch (_) {}
                    schedule();
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
                this.nextPreset = () => {
                    nextPmPreset();
                    restartPmCycle();
                };
                schedule();
                try {
                    if (typeof globalThis.updateSkipPresetButtonVisibility === 'function') {
                        globalThis.updateSkipPresetButtonVisibility();
                    }
                } catch (_) {}
                this._rvDigitalPmResize = resizePm;
                resizePm();
                const loop = () => {
                    this._rvDigitalPmAnimId = requestAnimationFrame(loop);
                    try {
                        if (this._rvDigitalPmVisualizer) this._rvDigitalPmVisualizer.render();
                    } catch (_) {}
                };
                loop();
                try {
                    mount.title = 'Double-click for fullscreen · Esc to exit';
                } catch (_) {}
            }

            _getDigitalStagingMountFullscreenEl() {
                const mount = this._stagingContentMount();
                if (!mount) return null;
                const fs = document.fullscreenElement || document.webkitFullscreenElement;
                if (!fs) return null;
                if (fs === mount || mount.contains(fs)) return fs;
                return null;
            }

            _isDigitalStagingMountOnlyFullscreen() {
                const mount = this._stagingContentMount();
                const fs = document.fullscreenElement || document.webkitFullscreenElement;
                return !!(mount && fs && fs === mount);
            }

            _requestStagingFullscreen(target) {
                if (!target) return Promise.resolve(false);
                const req = target.requestFullscreen || target.webkitRequestFullscreen;
                if (!req) return Promise.resolve(false);
                try {
                    return Promise.resolve(req.call(target)).then(() => true).catch(() => false);
                } catch (_) {
                    return Promise.resolve(false);
                }
            }

            _exitStagingFullscreen() {
                try {
                    if (!this._getDigitalStagingMountFullscreenEl()) return Promise.resolve();
                    if (document.exitFullscreen) {
                        return document.exitFullscreen()
                            .then(() => this._afterDigitalStagingMountFullscreen())
                            .catch(() => {
                                this._afterDigitalStagingMountFullscreen();
                            });
                    }
                    if (document.webkitExitFullscreen) {
                        document.webkitExitFullscreen();
                        this._afterDigitalStagingMountFullscreen();
                    }
                } catch (_) {}
                return Promise.resolve();
            }

            _afterDigitalStagingMountFullscreen() {
                setTimeout(() => {
                    try { if (this._rvDigitalPmResize) this._rvDigitalPmResize(); } catch (_) {}
                    try { this.onResize(); } catch (_) {}
                }, 200);
            }

            _exitDigitalStagingMountFullscreen() {
                try { this._exitStagingFullscreen(); } catch (_) {}
            }

            _toggleDigitalStagingMountFullscreen() {
                const mount = this._stagingContentMount();
                if (!mount || !mount.classList.contains('is-active')) return;
                const view = this._digitalStagingView;
                if (!RadioVisualEngine.digitalStagingViewSupportsMountFullscreen(view)) return;
                if (view === 'projectm' && !this._rvDigitalPmCanvas) return;
                const now = Date.now();
                if (now - (this._rvStagingPmFsToggleAt || 0) < 450) return;
                this._rvStagingPmFsToggleAt = now;
                if (this._isDigitalStagingMountOnlyFullscreen()) {
                    this._exitStagingFullscreen();
                    return;
                }
                const enterMountFs = () => {
                    this._requestStagingFullscreen(mount).then((ok) => {
                        if (ok) this._afterDigitalStagingMountFullscreen();
                    });
                };
                if (this._getDigitalStagingMountFullscreenEl()) {
                    this._exitStagingFullscreen().then(enterMountFs);
                    return;
                }
                enterMountFs();
            }

            _afterDigitalStagingPmFullscreen() {
                this._afterDigitalStagingMountFullscreen();
            }

            _exitDigitalStagingPmFullscreen() {
                this._exitDigitalStagingMountFullscreen();
            }

            _toggleDigitalStagingProjectMFullscreen() {
                this._toggleDigitalStagingMountFullscreen();
            }

            _isDigitalStagingProjectMTarget(el) {
                if (!el || typeof el.closest !== 'function') return false;
                if (this._digitalStagingView !== 'projectm') return false;
                const mount = this._stagingContentMount();
                if (!mount || !mount.classList.contains('is-active')) return false;
                return !!el.closest('.radio-visual-digital-staging-mount');
            }

            _showDigitalStagingAudioBars(mountEl) {
                const mount = mountEl || this._stagingContentMount();
                if (!mount || typeof THREE === 'undefined' || typeof sceneBars !== 'function') {
                    this._failDigitalStagingView();
                    return;
                }
                this._tearDownDigitalStagingView();
                mount.classList.add('is-active');
                try { initAudio(); } catch (_) {}
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
                camera.position.z = 8;
                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                try { renderer.setClearColor(0x000000, 0); } catch (_) {}
                const canvas = renderer.domElement;
                canvas.className = 'radio-visual-digital-deck-b-canvas radio-visual-digital-deck-b-canvas--bars';
                mount.appendChild(canvas);
                try { this._syncDigitalStagingBgVisibility(); } catch (_) {}
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

            _showDigitalStagingQueue(mountEl) {
                const mount = mountEl || this._stagingContentMount();
                if (!mount) return;
                this._tearDownDigitalStagingView();
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
                this._scrollDigitalStationsAfterDeckChange('a');
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
                this._scrollDigitalStationsAfterDeckChange('b');
            }

            /** A▶: next queued local track, or random station when the Deck A queue is empty. */
            _deckANextOrStation() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                const q = (typeof deckFileQueues !== 'undefined' && deckFileQueues.a) ? deckFileQueues.a : [];
                if (q.length > 0) {
                    try {
                        if (typeof playDeckATrackFromQueue === 'function') {
                            playDeckATrackFromQueue({ forceImmediate: true });
                        }
                    } catch (_) {}
                    try { this._updateStationUi(); } catch (_) {}
                    try { this._syncDeckSwitches(); } catch (_) {}
                    return;
                }
                this._stationRand();
            }

            _deckAPrevOrStation() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    if (typeof globalThis.cancelActiveAutoFade === 'function') {
                        globalThis.cancelActiveAutoFade();
                    }
                } catch (_) {}
                try {
                    if (typeof globalThis.goPreviousDeckPlayback === 'function') {
                        globalThis.goPreviousDeckPlayback('a');
                    } else if (typeof globalThis.goPreviousStation === 'function') {
                        globalThis.goPreviousStation();
                    } else {
                        this._stationPrev();
                    }
                } catch (_) {
                    this._stationPrev();
                }
                try { this._updateStationUi(); } catch (_) {}
                try { this._syncDeckSwitches(); } catch (_) {}
            }

            _deckBPrevOrStation() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    if (typeof globalThis.cancelActiveAutoFade === 'function') {
                        globalThis.cancelActiveAutoFade();
                    }
                } catch (_) {}
                try {
                    if (typeof globalThis.goPreviousDeckPlayback === 'function') {
                        globalThis.goPreviousDeckPlayback('b');
                    } else {
                        this._stationBPrev();
                    }
                } catch (_) {
                    this._stationBPrev();
                }
                try { this._updateStationUi(); } catch (_) {}
                try { this._syncDeckSwitches(); } catch (_) {}
            }

            _pauseDeckOutput(deckKey) {
                const dk = deckKey === 'b' ? 'b' : 'a';
                try {
                    if (typeof globalThis.cancelActiveAutoFade === 'function') {
                        globalThis.cancelActiveAutoFade();
                    }
                } catch (_) {}
                try {
                    if (typeof globalThis.clearCrossfadeResumeSuppress === 'function') {
                        globalThis.clearCrossfadeResumeSuppress();
                    }
                } catch (_) {}
                try {
                    if (dk === 'b') {
                        const mediaB = this._deckBPlaybackMedia();
                        if (mediaB && !mediaB.paused) mediaB.pause();
                    } else {
                        const mediaA = (typeof getDeckAMediaForPlaybackState === 'function')
                            ? getDeckAMediaForPlaybackState()
                            : audioEl;
                        if (mediaA && !mediaA.paused) mediaA.pause();
                    }
                } catch (_) {}
                try { this._syncDeckSwitches(); } catch (_) {}
            }

            /** A▶ / B▶ toolbar tap: resume paused output, else next local or random station. */
            async _deckTransportTap(deckKey) {
                const dk = deckKey === 'b' ? 'b' : 'a';
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    if (typeof globalThis.cancelActiveAutoFade === 'function') {
                        globalThis.cancelActiveAutoFade();
                    }
                } catch (_) {}
                if (dk === 'a') {
                    const media = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    if (media && this._deckHasSource(media) && media.paused && !media.ended) {
                        this._deckEngClearSuppress();
                        try {
                            if (typeof releaseAutoMixDeferredLocal === 'function') {
                                releaseAutoMixDeferredLocal('a', 'play');
                            }
                        } catch (_) {}
                        try {
                            await media.play();
                        } catch (_) {
                            try {
                                if (typeof playDeckATrackFromQueue === 'function') {
                                    playDeckATrackFromQueue({ forceImmediate: true });
                                } else if (typeof playRadio === 'function') {
                                    playRadio();
                                }
                            } catch (_) {}
                        }
                        try { this._updateStationUi(); } catch (_) {}
                        try { this._syncDeckSwitches(); } catch (_) {}
                        return;
                    }
                    this._deckANextOrStation();
                    return;
                }
                const mediaB = this._deckBPlaybackMedia();
                if (mediaB && this._deckHasSource(mediaB) && mediaB.paused && !mediaB.ended) {
                    this._deckEngClearSuppress();
                    try {
                        if (typeof releaseAutoMixDeferredLocal === 'function') {
                            releaseAutoMixDeferredLocal('b', 'play');
                        }
                    } catch (_) {}
                    try {
                        await mediaB.play();
                        try {
                            if (typeof connectDeckMediaToEq === 'function') connectDeckMediaToEq('b');
                        } catch (_) {}
                    } catch (_) {
                        try {
                            if (typeof playDeckBTrackFromQueue === 'function') {
                                playDeckBTrackFromQueue({ forceImmediate: true });
                            } else if (typeof playRadioB === 'function') {
                                playRadioB();
                            }
                        } catch (_) {}
                    }
                    try { this._updateStationUi(); } catch (_) {}
                    try { this._syncDeckSwitches(); } catch (_) {}
                    return;
                }
                this._deckBNextOrStation();
            }

            /** B▶: next queued local track, or random station when the Deck B queue is empty. */
            _deckBNextOrStation() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                const q = (typeof deckFileQueues !== 'undefined' && deckFileQueues.b) ? deckFileQueues.b : [];
                if (q.length > 0) {
                    try {
                        if (typeof playDeckBTrackFromQueue === 'function') {
                            playDeckBTrackFromQueue({ forceImmediate: true });
                        }
                    } catch (_) {}
                    try { this._updateStationUi(); } catch (_) {}
                    try { this._syncDeckSwitches(); } catch (_) {}
                    return;
                }
                this._stationBRand();
            }

            /** Crossfader-winning deck: next queued local track, else random station (N shortcut, like A▶ / B▶). */
            _crossfadedDeckNextOrStation() {
                const x = this._getCrossfadeX();
                if (x < 0.5) this._deckANextOrStation();
                else this._deckBNextOrStation();
            }

            /** Crossfader-winning deck: previous station or local file from deck history. */
            _crossfadedDeckPrevOrStation() {
                const x = this._getCrossfadeX();
                const deck = x < 0.5 ? 'a' : 'b';
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    if (typeof globalThis.cancelActiveAutoFade === 'function') {
                        globalThis.cancelActiveAutoFade();
                    }
                } catch (_) {}
                try {
                    if (typeof globalThis.goPreviousDeckPlayback === 'function') {
                        globalThis.goPreviousDeckPlayback(deck);
                    } else if (deck === 'a') {
                        this._deckAPrevOrStation();
                    } else {
                        this._deckBPrevOrStation();
                    }
                } catch (_) {
                    if (deck === 'a') this._deckAPrevOrStation();
                    else this._deckBPrevOrStation();
                }
                try { this._updateStationUi(); } catch (_) {}
                try { this._syncDeckSwitches(); } catch (_) {}
            }

            triggerNextFromShortcut() {
                try { this._crossfadedDeckNextOrStation(); } catch (_) {}
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
                try {
                    if (this.skin === 'digital') {
                        const dig = this.els.crossDigital || document.getElementById('radio-visual-cross-digital');
                        if (dig) {
                            return Math.max(0, Math.min(1, Number(dig.value) || 0));
                        }
                    }
                } catch (_) {}
                const dc = document.getElementById('dj-crossfader');
                const mc = document.getElementById('mix-crossfader');
                const rd = document.getElementById('radio-visual-cross-digital');
                const raw = (dc && dc.value) || (mc && mc.value) || (rd && rd.value) || 0;
                return Math.max(0, Math.min(1, Number(raw) || 0));
            }

            _setCrossfadeX(x) {
                const v = Math.max(0, Math.min(1, Number(x) || 0));
                try {
                    if (typeof applyCrossfade === 'function') applyCrossfade(v);
                } catch (_) {}
                if (this.els.crossDigital) this.els.crossDigital.value = String(v);
                this._syncCrossfadeKnob();
                try { this._applyDigitalStagingVideoCrossfadeOpacities(); } catch (_) {}
                if (this.els.digitalToolbarCenterXfade) {
                    this._syncDigitalToolbarCenterXfadeSlider();
                }
                if (this._digitalToolbarCenterMode !== 'crossfade') {
                    this._syncDigitalToolbarCenterReadout();
                }
            }
            _wireDigitalCrossfadeCutHold(crossEl, sig) {
                if (!crossEl || crossEl.dataset.cutFadeHold === '1') return;
                crossEl.dataset.cutFadeHold = '1';
                const valueFromPointer = (typeof crossfadeValueFromPointerOnRange === 'function')
                    ? crossfadeValueFromPointerOnRange
                    : (typeof globalThis.crossfadeValueFromPointerOnRange === 'function')
                        ? globalThis.crossfadeValueFromPointerOnRange
                        : (rangeEl, clientX) => {
                            if (!rangeEl || !Number.isFinite(Number(clientX))) return 0;
                            const rect = rangeEl.getBoundingClientRect();
                            const w = Math.max(1, rect.width);
                            const t = (Number(clientX) - rect.left) / w;
                            const min = Number(rangeEl.min) || 0;
                            const max = Number(rangeEl.max) || 1;
                            return Math.max(min, Math.min(max, min + t * (max - min)));
                        };
                let cutFadeHoldActive = false;
                let cutFadeHoldRestore = 0;
                let cutFadeHoldPointerId = null;
                const restoreCutFadeHold = () => {
                    if (!cutFadeHoldActive) return;
                    cutFadeHoldActive = false;
                    cutFadeHoldPointerId = null;
                    try { this._setCrossfadeX(cutFadeHoldRestore); } catch (_) {}
                    try { this._resumeDecksForCrossfade(); } catch (_) {}
                };
                const onCutFadePointerDown = (ev) => {
                    if (ev.button !== 2) return;
                    try { ev.preventDefault(); } catch (_) {}
                    cutFadeHoldRestore = this._getCrossfadeX();
                    const cutTo = valueFromPointer(crossEl, ev.clientX);
                    cutFadeHoldActive = true;
                    cutFadeHoldPointerId = ev.pointerId;
                    try { this._setCrossfadeX(cutTo); } catch (_) {}
                    try { this._resumeDecksForCrossfade(); } catch (_) {}
                    try { crossEl.setPointerCapture(ev.pointerId); } catch (_) {}
                };
                const onCutFadePointerEnd = (ev) => {
                    if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                    try { crossEl.releasePointerCapture(ev.pointerId); } catch (_) {}
                    restoreCutFadeHold();
                };
                const opts = sig || undefined;
                crossEl.addEventListener('pointerdown', onCutFadePointerDown, opts);
                crossEl.addEventListener('pointerup', onCutFadePointerEnd, opts);
                crossEl.addEventListener('pointercancel', onCutFadePointerEnd, opts);
                crossEl.addEventListener('lostpointercapture', (ev) => {
                    if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                    restoreCutFadeHold();
                }, opts);
                crossEl.addEventListener('contextmenu', (e) => {
                    try { e.preventDefault(); } catch (_) {}
                }, opts);
            }

            _syncCrossfadeKnob() {
                const x = this._getCrossfadeX();
                if (this.els.crossKnob) {
                    this._setKnobRotation(this.els.crossKnob, (x * 270) - 135);
                }
                if (this.els.crossDigital) this.els.crossDigital.value = String(x);
                try { this._syncDigitalCrossfadeLabels(); } catch (_) {}
                if (this._isRadioVisualActive()) this._updateHudModeLines();
            }

            /** Scale A/B labels beside the digital dash crossfader with fader position. */
            _syncDigitalCrossfadeLabels() {
                const wrap = this.els.digitalDashXfade;
                if (!wrap) return;
                const x = Math.max(0, Math.min(1, this._getCrossfadeX()));
                const minScale = 0.88;
                const maxScale = 1.42;
                const span = maxScale - minScale;
                wrap.style.setProperty('--xf-label-scale-a', String(minScale + ((1 - x) * span)));
                wrap.style.setProperty('--xf-label-scale-b', String(minScale + (x * span)));
            }

            /** Deck B has a local track cued/loaded (may be paused while crossfader favours A). */
            _deckBLcdLineReady() {
                if (this._deckBActive()) return true;
                try {
                    if (state.deckSourceMode && state.deckSourceMode.b === 'local') {
                        const el = this._deckBPlaybackMedia();
                        if (this._deckHasSource(el)) return true;
                        const dn = state.deckLocalDisplayName && state.deckLocalDisplayName.b;
                        if (dn && String(dn).trim()) return true;
                    }
                } catch (_) {}
                return false;
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

            _audibleDeckStationName() {
                try {
                    const dk = this._crossfaderAudibleDeckKey();
                    if (typeof globalThis.getDeckStationDisplayName === 'function') {
                        return String(globalThis.getDeckStationDisplayName(dk) || '').trim();
                    }
                    return String(this._stationNameForDeck(dk) || '').trim();
                } catch (_) {}
                return '';
            }

            _deckStationDisplayName(deck) {
                try {
                    const dk = deck === 'b' ? 'b' : 'a';
                    if (typeof globalThis.getDeckStationDisplayName === 'function') {
                        return String(globalThis.getDeckStationDisplayName(dk) || '').trim();
                    }
                    return String(this._stationNameForDeck(dk) || '').trim();
                } catch (_) {}
                return '';
            }

            _audibleDeckElement() {
                try {
                    const dk = this._crossfaderAudibleDeckKey();
                    if (dk === 'b') {
                        if (typeof globalThis.getDeckBRadioAudibleEl === 'function') {
                            return globalThis.getDeckBRadioAudibleEl();
                        }
                        return (typeof audioElB !== 'undefined' ? audioElB : null);
                    }
                    if (typeof globalThis.getDeckARadioAudibleEl === 'function') {
                        return globalThis.getDeckARadioAudibleEl();
                    }
                    return (typeof audioEl !== 'undefined' ? audioEl : null);
                } catch (_) {}
                return null;
            }

            _audibleElementIsPlaying(el) {
                if (!el) return false;
                try {
                    if (!el.paused && el.readyState >= 2) return true;
                    if (el.currentTime > 0 && !el.ended) return true;
                } catch (_) {}
                return false;
            }

            _analyserMeanLevel() {
                try {
                    if (!state.analyserNode || !state.audioCtx) return 0;
                    const fft = state.analyserNode.fftSize || 256;
                    const binCount = state.analyserNode.frequencyBinCount || Math.floor(fft / 2);
                    if (!this._vuBuf || this._vuBuf.length < binCount) {
                        this._vuBuf = new Uint8Array(binCount);
                    }
                    state.analyserNode.getByteFrequencyData(this._vuBuf);
                    let sum = 0;
                    for (let i = 0; i < binCount; i++) sum += this._vuBuf[i] || 0;
                    return sum / Math.max(1, binCount);
                } catch (_) {}
                return 0;
            }

            _isSpectrumAudioActive() {
                try {
                    if (this._audibleElementIsPlaying(this._audibleDeckElement())) return true;
                    if (this._analyserMeanLevel() > 5) return true;
                    if (state && state.isPlaying && this._analyserMeanLevel() > 2) return true;
                } catch (_) {}
                return false;
            }

            _isAudiblePlaybackActive() {
                return this._isSpectrumAudioActive();
            }

            _digitalLcdPrimaryLine() {
                const name = this._deckStationDisplayName('a');
                if (name && name !== '—') return name;
                return 'DIGITAL TUNER';
            }

            _digitalLcdSecondaryLine() {
                if (!this._deckBLcdLineReady()) return 'STEREO · EQ';
                const name = this._deckStationDisplayName('b');
                if (name && name !== '—') return name;
                return 'STEREO · EQ';
            }

            _fullBandRadii(n, maxRing) {
                const fill = maxRing ?? 0.66;
                return new Array(n).fill(fill);
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
                    const visName = this.name || 'Radio';
                    if (titleEl) titleEl.textContent = visName;
                    if (subEl) {
                        const deck = this._crossfaderAudibleDeckKey();
                        subEl.textContent = this._stationNameForDeck(deck);
                    }
                } catch (_) {}
            }

            _isRadioVisualActive() {
                const vis = state && state.activeVisualizer;
                return !!(vis && (vis === this || RadioVisualEngine.isRadioModeName(vis.name)));
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
                if (this._suppressCrossfadeResume) return;
                try {
                    const x = this._getCrossfadeX();
                    const ga = 1 - x;
                    const gb = x;
                    const thresh = 0.03;
                    const mediaA = (typeof getDeckAMediaForPlaybackState === 'function')
                        ? getDeckAMediaForPlaybackState()
                        : audioEl;
                    const mediaB = (typeof getDeckBRadioAudibleEl === 'function')
                        ? getDeckBRadioAudibleEl()
                        : audioElB;
                    if (ga > thresh && mediaA && mediaA.src && mediaA.paused) {
                        mediaA.play().catch(() => {});
                    }
                    if (gb > thresh && mediaB && mediaB.src && mediaB.paused) {
                        mediaB.play().catch(() => {});
                    }
                } catch (_) {}
            }

            /**
             * Space-bar short tap: if nothing is playing, start the deck the crossfader
             * favours; once audio is flowing, run auto-fade (same as DJ Decks Space tap).
             */
            triggerAutoFadeFromShortcut() {
                try {
                    this._clearSuppressCrossfadeResume();
                    if (this._bothDecksSilent()) {
                        this._startActiveDeckByCrossfader();
                        return;
                    }
                    this._triggerAutoFade();
                } catch (_) {}
            }

            triggerFadeFromShortcut() {
                this.triggerAutoFadeFromShortcut();
            }

            triggerVisBgFromShortcut() {
                try { this._onDigitalVisBgTap(); } catch (_) {}
            }

            /** / — toggle Digital Radio Ai button (auto-mix + AI video hub). */
            toggleDigitalAiFromShortcut() {
                if (this.skin !== 'digital') return;
                this._toggleAutoMix();
                if (this._isAutoMixEnabled()) this._enterDigitalHubAiMode();
                else this._exitDigitalHubAiMode();
            }

            /** C: ProjectM staging → next preset; otherwise cycle centre hub (Equaliser → … → Off). */
            triggerCFromShortcut() {
                if (this._digitalStagingView === 'projectm' && typeof this.nextPreset === 'function') {
                    try { this.nextPreset(); } catch (_) {}
                    return;
                }
                if (this.digitalCenterMode !== 'spectrum' || this._digitalStagingView) {
                    try { this._returnToDefaultDigitalSpectrumView(); } catch (_) {}
                    return;
                }
                try { this._cycleDigitalSpectrumLayout(); } catch (_) {}
            }

            /** Long-hold 🔆 / I: toggle background GIF layer off or on (same as pointer long-press on the button). */
            triggerVisBgLongHoldFromShortcut() {
                try {
                    this._toggleDigitalBgGifEnabled();
                    this._syncDigitalVisBgButton();
                } catch (_) {}
            }

            _digitalKeyboardBlocksKey(ev) {
                try {
                    if (globalThis.uiLocked || globalThis.shortcutsLocked) return true;
                    const ae = document.activeElement;
                    if (!ae) return false;
                    if (ae.tagName === 'TEXTAREA' || ae.tagName === 'SELECT' || ae.isContentEditable
                        || (ae.closest && ae.closest('#textin-panel'))) {
                        return true;
                    }
                    if (ae.tagName === 'INPUT') {
                        const it = String(ae.type || '').toLowerCase();
                        if (it === 'range' && ae.closest
                            && (ae.closest('#dj-visual-root') || ae.closest('#radio-visual-root'))) {
                            return false;
                        }
                        return true;
                    }
                } catch (_) {}
                return false;
            }

            _resetDigitalSpaceShortcutState() {
                this._digitalSpaceHeld = false;
                this._digitalSpaceAt = 0;
                this._digitalSpaceLongFired = false;
                if (this._digitalSpaceTimer) {
                    try { clearTimeout(this._digitalSpaceTimer); } catch (_) {}
                    this._digitalSpaceTimer = null;
                }
            }

            _resetDigitalIShortcutState() {
                this._digitalIHeld = false;
                this._digitalIAt = 0;
                this._digitalILongFired = false;
                if (this._digitalITimer) {
                    try { clearTimeout(this._digitalITimer); } catch (_) {}
                    this._digitalITimer = null;
                }
            }

            /** Digital Radio only: Space = play/fade / long-hold pause; I = 🔆 tap / long-hold GIF off-on. */
            _wireDigitalKeyboardShortcuts(sig) {
                const HOLD_MS = 500;
                this._resetDigitalSpaceShortcutState();
                this._resetDigitalIShortcutState();
                const onKeyDown = (ev) => {
                    if (!ev || this._digitalKeyboardBlocksKey(ev)) return;
                    const isSpace = ev.code === 'Space' || ev.key === ' ';
                    const isI = (ev.key === 'i' || ev.key === 'I')
                        && !ev.ctrlKey && !ev.metaKey && !ev.altKey && !ev.shiftKey;
                    const isC = (ev.key === 'c' || ev.key === 'C')
                        && !ev.ctrlKey && !ev.metaKey && !ev.altKey && !ev.shiftKey;
                    const isN = (ev.key === 'n' || ev.key === 'N')
                        && !ev.ctrlKey && !ev.metaKey && !ev.altKey && !ev.shiftKey;
                    if (!isSpace) {
                        this._resetDigitalSpaceShortcutState();
                    }
                    if (!isI) {
                        this._resetDigitalIShortcutState();
                    }
                    if (isN) {
                        if (!this._isRadioVisualActive() || this.skin !== 'digital') return;
                        try {
                            ev.preventDefault();
                            ev.stopPropagation();
                        } catch (_) {}
                        if (ev.repeat) return;
                        try { this.triggerNextFromShortcut(); } catch (_) {}
                        return;
                    }
                    if (isC) {
                        if (!this._isRadioVisualActive() || this.skin !== 'digital') return;
                        try {
                            ev.preventDefault();
                            ev.stopPropagation();
                        } catch (_) {}
                        if (ev.repeat) return;
                        try { this.triggerCFromShortcut(); } catch (_) {}
                        return;
                    }
                    if (isI) {
                        if (!this._isRadioVisualActive() || this.skin !== 'digital') return;
                        try {
                            ev.preventDefault();
                            ev.stopPropagation();
                        } catch (_) {}
                        if (ev.repeat) return;
                        this._resetDigitalIShortcutState();
                        this._digitalIHeld = true;
                        this._digitalIAt = performance.now();
                        this._digitalITimer = setTimeout(() => {
                            this._digitalITimer = null;
                            if (!this._digitalIHeld) return;
                            this._digitalILongFired = true;
                            try { this.triggerVisBgLongHoldFromShortcut(); } catch (_) {}
                        }, HOLD_MS);
                        return;
                    }
                    if (!isSpace) {
                        return;
                    }
                    if (!this._isRadioVisualActive() || this.skin !== 'digital') return;
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    if (ev.repeat) return;
                    try {
                        const ae = document.activeElement;
                        if (ae && ae !== document.body && typeof ae.blur === 'function'
                            && ae.closest && ae.closest('#radio-visual-root')) {
                            ae.blur();
                        }
                    } catch (_) {}
                    this._resetDigitalSpaceShortcutState();
                    this._digitalSpaceHeld = true;
                    this._digitalSpaceAt = performance.now();
                    this._digitalSpaceTimer = setTimeout(() => {
                        this._digitalSpaceTimer = null;
                        if (!this._digitalSpaceHeld) return;
                        this._digitalSpaceLongFired = true;
                        try { this.pauseBothDecksOrStartActive(); } catch (_) {}
                    }, HOLD_MS);
                };
                const onKeyUp = (ev) => {
                    if (!ev || !this._isRadioVisualActive() || this.skin !== 'digital') {
                        this._resetDigitalSpaceShortcutState();
                        this._resetDigitalIShortcutState();
                        return;
                    }
                    const isI = ev.key === 'i' || ev.key === 'I';
                    const isSpace = ev.code === 'Space' || ev.key === ' ';
                    if (isI && (this._digitalIAt || this._digitalIHeld)) {
                        try {
                            ev.preventDefault();
                            ev.stopPropagation();
                        } catch (_) {}
                        const held = this._digitalIAt ? (performance.now() - this._digitalIAt) : HOLD_MS;
                        const longFired = this._digitalILongFired;
                        this._resetDigitalIShortcutState();
                        if (!longFired && held < HOLD_MS) {
                            try { this.triggerVisBgFromShortcut(); } catch (_) {}
                        }
                        return;
                    }
                    if (!isSpace) return;
                    if (!this._digitalSpaceAt && !this._digitalSpaceHeld) return;
                    try {
                        ev.preventDefault();
                        ev.stopPropagation();
                    } catch (_) {}
                    const held = this._digitalSpaceAt ? (performance.now() - this._digitalSpaceAt) : HOLD_MS;
                    const longFired = this._digitalSpaceLongFired;
                    this._resetDigitalSpaceShortcutState();
                    if (!longFired && held < HOLD_MS) {
                        try { this.triggerFadeFromShortcut(); } catch (_) {}
                    }
                };
                document.addEventListener('keydown', onKeyDown, { capture: true, signal: sig.signal });
                document.addEventListener('keyup', onKeyUp, { capture: true, signal: sig.signal });
            }

            /** Stop in-flight radio auto-fade so manual play/pause (V / B) is not overridden each frame. */
            cancelAutoFade() {
                if (this._rvAutoFadeRaf) {
                    try { cancelAnimationFrame(this._rvAutoFadeRaf); } catch (_) {}
                    this._rvAutoFadeRaf = null;
                }
                this._rvFadeTargetDeck = null;
                this._rvFadeActive = false;
                if (this._rvFadeLedTimer) {
                    try { clearTimeout(this._rvFadeLedTimer); } catch (_) {}
                    this._rvFadeLedTimer = null;
                }
                try { this._syncFadeKnobs(); } catch (_) {}
            }

            _clearSuppressCrossfadeResume() {
                this._suppressCrossfadeResume = false;
            }

            _bothDecksSilent() {
                return !this._deckAActive() && !this._deckBActive();
            }

            _startActiveDeckByCrossfader() {
                this._clearSuppressCrossfadeResume();
                const x = this._getCrossfadeX();
                if (x < 0.5) {
                    try { this._startDeckA(); } catch (_) {}
                } else {
                    try { this._startDeckB(); } catch (_) {}
                }
            }

            /** Space long-hold: pause both decks (and cancel fade), or start the crossfader-favoured deck if silent. */
            pauseBothDecksOrStartActive() {
                try {
                    if (!this._bothDecksSilent()) {
                        this._suppressCrossfadeResume = true;
                        this.cancelAutoFade();
                        const mediaA = (typeof getDeckAMediaForPlaybackState === 'function')
                            ? getDeckAMediaForPlaybackState()
                            : audioEl;
                        const mediaB = this._deckBPlaybackMedia();
                        try { if (mediaA && !mediaA.paused) mediaA.pause(); } catch (_) {}
                        try { if (mediaB && !mediaB.paused) mediaB.pause(); } catch (_) {}
                        return;
                    }
                    this._startActiveDeckByCrossfader();
                } catch (_) {}
            }

            _runLocalAutoFade() {
                if (this._isAutoMixEnabled()) {
                    this._clearRadioAutoMixTimeoutOnly();
                }
                const fadeInFlight = !!(this._rvAutoFadeRaf && this._rvFadeTargetDeck);
                const prevFadeTarget = this._rvFadeTargetDeck;
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
                if (fadeInFlight && prevFadeTarget) {
                    const destX = prevFadeTarget === 'b' ? 1 : 0;
                    if (Math.abs(x - destX) > 0.001) {
                        targetDeck = prevFadeTarget === 'a' ? 'b' : 'a';
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
                this._ensureIncomingDeckPlaybackForAutoFade(targetDeck);
                const fullDurMs = this._readAutoFadeDurationMs();
                const travel = Math.abs(endVal - startVal);
                const durMs = Math.max(300, Math.round(fullDurMs * travel));
                if (this._rvFadeLedTimer) {
                    try { clearTimeout(this._rvFadeLedTimer); } catch (_) {}
                    this._rvFadeLedTimer = null;
                }
                this._rvFadeLedTimer = setTimeout(() => {
                    if (!this._rvAutoFadeRaf) {
                        this._rvFadeActive = false;
                        this._syncFadeKnobs();
                    }
                    this._rvFadeLedTimer = null;
                }, durMs + 120);
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
                        this._rvAutoMixCyclePending = false;
                        if (this._isAutoMixEnabled()) {
                            try { this._scheduleRadioAutoMix(); } catch (_) {}
                        }
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
                    if (this._isAutoMixEnabled()) {
                        try { this._scheduleRadioAutoMix(); } catch (_) {}
                    }
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

            _formatAutoMixCountdown(ms) {
                if (!Number.isFinite(ms) || ms <= 0) return '0:00';
                const totalSec = Math.max(0, Math.ceil(ms / 1000));
                const h = Math.floor(totalSec / 3600);
                const m = Math.floor((totalSec % 3600) / 60);
                const sec = totalSec % 60;
                if (h > 0) {
                    return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
                }
                return `${m}:${String(sec).padStart(2, '0')}`;
            }

            _clearAutoMixNextFadeTick() {
                if (this._rvAutoMixNextFadeIntervalId) {
                    try { clearInterval(this._rvAutoMixNextFadeIntervalId); } catch (_) {}
                    this._rvAutoMixNextFadeIntervalId = null;
                }
            }

            _clearAutoMixNextFadeUi() {
                if (this.els.digitalAutoMixTimer) {
                    this.els.digitalAutoMixTimer.textContent = '';
                    this.els.digitalAutoMixTimer.classList.add('display-none');
                    this.els.digitalAutoMixTimer.setAttribute('aria-hidden', 'true');
                }
                if (this._digitalToolbarCenterMode === 'automix' && !this._digitalToolbarVolumePeekActive) {
                    this._syncDigitalToolbarCenterReadout();
                }
            }

            _updateAutoMixNextFadeUi() {
                if (!this._isAutoMixEnabled() || !this._rvAutoMixNextFadeAt) {
                    this._clearAutoMixNextFadeUi();
                    return;
                }
                if (this._digitalToolbarCenterMode === 'automix' && !this._digitalToolbarVolumePeekActive) {
                    this._syncDigitalToolbarCenterReadout();
                }
            }

            _syncAutoMixNextFadeDisplay() {
                this._clearAutoMixNextFadeTick();
                this._updateAutoMixNextFadeUi();
                if (!this._isAutoMixEnabled() || !this._rvAutoMixNextFadeAt) return;
                this._rvAutoMixNextFadeIntervalId = setInterval(() => {
                    try { this._updateAutoMixNextFadeUi(); } catch (_) {}
                }, 1000);
            }

            _clearRadioAutoMixTimeoutOnly() {
                if (this._rvAutoMixTimerId) {
                    try { clearTimeout(this._rvAutoMixTimerId); } catch (_) {}
                    this._rvAutoMixTimerId = null;
                }
            }

            _clearRadioAutoMixTimer() {
                this._clearRadioAutoMixTimeoutOnly();
                this._rvAutoMixNextFadeAt = 0;
                this._clearAutoMixNextFadeTick();
                this._clearAutoMixNextFadeUi();
            }

            _scheduleRadioAutoMix() {
                this._clearRadioAutoMixTimer();
                if (!this._isAutoMixEnabled() || !this._isRadioVisualActive()) return;
                const maxMin = this._readAutoMixMaxMin();
                const minMs = RadioVisualEngine.AUTOMIX_MIN_MIN * 60 * 1000;
                const maxMs = Math.max(minMs, maxMin * 60 * 1000);
                const waitMs = minMs + Math.floor(Math.random() * Math.max(1, maxMs - minMs + 1));
                this._rvAutoMixNextFadeAt = Date.now() + waitMs;
                this._syncAutoMixNextFadeDisplay();
                this._rvAutoMixTimerId = setTimeout(() => {
                    this._rvAutoMixTimerId = null;
                    if (!this._isAutoMixEnabled() || !this._isRadioVisualActive()) return;
                    this._rvAutoMixCyclePending = true;
                    this._runLocalAutoFade();
                }, waitMs);
            }

            _toggleAutoMix() {
                if (this._isRadioVisualActive()) {
                    const next = !this._isAutoMixEnabled();
                    try { localStorage.setItem(RadioVisualEngine.AUTOMIX_ENABLED_KEY, next ? '1' : '0'); } catch (_) {}
                    try { state.autoMixEnabled = next; } catch (_) {}
                    if (next) {
                        try { this._setAutoFadeChangeStationEnabled(true); } catch (_) {}
                        this._scheduleRadioAutoMix();
                        try { this._showDigitalToolbarAutomixCountdown(); } catch (_) {}
                    } else {
                        this._clearRadioAutoMixTimer();
                        this._rvAutoMixCyclePending = false;
                        if (this._digitalToolbarCenterMode === 'automix') {
                            try { this._setDigitalToolbarCenterMode('clock'); } catch (_) {}
                        }
                    }
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
                if (cb) {
                    cb.checked = next;
                    try { cb.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
                }
                this._syncAutoFadeChangeStationKnob();
            }

            _retuneIncomingDeckForAutoFade(targetDeck) {
                if (!this._isAutoFadeChangeStationEnabled()) return;
                try {
                    if (state.deckSourceMode && state.deckSourceMode[targetDeck] === 'local') return;
                } catch (_) {}
                try {
                    if (targetDeck === 'b') {
                        if (typeof pickRandomStationB === 'function') pickRandomStationB();
                    } else if (typeof pickRandomStation === 'function') pickRandomStation();
                } catch (_) {}
            }

            _ensureIncomingDeckPlaybackForAutoFade(targetDeck) {
                const dk = targetDeck === 'b' ? 'b' : 'a';
                try {
                    const localMode = !!(state && state.deckSourceMode && state.deckSourceMode[dk] === 'local');
                    const media = dk === 'b'
                        ? this._deckBPlaybackMedia()
                        : ((typeof getDeckAMediaForPlaybackState === 'function')
                            ? getDeckAMediaForPlaybackState()
                            : audioEl);
                    const src = media ? String(media.currentSrc || media.src || '') : '';
                    if (localMode && src && src !== 'about:blank') {
                        if (media.paused) media.play().catch(() => {});
                        return;
                    }
                    if (!media || !src || src === 'about:blank') {
                        if (dk === 'b') {
                            if (typeof playRadioB === 'function') playRadioB();
                        } else if (typeof playRadio === 'function') playRadio();
                        return;
                    }
                    if (media.paused) media.play().catch(() => {});
                } catch (_) {
                    try {
                        if (dk === 'b') {
                            if (typeof playRadioB === 'function') playRadioB();
                        } else if (typeof playRadio === 'function') playRadio();
                    } catch (_) {}
                }
            }

            _setAutoFadeChangeStationEnabled(on) {
                try {
                    localStorage.setItem(RadioVisualEngine.AUTOFADE_CHANGE_STATION_KEY, on ? '1' : '0');
                } catch (_) {}
                const cb = document.getElementById('dj-autofade-change-station');
                if (cb) cb.checked = !!on;
                this._syncAutoFadeChangeStationKnob();
            }

            _syncAutoMixKnob() {
                const on = this._isAutoMixEnabled();
                if (this.els.autoMixKnob) {
                    this.els.autoMixKnob.classList.toggle('is-on', on);
                    this.els.autoMixKnob.setAttribute('aria-pressed', on ? 'true' : 'false');
                }
                if (this.els.btnDigitalAi) {
                    if (!this._digitalAiIntroRunning) {
                        this.els.btnDigitalAi.classList.toggle('is-active', on);
                        this.els.btnDigitalAi.setAttribute('aria-pressed', on ? 'true' : 'false');
                    }
                }
            }

            _cancelDigitalAiButtonIntroHint() {
                if (this._digitalAiIntroTimeouts) {
                    this._digitalAiIntroTimeouts.forEach((t) => clearTimeout(t));
                    this._digitalAiIntroTimeouts = null;
                }
                this._digitalAiIntroRunning = false;
            }

            _scheduleDigitalAiButtonIntroHint() {
                if (this.skin !== 'digital') return;
                this._cancelDigitalAiButtonIntroHint();
                try {
                    if (localStorage.getItem(RadioVisualEngine.DIGITAL_AI_INTRO_SEEN_KEY) === '1') return;
                } catch (_) { return; }
                const t = setTimeout(() => {
                    try { this._playDigitalAiButtonIntroHint(); } catch (_) {}
                }, 900);
                this._digitalAiIntroTimeouts = [t];
            }

            _playDigitalAiButtonIntroHint() {
                const btn = this.els.btnDigitalAi;
                if (!btn || this._digitalAiIntroRunning) return;
                try {
                    if (localStorage.getItem(RadioVisualEngine.DIGITAL_AI_INTRO_SEEN_KEY) === '1') return;
                } catch (_) { return; }
                this._digitalAiIntroRunning = true;
                const timeouts = [];
                this._digitalAiIntroTimeouts = timeouts;
                const onMs = 600;
                const offMs = 380;
                let step = 0;
                const pulse = () => {
                    const b = this.els.btnDigitalAi;
                    if (!b) {
                        this._cancelDigitalAiButtonIntroHint();
                        return;
                    }
                    const highlight = step % 2 === 0;
                    b.classList.toggle('is-active', highlight);
                    b.setAttribute('aria-pressed', highlight ? 'true' : 'false');
                    step += 1;
                    if (step >= 6) {
                        this._digitalAiIntroRunning = false;
                        this._digitalAiIntroTimeouts = null;
                        try { localStorage.setItem(RadioVisualEngine.DIGITAL_AI_INTRO_SEEN_KEY, '1'); } catch (_) {}
                        try { this._syncAutoMixKnob(); } catch (_) {}
                        return;
                    }
                    const t = setTimeout(pulse, highlight ? onMs : offMs);
                    timeouts.push(t);
                };
                pulse();
            }

            _closeDigitalToolbarSliderPanel(panel) {
                if (!panel) return;
                panel.classList.remove('is-open');
                panel.setAttribute('aria-hidden', 'true');
                panel.classList.remove('is-fixed-popup');
                panel.style.position = '';
                panel.style.left = '';
                panel.style.top = '';
                panel.style.right = '';
            }

            _closeDigitalAutoMixPanel() {
                this._closeDigitalToolbarSliderPanel(this.els.digitalAutoMixPanel);
            }

            _closeDigitalAutoFadePanel() {
                this._closeDigitalToolbarSliderPanel(this.els.digitalAutoFadePanel);
            }

            _openDigitalToolbarSliderPanel(panel, syncFn, clientX, clientY) {
                if (!panel) return;
                try { syncFn(); } catch (_) {}
                panel.classList.add('is-open');
                panel.setAttribute('aria-hidden', 'false');
                const px = Number(clientX);
                const py = Number(clientY);
                panel.classList.add('is-fixed-popup');
                panel.style.position = 'fixed';
                panel.style.right = 'auto';
                const place = () => {
                    const rect = panel.getBoundingClientRect();
                    const pad = 8;
                    let left = (Number.isFinite(px) ? px : window.innerWidth / 2) + 4;
                    let top = (Number.isFinite(py) ? py : window.innerHeight / 2) + 4;
                    if (left + rect.width > window.innerWidth - pad) left = window.innerWidth - pad - rect.width;
                    if (left < pad) left = pad;
                    if (top + rect.height > window.innerHeight - pad) top = window.innerHeight - pad - rect.height;
                    if (top < pad) top = pad;
                    panel.style.left = `${Math.round(left)}px`;
                    panel.style.top = `${Math.round(top)}px`;
                };
                requestAnimationFrame(() => requestAnimationFrame(place));
            }

            _syncDigitalAutoMixPanelUi() {
                const slider = this.els.digitalAutoMixSlider;
                if (slider) slider.value = String(this._readAutoMixMaxMin());
                const readout = this.els.digitalAutoMixReadout;
                if (readout) readout.textContent = `${this._readAutoMixMaxMin()}m`;
            }

            _openDigitalAutoMixPanel(clientX, clientY) {
                this._openDigitalToolbarSliderPanel(
                    this.els.digitalAutoMixPanel,
                    () => this._syncDigitalAutoMixPanelUi(),
                    clientX,
                    clientY
                );
            }

            _syncDigitalAutoFadePanelUi() {
                const ms = this._readAutoFadeDurationMs();
                const slider = this.els.digitalAutoFadeSlider;
                if (slider) slider.value = String(ms / 1000);
                const readout = this.els.digitalAutoFadeReadout;
                if (readout) readout.textContent = `${(ms / 1000).toFixed(1)}s`;
            }

            _openDigitalAutoFadePanel(clientX, clientY) {
                this._openDigitalToolbarSliderPanel(
                    this.els.digitalAutoFadePanel,
                    () => this._syncDigitalAutoFadePanelUi(),
                    clientX,
                    clientY
                );
            }

            _wireDigitalDeckTransportBtn(btn, deckKey, sig) {
                if (!btn) return;
                const dk = deckKey === 'b' ? 'b' : 'a';
                const HOLD_MS = 500;
                let longPressTimer = null;
                let longPressHandled = false;
                const clearLongPress = () => {
                    if (longPressTimer) {
                        clearTimeout(longPressTimer);
                        longPressTimer = null;
                    }
                };
                try { btn.removeAttribute('title'); } catch (_) {}
                btn.setAttribute('aria-label', dk === 'b' ? 'Deck B transport' : 'Deck A transport');
                btn.addEventListener('contextmenu', (ev) => {
                    try {
                        ev.preventDefault();
                        ev.stopPropagation();
                    } catch (_) {}
                    if (dk === 'b') this._deckBPrevOrStation();
                    else this._deckAPrevOrStation();
                }, sig);
                btn.addEventListener('pointerdown', (ev) => {
                    if (ev.button !== 0) return;
                    this._stopClick(ev);
                    longPressHandled = false;
                    clearLongPress();
                    longPressTimer = setTimeout(() => {
                        longPressTimer = null;
                        longPressHandled = true;
                        this._pauseDeckOutput(dk);
                    }, HOLD_MS);
                }, sig);
                btn.addEventListener('pointerup', (ev) => {
                    this._stopClick(ev);
                    clearLongPress();
                    if (!longPressHandled) {
                        try { this._deckTransportTap(dk); } catch (_) {}
                    }
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('pointercancel', () => {
                    clearLongPress();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _wireDigitalAiButton(btn, sig) {
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
                const pointer = { x: 0, y: 0 };
                btn.setAttribute('aria-label', 'Toggle auto-mix and AI video; hold for max interval');
                btn.addEventListener('pointerdown', (ev) => {
                    this._stopClick(ev);
                    longPressHandled = false;
                    pointer.x = ev.clientX;
                    pointer.y = ev.clientY;
                    clearLongPress();
                    longPressTimer = setTimeout(() => {
                        longPressTimer = null;
                        longPressHandled = true;
                        this._openDigitalAutoMixPanel(pointer.x, pointer.y);
                    }, longPressMs);
                }, sig);
                btn.addEventListener('pointerup', (ev) => {
                    this._stopClick(ev);
                    clearLongPress();
                    if (!longPressHandled) {
                        this._toggleAutoMix();
                        if (this._isAutoMixEnabled()) this._enterDigitalHubAiMode();
                        else this._exitDigitalHubAiMode();
                    }
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('pointercancel', () => {
                    clearLongPress();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _wireDigitalFadeButton(btn, sig) {
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
                const pointer = { x: 0, y: 0 };
                btn.setAttribute('aria-label', 'Auto-fade; hold for fade duration');
                btn.addEventListener('pointerdown', (ev) => {
                    this._stopClick(ev);
                    longPressHandled = false;
                    pointer.x = ev.clientX;
                    pointer.y = ev.clientY;
                    clearLongPress();
                    longPressTimer = setTimeout(() => {
                        longPressTimer = null;
                        longPressHandled = true;
                        this._openDigitalAutoFadePanel(pointer.x, pointer.y);
                    }, longPressMs);
                }, sig);
                btn.addEventListener('pointerup', (ev) => {
                    this._stopClick(ev);
                    clearLongPress();
                    if (!longPressHandled) this._triggerAutoFade();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('pointercancel', () => {
                    clearLongPress();
                    longPressHandled = false;
                }, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _wireDigitalAutoMixPanelDismiss(sig) {
                const dismissIfOutside = (ev) => {
                    const t = ev.target;
                    if (!t || typeof t.closest !== 'function') return;
                    const panels = [
                        { panel: this.els.digitalAutoMixPanel, btn: this.els.btnDigitalAi, close: () => this._closeDigitalAutoMixPanel() },
                        { panel: this.els.digitalAutoFadePanel, btn: this.els.btnDigitalFade, close: () => this._closeDigitalAutoFadePanel() }
                    ];
                    panels.forEach(({ panel, btn, close }) => {
                        if (!panel || !panel.classList.contains('is-open')) return;
                        if (panel.contains(t)) return;
                        if (btn && (btn === t || btn.contains(t))) return;
                        close();
                    });
                };
                document.addEventListener('pointerdown', dismissIfOutside, { capture: true, ...sig });
                document.addEventListener('click', dismissIfOutside, { capture: true, ...sig });
                document.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Escape') {
                        this._closeDigitalAutoMixPanel();
                        this._closeDigitalAutoFadePanel();
                    }
                }, sig);
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
                    const el = (typeof globalThis.getDeckBRadioAudibleEl === 'function')
                        ? globalThis.getDeckBRadioAudibleEl()
                        : ((typeof audioElB !== 'undefined') ? audioElB : null);
                    return !!(el && el.src && el.src !== 'about:blank' && !el.paused && !el.ended);
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
                    if (this._isRadioVisualActive()) this.cancelAutoFade();
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    if (eng && eng !== this && typeof eng.cancelAutoFade === 'function') eng.cancelAutoFade();
                } catch (_) {}
            }

            _deckEngClearSuppress() {
                try { this._clearSuppressCrossfadeResume(); } catch (_) {}
                try {
                    const eng = state && state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks'
                        ? state.activeVisualizer : null;
                    if (eng && eng !== this && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
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

            _deckBPlaybackMedia() {
                try {
                    if (typeof getDeckBRadioAudibleEl === 'function') return getDeckBRadioAudibleEl();
                } catch (_) {}
                return (typeof audioElB !== 'undefined') ? audioElB : null;
            }

            async _startDeckB() {
                try { if (typeof initAudio === 'function') initAudio(); } catch (_) {}
                try {
                    this._deckEngCancelAutoFade();
                    const mediaB = this._deckBPlaybackMedia();
                    if (!mediaB || !this._deckHasSource(mediaB)) {
                        this._deckEngClearSuppress();
                        if (typeof playRadioB === 'function') playRadioB();
                        return;
                    }
                    if (mediaB.paused) {
                        this._deckEngClearSuppress();
                        await mediaB.play().catch(() => {
                            try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                        });
                        try { if (typeof connectDeckMediaToEq === 'function') connectDeckMediaToEq('b'); } catch (_) {}
                    }
                } catch (_) {}
            }

            async _stopDeckB() {
                try {
                    this._deckEngCancelAutoFade();
                    this._deckEngClearSuppress();
                    const mediaB = this._deckBPlaybackMedia();
                    if (mediaB && !mediaB.paused) mediaB.pause();
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
            }

            _formatDigitalVolumeReadout(v) {
                return String(Math.round(Math.max(0, Math.min(1, Number(v) || 0)) * 100));
            }

            _cancelDigitalToolbarVolumePeek() {
                if (this._digitalToolbarVolumePeekTimer) {
                    try { clearTimeout(this._digitalToolbarVolumePeekTimer); } catch (_) {}
                    this._digitalToolbarVolumePeekTimer = null;
                }
                this._digitalToolbarVolumePeekActive = false;
            }

            _crossfadedDeckPlaybackMedia() {
                const deck = this._crossfaderAudibleDeckKey();
                try {
                    if (deck === 'a') {
                        if (typeof getDeckAMediaForPlaybackState === 'function') {
                            return getDeckAMediaForPlaybackState();
                        }
                    } else if (typeof getDeckBJogSeekMedia === 'function') {
                        return getDeckBJogSeekMedia();
                    } else if (typeof getDeckBRadioAudibleEl === 'function') {
                        return getDeckBRadioAudibleEl();
                    }
                } catch (_) {}
                try {
                    return deck === 'b'
                        ? (typeof audioElB !== 'undefined' ? audioElB : null)
                        : (typeof audioEl !== 'undefined' ? audioEl : null);
                } catch (_) {}
                return null;
            }

            _digitalTrackRemainingReadoutText() {
                const media = this._crossfadedDeckPlaybackMedia();
                if (!media) return '—:—';
                try {
                    if (!media.src && !media.currentSrc) return '—:—';
                    const dur = Number(media.duration);
                    if (!Number.isFinite(dur) || dur <= 0 || dur >= 86400) return '—:—';
                    const t = Math.max(0, Number(media.currentTime) || 0);
                    const leftMs = Math.max(0, (dur - t) * 1000);
                    return `-${this._formatAutoMixCountdown(leftMs)}`;
                } catch (_) {
                    return '—:—';
                }
            }

            _peekDigitalToolbarVolumeReadout() {
                const mode = this._digitalToolbarCenterMode;
                if (mode !== 'volume' && mode !== 'automix') return;
                if (this._digitalToolbarVolumePeekTimer) {
                    try { clearTimeout(this._digitalToolbarVolumePeekTimer); } catch (_) {}
                }
                this._digitalToolbarVolumePeekActive = true;
                this._syncDigitalToolbarCenterReadout();
                this._digitalToolbarVolumePeekTimer = setTimeout(() => {
                    this._digitalToolbarVolumePeekTimer = null;
                    this._digitalToolbarVolumePeekActive = false;
                    if (this._digitalToolbarCenterMode === 'volume'
                        || this._digitalToolbarCenterMode === 'automix') {
                        this._syncDigitalToolbarCenterReadout();
                    }
                }, 5000);
            }

            _syncDigitalVolumeUi(options = {}) {
                const mode = this._digitalToolbarCenterMode;
                if (mode === 'crossfade') {
                    this._syncDigitalToolbarCenterReadout();
                } else if (mode === 'volume' || mode === 'automix') {
                    if (options.peekVolume) this._peekDigitalToolbarVolumeReadout();
                    else this._syncDigitalToolbarCenterReadout();
                }
            }

            _formatDigitalClockReadout(date = new Date(), opts = {}) {
                const compact = !!(opts && opts.compact);
                const format = this._digitalClockFormat();
                try {
                    const time24 = date.toLocaleTimeString(undefined, {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                    });
                    if (compact && format !== 'date-time' && format !== 'date-short') {
                        return time24;
                    }
                    if (format === 'time-24') return time24;
                    if (format === 'time-12') {
                        return date.toLocaleTimeString(undefined, {
                            hour: 'numeric',
                            minute: '2-digit',
                            hour12: true
                        });
                    }
                    if (format === 'date-time') {
                        return date.toLocaleString(undefined, {
                            weekday: 'short',
                            day: 'numeric',
                            month: 'short',
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: false
                        });
                    }
                    if (format === 'date-short') {
                        return date.toLocaleString(undefined, {
                            day: '2-digit',
                            month: '2-digit',
                            year: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: false
                        });
                    }
                    const day = date.toLocaleDateString(undefined, { weekday: 'short' }).toUpperCase();
                    return `${day} ${time24}`;
                } catch (_) {
                    return '—';
                }
            }

            _digitalClockFormat() {
                const root = this.root || document.getElementById('radio-visual-root');
                const fmt = root && root.dataset ? root.dataset.rvClockFormat : '';
                return fmt || 'weekday-time';
            }

            _readRvThemeCssNumber(varName, fallback = 1) {
                try {
                    const root = this.root || document.getElementById('radio-visual-root');
                    if (!root) return fallback;
                    const raw = getComputedStyle(root).getPropertyValue(varName).trim();
                    const v = Number(raw);
                    return Number.isFinite(v) ? v : fallback;
                } catch (_) {
                    return fallback;
                }
            }

            _reflowDigitalThemeUi() {
                try { this._tickClock(); } catch (_) {}
                try {
                    const toolbar = this.els.digitalToolbar || document.getElementById('radio-visual-digital-toolbar');
                    if (toolbar) {
                        this._fitDigitalToolbarButtonLabels(toolbar);
                        this._fitDigitalToolbarCenterReadout(toolbar);
                    }
                } catch (_) {}
                try {
                    if (this.els.digitalBtns) this._fitDigitalFeatureButtonLabels(this.els.digitalBtns);
                } catch (_) {}
            }

            _digitalToolbarClockReadoutCompact() {
                const readout = this.els.volDigitalReadout;
                if (!readout) return false;
                const w = readout.clientWidth;
                return w > 0 && w < 72;
            }

            _digitalAutoMixToolbarReadoutText() {
                if (!this._isAutoMixEnabled() || !this._rvAutoMixNextFadeAt) return '—:—';
                const left = Math.max(0, this._rvAutoMixNextFadeAt - Date.now());
                return this._formatAutoMixCountdown(left);
            }

            _syncDigitalToolbarCenterXfadeSlider() {
                const wrap = this.els.digitalToolbarCenterXfadeWrap;
                const range = this.els.digitalToolbarCenterXfade;
                if (!wrap || !range) return;
                const x = Math.max(0, Math.min(1, this._getCrossfadeX()));
                range.value = String(x);
                try { wrap.style.setProperty('--cross-x', String(x)); } catch (_) {}
            }

            _syncDigitalToolbarCenterReadout() {
                const label = this.els.digitalToolbarCenterReadoutLabel;
                const readout = this.els.volDigitalReadout;
                if (!label || !readout) return;
                const mode = this._digitalToolbarCenterMode || 'clock';
                if (mode === 'crossfade') {
                    readout.dataset.centerMode = mode;
                    this._syncDigitalToolbarCenterXfadeSlider();
                    return;
                }
                let text = '—';
                const clockCompact = this._digitalToolbarClockReadoutCompact();
                if (mode === 'clock') {
                    text = this._formatDigitalClockReadout(undefined, { compact: clockCompact });
                } else if (mode === 'volume') {
                    if (this._digitalToolbarVolumePeekActive) {
                        const vs = document.getElementById('volume-slider');
                        text = this._formatDigitalVolumeReadout(vs ? Number(vs.value) : 0.5);
                    } else {
                        text = this._formatDigitalClockReadout(undefined, { compact: clockCompact });
                    }
                } else if (mode === 'automix') {
                    if (this._digitalToolbarVolumePeekActive) {
                        const vs = document.getElementById('volume-slider');
                        text = this._formatDigitalVolumeReadout(vs ? Number(vs.value) : 0.5);
                    } else {
                        text = this._digitalAutoMixToolbarReadoutText();
                    }
                } else if (mode === 'remaining') {
                    text = this._digitalTrackRemainingReadoutText();
                }
                label.textContent = text;
                readout.dataset.centerMode = mode;
                if (mode === 'volume' || mode === 'automix') {
                    readout.dataset.volumePeek = this._digitalToolbarVolumePeekActive ? '1' : '0';
                } else if (readout.dataset.volumePeek != null) {
                    delete readout.dataset.volumePeek;
                }
                try {
                    const toolbar = this.els.digitalToolbar || document.getElementById('radio-visual-digital-toolbar');
                    if (toolbar) this._fitDigitalToolbarCenterReadout(toolbar);
                } catch (_) {}
            }

            _updateDigitalToolbarStepButton(btn, text, ariaLabel) {
                if (!btn) return;
                let label = btn.querySelector('.radio-visual-btn-label');
                if (!label) label = this._appendRvButtonLabel(btn, text);
                else label.textContent = text;
                if (ariaLabel) btn.setAttribute('aria-label', ariaLabel);
            }

            _syncDigitalToolbarCenterMode() {
                const mode = this._digitalToolbarCenterMode || 'clock';
                const group = this.els.digitalToolbarCenter;
                const readout = this.els.volDigitalReadout;
                if (group) group.dataset.centerMode = mode;
                if (readout) {
                    const ariaLabels = {
                        clock: 'Date and time — tap to change centre display',
                        volume: 'Volume — date and time; level shown briefly when adjusted — tap to change centre display',
                        crossfade: 'Crossfade position — tap to change centre display',
                        automix: 'Auto-mix countdown — tap for date and time; volume shown briefly when adjusted',
                        remaining: 'Track time remaining on crossfader deck — tap to change centre display'
                    };
                    readout.setAttribute('aria-label', ariaLabels[mode] || 'Centre display — tap to change mode');
                }
                const xfadeRange = this.els.digitalToolbarCenterXfade;
                if (xfadeRange) {
                    xfadeRange.hidden = mode !== 'crossfade';
                    xfadeRange.tabIndex = mode === 'crossfade' ? 0 : -1;
                }
                if (mode === 'crossfade') {
                    this._updateDigitalToolbarStepButton(this.els.volDown, 'A', 'Crossfade toward deck A');
                    this._updateDigitalToolbarStepButton(this.els.volUp, 'B', 'Crossfade toward deck B');
                } else if (mode === 'clock') {
                    this._updateDigitalToolbarStepButton(this.els.volDown, '<', 'Previous station or file on crossfader deck');
                    this._updateDigitalToolbarStepButton(this.els.volUp, '>', 'Next station or file on crossfader deck');
                } else {
                    this._updateDigitalToolbarStepButton(this.els.volDown, '−', 'Volume down');
                    this._updateDigitalToolbarStepButton(this.els.volUp, '+', 'Volume up');
                }
                this._syncDigitalToolbarCenterReadout();
                try {
                    const toolbar = this.els.digitalToolbar || document.getElementById('radio-visual-digital-toolbar');
                    if (toolbar) this._fitDigitalToolbarButtonLabels(toolbar);
                } catch (_) {}
            }

            _setDigitalToolbarCenterMode(mode) {
                this._cancelDigitalToolbarVolumePeek();
                this._digitalToolbarCenterMode = mode;
                this._syncDigitalToolbarCenterMode();
                if (mode === 'volume') this._peekDigitalToolbarVolumeReadout();
            }

            _showDigitalToolbarAutomixCountdown() {
                if (this.skin !== 'digital') return;
                this._setDigitalToolbarCenterMode('automix');
            }

            _cycleDigitalToolbarCenterMode() {
                if (this._digitalToolbarCenterMode === 'automix') {
                    this._setDigitalToolbarCenterMode('clock');
                    return;
                }
                const cycle = RadioVisualEngine.DIGITAL_TOOLBAR_CENTER_CYCLE;
                let i = cycle.indexOf(this._digitalToolbarCenterMode);
                if (i < 0) i = 0;
                this._setDigitalToolbarCenterMode(cycle[(i + 1) % cycle.length]);
            }

            _nudgeDigitalToolbarCrossfade(towardDeck) {
                const step = RadioVisualEngine.DIGITAL_TOOLBAR_XFADE_STEP;
                const x = this._getCrossfadeX();
                if (towardDeck === 'a') this._setCrossfadeX(x - step);
                else this._setCrossfadeX(x + step);
            }

            _onDigitalToolbarCenterStep(side) {
                const mode = this._digitalToolbarCenterMode || 'clock';
                if (mode === 'clock') {
                    if (side < 0) this._crossfadedDeckPrevOrStation();
                    else this._crossfadedDeckNextOrStation();
                    return;
                }
                if (mode === 'crossfade') {
                    if (side < 0) this._nudgeDigitalToolbarCrossfade('a');
                    else this._nudgeDigitalToolbarCrossfade('b');
                    return;
                }
                this._stepDigitalVolume(side);
            }

            _wireDigitalToolbarCenterCrossfadeSlider(rangeEl, sig) {
                if (!rangeEl || rangeEl.dataset.rvToolbarXfadeWired === '1') return;
                rangeEl.dataset.rvToolbarXfadeWired = '1';
                let dragActive = false;
                const markDragEnd = (ev) => {
                    if (!dragActive) return;
                    dragActive = false;
                    try { ev.stopPropagation(); } catch (_) {}
                    try { window.__suppressNextClick = true; } catch (_) {}
                };
                rangeEl.addEventListener('input', () => {
                    dragActive = true;
                    this._setCrossfadeX(rangeEl.value);
                    try { this._resumeDecksForCrossfade(); } catch (_) {}
                }, sig);
                rangeEl.addEventListener('pointerdown', (ev) => {
                    if (ev.button !== 0) return;
                    dragActive = false;
                    try { ev.stopPropagation(); } catch (_) {}
                }, sig);
                rangeEl.addEventListener('pointerup', markDragEnd, sig);
                rangeEl.addEventListener('pointercancel', markDragEnd, sig);
                rangeEl.addEventListener('change', markDragEnd, sig);
                this._wireDigitalCrossfadeCutHold(rangeEl, sig);
            }

            _wireDigitalToolbarCenterSlotCycle(cycleHitEl, sig) {
                if (!cycleHitEl || cycleHitEl.dataset.rvCenterSlotCycleWired === '1') return;
                cycleHitEl.dataset.rvCenterSlotCycleWired = '1';
                cycleHitEl.addEventListener('click', (ev) => {
                    if (this._digitalToolbarCenterMode !== 'crossfade') return;
                    try { if (window.__suppressNextClick) return; } catch (_) {}
                    try { this._stopClick(ev); } catch (_) {}
                    try { this._cycleDigitalToolbarCenterMode(); } catch (_) {}
                }, sig);
            }

            _clearDigitalToolbarCenterStepRepeat(btn) {
                const r = btn && btn._rvCenterStepRepeat;
                if (!r) return;
                if (r.timer) {
                    clearTimeout(r.timer);
                    r.timer = null;
                }
                if (r.interval) {
                    clearInterval(r.interval);
                    r.interval = null;
                }
            }

            _wireDigitalToolbarCenterStepButton(btn, side, sig) {
                if (!btn || btn.dataset.rvCenterStepWired === '1') return;
                btn.dataset.rvCenterStepWired = '1';
                btn._rvCenterStepRepeat = { timer: null, interval: null };
                const initialDelayMs = 400;
                const repeatMs = 70;
                const clearRepeat = () => this._clearDigitalToolbarCenterStepRepeat(btn);
                if (sig && sig.signal) {
                    sig.signal.addEventListener('abort', clearRepeat, { once: true });
                }
                const runStep = () => {
                    try { this._onDigitalToolbarCenterStep(side); } catch (_) {}
                };
                const startRepeat = () => {
                    clearRepeat();
                    runStep();
                    btn._rvCenterStepRepeat.timer = setTimeout(() => {
                        btn._rvCenterStepRepeat.timer = null;
                        btn._rvCenterStepRepeat.interval = setInterval(runStep, repeatMs);
                    }, initialDelayMs);
                };
                const stopRepeat = (ev) => {
                    clearRepeat();
                    try {
                        if (ev && ev.pointerId != null) btn.releasePointerCapture(ev.pointerId);
                    } catch (_) {}
                };
                btn.addEventListener('pointerdown', (ev) => {
                    if (ev.button !== 0) return;
                    try { ev.stopPropagation(); } catch (_) {}
                    try { btn.setPointerCapture(ev.pointerId); } catch (_) {}
                    startRepeat();
                }, sig);
                btn.addEventListener('pointerup', stopRepeat, sig);
                btn.addEventListener('pointercancel', stopRepeat, sig);
                btn.addEventListener('lostpointercapture', stopRepeat, sig);
                btn.addEventListener('click', (ev) => this._stopClick(ev), sig);
            }

            _fitDigitalToolbarCenterReadout(toolbarEl) {
                const readout = toolbarEl?.querySelector('.radio-visual-digital-center-readout');
                const label = readout?.querySelector('.radio-visual-center-readout-text');
                if (!readout || !label) return;
                label.style.fontSize = '';
                label.style.letterSpacing = '';
                if (readout.clientWidth < 8 || readout.clientHeight < 6) return;
                const mode = this._digitalToolbarCenterMode || 'clock';
                if (mode === 'crossfade') return;
                const volumeShowsClock = mode === 'volume' && !this._digitalToolbarVolumePeekActive;
                const fitAsClock = mode === 'clock' || volumeShowsClock;
                if (fitAsClock) {
                    const clockCompact = readout.clientWidth > 0 && readout.clientWidth < 72;
                    const current = String(label.textContent || '');
                    const next = this._formatDigitalClockReadout(undefined, { compact: clockCompact });
                    if (current !== next) label.textContent = next;
                }
                const maxW = Math.max(6, readout.clientWidth - 1);
                const maxH = Math.max(6, readout.clientHeight - 1);
                const clockScale = this._readRvThemeCssNumber('--rv-digital-clock-font-scale', 1);
                const minPx = fitAsClock ? 4 : 6;
                const maxCap = (fitAsClock ? 11 : 13) * clockScale;
                let size = Math.min(maxH * 0.9, maxCap);
                label.style.fontSize = `${size}px`;
                if (fitAsClock) label.style.letterSpacing = '0.03em';
                for (let guard = 0; guard < 200 && size > minPx; guard++) {
                    if (label.scrollWidth <= maxW && label.scrollHeight <= maxH) break;
                    size -= 0.2;
                    label.style.fontSize = `${size}px`;
                }
                if (fitAsClock && label.scrollWidth > maxW) {
                    let spacing = 0.03;
                    for (let i = 0; i < 30 && label.scrollWidth > maxW && spacing > -0.04; i++) {
                        spacing -= 0.003;
                        label.style.letterSpacing = `${spacing}em`;
                    }
                }
                if (!fitAsClock) {
                    this._computeRvButtonLabelFitPx(readout, label, {
                        fill: true,
                        maxCap,
                        heightFactor: 0.88,
                        widthFactor: (mode === 'automix' || mode === 'remaining') ? 0.92 : 0.55,
                        minPx,
                        pad: 2
                    });
                }
            }

            _scheduleDigitalSpectrumLayoutSync() {
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        try { this._resizeCanvases(); } catch (_) {}
                    });
                });
            }

            _resolvedSpectrumDrawSettings() {
                const s = this._spectrumSettings || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                const d = RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                if (this._digitalHubMode === 'spectrum') {
                    return {
                        colorStreamId: s.colorStreamId || d.colorStreamId,
                        scale: s.scale ?? d.scale,
                        opacity: s.opacity ?? d.opacity,
                        audioStrength: s.audioStrength ?? d.audioStrength,
                        colorFlow: s.colorFlow ?? d.colorFlow
                    };
                }
                return {
                    colorStreamId: s.eqColorStreamId || s.colorStreamId || d.eqColorStreamId,
                    scale: s.eqScale ?? d.eqScale,
                    opacity: s.eqOpacity ?? s.opacity ?? d.eqOpacity,
                    audioStrength: s.eqAudioStrength ?? s.audioStrength ?? d.eqAudioStrength,
                    colorFlow: s.eqColorFlow ?? s.colorFlow ?? d.eqColorFlow
                };
            }

            _clampSpectrumSideScale(raw, fallback = 1) {
                const min = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MIN;
                const max = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MAX;
                let scale = Number(raw);
                if (!Number.isFinite(scale)) scale = fallback;
                return Math.max(min, Math.min(max, scale));
            }

            _getSpectrumHubSideScale() {
                const s = this._spectrumSettings || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                const d = RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                return this._clampSpectrumSideScale(s.scale, d.scale);
            }

            _getEqualiserSideScale() {
                const s = this._spectrumSettings || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                const d = RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                return this._clampSpectrumSideScale(s.eqScale, d.eqScale);
            }

            _getActiveSpectrumSideScale() {
                if (this._digitalHubMode === 'spectrum') return this._getSpectrumHubSideScale();
                return this._getEqualiserSideScale();
            }

            _getSpectrumOpacityMult() {
                const v = Number(this._resolvedSpectrumDrawSettings().opacity);
                if (!Number.isFinite(v)) return 1;
                return Math.max(0.15, Math.min(1, v));
            }

            _getSpectrumAudioStrength() {
                const v = Number(this._resolvedSpectrumDrawSettings().audioStrength);
                if (!Number.isFinite(v)) return 1;
                return Math.max(0.25, Math.min(3, v));
            }

            _getSpectrumColorFlow() {
                const v = Number(this._resolvedSpectrumDrawSettings().colorFlow);
                if (!Number.isFinite(v)) return 1;
                return Math.max(0.25, Math.min(3, v));
            }

            _getSpectrumColorPalette() {
                const id = this._resolvedSpectrumDrawSettings().colorStreamId
                    || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS.colorStreamId;
                const preset = RadioVisualEngine.SPECTRUM_COLOR_STREAM_PRESETS[id];
                if (preset && Array.isArray(preset.palette) && preset.palette.length) return preset.palette;
                return RadioVisualEngine.SPECTRUM_COLOR_STREAM_PRESETS.aurora.palette;
            }

            _spectrumRibbonRotateKey(side, bandKey) {
                const sideKey = side === 'R' ? 'R' : 'L';
                const band = String(bandKey || 'high');
                const cap = band.charAt(0).toUpperCase() + band.slice(1).toLowerCase();
                return `rotate${sideKey}${cap}`;
            }

            _spectrumRibbonRotateDeg(side, bandKey) {
                const s = this._spectrumSettings || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS;
                const key = this._spectrumRibbonRotateKey(side, bandKey);
                const v = Number(s[key]);
                if (!Number.isFinite(v)) return 0;
                return Math.max(0, Math.min(360, v));
            }

            _effectiveRibbonPhaseBins(basePhaseBins, side, bandKey) {
                const n = RadioVisualEngine.SPECTRUM_ANGULAR_BINS;
                const base = Number(basePhaseBins) || 0;
                const deg = this._spectrumRibbonRotateDeg(side, bandKey);
                return base + (deg / 360) * n;
            }

            _spectrumScaleToSliderPct(scale) {
                const min = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MIN;
                const max = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MAX;
                const clamped = Math.max(min, Math.min(max, Number(scale) || 1));
                return Math.round(clamped * 100);
            }

            applySpectrumSettings(settings, opts = {}) {
                const next = RadioVisualEngine.clampSpectrumSettings({
                    ...(this._spectrumSettings || RadioVisualEngine.DEFAULT_SPECTRUM_SETTINGS),
                    ...(settings || {})
                });
                this._spectrumSettings = next;
                if (this._digitalHubMode === 'spectrum') {
                    this._spectrumStagingScale = this._getSpectrumHubSideScale();
                }
                if (!opts.skipSave) {
                    RadioVisualEngine.saveSpectrumSettingsToStorage(next);
                }
                this._syncSpectrumStagingScale();
                const slider = this.els && this.els.digitalSpectrumScaleSlider;
                if (slider && !opts.skipSlider && this._digitalHubMode === 'spectrum') {
                    slider.value = String(this._spectrumScaleToSliderPct(this._getSpectrumHubSideScale()));
                }
            }

            _persistSpectrumScaleFromStaging() {
                const scale = Math.max(
                    RadioVisualEngine.SPECTRUM_STAGING_SCALE_MIN,
                    Math.min(
                        RadioVisualEngine.SPECTRUM_STAGING_SCALE_MAX,
                        Number(this._spectrumStagingScale) || 1
                    )
                );
                this.applySpectrumSettings({ scale }, { skipSlider: true });
            }

            _showDigitalSpectrumStagingSlider() {
                const wrap = this.els.digitalSpectrumStagingSlider;
                if (!wrap) return;
                wrap.classList.add('is-visible');
                if (this._spectrumSliderHideTimer) {
                    clearTimeout(this._spectrumSliderHideTimer);
                    this._spectrumSliderHideTimer = null;
                }
            }

            _hideDigitalSpectrumStagingSlider(immediate = false) {
                const wrap = this.els.digitalSpectrumStagingSlider;
                if (!wrap) return;
                if (immediate) {
                    if (this._spectrumSliderHideTimer) {
                        clearTimeout(this._spectrumSliderHideTimer);
                        this._spectrumSliderHideTimer = null;
                    }
                    wrap.classList.remove('is-visible');
                    return;
                }
                if (this._spectrumSliderHideTimer) clearTimeout(this._spectrumSliderHideTimer);
                this._spectrumSliderHideTimer = setTimeout(() => {
                    this._spectrumSliderHideTimer = null;
                    if (this._digitalHubMode === 'spectrum') wrap.classList.remove('is-visible');
                }, 1400);
            }

            _scheduleHideDigitalSpectrumStagingSlider() {
                this._hideDigitalSpectrumStagingSlider(false);
            }

            /** Scale side spectrum flowers only — centre equaliser / dash stack stays fixed. */
            _syncSpectrumStagingScale() {
                const row = this.els.digitalSpectrumRow;
                if (!row) return;
                const sideScale = this._digitalHubMode === 'spectrum'
                    ? this._clampSpectrumSideScale(
                        this._spectrumStagingScale,
                        this._getSpectrumHubSideScale()
                    )
                    : this._getEqualiserSideScale();
                try {
                    row.style.setProperty('--rv-spectrum-side-scale', String(sideScale));
                } catch (_) {}
                this._syncSpectrumStagingPan();
            }

            _syncSpectrumStagingPan() {
                const row = this.els.digitalSpectrumRow;
                const pane = this.els.digitalCenterSpectrum;
                if (!row) return;
                if (this._digitalHubMode !== 'spectrum') {
                    try { row.style.setProperty('--rv-spectrum-staging-spread', '0px'); } catch (_) {}
                    return;
                }
                const spreadNorm = Math.max(0, Math.min(1, Number(this._spectrumStagingPanNorm) || 0));
                const scale = this._getSpectrumHubSideScale();
                const paneW = pane ? Math.max(1, pane.clientWidth || 0) : 400;
                const maxSpread = Math.max(24, paneW * 0.22 * Math.max(1, scale * 0.85));
                const spreadPx = spreadNorm * maxSpread;
                try { row.style.setProperty('--rv-spectrum-staging-spread', `${spreadPx}px`); } catch (_) {}
            }

            _applySpectrumStagingPanNorm(panNorm, opts = {}) {
                const n = Math.max(0, Math.min(1, Number(panNorm) || 0));
                this._spectrumStagingPanNorm = n;
                const panSlider = this.els.digitalSpectrumPanSlider;
                if (panSlider && !opts.skipSlider) {
                    panSlider.value = String(Math.round(n * 100));
                }
                this._syncSpectrumStagingPan();
            }

            _wireDigitalSpectrumStagingSlider(abortSignal) {
                const pane = this.els.digitalCenterSpectrum;
                const wrap = this.els.digitalSpectrumStagingSlider;
                const slider = this.els.digitalSpectrumScaleSlider;
                const panSlider = this.els.digitalSpectrumPanSlider;
                if (!pane || !wrap || !slider) return;
                const opts = abortSignal ? { signal: abortSignal } : {};
                const scaleWrap = wrap.querySelector('.radio-visual-digital-spectrum-staging-scale-wrap');
                const panWrap = wrap.querySelector('.radio-visual-digital-spectrum-staging-pan-wrap');

                const clearStagingDrag = () => {
                    wrap.classList.remove('is-dragging-scale', 'is-dragging-pan');
                    if (panWrap) panWrap.style.pointerEvents = '';
                    if (scaleWrap) scaleWrap.style.pointerEvents = '';
                };

                const applyScaleFromSlider = () => {
                    const v = Number(slider.value);
                    const min = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MIN;
                    const max = RadioVisualEngine.SPECTRUM_STAGING_SCALE_MAX;
                    this._spectrumStagingScale = Math.max(min, Math.min(max, v / 100));
                    this._syncSpectrumStagingScale();
                };
                const applyScaleFromSliderAndSave = () => {
                    applyScaleFromSlider();
                    try { this._persistSpectrumScaleFromStaging(); } catch (_) {}
                };
                this._spectrumStagingScale = this._getSpectrumHubSideScale();
                slider.value = String(this._spectrumScaleToSliderPct(this._spectrumStagingScale));
                this._applySpectrumStagingPanNorm(this._spectrumStagingPanNorm, { skipSlider: !panSlider });

                slider.addEventListener('input', () => {
                    applyScaleFromSliderAndSave();
                    this._showDigitalSpectrumStagingSlider();
                }, opts);

                if (panSlider) {
                    const applyPanFromSlider = () => {
                        this._applySpectrumStagingPanNorm(Number(panSlider.value) / 100, { skipSlider: true });
                        this._showDigitalSpectrumStagingSlider();
                    };
                    panSlider.addEventListener('input', applyPanFromSlider, opts);
                    panSlider.addEventListener('change', applyPanFromSlider, opts);
                    panSlider.addEventListener('pointerdown', (ev) => {
                        try { ev.stopPropagation(); } catch (_) {}
                        wrap.classList.add('is-dragging-pan');
                        if (scaleWrap) scaleWrap.style.pointerEvents = 'none';
                        try { slider.blur(); } catch (_) {}
                        try { panSlider.setPointerCapture(ev.pointerId); } catch (_) {}
                        this._showDigitalSpectrumStagingSlider();
                    }, opts);
                }

                slider.addEventListener('pointerdown', (ev) => {
                    try { ev.stopPropagation(); } catch (_) {}
                    wrap.classList.add('is-dragging-scale');
                    if (panWrap) panWrap.style.pointerEvents = 'none';
                    if (panSlider) {
                        try { panSlider.blur(); } catch (_) {}
                    }
                    try { slider.setPointerCapture(ev.pointerId); } catch (_) {}
                    this._showDigitalSpectrumStagingSlider();
                }, opts);

                document.addEventListener('pointerup', clearStagingDrag, opts);
                document.addEventListener('pointercancel', clearStagingDrag, opts);

                const onPaneMove = (ev) => {
                    if (this._digitalHubMode !== 'spectrum') return;
                    if (ev.target && wrap.contains(ev.target)) return;
                    this._showDigitalSpectrumStagingSlider();
                    this._scheduleHideDigitalSpectrumStagingSlider();
                };
                pane.addEventListener('pointermove', onPaneMove, opts);
                wrap.addEventListener('pointermove', () => {
                    this._showDigitalSpectrumStagingSlider();
                    this._scheduleHideDigitalSpectrumStagingSlider();
                }, opts);
            }

            _digitalHubModeLabel(mode) {
                return RadioVisualEngine.DIGITAL_HUB_LABELS[mode] || 'EQUALISER';
            }

            _hubSpectrumLayout(hubMode) {
                switch (hubMode) {
                    case 'equaliser': return 'full';
                    case 'spectrum': return 'focus';
                    case 'volume':
                    case 'effects':
                    case 'live':
                    case 'video': return 'full';
                    case 'ai-off': return 'blank';
                    default: return 'full';
                }
            }

            _enterDigitalHubLiveMode() {
                if (this._digitalHubMode === 'live') return;
                if (this.digitalCenterMode !== 'spectrum') {
                    try { this._setDigitalCenterMode('spectrum'); } catch (_) {}
                }
                this._digitalHubMode = 'live';
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
            }

            _exitDigitalHubLiveToEqualiser() {
                if (this._digitalHubMode !== 'live') return;
                this._digitalHubMode = 'equaliser';
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
            }

            _enterDigitalHubAiMode() {
                if (this._digitalHubMode === 'ai') return;
                this._digitalHubAiReturnMode = this._digitalHubMode || 'equaliser';
                if (this.digitalCenterMode !== 'spectrum') {
                    try { this._setDigitalCenterMode('spectrum'); } catch (_) {}
                }
                this._digitalHubMode = 'ai';
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
            }

            _exitDigitalHubAiMode() {
                if (this._digitalHubMode !== 'ai') return;
                const back = this._digitalHubAiReturnMode || 'equaliser';
                this._digitalHubMode = back;
                this._digitalHubAiReturnMode = 'equaliser';
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
            }

            _toggleDigitalHubAiMode() {
                if (this._digitalHubMode === 'ai') this._exitDigitalHubAiMode();
                else this._enterDigitalHubAiMode();
            }

            _buildDigitalHubPanel() {
                const panel = document.createElement('div');
                panel.className = 'radio-visual-digital-hub-panel';
                panel.setAttribute('aria-hidden', 'true');

                const volume = document.createElement('div');
                volume.className = 'radio-visual-digital-hub-volume';

                const mkKnob = (mixId, label, color) => {
                    const wrap = document.createElement('div');
                    wrap.className = 'radio-visual-digital-hub-knob-wrap';
                    const knob = document.createElement('div');
                    knob.className = 'knob-wrap radio-visual-digital-hub-knob';
                    knob.id = `digital-hub-${mixId}`;
                    knob.style.setProperty('--knob-color', color);
                    const indicator = document.createElement('div');
                    indicator.className = 'knob-indicator';
                    const val = document.createElement('div');
                    val.className = 'knob-value';
                    knob.append(indicator, val);
                    const lbl = document.createElement('span');
                    lbl.className = 'radio-visual-digital-hub-knob-label';
                    lbl.textContent = label;
                    lbl.style.color = color;
                    wrap.append(knob, lbl);
                    return wrap;
                };

                const deckA = document.createElement('div');
                deckA.className = 'radio-visual-digital-hub-deck radio-visual-digital-hub-deck--a';
                deckA.append(
                    mkKnob('knob-a-high', 'HI', '#F94C10'),
                    mkKnob('knob-a-mid', 'MED', '#9BEC00'),
                    mkKnob('knob-a-low', 'LO', '#06D001')
                );
                deckA.setAttribute('aria-label', 'Deck A EQ');

                const gainCol = document.createElement('div');
                gainCol.className = 'radio-visual-digital-hub-gain-col';
                const mkGain = (deck, label, color) => {
                    const wrap = document.createElement('div');
                    wrap.className = 'radio-visual-digital-hub-gain-wrap radio-visual-digital-hub-gain-wrap--' + deck;
                    const lbl = document.createElement('span');
                    lbl.className = 'radio-visual-digital-hub-gain-label';
                    lbl.textContent = label;
                    const slider = document.createElement('input');
                    slider.type = 'range';
                    slider.className = 'radio-visual-digital-hub-gain-slider radio-visual-digital-range';
                    slider.min = '0';
                    slider.max = '200';
                    slider.step = '1';
                    slider.dataset.deckGain = deck;
                    slider.style.setProperty('--gain-color', color);
                    slider.setAttribute('aria-label', `${label} gain`);
                    wrap.append(slider, lbl);
                    return wrap;
                };
                gainCol.append(
                    mkGain('a', 'A', '#FF0B55'),
                    mkGain('b', 'B', '#5800FF')
                );

                const deckB = document.createElement('div');
                deckB.className = 'radio-visual-digital-hub-deck radio-visual-digital-hub-deck--b';
                deckB.append(
                    mkKnob('knob-b-high', 'HI', '#f0f'),
                    mkKnob('knob-b-mid', 'MED', '#0ff'),
                    mkKnob('knob-b-low', 'LO', '#9AB3F5')
                );
                deckB.setAttribute('aria-label', 'Deck B EQ');

                volume.append(deckA, gainCol, deckB);

                const mkBeatFxKnob = (deck, suffix, label, color) => {
                    const wrap = document.createElement('div');
                    wrap.className = 'radio-visual-digital-hub-knob-wrap radio-visual-digital-hub-beatfx-wrap';
                    const knob = document.createElement('div');
                    knob.className = 'knob-wrap radio-visual-digital-hub-knob radio-visual-digital-hub-beatfx-knob';
                    knob.id = `digital-hub-dj-knob-${deck}-${suffix}`;
                    knob.style.setProperty('--knob-color', color);
                    const indicator = document.createElement('div');
                    indicator.className = 'knob-indicator';
                    const val = document.createElement('div');
                    val.className = 'knob-value';
                    knob.append(indicator, val);
                    const lbl = document.createElement('span');
                    lbl.className = 'radio-visual-digital-hub-knob-label';
                    lbl.textContent = label;
                    wrap.append(knob, lbl);
                    return wrap;
                };

                const effects = document.createElement('div');
                effects.className = 'radio-visual-digital-hub-effects';
                const fxKnobs = RadioVisualEngine.DIGITAL_HUB_FX_KNOBS;
                const fxDeckA = document.createElement('div');
                fxDeckA.className = 'radio-visual-digital-hub-fx-deck radio-visual-digital-hub-fx-deck--a';
                fxDeckA.setAttribute('aria-label', 'Deck A beat FX');
                fxKnobs.a.forEach((k) => fxDeckA.appendChild(mkBeatFxKnob('a', k.suffix, k.label, k.color)));
                const fxDeckB = document.createElement('div');
                fxDeckB.className = 'radio-visual-digital-hub-fx-deck radio-visual-digital-hub-fx-deck--b';
                fxDeckB.setAttribute('aria-label', 'Deck B beat FX');
                fxKnobs.b.forEach((k) => fxDeckB.appendChild(mkBeatFxKnob('b', k.suffix, k.label, k.color)));
                const fxCenter = document.createElement('div');
                fxCenter.className = 'radio-visual-digital-hub-fx-center';
                fxCenter.setAttribute('aria-label', 'Mixer effects and samples');
                RadioVisualEngine.DIGITAL_HUB_MIX_FX_COLS.forEach((col, colIdx) => {
                    const colEl = document.createElement('div');
                    colEl.className = 'radio-visual-digital-hub-fx-col';
                    colEl.dataset.fxCol = String(colIdx);
                    col.forEach((fx) => {
                        const btn = document.createElement('button');
                        btn.type = 'button';
                        btn.className = 'radio-visual-digital-hub-fx-btn';
                        btn.textContent = fx.label;
                        if (fx.mixId) btn.dataset.mixFx = fx.mixId;
                        if (fx.sample) btn.dataset.hubSample = fx.sample;
                        colEl.appendChild(btn);
                    });
                    fxCenter.appendChild(colEl);
                });
                effects.append(fxDeckA, fxCenter, fxDeckB);

                const aiWrap = document.createElement('div');
                aiWrap.className = 'radio-visual-digital-hub-ai';
                const aiStack = document.createElement('div');
                aiStack.className = 'radio-visual-digital-hub-ai-stack';
                const mkHubAiVideo = (slot) => {
                    const v = document.createElement('video');
                    v.className = 'radio-visual-digital-hub-ai-video radio-visual-digital-hub-ai-video--' + slot;
                    v.muted = true;
                    v.playsInline = true;
                    v.setAttribute('playsinline', '');
                    v.preload = 'auto';
                    return v;
                };
                const aiVideoA = mkHubAiVideo('a');
                const aiVideoB = mkHubAiVideo('b');
                aiVideoB.classList.add('is-hidden');
                aiStack.append(aiVideoA, aiVideoB);
                const aiLoader = document.createElement('div');
                aiLoader.className = 'radio-visual-digital-hub-ai-loader';
                aiLoader.setAttribute('aria-hidden', 'true');
                const aiLoaderRing = document.createElement('div');
                aiLoaderRing.className = 'radio-visual-digital-hub-ai-loader-ring';
                aiLoader.appendChild(aiLoaderRing);
                aiWrap.append(aiStack, aiLoader);

                const deckVideoWrap = document.createElement('div');
                deckVideoWrap.className = 'radio-visual-digital-hub-deck-video';
                const deckVideoEl = document.createElement('video');
                deckVideoEl.className = 'radio-visual-digital-hub-deck-video-el';
                deckVideoEl.muted = true;
                deckVideoEl.playsInline = true;
                deckVideoEl.setAttribute('playsinline', '');
                deckVideoEl.preload = 'auto';
                deckVideoWrap.appendChild(deckVideoEl);

                const off = document.createElement('div');
                off.className = 'radio-visual-digital-hub-off';
                off.textContent = 'OFF';

                panel.append(volume, effects, aiWrap, deckVideoWrap, off);
                return panel;
            }

            _wireDigitalHubPanel(abortSignal) {
                const wireKnob = globalThis.wireDjKnobMirror;
                if (typeof wireKnob === 'function') {
                    ['knob-a-low', 'knob-a-mid', 'knob-a-high', 'knob-b-low', 'knob-b-mid', 'knob-b-high'].forEach((id) => {
                        const hub = document.getElementById(`digital-hub-${id}`);
                        if (hub) wireKnob(hub, id, abortSignal);
                    });
                }

                const eqState = globalThis.eqState;
                const opts = abortSignal ? { signal: abortSignal } : {};
                this.els.digitalHubPanel?.querySelectorAll('[data-deck-gain]').forEach((slider) => {
                    const deck = slider.dataset.deckGain;
                    const cfgKey = `knob-${deck}-gain`;
                    const cfg = globalThis.__mixKnobConfig?.[cfgKey];
                    const deckState = eqState?.[deck];
                    const initial = deckState?.gain ?? cfg?.stateObj?.[cfg?.key] ?? 1;
                    slider.value = String(Math.round(initial * 100));

                    slider.addEventListener('input', () => {
                        const v = parseInt(slider.value, 10) / 100;
                        if (deckState) deckState.gain = v;
                        if (cfg) {
                            cfg.stateObj[cfg.key] = v;
                            const mixKnob = document.getElementById(cfgKey);
                            if (mixKnob && typeof globalThis.setKnobUi === 'function') {
                                globalThis.setKnobUi(mixKnob, cfg.min, cfg.max, v);
                            }
                            if (cfg.callback) cfg.callback(v);
                        }
                    }, opts);
                });

                try {
                    const fxRoot = this.els.digitalHubEffects;
                    if (fxRoot && typeof globalThis.wireDjBeatFxKnobs === 'function') {
                        globalThis.wireDjBeatFxKnobs(fxRoot);
                        try { globalThis.syncDjBeatFxKnobActiveDom?.(this.root); } catch (_) {}
                    }
                } catch (_) {}
                this.els.digitalHubMixFx?.querySelectorAll('.radio-visual-digital-hub-fx-btn').forEach((btn) => {
                    btn.addEventListener('click', (ev) => {
                        try { ev.stopPropagation(); } catch (_) {}
                        if (btn.dataset.hubSample) {
                            try {
                                const a = new Audio(btn.dataset.hubSample);
                                a.play().catch(() => {});
                            } catch (_) {}
                            return;
                        }
                        const mixId = btn.dataset.mixFx;
                        if (!mixId) return;
                        document.getElementById(mixId)?.click();
                        requestAnimationFrame(() => this._syncDigitalHubMixFxButtons());
                    }, opts);
                });
                try { this._syncDigitalHubMixFxButtons(); } catch (_) {}
            }

            _syncDigitalHubMixFxButtons() {
                const root = this.els.digitalHubMixFx;
                if (!root) return;
                root.querySelectorAll('[data-mix-fx]').forEach((btn) => {
                    const mix = document.getElementById(btn.dataset.mixFx);
                    btn.classList.toggle('on', !!(mix && mix.classList.contains('on')));
                });
            }

            _digitalHubAiActiveVideo() {
                const slot = this._digitalHubAiVideoSlot || 0;
                return slot === 1 ? this.els.digitalHubAiVideoB : this.els.digitalHubAiVideoA;
            }

            _digitalHubAiIdleVideo() {
                const slot = this._digitalHubAiVideoSlot || 0;
                return slot === 1 ? this.els.digitalHubAiVideoA : this.els.digitalHubAiVideoB;
            }

            _syncDigitalHubAiVideoAlias() {
                this.els.digitalHubAiVideo = this._digitalHubAiActiveVideo();
            }

            _resetDigitalHubAiVideoSlots() {
                const a = this.els.digitalHubAiVideoA;
                const b = this.els.digitalHubAiVideoB;
                this._digitalHubAiVideoSlot = 0;
                if (a) {
                    try { a.classList.remove('is-hidden'); } catch (_) {}
                }
                if (b) {
                    try { b.classList.add('is-hidden'); } catch (_) {}
                }
                this._syncDigitalHubAiVideoAlias();
            }

            _showDigitalHubAiVideoEl(showEl, hideEl) {
                if (showEl) {
                    try { showEl.classList.remove('is-hidden'); } catch (_) {}
                }
                if (hideEl) {
                    try { hideEl.classList.add('is-hidden'); } catch (_) {}
                    try { hideEl.pause(); } catch (_) {}
                }
                this._syncDigitalHubAiVideoAlias();
            }

            _digitalAiVideoFilesFromStatic() {
                return RadioVisualEngine.DIGITAL_AI_VIDEO_LIST.slice();
            }

            _sortDigitalAiVideoNames(names) {
                const list = (names || [])
                    .map((f) => String(f || '').trim())
                    .filter((f) => /\.mp4$/i.test(f));
                const primary = list.filter((f) => f.toLowerCase() === 'ai.mp4');
                const rest = list
                    .filter((f) => f.toLowerCase() !== 'ai.mp4')
                    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
                return [...primary, ...rest];
            }

            _digitalAiVideoFiles() {
                if (this._digitalAiVideoFilesList && this._digitalAiVideoFilesList.length) {
                    return this._digitalAiVideoFilesList;
                }
                return this._digitalAiVideoFilesFromStatic();
            }

            _refreshDigitalAiVideoList() {
                if (this._digitalAiVideoManifestPromise) return this._digitalAiVideoManifestPromise;
                this._digitalAiVideoManifestPromise = (async () => {
                    try {
                        const url = `${RadioVisualEngine.DIGITAL_AI_VIDEO_MANIFEST}?t=${Date.now()}`;
                        const res = await fetch(url, { cache: 'no-store' });
                        if (res.ok) {
                            const data = await res.json();
                            const raw = Array.isArray(data)
                                ? data
                                : (data && (data.videos || data.files)) || [];
                            const sorted = this._sortDigitalAiVideoNames(raw);
                            if (sorted.length) {
                                this._digitalAiVideoFilesList = sorted;
                                return sorted;
                            }
                        }
                    } catch (_) {}
                    this._digitalAiVideoFilesList = this._digitalAiVideoFilesFromStatic();
                    return this._digitalAiVideoFilesList;
                })().finally(() => {
                    this._digitalAiVideoManifestPromise = null;
                });
                return this._digitalAiVideoManifestPromise;
            }

            _digitalHubAiVideoPool() {
                const dir = RadioVisualEngine.DIGITAL_AI_VIDEO_DIR;
                return this._digitalAiVideoFiles().map((file) => dir + file);
            }

            _pickRandomDigitalHubAiVideoSrc(excludeSrc) {
                const pool = this._digitalHubAiVideoPool();
                let choices = excludeSrc ? pool.filter((src) => src !== excludeSrc) : pool.slice();
                if (!choices.length) choices = pool.slice();
                return choices[Math.floor(Math.random() * choices.length)];
            }

            _prepareDigitalHubAiClip(video, src) {
                if (!video || !src) return Promise.resolve(false);
                this._digitalHubAiVideoLoadGen = (this._digitalHubAiVideoLoadGen || 0) + 1;
                const gen = this._digitalHubAiVideoLoadGen;
                video.loop = false;
                try { video.pause(); } catch (_) {}

                return new Promise((resolve) => {
                    let settled = false;
                    const done = (ok) => {
                        if (settled) return;
                        settled = true;
                        resolve(ok);
                    };
                    const onReady = () => {
                        if (gen !== this._digitalHubAiVideoLoadGen) return;
                        done(true);
                    };
                    const onErr = () => {
                        if (gen !== this._digitalHubAiVideoLoadGen) return;
                        done(false);
                    };
                    try { video.addEventListener('canplaythrough', onReady, { once: true }); } catch (_) {}
                    try { video.addEventListener('loadeddata', onReady, { once: true }); } catch (_) {}
                    try { video.addEventListener('error', onErr, { once: true }); } catch (_) {}
                    try { video.src = src; } catch (_) {}
                    try {
                        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                            queueMicrotask(onReady);
                        }
                    } catch (_) {}
                });
            }

            _playPreparedDigitalHubAiClip(video) {
                if (!video || !RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                    return Promise.resolve(false);
                }
                this._setDigitalHubAiVideoLoading(true);
                try { video.loop = false; } catch (_) {}
                try { video.currentTime = 0; } catch (_) {}
                let p;
                try { p = video.play(); } catch (_) {
                    return Promise.resolve(false);
                }
                if (p && typeof p.then === 'function') {
                    return p.then(() => {
                        this._setDigitalHubAiVideoLoading(false);
                        return true;
                    }).catch(() => {
                        this._syncDigitalHubAiVideoLoadingState();
                        return false;
                    });
                }
                return Promise.resolve(true);
            }

            _startDigitalHubAiClip(video, src) {
                if (!video || !src) return Promise.resolve(false);
                this._setDigitalHubAiVideoLoading(true);
                return this._prepareDigitalHubAiClip(video, src).then((ok) => {
                    if (!ok) {
                        this._setDigitalHubAiVideoLoading(false);
                        return false;
                    }
                    return this._playPreparedDigitalHubAiClip(video);
                });
            }

            _onDigitalHubAiVideoFinished(fromVideo) {
                if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return;
                if (fromVideo && fromVideo !== this._digitalHubAiActiveVideo()) return;
                if (this._digitalHubAiAdvancing || this._digitalHubAiEndGuard) return;
                this._digitalHubAiEndGuard = true;
                Promise.resolve()
                    .then(() => this._advanceDigitalHubAiVideo())
                    .finally(() => { this._digitalHubAiEndGuard = false; });
            }

            _advanceDigitalHubAiVideo() {
                if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return Promise.resolve(false);
                if (this._digitalHubAiAdvancing) return Promise.resolve(false);
                this._digitalHubAiAdvancing = true;
                this._setDigitalHubAiVideoLoading(true);

                const idle = this._digitalHubAiIdleVideo();
                const active = this._digitalHubAiActiveVideo();
                const next = this._pickRandomDigitalHubAiVideoSrc(this._digitalHubAiVideoSrc);
                this._digitalHubAiVideoSrc = next;

                return this._prepareDigitalHubAiClip(idle, next).then((ok) => {
                    if (!ok || !RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                        return false;
                    }
                    this._showDigitalHubAiVideoEl(idle, active);
                    this._digitalHubAiVideoSlot = 1 - (this._digitalHubAiVideoSlot || 0);
                    this._syncDigitalHubAiVideoAlias();
                    return this._playPreparedDigitalHubAiClip(idle);
                }).then((played) => {
                    if (!played && RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                        this._syncDigitalHubAiVideoLoadingState();
                    }
                    return played;
                }).finally(() => {
                    this._digitalHubAiAdvancing = false;
                });
            }

            _setDigitalHubAiVideoLoading(loading) {
                const loader = this.els.digitalHubAiLoader;
                if (!loader) return;
                const on = !!loading;
                loader.classList.toggle('is-visible', on);
                loader.setAttribute('aria-hidden', on ? 'false' : 'true');
            }

            _syncDigitalHubAiVideoLoadingState() {
                const video = this._digitalHubAiActiveVideo();
                if (!video || !RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                    this._setDigitalHubAiVideoLoading(false);
                    return;
                }
                const buffering = video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA
                    || video.networkState === HTMLMediaElement.NETWORK_LOADING
                    || (video.paused && !video.ended && video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA);
                this._setDigitalHubAiVideoLoading(buffering);
            }

            _syncDigitalHubAiVideo() {
                const videos = [this.els.digitalHubAiVideoA, this.els.digitalHubAiVideoB].filter(Boolean);
                if (!videos.length) return;
                if (RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                    if (!this._digitalHubAiVideoSrc) {
                        this._resetDigitalHubAiVideoSlots();
                        const src = this._pickRandomDigitalHubAiVideoSrc();
                        this._digitalHubAiVideoSrc = src;
                        this._startDigitalHubAiClip(this._digitalHubAiActiveVideo(), src);
                        return;
                    }
                    const active = this._digitalHubAiActiveVideo();
                    if (active) {
                        this._startDigitalHubAiClip(active, this._digitalHubAiVideoSrc);
                    }
                } else {
                    videos.forEach((video) => {
                        try { video.pause(); } catch (_) {}
                        try { video.currentTime = 0; } catch (_) {}
                        try { video.removeAttribute('src'); } catch (_) {}
                    });
                    this._digitalHubAiVideoSrc = null;
                    this._digitalHubAiAdvancing = false;
                    this._resetDigitalHubAiVideoSlots();
                    this._setDigitalHubAiVideoLoading(false);
                }
            }

            _dismissDigitalHubAiVideoOverlay() {
                if (this._digitalHubMode === 'live') this._exitDigitalHubLiveToEqualiser();
                else if (this._digitalHubMode === 'video') this._exitDigitalHubVideoToEqualiser();
                else if (this._digitalHubMode === 'ai') this._exitDigitalHubAiMode();
            }

            _exitDigitalHubVideoToEqualiser() {
                if (this._digitalHubMode !== 'video') return;
                this._pauseDigitalHubDeckVideo();
                this._digitalHubMode = 'equaliser';
                if ((this._digitalVideoToolbarStep || 0) === 1) this._resetDigitalVideoToolbarStep();
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
            }

            _isDigitalHubDeckVideoTarget(el) {
                if (!el || !el.closest) return false;
                return !!el.closest('.radio-visual-digital-hub-deck-video, .radio-visual-digital-hub-deck-video-el');
            }

            _clearDigitalHubDeckVideoTapTimer() {
                if (this._digitalHubDeckVideoTapTimer) {
                    clearTimeout(this._digitalHubDeckVideoTapTimer);
                    this._digitalHubDeckVideoTapTimer = null;
                }
            }

            _onDigitalHubDeckVideoClick(ev) {
                if (this.skin !== 'digital') return false;
                if (!RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode)) return false;
                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                try { window.__suppressNextClick = true; } catch (_) {}
                if (ev.detail >= 2) {
                    this._clearDigitalHubDeckVideoTapTimer();
                    try { this._toggleDigitalHubDeckVideoFullscreen(); } catch (_) {}
                    return true;
                }
                this._clearDigitalHubDeckVideoTapTimer();
                this._digitalHubDeckVideoTapTimer = setTimeout(() => {
                    this._digitalHubDeckVideoTapTimer = null;
                    try { this._exitDigitalHubVideoToEqualiser(); } catch (_) {}
                }, 280);
                return true;
            }

            _toggleDigitalHubDeckVideoFullscreen() {
                const video = this.els.digitalHubDeckVideo;
                const wrap = this.els.digitalHubDeckVideoWrap;
                if (!video || !wrap) return;
                try {
                    if (typeof toggleVideoSurfaceFullscreen === 'function') {
                        toggleVideoSurfaceFullscreen(video, wrap);
                        return;
                    }
                } catch (_) {}
                try {
                    const fs = document.fullscreenElement || document.webkitFullscreenElement;
                    if (fs && (fs === wrap || wrap.contains(fs))) {
                        if (document.exitFullscreen) document.exitFullscreen();
                        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
                        return;
                    }
                    const req = wrap.requestFullscreen || wrap.webkitRequestFullscreen;
                    if (req) req.call(wrap);
                } catch (_) {}
            }

            _wireDigitalHubDeckVideoInteractions(abortSignal) {
                const wrap = this.els.digitalHubDeckVideoWrap;
                const video = this.els.digitalHubDeckVideo;
                if (!wrap || !video || wrap.dataset.rvHubDeckVideoWired === '1') return;
                const sig = abortSignal && abortSignal.signal ? abortSignal.signal : abortSignal;
                const capOpts = sig ? { signal: sig, capture: true } : { capture: true };
                try { wrap.title = 'Double-click for fullscreen · Esc to exit'; } catch (_) {}
                const onClick = (ev) => { this._onDigitalHubDeckVideoClick(ev); };
                const onDbl = (ev) => {
                    if (this.skin !== 'digital') return;
                    if (!RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode)) return;
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    this._clearDigitalHubDeckVideoTapTimer();
                    try { this._toggleDigitalHubDeckVideoFullscreen(); } catch (_) {}
                };
                wrap.addEventListener('click', onClick, capOpts);
                wrap.addEventListener('dblclick', onDbl, capOpts);
                video.addEventListener('click', onClick, capOpts);
                video.addEventListener('dblclick', onDbl, capOpts);
                wrap.dataset.rvHubDeckVideoWired = '1';
            }

            _isDigitalHubAiVideoTarget(el) {
                if (!el || !el.closest) return false;
                return !!el.closest('.radio-visual-digital-hub-ai-video, .radio-visual-digital-hub-ai');
            }

            _clearDigitalHubAiVideoTapTimer() {
                if (this._digitalHubAiTapTimer) {
                    clearTimeout(this._digitalHubAiTapTimer);
                    this._digitalHubAiTapTimer = null;
                }
            }

            /** Single click → dismiss overlay; double click → fullscreen the AI video surface. */
            _onDigitalHubAiVideoClick(ev) {
                if (this.skin !== 'digital') return false;
                if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return false;
                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                try { window.__suppressNextClick = true; } catch (_) {}
                if (ev.detail >= 2) {
                    this._clearDigitalHubAiVideoTapTimer();
                    try { this._toggleDigitalHubAiVideoFullscreen(); } catch (_) {}
                    return true;
                }
                this._clearDigitalHubAiVideoTapTimer();
                this._digitalHubAiTapTimer = setTimeout(() => {
                    this._digitalHubAiTapTimer = null;
                    try { this._dismissDigitalHubAiVideoOverlay(); } catch (_) {}
                }, 280);
                return true;
            }

            _toggleDigitalHubAiVideoFullscreen() {
                const video = this._digitalHubAiActiveVideo();
                const wrap = this.els.digitalHubAi;
                if (!video || !wrap) return;
                try {
                    if (typeof toggleVideoSurfaceFullscreen === 'function') {
                        toggleVideoSurfaceFullscreen(video, wrap);
                        return;
                    }
                } catch (_) {}
                try {
                    const fs = document.fullscreenElement || document.webkitFullscreenElement;
                    if (fs && (fs === wrap || wrap.contains(fs))) {
                        if (document.exitFullscreen) document.exitFullscreen();
                        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
                        return;
                    }
                    const req = wrap.requestFullscreen || wrap.webkitRequestFullscreen;
                    if (req) req.call(wrap);
                } catch (_) {}
            }

            _finalizeDigitalRadioInit() {
                this._digitalHubMode = 'equaliser';
                this._digitalSpectrumLayout = 'full';
                this.digitalCenterMode = 'spectrum';
                const pane = this.els.digitalCenterSpectrum;
                const deckB = this.els.digitalCenterDeckB;
                if (pane) {
                    pane.classList.add('is-active');
                    pane.dataset.hubMode = 'equaliser';
                    pane.classList.remove('is-spectrum-hud-hidden', 'is-spectrum-sides-hidden');
                }
                if (deckB) deckB.classList.remove('is-active');
                const dash = this.els.digitalDashStack;
                if (dash) dash.classList.remove('is-spectrum-hud-hidden');
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                try { this._syncSpectrumStagingScale(); } catch (_) {}
                try { this.applySpectrumSettings(RadioVisualEngine.loadSpectrumSettingsFromStorage(), { skipSave: true }); } catch (_) {}
                try {
                    if (typeof globalThis.applyDigitalRadioTheme === 'function') {
                        globalThis.applyDigitalRadioTheme();
                    }
                } catch (_) {}
                this._scheduleDigitalSpectrumLayoutSync();
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        try { this._resizeCanvases(); } catch (_) {}
                        try { this._drawDigitalSpectrum(); } catch (_) {}
                    });
                });
            }

            _syncDigitalHubPanel() {
                const mode = this._digitalHubMode || 'equaliser';
                const panel = this.els.digitalHubPanel;
                const pane = this.els.digitalCenterSpectrum;
                if (!panel || !pane) return;

                pane.dataset.hubMode = mode;
                pane.classList.toggle('is-hub-equaliser', mode === 'equaliser');
                pane.classList.toggle('is-hub-spectrum', mode === 'spectrum');
                pane.classList.toggle('is-hub-volume', mode === 'volume');
                pane.classList.toggle('is-hub-effects', mode === 'effects');
                pane.classList.toggle('is-hub-ai', mode === 'ai');
                pane.classList.toggle('is-hub-live', mode === 'live');
                pane.classList.toggle('is-hub-video', mode === 'video');
                pane.classList.toggle('is-hub-off', mode === 'ai-off');

                const showHub = mode === 'volume' || mode === 'effects'
                    || RadioVisualEngine.digitalHubShowsCenterVideo(mode) || mode === 'ai-off';
                panel.setAttribute('aria-hidden', showHub ? 'false' : 'true');

                if (mode === 'volume') {
                    const eqState = globalThis.eqState;
                    this.els.digitalHubPanel?.querySelectorAll('[data-deck-gain]').forEach((slider) => {
                        const deck = slider.dataset.deckGain;
                        const deckState = eqState?.[deck];
                        const cfg = globalThis.__mixKnobConfig?.[`knob-${deck}-gain`];
                        const initial = deckState?.gain ?? cfg?.stateObj?.[cfg?.key] ?? 1;
                        slider.value = String(Math.round(initial * 100));
                    });
                }

                if (mode === 'effects') {
                    try { globalThis.syncDjBeatFxKnobActiveDom?.(this.root || document.getElementById('radio-visual-root')); } catch (_) {}
                    try { this._syncDigitalHubMixFxButtons(); } catch (_) {}
                }
                if (mode === 'spectrum') {
                    this._spectrumStagingScale = this._getSpectrumHubSideScale();
                    const scaleSlider = this.els && this.els.digitalSpectrumScaleSlider;
                    if (scaleSlider) {
                        scaleSlider.value = String(this._spectrumScaleToSliderPct(this._spectrumStagingScale));
                    }
                    try { this._syncSpectrumStagingScale(); } catch (_) {}
                    try { this._hideDigitalSpectrumStagingSlider(true); } catch (_) {}
                } else {
                    try { this._syncSpectrumStagingScale(); } catch (_) {}
                    try { this._hideDigitalSpectrumStagingSlider(true); } catch (_) {}
                }
                if (RadioVisualEngine.digitalHubShowsDeckVideo(mode)) {
                    try { this._syncDigitalHubDeckVideo(true); } catch (_) {}
                } else {
                    this._pauseDigitalHubDeckVideo();
                }
                this._syncDigitalHubAiVideo();
            }

            _syncDigitalSpectrumLayout() {
                const hubMode = this._digitalHubMode;
                const layout = this._hubSpectrumLayout(hubMode);
                this._digitalSpectrumLayout = layout;
                const hudHidden = hubMode === 'spectrum';
                const sidesHidden = hubMode === 'ai-off';
                const pane = this.els.digitalCenterSpectrum;
                if (pane) {
                    pane.classList.toggle('is-spectrum-hud-hidden', hudHidden);
                    pane.classList.toggle('is-spectrum-sides-hidden', sidesHidden);
                }
                const dash = this.els.digitalDashStack;
                if (dash) dash.classList.toggle('is-spectrum-hud-hidden', hudHidden);
                this._syncDigitalHubPanel();
                this._syncDigitalSpectrumButtonState();
                if (this.skin === 'digital' && this.digitalCenterMode === 'spectrum') {
                    this._scheduleDigitalSpectrumLayoutSync();
                }
            }

            _syncDigitalSpectrumButtonState() {
                const btn = this.els.btnDigitalSpectrum;
                if (!btn) return;
                const inSpectrumCenter = this.digitalCenterMode === 'spectrum' && !this._digitalStagingView;
                const hubMode = this._digitalHubMode;
                const labelMode = (hubMode === 'ai' || hubMode === 'live') ? 'live' : hubMode;
                const spectrumsVisible = hubMode !== 'ai-off';
                btn.classList.toggle('is-active', inSpectrumCenter && spectrumsVisible);
                btn.setAttribute('aria-pressed', inSpectrumCenter ? 'true' : 'false');
                btn.setAttribute('aria-label', `Centre display: ${this._digitalHubModeLabel(labelMode)}`);
                const labelEl = btn.querySelector('.radio-visual-btn-label');
                const hubLabel = this._digitalHubModeLabel(labelMode);
                if (labelEl) labelEl.textContent = hubLabel;
                else this._appendRvButtonLabel(btn, hubLabel);
                try { btn.removeAttribute('title'); } catch (_) {}
            }

            /** equaliser → volume → effects → live → video → off → spectrum → equaliser */
            _setDigitalHubModeOff() {
                if (this._digitalStagingView) {
                    this._digitalStagingView = null;
                    try { this._tearDownDigitalStagingView(); } catch (_) {}
                    try { this._syncDigitalStagingButtons(); } catch (_) {}
                }
                if (this.digitalCenterMode !== 'spectrum') {
                    try { this._setDigitalCenterMode('spectrum'); } catch (_) {}
                }
                const prev = this._digitalHubMode;
                if (prev === 'video') {
                    this._pauseDigitalHubDeckVideo();
                    if ((this._digitalVideoToolbarStep || 0) <= 1) this._resetDigitalVideoToolbarStep();
                }
                this._digitalHubMode = 'ai-off';
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                try { resetIdleTimer(); } catch (_) {}
            }

            _cycleDigitalSpectrumLayout() {
                if (this._digitalHubMode === 'ai') {
                    this._exitDigitalHubAiMode();
                    return;
                }
                const cycle = RadioVisualEngine.DIGITAL_HUB_CYCLE;
                let i = cycle.indexOf(this._digitalHubMode);
                if (i < 0) i = 0;
                const prev = this._digitalHubMode;
                this._digitalHubMode = cycle[(i + 1) % cycle.length];
                if (prev === 'video' && this._digitalHubMode !== 'video') {
                    this._pauseDigitalHubDeckVideo();
                    if ((this._digitalVideoToolbarStep || 0) <= 1) this._resetDigitalVideoToolbarStep();
                }
                if (this._digitalHubMode === 'video' && this.digitalCenterMode !== 'spectrum') {
                    this._setDigitalCenterMode('spectrum');
                }
                this._syncDigitalSpectrumLayout();
            }

            /** Default spectrum layout: no staging overlay, central dash visible. */
            _returnToDefaultDigitalSpectrumView() {
                try { this._closeDigitalLocalQueuePanel(); } catch (_) {}
                try { this._closeDigitalStationsPanel(); } catch (_) {}
                if (this._digitalStagingView) {
                    this._digitalStagingView = null;
                    try { this._tearDownDigitalStagingView(); } catch (_) {}
                    try { this._syncDigitalStagingButtons(); } catch (_) {}
                }
                if (this.digitalCenterMode !== 'spectrum') {
                    this._setDigitalCenterMode('spectrum');
                    return;
                }
                this._digitalHubMode = 'equaliser';
                this._digitalSpectrumLayout = 'full';
                this._resetDigitalVideoToolbarStep();
                this._pauseDigitalHubDeckVideo();
                try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                const mount = this.els.digitalStagingMount;
                if (mount) {
                    mount.classList.remove('is-active');
                    mount.setAttribute('aria-hidden', 'true');
                }
            }

            _setDigitalCenterMode(mode) {
                const next = (mode === 'deckB') ? 'deckB' : 'spectrum';
                const wasDeckB = this.digitalCenterMode === 'deckB';
                if (next !== 'spectrum') {
                    try { this._closeDigitalLocalQueuePanel(); } catch (_) {}
                    try { this._closeDigitalStationsPanel(); } catch (_) {}
                }
                this.digitalCenterMode = next;
                try { localStorage.setItem('radioVisual.digitalCenter.v1', next); } catch (_) {}
                if (next === 'spectrum') {
                    if (wasDeckB) {
                        if ((this._digitalVideoToolbarStep || 0) === 2) this._resetDigitalVideoToolbarStep();
                        this._digitalHubMode = 'equaliser';
                    }
                    try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                }
                if (this.els.digitalCenterSpectrum) {
                    this.els.digitalCenterSpectrum.classList.toggle('is-active', next === 'spectrum');
                }
                if (this.els.digitalCenterDeckB) {
                    this.els.digitalCenterDeckB.classList.toggle('is-active', next === 'deckB');
                }
                this._syncDigitalSpectrumButtonState();
                if (next === 'deckB') {
                    this._setDigitalDeckBView(this._digitalDeckBView || 'video');
                } else if (this._digitalStagingView) {
                    this._setDigitalStagingView(this._digitalStagingView);
                } else if (this.els.digitalStagingMount) {
                    this.els.digitalStagingMount.classList.remove('is-active');
                }
            }

            _tearDownDigitalDeckBPlayer() {
                try { this._tearDownDigitalStagingView(); } catch (_) {}
                this._digitalDeckBVideoSrc = '';
                this._digitalDeckBVideoMode = 'idle';
                try {
                    if (this.els.digitalDeckBVideo) {
                        this.els.digitalDeckBVideo.pause();
                        this.els.digitalDeckBVideo.removeAttribute('src');
                    }
                } catch (_) {}
            }

            _syncDigitalDeckBVideo(forceLoad) {
                const vid = this.els.digitalDeckBVideo;
                if (!vid || this.digitalCenterMode !== 'deckB') return;
                try {
                    const payload = this._resolveDigitalStagingVideoPayload();
                    const want = String(payload.url || '');
                    if (!payload.syncFrom && payload.mode === 'idle') {
                        const same = this._digitalDeckBVideoSrc && want && (
                            (typeof urlsMediaMatch === 'function')
                                ? urlsMediaMatch(this._digitalDeckBVideoSrc, want)
                                : this._digitalDeckBVideoSrc === want
                        );
                        if (!forceLoad && same && !vid.paused) return;
                        this._applyDigitalIdleLogoVideo(vid, want);
                        this._digitalDeckBVideoSrc = want;
                        this._digitalDeckBVideoMode = 'idle';
                        return;
                    }
                    if (!forceLoad && this._digitalDeckBVideoSrc && want && (
                        (typeof urlsMediaMatch === 'function')
                            ? urlsMediaMatch(this._digitalDeckBVideoSrc, want)
                            : this._digitalDeckBVideoSrc === want
                    )) {
                        if (payload.syncFrom) {
                            this._applyDigitalStagingVideoPayload(vid, payload);
                        } else if (vid.paused) {
                            vid.play().catch(() => {});
                        }
                        this._digitalDeckBVideoMode = payload.mode || 'idle';
                        return;
                    }
                    this._digitalDeckBVideoMode = payload.mode || 'idle';
                    if (!payload.syncFrom && payload.loop) {
                        this._applyDigitalIdleLogoVideo(vid, want);
                    } else {
                        this._applyDigitalStagingVideoPayload(vid, payload);
                    }
                    this._digitalDeckBVideoSrc = want;
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
                const blended = this._blendSpectrumCircularEnds(radii, 3);
                const n = blended.length;
                if (n < 2) return blended;
                const tie = (blended[0] + blended[n - 1]) * 0.5;
                blended[0] = tie;
                blended[n - 1] = tie;
                return blended;
            }

            _spectrumPetalNorm(ii, radii, n, maxRing, petalFloor) {
                const raw = (radii[ii] || 0) / maxRing;
                return Math.min(1, Math.max(petalFloor, raw));
            }

            _spectrumRibbonSeamAngle(n, phaseBins) {
                return this._spectrumEdgeAngle(0, n, phaseBins);
            }

            _spectrumConicStop(u, opts) {
                const {
                    useOuterPalette, isHigh, hue, sat, lit, drift, layerSpec, bass, phaseBase, shimmerT
                } = opts;
                const opacityMult = this._getSpectrumOpacityMult();
                let h;
                let satUse = sat;
                let litUse = lit;
                let a0 = layerSpec.alpha * (0.5 + 0.5 * u) * opacityMult;
                if (useOuterPalette) {
                    const phaseWrap = phaseBase + u * 1.1 + bass * 0.35;
                    h = this._paletteHueAt(phaseWrap);
                } else if (isHigh) {
                    const ripple = Math.sin(u * Math.PI * 6 + shimmerT * 5.8);
                    const pulse = 0.65 + 0.35 * Math.sin(shimmerT * 4.4 + u * Math.PI * 2);
                    h = (hue + u * drift + shimmerT * 48 + ripple * 32) % 360;
                    satUse = Math.min(100, sat + 6 + pulse * 10);
                    litUse = lit + 10 + pulse * 22 + ripple * 8;
                    a0 = layerSpec.alpha * (0.62 + 0.38 * u) * (0.88 + pulse * 0.12) * opacityMult;
                } else {
                    const cycle = ((u * drift) % 360 + 360) % 360;
                    h = (hue + cycle) % 360;
                }
                return `hsla(${h}, ${satUse}%, ${litUse}%, ${a0})`;
            }

            _spectrumPetalRadialFill(ctx, cx, cy, zoneInner, zoneOuter, layer, coreHue, t) {
                const hue = this._spectrumPetalBaseHue(layer, coreHue);
                const sat = layer.sat ?? 88;
                const lit = layer.light ?? 54;
                const drift = layer.hueDrift ?? 40;
                const opacityMult = this._getSpectrumOpacityMult();
                const petal = ctx.createRadialGradient(cx, cy, zoneInner, cx, cy, zoneOuter);
                const shimmerT = typeof t === 'number' ? t : performance.now() * 0.001;
                if (layer.key === 'high') {
                    const pulse = 0.5 + 0.5 * Math.sin(shimmerT * 4.4);
                    const ripple = Math.sin(shimmerT * 5.8);
                    const h0 = (hue + shimmerT * 42 + ripple * 18) % 360;
                    const h1 = (hue + drift * 0.35 + shimmerT * 28 + ripple * 12) % 360;
                    const h2 = (hue + drift * 0.7 + shimmerT * 55 - ripple * 10) % 360;
                    petal.addColorStop(0, `hsla(${h0}, ${Math.min(100, sat + 10)}%, ${lit + 14 + pulse * 18}%, ${layer.alpha * (0.6 + pulse * 0.15) * opacityMult})`);
                    petal.addColorStop(0.55, `hsla(${h1}, ${sat + 6}%, ${lit + 8 + pulse * 16}%, ${layer.alpha * (0.92 + pulse * 0.08) * opacityMult})`);
                    petal.addColorStop(1, `hsla(${h2}, ${sat}%, ${lit + pulse * 12}%, ${layer.alpha * 0.9 * opacityMult})`);
                } else {
                    petal.addColorStop(0, `hsla(${hue}, ${sat}%, ${lit + 6}%, ${layer.alpha * 0.55 * opacityMult})`);
                    petal.addColorStop(0.55, `hsla(${(hue + drift * 0.35) % 360}, ${sat}%, ${lit}%, ${layer.alpha * opacityMult})`);
                    petal.addColorStop(1, `hsla(${(hue + drift * 0.7) % 360}, ${sat - 4}%, ${lit - 8}%, ${layer.alpha * 0.85 * opacityMult})`);
                }
                return petal;
            }

            _bassLevelFromSmooth(smooth) {
                if (!smooth || !smooth.length) return 0;
                let sum = 0;
                let peak = 0;
                for (let i = 0; i < smooth.length; i++) {
                    const v = smooth[i] || 0;
                    sum += v;
                    if (v > peak) peak = v;
                }
                return Math.min(1, Math.max(peak, (sum / smooth.length) * 2.2));
            }

            _paletteHueAt(phase) {
                const palette = this._getSpectrumColorPalette();
                const n = palette.length;
                if (n < 2) return palette[0] || 0;
                const p = ((phase % n) + n) % n;
                const i0 = Math.floor(p);
                const i1 = (i0 + 1) % n;
                const frac = p - i0;
                const h0 = palette[i0];
                const h1 = palette[i1];
                let delta = h1 - h0;
                if (delta > 180) delta -= 360;
                if (delta < -180) delta += 360;
                return (h0 + delta * frac + 360) % 360;
            }

            _advanceOuterHuePhase(t, bassLevel) {
                const dt = this._outerHueLastT ? Math.min(0.12, Math.max(0, t - this._outerHueLastT)) : 0.016;
                this._outerHueLastT = t;
                const flow = this._getSpectrumColorFlow();
                this._outerHuePhase = (this._outerHuePhase || 0) + dt * (0.04 + bassLevel * 3.2) * flow;
            }

            static get SPECTRUM_ANGULAR_BINS() { return 72; }

            static get SPECTRUM_OUTER_HUE_PALETTE() {
                return [18, 35, 55, 95, 140, 185, 220, 265, 310, 350];
            }

            /** Radial order: 0 = inner (high), 1 = mid, 2 = outer (low). */
            static get SPECTRUM_FLOWER_LAYERS() {
                return [
                    {
                        key: 'high',
                        layerIndex: 0,
                        phaseBins: 0,
                        hue: 198,
                        hueDrift: 58,
                        sat: 90,
                        light: 56,
                        alpha: 0.62,
                        maxRing: 0.66
                    },
                    {
                        key: 'mid',
                        layerIndex: 1,
                        phaseBins: 2.5,
                        hue: 152,
                        hueDrift: 65,
                        sat: 90,
                        light: 52,
                        alpha: 0.58,
                        maxRing: 0.64
                    },
                    {
                        key: 'low',
                        layerIndex: 2,
                        phaseBins: 5,
                        hue: 24,
                        hueDrift: 70,
                        sat: 92,
                        light: 54,
                        alpha: 0.54,
                        maxRing: 0.62
                    }
                ];
            }

            _spectrumPetalBaseHue(layer, coreHue) {
                if (typeof layer.hue === 'number') return ((layer.hue % 360) + 360) % 360;
                return (((Number(coreHue) || 0) + (layer.hueOff || 0)) % 360 + 360) % 360;
            }

            _sampleBandFromFftRange(buf, start, end, n) {
                const levels = [];
                const span = Math.max(1, end - start);
                for (let i = 0; i < n; i++) {
                    const f = start + (i / Math.max(1, n - 1)) * span;
                    const i0 = Math.floor(f);
                    const i1 = Math.min(buf.length - 1, i0 + 1);
                    const tt = f - i0;
                    const v = (buf[i0] || 0) * (1 - tt) + (buf[i1] || 0) * tt;
                    levels.push(v / 255);
                }
                return levels;
            }

            _sampleDigitalSpectrumBandLevels(t) {
                const n = RadioVisualEngine.SPECTRUM_ANGULAR_BINS;
                const out = { low: [], mid: [], high: [], fromAnalyser: false };
                try {
                    if (state.analyserNode && state.audioCtx) {
                        const fft = state.analyserNode.fftSize || 256;
                        const binCount = state.analyserNode.frequencyBinCount || Math.floor(fft / 2);
                        if (!this._vuBuf || this._vuBuf.length < binCount) {
                            this._vuBuf = new Uint8Array(binCount);
                        }
                        state.analyserNode.getByteFrequencyData(this._vuBuf);
                        const len = this._vuBuf.length;
                        const lowEnd = Math.floor(len * 0.33);
                        const midEnd = Math.floor(len * 0.66);
                        const highStart = Math.floor(len * 0.58);
                        out.low = this._sampleBandFromFftRange(this._vuBuf, 0, lowEnd, n);
                        out.mid = this._sampleBandFromFftRange(this._vuBuf, lowEnd, midEnd, n);
                        out.high = this._sampleBandFromFftRange(this._vuBuf, highStart, len, n);
                        out.fromAnalyser = true;
                        return out;
                    }
                } catch (_) {}
                const keys = ['low', 'mid', 'high'];
                const phase = { low: 0, mid: 1.15, high: 2.35 };
                for (let b = 0; b < 3; b++) {
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
                const maxRing = layerOpts.maxRing ?? 0.64;
                const floor = 0.04;
                const target = [];
                const sorted = smooth.slice().sort((a, b) => a - b);
                const p78 = sorted[Math.min(n - 1, Math.floor(n * 0.78))] || 0.001;
                const norm = 1 / Math.max(0.05, p78);
                const strength = this._getSpectrumAudioStrength();
                const gain = (bandKey === 'high' ? 0.88 : 0.76) * strength;
                const gamma = bandKey === 'high' ? 1.12 : 1.26;
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
                const attack = bandKey === 'high' ? 0.55 : 0.28;
                const release = bandKey === 'high' ? 0.14 : 0.12;
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
                const spectrumActive = this._isSpectrumAudioActive();
                let lowSmoothForHue = null;
                for (const layer of RadioVisualEngine.SPECTRUM_FLOWER_LAYERS) {
                    const levels = sampled[layer.key] || [];
                    const smooth = this._smoothSpectrumBandLevels(levels, sampled.fromAnalyser);
                    if (layer.key === 'low') lowSmoothForHue = smooth;
                    let radii;
                    if (layer.key === 'high' && !spectrumActive) {
                        radii = this._fullBandRadii(n, layer.maxRing);
                        if (!this._spectrumRingSmooth.high || this._spectrumRingSmooth.high.length !== n) {
                            this._spectrumRingSmooth.high = radii.slice();
                        } else {
                            for (let i = 0; i < n; i++) this._spectrumRingSmooth.high[i] = layer.maxRing;
                        }
                    } else {
                        radii = this._radiiFromSpectrumSmooth(smooth, t, layer.key, layer);
                    }
                    const bassLevel = layer.key === 'low' ? this._bassLevelFromSmooth(smooth) : 0;
                    layersL.push({
                        ...layer,
                        phaseBins: this._effectiveRibbonPhaseBins(layer.phaseBins, 'L', layer.key),
                        radii,
                        bassLevel
                    });
                    layersR.push({
                        ...layer,
                        phaseBins: this._effectiveRibbonPhaseBins(layer.phaseBins, 'R', layer.key),
                        radii: this._mirrorSpectrumRadii(radii),
                        bassLevel
                    });
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
                if (lowSmoothForHue) {
                    this._advanceOuterHuePhase(t, this._bassLevelFromSmooth(lowSmoothForHue));
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

            /** Bin boundary angle — outer ring vertices here close without a chord gap. */
            _spectrumEdgeAngle(ii, n, phaseBins = 0) {
                const phase = (phaseBins / n) * Math.PI * 2;
                return (ii / n) * Math.PI * 2 - Math.PI / 2 + phase;
            }

            _spectrumOuterNormAtEdge(edgeIi, radii, n, maxRing, petalFloor) {
                const prev = (edgeIi + n - 1) % n;
                const raw = ((radii[edgeIi] || 0) + (radii[prev] || 0)) / (2 * maxRing);
                return Math.min(1, Math.max(petalFloor, raw));
            }

            _fillDigitalSpectrumPetal(ctx, cx, cy, innerR, outerR, coreR, radii, n, layer, coreHue, t) {
                const li = layer.layerIndex;
                const layerCount = 3;
                const span = (outerR - innerR) / layerCount;
                /** All bands share the donut edge; each expands to its own outer guide ring. */
                const zoneInner = coreR;
                const zoneOuter = innerR + (li + 1) * span;
                const petalFloor = 0.1;
                const maxRing = layer.maxRing || 0.64;
                const hue = this._spectrumPetalBaseHue(layer, coreHue);
                const sat = layer.sat ?? 88;
                const lit = layer.light ?? 54;
                const drift = layer.hueDrift ?? 40;
                ctx.beginPath();
                for (let i = 0; i <= n; i++) {
                    const edge = i % n;
                    const a = this._spectrumEdgeAngle(edge, n, layer.phaseBins);
                    const norm = this._spectrumOuterNormAtEdge(edge, radii, n, maxRing, petalFloor);
                    const r = zoneInner + (zoneOuter - zoneInner) * norm;
                    const x = cx + Math.cos(a) * r;
                    const y = cy + Math.sin(a) * r;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                const innerPad = coreR;
                for (let i = 0; i <= n; i++) {
                    const edge = i % n;
                    const a = this._spectrumEdgeAngle(edge, n, layer.phaseBins);
                    ctx.lineTo(cx + Math.cos(a) * innerPad, cy + Math.sin(a) * innerPad);
                }
                ctx.closePath();
                const canConic = typeof ctx.createConicGradient === 'function';
                const useOuterPalette = layer.key === 'low' && canConic;
                const useConic = useOuterPalette;
                const ribbonSeam = this._spectrumRibbonSeamAngle(n, layer.phaseBins);
                if (useConic) {
                    const rim = ctx.createConicGradient(ribbonSeam + Math.PI, cx, cy);
                    const steps = useOuterPalette ? 20 : (layer.key === 'high' ? 24 : 16);
                    const bass = layer.bassLevel ?? 0;
                    const phaseBase = this._outerHuePhase || 0;
                    const shimmerT = typeof t === 'number' ? t : performance.now() * 0.001;
                    const isHigh = layer.key === 'high';
                    const stopOpts = {
                        useOuterPalette, isHigh, hue, sat, lit, drift, layerSpec: layer, bass, phaseBase, shimmerT
                    };
                    const wrap = this._spectrumConicStop(0, stopOpts);
                    rim.addColorStop(0, wrap);
                    for (let k = 1; k < steps; k++) {
                        const u = k / steps;
                        rim.addColorStop(u, this._spectrumConicStop(u, stopOpts));
                    }
                    const endBlend = this._spectrumConicStop(1 - 1 / steps, stopOpts);
                    rim.addColorStop(1 - 1 / steps, endBlend);
                    rim.addColorStop(1, wrap);
                    ctx.fillStyle = rim;
                } else {
                    ctx.fillStyle = this._spectrumPetalRadialFill(ctx, cx, cy, zoneInner, zoneOuter, layer, coreHue, t);
                }
                ctx.fill();
            }

            _drawDigitalSpectrumFlower(canvas, layers, n, coreHue = 175, drawT) {
                if (!canvas || !layers || !layers.length) return;
                const ctx = canvas.getContext('2d');
                if (!ctx) return;
                const w = canvas.width;
                const h = canvas.height;
                if (w < 8 || h < 8) return;
                const cx = w * 0.5;
                const cy = h * 0.5;
                ctx.clearRect(0, 0, w, h);
                const opacityMult = this._getSpectrumOpacityMult();
                const innerR = Math.min(w, h) * 0.14;
                const outerR = Math.min(w, h) * 0.44;
                const coreR = innerR * 1.08;
                ctx.strokeStyle = `rgba(255, 255, 255, ${0.05 * opacityMult})`;
                ctx.lineWidth = 1;
                for (let ring = 1; ring <= 3; ring++) {
                    ctx.beginPath();
                    ctx.arc(cx, cy, innerR + ((outerR - innerR) * ring) / 3, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                const ordered = layers.slice().sort((a, b) => a.layerIndex - b.layerIndex);
                const tDraw = typeof drawT === 'number' ? drawT : performance.now() * 0.001;
                for (const layer of ordered) {
                    if (!layer.radii || !layer.radii.length) continue;
                    this._fillDigitalSpectrumPetal(ctx, cx, cy, innerR, outerR, coreR, layer.radii, n, layer, coreHue, tDraw);
                }
                const hue = ((Number(coreHue) || 0) % 360 + 360) % 360;
                ctx.beginPath();
                ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
                const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
                core.addColorStop(0, `hsla(${hue}, 88%, 62%, ${0.88 * opacityMult})`);
                core.addColorStop(0.45, `hsla(${hue}, 76%, 40%, ${0.78 * opacityMult})`);
                core.addColorStop(1, `hsla(${(hue + 28) % 360}, 68%, 24%, ${0.62 * opacityMult})`);
                ctx.fillStyle = core;
                ctx.fill();
            }

            _lcdMarqueeAnimTime(slotKey, text, animT) {
                if (!this._lcdMarqueeSlots) this._lcdMarqueeSlots = {};
                const key = String(slotKey || 'lcd');
                const line = String(text || '');
                let slot = this._lcdMarqueeSlots[key];
                if (!slot || slot.text !== line) {
                    slot = { text: line, t0: Number(animT) || 0 };
                    this._lcdMarqueeSlots[key] = slot;
                }
                return Math.max(0, (Number(animT) || 0) - slot.t0);
            }

            /** LCD line: fixed size; slow back-and-forth scroll when wider than the display. */
            _drawCanvasLcdScrollText(ctx, text, clipX, clipY, clipW, clipH, y, basePx, animT, opts = {}) {
                const weight = opts.weight ?? 700;
                const family = 'ui-monospace, Menlo, Consolas, monospace';
                const line = String(text || '').trim() || '—';
                const fontPx = Math.max(6, Number(basePx) || 12);
                const padX = 3;
                const innerW = Math.max(8, clipW - padX * 2);
                const marqueeT = this._lcdMarqueeAnimTime(opts.marqueeKey || 'lcd', line, animT);

                ctx.save();
                ctx.beginPath();
                ctx.rect(clipX, clipY, clipW, clipH);
                ctx.clip();
                if (opts.fillStyle) ctx.fillStyle = opts.fillStyle;
                if (opts.shadowColor) {
                    ctx.shadowColor = opts.shadowColor;
                    ctx.shadowBlur = opts.shadowBlur ?? 0;
                }
                ctx.font = `${weight} ${fontPx}px ${family}`;
                ctx.textBaseline = 'middle';
                const textW = ctx.measureText(line).width;

                if (textW <= innerW) {
                    ctx.textAlign = 'center';
                    ctx.fillText(line, clipX + clipW * 0.5, y);
                    ctx.restore();
                    return line;
                }

                const overflow = textW - innerW;
                const pxPerSec = 20;
                const pauseSec = 1.15;
                const travelSec = Math.max(2, overflow / pxPerSec);
                const cycle = pauseSec * 2 + travelSec * 2;
                const u = marqueeT % cycle;
                let scroll = 0;
                if (u < pauseSec) {
                    scroll = 0;
                } else if (u < pauseSec + travelSec) {
                    scroll = ((u - pauseSec) / travelSec) * overflow;
                } else if (u < pauseSec * 2 + travelSec) {
                    scroll = overflow;
                } else {
                    scroll = overflow - ((u - pauseSec * 2 - travelSec) / travelSec) * overflow;
                }

                ctx.textAlign = 'left';
                ctx.fillText(line, clipX + padX - scroll, y);
                ctx.restore();
                return line;
            }

            _drawDigitalCarDash(eqHeights, t, lcdPrimaryLine, lcdSecondaryLine) {
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
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = 'rgba(0, 255, 200, 0.55)';
                ctx.shadowColor = 'rgba(0, 255, 200, 0.35)';
                ctx.shadowBlur = 6;
                const lcdClipX = ix + lcdPad;
                const lcdClipW = iw - lcdPad * 2;
                const lcdLine = lcdPrimaryLine || 'DIGITAL TUNER';
                const lcdStationPx = Math.max(10, Math.min(w, h) * 0.10);
                this._drawCanvasLcdScrollText(
                    ctx,
                    lcdLine,
                    lcdClipX,
                    lcdY,
                    lcdClipW,
                    lcdH,
                    lcdY + lcdH * 0.38,
                    lcdStationPx,
                    t,
                    {
                        marqueeKey: 'digital-lcd-primary',
                        fillStyle: 'rgba(0, 255, 200, 0.55)',
                        shadowColor: 'rgba(0, 255, 200, 0.35)',
                        shadowBlur: 6
                    }
                );
                ctx.shadowBlur = 0;
                const lcdSub = lcdSecondaryLine || 'STEREO · EQ';
                const lcdSubIsStation = lcdSub !== 'STEREO · EQ';
                const lcdStatusPx = Math.max(8, Math.min(w, h) * 0.062);
                if (lcdSubIsStation) {
                    this._drawCanvasLcdScrollText(
                        ctx,
                        lcdSub,
                        lcdClipX,
                        lcdY,
                        lcdClipW,
                        lcdH,
                        lcdY + lcdH * 0.72,
                        lcdStationPx,
                        t,
                        {
                            weight: 600,
                            marqueeKey: 'digital-lcd-secondary',
                            fillStyle: 'rgba(0, 255, 200, 0.55)',
                            shadowColor: 'rgba(0, 255, 200, 0.35)',
                            shadowBlur: 6
                        }
                    );
                } else {
                    ctx.font = `600 ${lcdStatusPx}px ui-monospace, Menlo, Consolas, monospace`;
                    ctx.fillStyle = 'rgba(120, 255, 200, 0.35)';
                    ctx.fillText(lcdSub, ix + iw * 0.5, lcdY + lcdH * 0.72);
                }
                ctx.shadowBlur = 0;
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
                if (this._digitalSpectrumLayout === 'blank') return;
                const pack = this._computeDigitalSpectrumRadiiAndEq();
                this._syncDonutCoreHues();
                this._drawDigitalSpectrumFlower(cL, pack.layersL, pack.n, this._donutCoreHueA, pack.t);
                this._drawDigitalSpectrumFlower(cR, pack.layersR, pack.n, this._donutCoreHueB, pack.t);
                const hubMode = this._digitalHubMode || 'equaliser';
                const dashCanvas = this.els.digitalCarDashCanvas;
                if (hubMode === 'equaliser') {
                    try { this._syncDigitalCrossfadeLabels(); } catch (_) {}
                    this._drawDigitalCarDash(
                        pack.eqHeights,
                        pack.t,
                        this._digitalLcdPrimaryLine(),
                        this._digitalLcdSecondaryLine()
                    );
                } else if (dashCanvas) {
                    const ctx = dashCanvas.getContext('2d');
                    if (ctx && dashCanvas.width > 0 && dashCanvas.height > 0) {
                        ctx.clearRect(0, 0, dashCanvas.width, dashCanvas.height);
                    }
                }
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
                const prev = this._rvLastGlobalVolumeNorm;
                const changed = prev !== undefined && Math.abs(prev - v) > 0.0001;
                this._rvLastGlobalVolumeNorm = v;
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
                this._syncDigitalVolumeUi({
                    peekVolume: changed && (this._digitalToolbarCenterMode === 'volume'
                        || this._digitalToolbarCenterMode === 'automix')
                });
                this._syncVolumeMuteLed();
            }

            _wireGlobalVolumeSliderSync(sig) {
                const vs = document.getElementById('volume-slider');
                if (!vs) return;
                const sync = () => {
                    try { this._syncVolumeFromGlobal(); } catch (_) {}
                };
                vs.addEventListener('input', sync, sig);
                vs.addEventListener('change', sync, sig);
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
                const vs = document.getElementById('volume-slider');
                const prev = vs ? Number(vs.value) : 0.5;
                const changed = Math.abs(prev - v) > 0.0001;
                if (v > 0.001) this._volMuted = false;
                try {
                    if (typeof setVolume === 'function') setVolume(v);
                } catch (_) {}
                if (vs) vs.value = String(v);
                this._rvLastGlobalVolumeNorm = v;
                if (this.els.volAnalog) this.els.volAnalog.value = String(v);
                this._setKnobRotation(this.els.volKnob, (v * 270) - 135);
                this._syncDigitalVolumeUi({
                    peekVolume: changed && (this._digitalToolbarCenterMode === 'volume'
                        || this._digitalToolbarCenterMode === 'automix')
                });
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
                if (this.els.digitalClock) {
                    try {
                        this.els.digitalClock.textContent = this._formatDigitalClockReadout();
                    } catch (_) {}
                }
                const mode = this._digitalToolbarCenterMode;
                if (mode === 'clock' || (mode === 'volume' && !this._digitalToolbarVolumePeekActive)
                    || mode === 'remaining'
                    || (mode === 'automix' && !this._digitalToolbarVolumePeekActive)) {
                    this._syncDigitalToolbarCenterReadout();
                }
            }

            _isDigitalEqualiserLiveTapTarget(el) {
                if (!el || !el.closest) return false;
                if (el.closest('.radio-visual-digital-dash-xfade')) return false;
                if (el.closest('.radio-visual-digital-hub-panel')) return false;
                return !!el.closest(
                    '.radio-visual-digital-car-main, .radio-visual-digital-car-dash-canvas'
                );
            }

            _wireDigitalHubAiVideoInteractions(abortSignal) {
                const wrap = this.els.digitalHubAi;
                const videos = [this.els.digitalHubAiVideoA, this.els.digitalHubAiVideoB].filter(Boolean);
                if (!videos.length || !wrap || wrap.dataset.rvHubAiVideoWired === '1') return;
                const sig = abortSignal && abortSignal.signal ? abortSignal.signal : abortSignal;
                const opts = sig ? { signal: sig } : {};
                const capOpts = sig ? { signal: sig, capture: true } : { capture: true };
                videos.forEach((video) => {
                    try { video.removeAttribute('title'); } catch (_) {}
                });
                try { wrap.removeAttribute('title'); } catch (_) {}
                const onClick = (ev) => { this._onDigitalHubAiVideoClick(ev); };
                const onDbl = (ev) => {
                    if (this.skin !== 'digital') return;
                    if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return;
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    this._clearDigitalHubAiVideoTapTimer();
                    try { this._toggleDigitalHubAiVideoFullscreen(); } catch (_) {}
                };
                const syncLoad = () => {
                    try { this._syncDigitalHubAiVideoLoadingState(); } catch (_) {}
                };
                const onFinished = (ev) => {
                    try { this._onDigitalHubAiVideoFinished(ev.currentTarget); } catch (_) {}
                };
                const onTimeUpdate = (ev) => {
                    const video = ev.currentTarget;
                    if (!video || video !== this._digitalHubAiActiveVideo()) return;
                    if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return;
                    const dur = Number(video.duration);
                    if (!Number.isFinite(dur) || dur <= 0) return;
                    if (video.currentTime < dur - 0.35) return;
                    onFinished(ev);
                };
                wrap.addEventListener('click', onClick, capOpts);
                wrap.addEventListener('dblclick', onDbl, capOpts);
                videos.forEach((video) => {
                    video.addEventListener('click', onClick, capOpts);
                    video.addEventListener('dblclick', onDbl, capOpts);
                    video.addEventListener('loadstart', () => {
                        if (RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                            this._setDigitalHubAiVideoLoading(true);
                        }
                        syncLoad();
                    }, opts);
                    video.addEventListener('waiting', syncLoad, opts);
                    video.addEventListener('stalled', syncLoad, opts);
                    video.addEventListener('seeking', syncLoad, opts);
                    video.addEventListener('canplay', syncLoad, opts);
                    video.addEventListener('playing', () => this._setDigitalHubAiVideoLoading(false), opts);
                    video.addEventListener('ended', onFinished, opts);
                    video.addEventListener('timeupdate', onTimeUpdate, opts);
                    video.addEventListener('error', () => {
                        this._setDigitalHubAiVideoLoading(false);
                        if (RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                            try { this._onDigitalHubAiVideoFinished(video); } catch (_) {}
                        }
                    }, opts);
                });
                wrap.dataset.rvHubAiVideoWired = '1';
            }

            _tickDigitalHubAiVideo() {
                if (!RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) return;
                const video = this._digitalHubAiActiveVideo();
                if (!video || this._digitalHubAiAdvancing || this._digitalHubAiEndGuard) return;
                const dur = Number(video.duration);
                if (!Number.isFinite(dur) || dur <= 0) return;
                if (video.ended || video.currentTime >= dur - 0.12) {
                    this._onDigitalHubAiVideoFinished(video);
                }
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
                    if (this._digitalHubMode === 'equaliser' && this._isDigitalEqualiserLiveTapTarget(ev.target)) {
                        try { this._enterDigitalHubLiveMode(); } catch (_) {}
                        return;
                    }
                    if (this._digitalHubMode === 'live') {
                        if (this._isDigitalHubAiVideoTarget(ev.target)) {
                            this._onDigitalHubAiVideoClick(ev);
                        }
                        return;
                    }
                    if (this._digitalHubMode === 'video') {
                        if (this._isDigitalHubDeckVideoTarget(ev.target)) {
                            this._onDigitalHubDeckVideoClick(ev);
                        }
                        return;
                    }
                    if (this._digitalHubMode === 'ai') {
                        const t = ev.target;
                        if (t && typeof t.closest === 'function') {
                            if (t.closest('.radio-visual-digital-hub-ai-video')) {
                                this._onDigitalHubAiVideoClick(ev);
                                return;
                            }
                            if (t.closest('.radio-visual-digital-hub-ai')) {
                                this._onDigitalHubAiVideoClick(ev);
                                return;
                            }
                            if (t.closest('[data-rv-digital="ai"]')) return;
                        }
                        this._exitDigitalHubAiMode();
                        return;
                    }
                    if (this._isDigitalStageUiTarget(ev.target)) return;
                    clearTimeout(this._digitalStageClickTimer);
                    this._digitalStageClickTimer = setTimeout(() => {
                        this._digitalStageClickTimer = null;
                        if (this.digitalCenterMode === 'deckB') {
                            if ((this._digitalVideoToolbarStep || 0) === 2) this._resetDigitalVideoToolbarStep();
                            this._setDigitalCenterMode('spectrum');
                        }
                    }, 280);
                }, { signal: sig });
                digitalCenterEl.addEventListener('dblclick', (ev) => {
                    if (this.skin !== 'digital') return;
                    const t = ev.target;
                    if (t && typeof t.closest === 'function' && t.closest('.radio-visual-digital-hub-ai')) {
                        return;
                    }
                    if (t && typeof t.closest === 'function' && t.closest('.radio-visual-digital-hub-deck-video')) {
                        return;
                    }
                    if (this._isDigitalStageUiTarget(ev.target)) return;
                    try { ev.preventDefault(); } catch (_) {}
                    clearTimeout(this._digitalStageClickTimer);
                    this._digitalStageClickTimer = null;
                    if (this._isDigitalStagingProjectMTarget(ev.target)) {
                        this._toggleDigitalStagingMountFullscreen();
                        return;
                    }
                    try {
                        if (RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode) && ev.target && ev.target.closest) {
                            const hubVid = ev.target.closest('.radio-visual-digital-hub-deck-video-el, .radio-visual-digital-hub-deck-video');
                            if (hubVid) {
                                const vid = this.els.digitalHubDeckVideo;
                                const mount = this.els.digitalHubDeckVideoWrap;
                                if (vid && mount && typeof toggleVideoSurfaceFullscreen === 'function') {
                                    toggleVideoSurfaceFullscreen(vid, mount);
                                    return;
                                }
                            }
                        }
                        if (this._digitalStagingView === 'video' && ev.target && ev.target.closest) {
                            const stagingVid = ev.target.closest('.radio-visual-digital-staging-video');
                            const stagingMount = ev.target.closest('.radio-visual-digital-staging-mount');
                            if (stagingVid || stagingMount) {
                                const vid = this.els.digitalStagingVideoStack || this.els.digitalStagingVideo;
                                const mount = this._stagingContentMount();
                                if (vid && typeof toggleVideoSurfaceFullscreen === 'function') {
                                    toggleVideoSurfaceFullscreen(vid, mount || stagingMount);
                                    return;
                                }
                            }
                        }
                        if (this.digitalCenterMode === 'deckB' && ev.target && ev.target.closest) {
                            const deckVid = ev.target.closest('.radio-visual-digital-deck-b-video');
                            if (deckVid && typeof toggleVideoSurfaceFullscreen === 'function') {
                                toggleVideoSurfaceFullscreen(deckVid, deckVid.parentElement);
                                return;
                            }
                        }
                    } catch (_) {}
                    try {
                        const fs = globalThis.toggleFullscreen;
                        if (typeof fs === 'function') fs();
                    } catch (_) {}
                }, { signal: sig });
                const onStagingPmFs = () => {
                    if (RadioVisualEngine.digitalStagingViewSupportsMountFullscreen(this._digitalStagingView)) {
                        this._afterDigitalStagingMountFullscreen();
                    }
                };
                document.addEventListener('fullscreenchange', onStagingPmFs, sig);
                document.addEventListener('webkitfullscreenchange', onStagingPmFs, sig);
            }

            _applySkinUi() {
                const next = this.skin === 'digital' ? 'digital' : 'analogue';
                if (this.els.stageAnalog) {
                    this.els.stageAnalog.classList.toggle('is-active', next === 'analogue');
                }
                if (this.els.stageDigital) {
                    this.els.stageDigital.classList.toggle('is-active', next === 'digital');
                }
                if (this.els.btnSkinAnalog) {
                    this.els.btnSkinAnalog.classList.toggle('is-active', next === 'analogue');
                }
                if (this.els.btnSkinDigital) {
                    this.els.btnSkinDigital.classList.toggle('is-active', next === 'digital');
                }
                if (this.root) {
                    this.root.classList.toggle('radio-visual-root--digital-active', next === 'digital');
                }
                try { this.onResize(); } catch (_) {}
                if (next === 'digital') {
                    try { this._syncDigitalSpectrumLayout(); } catch (_) {}
                    if (this.digitalCenterMode === 'deckB') {
                        this._syncDigitalDeckBVideo();
                    }
                }
                try { globalThis.syncModeInfoHudHintForDigitalRadio?.(); } catch (_) {}
            }

            _setSkin(skin) {
                if (this._skinLocked) return;
                this.skin = (skin === 'digital') ? 'digital' : 'analogue';
                try { localStorage.setItem('radioVisual.skin.v1', this.skin); } catch (_) {}
                this._applySkinUi();
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
                const stagingByLabel = {
                    ProjectM: 'projectm',
                    'Audio:Bar': 'bars',
                    KARAOKE: 'karaoke'
                };
                const items = [
                    { label: 'Mixer', fn: () => { try { g.toggleMixPanel?.(); } catch (_) {} } },
                    { label: 'Avatar', fn: () => {
                        try {
                            if (typeof g.toggleWebmOverlay === 'function') g.toggleWebmOverlay();
                        } catch (_) {}
                    }},
                    { label: 'KARAOKE', fn: () => {
                        if (deckBInPanel) {
                            this.toggleDigitalStagingKaraoke();
                            return;
                        }
                        this._withDjDeck((dj) => {
                            if (typeof dj.toggleDeckBKaraokeEmbed === 'function') dj.toggleDeckBKaraokeEmbed();
                        });
                    }},
                    ...(deckBInPanel ? [
                        { label: 'Audio:Bar', fn: () => { this._toggleDigitalStagingFeature('bars'); } },
                        { label: 'ProjectM', fn: () => { this._toggleDigitalStagingFeature('projectm'); } },
                        { label: 'CHAT', fn: () => { try { g.toggleTextInPanel?.(); } catch (_) {} } }
                    ] : [
                        { label: 'Video', fn: () => {
                            this._withDjDeck((dj) => {
                                if (dj.deckBVizMode === 'video') { dj.tearDownDeckBViz(); dj.syncDeckBVisualButtons(); }
                                else dj.startDeckBVideoVisual();
                            });
                        }},
                        { label: 'ProjectM', fn: () => { this._loadVisualByName('ProjectM v2'); } },
                        { label: 'Audio:Bar', fn: () => { this._loadVisualByName('Audio Bars'); } }
                    ]),
                    ...(deckBInPanel ? [{
                        label: 'DECKS',
                        fn: () => { this._loadVisualByName('DJ Decks'); }
                    }] : []),
                    { label: 'Queue', fn: () => {
                        if (deckBInPanel) {
                            this._toggleDigitalLocalQueuePanel();
                            return;
                        }
                        this._withDjDeck((dj) => dj.toggleDeckBQueuePanel());
                    }},
                    { label: 'Stations', fn: () => {
                        if (deckBInPanel) {
                            this._toggleDigitalStationsPanel();
                            return;
                        }
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
                    if (deckBInPanel) {
                        this._appendRvButtonLabel(b, it.label);
                    } else {
                        b.textContent = it.label;
                    }
                    if (it.label === 'DECKS') {
                        b.title = 'Open DJ Decks visual';
                        b.setAttribute('aria-label', 'Open DJ Decks visual');
                    }
                    if (deckBInPanel && it.label === 'CHAT') {
                        b.title = 'Open or close CHAT panel';
                        b.setAttribute('aria-label', 'CHAT panel');
                    }
                    const stagingKind = stagingByLabel[it.label];
                    if (deckBInPanel && stagingKind) {
                        b.dataset.rvStaging = stagingKind;
                        b.title = stagingKind === 'karaoke'
                            ? 'Toggle Karaoke Nerds in staging area'
                            : `Toggle ${it.label} in staging area`;
                    }
                    if (deckBInPanel && it.label === 'Queue') {
                        b.dataset.rvLocalQueue = '1';
                        b.title = 'Local playlists · Deck A & B';
                    }
                    if (deckBInPanel && it.label === 'Stations') {
                        b.dataset.rvStations = '1';
                        b.title = 'Radio stations · Deck A & B';
                        b.setAttribute('aria-label', 'Radio stations');
                    }
                    this._bindAction(b, it.fn);
                    gridEl.appendChild(b);
                });
                if (deckBInPanel) {
                    this._syncDigitalStagingButtons();
                    this._syncDigitalLocalQueueButton();
                    this._syncDigitalStationsButton();
                    this._wireDigitalFeatureButtonLabelFit(gridEl);
                }
            }

            _appendRvButtonLabel(btn, text) {
                const labelEl = document.createElement('span');
                labelEl.className = 'radio-visual-btn-label';
                labelEl.textContent = text;
                btn.appendChild(labelEl);
                return labelEl;
            }

            _computeRvButtonLabelFitPx(btn, label, opts = {}) {
                const pad = opts.pad ?? 4;
                const maxCap = opts.maxCap ?? 12;
                const minPx = opts.minPx ?? 5;
                const heightFactor = opts.heightFactor ?? 0.68;
                const widthFactor = opts.widthFactor ?? 0.22;
                const fill = !!opts.fill;
                const maxW = Math.max(8, btn.clientWidth - pad);
                const maxH = Math.max(8, btn.clientHeight - pad);
                let size = Math.min(maxW * widthFactor, maxH * heightFactor, maxCap);
                size = Math.max(size, minPx);
                label.style.fontSize = `${size}px`;
                if (fill) {
                    size = maxCap;
                    label.style.fontSize = `${size}px`;
                    for (let guard = 0; guard < 120 && size > minPx; guard++) {
                        if (label.scrollWidth <= maxW && label.scrollHeight <= maxH) break;
                        size -= 0.25;
                        label.style.fontSize = `${size}px`;
                    }
                } else {
                    for (let guard = 0; guard < 80 && size > minPx; guard++) {
                        if (label.scrollWidth <= maxW && label.scrollHeight <= maxH) break;
                        size -= 0.25;
                        label.style.fontSize = `${size}px`;
                    }
                }
                return size;
            }

            _rvLabelFitOpts(kind) {
                if (kind === 'toolbar-vol') {
                    const blueScale = this._readRvThemeCssNumber('--rv-digital-btn-blue-font-scale', 1);
                    return { fill: true, maxCap: 13 * blueScale, heightFactor: 0.82, widthFactor: 0.5, minPx: 7 };
                }
                if (kind === 'feature') {
                    const purpleScale = this._readRvThemeCssNumber('--rv-digital-btn-purple-font-scale', 1);
                    return { fill: true, maxCap: 12 * purpleScale, heightFactor: 0.88, widthFactor: 0.42, minPx: 5 };
                }
                const blueScale = this._readRvThemeCssNumber('--rv-digital-btn-blue-font-scale', 1);
                return { fill: true, maxCap: 12 * blueScale, heightFactor: 0.88, widthFactor: 0.42, minPx: 5 };
            }

            _fitDigitalFeatureButtonLabels(gridEl) {
                if (!gridEl) return;
                const opts = this._rvLabelFitOpts('feature');
                gridEl.querySelectorAll('.radio-visual-btn').forEach((btn) => {
                    const label = btn.querySelector('.radio-visual-btn-label');
                    if (!label) return;
                    label.style.fontSize = '';
                    this._computeRvButtonLabelFitPx(btn, label, opts);
                });
            }

            _fitDigitalToolbarButtonLabels(toolbarEl) {
                if (!toolbarEl) return;
                const blueOpts = this._rvLabelFitOpts('toolbar-blue');
                const volOpts = this._rvLabelFitOpts('toolbar-vol');
                const fitBtn = (btn, opts) => {
                    const label = btn.querySelector('.radio-visual-btn-label');
                    if (!label) return;
                    label.style.fontSize = '';
                    if (btn.clientWidth < 10 || btn.clientHeight < 8) return;
                    this._computeRvButtonLabelFitPx(btn, label, opts);
                };
                const main = toolbarEl.querySelector('.radio-visual-digital-toolbar-main');
                if (main) {
                    main.querySelectorAll('.radio-visual-digital-toolbar-text-btn').forEach((btn) => {
                        fitBtn(btn, blueOpts);
                    });
                }
                toolbarEl.querySelectorAll('.radio-visual-digital-toolbar-vol .radio-visual-digital-step-btn').forEach((btn) => {
                    fitBtn(btn, volOpts);
                });
                this._fitDigitalToolbarCenterReadout(toolbarEl);
            }

            _wireRvButtonLabelFit(rootEl, fitFn) {
                if (!rootEl || !this.abortCtrl) return;
                let rafId = 0;
                const run = () => {
                    if (rafId) cancelAnimationFrame(rafId);
                    rafId = requestAnimationFrame(() => {
                        rafId = requestAnimationFrame(() => {
                            rafId = 0;
                            try { fitFn.call(this, rootEl); } catch (_) {}
                        });
                    });
                };
                run();
                if (typeof ResizeObserver === 'undefined') {
                    globalThis.addEventListener('resize', run, { signal: this.abortCtrl.signal });
                    return;
                }
                const ro = new ResizeObserver(run);
                ro.observe(rootEl);
                const main = rootEl.querySelector('.radio-visual-digital-toolbar-main');
                if (main) ro.observe(main);
                rootEl.querySelectorAll('.radio-visual-btn').forEach((btn) => ro.observe(btn));
                const volGroup = rootEl.querySelector('.radio-visual-digital-toolbar-vol');
                if (volGroup) ro.observe(volGroup);
                const centerSlot = rootEl.querySelector('.radio-visual-digital-center-slot');
                if (centerSlot) ro.observe(centerSlot);
                const panel = rootEl.closest('.radio-visual-digital-panel')
                    || rootEl.closest('.radio-visual-stage.radio-visual-skin--digital');
                if (panel) ro.observe(panel);
                this.abortCtrl.signal.addEventListener('abort', () => {
                    try { ro.disconnect(); } catch (_) {}
                    if (rafId) cancelAnimationFrame(rafId);
                }, { once: true });
            }

            _wireDigitalFeatureButtonLabelFit(gridEl) {
                this._wireRvButtonLabelFit(gridEl, this._fitDigitalFeatureButtonLabels);
            }

            _wireDigitalToolbarLabelFit(toolbarEl) {
                this._wireRvButtonLabelFit(toolbarEl, this._fitDigitalToolbarButtonLabels);
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
                try { this._cancelDigitalAiButtonIntroHint(); } catch (_) {}
                try { if (this.abortCtrl) this.abortCtrl.abort(); } catch (_) {}
                container.innerHTML = '';
                try { initAudio(); } catch (_) {}
                try { if (!state.isPlaying && typeof playRadio === 'function') playRadio(); } catch (_) {}
                this.abortCtrl = new AbortController();
                const sig = { signal: this.abortCtrl.signal };
                if (!this._skinLocked) {
                    try {
                        const stored = localStorage.getItem('radioVisual.skin.v1');
                        if (stored === 'digital' || stored === 'analogue') this.skin = stored;
                    } catch (_) {}
                }
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
                root.setAttribute('aria-label', this.name || 'Radio');
                container.appendChild(root);
                this.root = root;

                const showAnalogue = !this._skinLocked || this.skin === 'analogue';
                const showDigital = !this._skinLocked || this.skin === 'digital';
                if (showAnalogue) root.classList.add('radio-visual-root--analogue-only');
                if (showDigital) {
                    root.classList.add('radio-visual-root--digital-only');
                    root.classList.add('radio-visual-root--digital-active');
                }

                let btnA = null;
                let btnD = null;
                if (!this._skinLocked) {
                    const skinToggle = document.createElement('div');
                    skinToggle.className = 'radio-visual-skin-toggle';
                    btnA = document.createElement('button');
                    btnA.type = 'button';
                    btnA.className = 'radio-visual-skin-btn';
                    btnA.dataset.skin = 'analogue';
                    btnA.textContent = 'Analogue';
                    btnD = document.createElement('button');
                    btnD.type = 'button';
                    btnD.className = 'radio-visual-skin-btn';
                    btnD.dataset.skin = 'digital';
                    btnD.textContent = 'Digital';
                    skinToggle.appendChild(btnA);
                    skinToggle.appendChild(btnD);
                    root.appendChild(skinToggle);
                }

                let stageA = null;
                let volKnob = null;
                let deckAKnob = null;
                let deckBKnob = null;
                let crossKnob = null;
                let autoFadeKnob = null;
                let autoMixKnob = null;
                let autoFadeReadout = null;
                let autoMixReadout = null;
                let tunerRail = null;
                let analogBtns = null;
                let ticks = null;
                let glow = null;
                let glowB = null;
                let needle = null;
                let needleB = null;
                let vuCanvas = null;
                let dClk = null;
                let digitalAutoMixTimer = null;
                let digBtns = null;
                let digitalCenter = null;
                let crossDig = null;
                let btnDigitalSpectrum = null;
                let btnDigitalVideo = null;
                let btnVis = null;
                let btnXfadeStation = null;
                let spectrumBg = null;
                let dashStack = null;
                let dashXfade = null;
                let digitalCenterSpectrum = null;
                let digitalCenterDeckB = null;
                let spectrumSideL = null;
                let spectrumSideR = null;
                let digitalSpectrumCanvasL = null;
                let digitalSpectrumCanvasR = null;
                let digitalCarDashCanvas = null;
                let digitalCarDisplay = null;
                let digitalHubPanel = null;
                let spectrumStagingSlider = null;
                let spectrumScaleSlider = null;
                let spectrumPanSlider = null;
                let digitalSpectrumRow = null;
                let digitalDeckBVideo = null;
                let digitalDeckBMount = null;
                let digitalDeckBContent = null;
                let volDigitalReadout = null;
                let volReadoutLabel = null;
                let volGroup = null;
                let digitalToolbarCenterSlot = null;
                let digitalToolbarCenterXfadeWrap = null;
                let digitalToolbarCenterXfadeCycleHit = null;
                let digitalToolbarCenterXfade = null;
                let volDown = null;
                let volUp = null;
                let digitalToolbar = null;
                let btnDigitalAi = null;
                let btnDigitalFade = null;
                let digitalStagingMount = null;
                let digitalLocalQueuePanel = null;
                let digitalStationsPanel = null;
                let digitalAutoMixPanel = null;
                let digitalAutoMixSlider = null;
                let digitalAutoMixReadout = null;
                let digitalAutoFadePanel = null;
                let digitalAutoFadeSlider = null;
                let digitalAutoFadeReadout = null;
                if (showAnalogue) {
                stageA = document.createElement('section');
                stageA.className = 'radio-visual-stage radio-visual-skin--analogue is-active';
                stageA.setAttribute('aria-label', 'Analogue radio');
                const tunerShell = document.createElement('div');
                tunerShell.className = 'radio-visual-tuner-shell';
                tunerRail = document.createElement('div');
                tunerRail.className = 'radio-visual-tuner-rail';
                tunerRail.id = 'radio-visual-tuner-rail';
                ticks = document.createElement('div');
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
                glow = document.createElement('div');
                glow.className = 'radio-visual-tuner-glow';
                glow.id = 'radio-visual-tuner-glow';
                glowB = document.createElement('div');
                glowB.className = 'radio-visual-tuner-glow radio-visual-tuner-glow--deck-b';
                glowB.id = 'radio-visual-tuner-glow-b';
                needle = document.createElement('div');
                needle.className = 'radio-visual-tuner-needle radio-visual-tuner-needle--deck-a';
                needle.id = 'radio-visual-needle';
                needleB = document.createElement('div');
                needleB.className = 'radio-visual-tuner-needle radio-visual-tuner-needle--deck-b';
                needleB.id = 'radio-visual-needle-b';
                const vuWrap = document.createElement('div');
                vuWrap.className = 'radio-visual-vu-wrap';
                vuCanvas = document.createElement('canvas');
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
                volKnob = mkControlKnob('radio-visual-vol-knob', 'Volume');
                volKnob.setAttribute('role', 'slider');
                volKnob.classList.add('radio-visual-knob--switch', 'radio-visual-knob--vol-mute');
                volKnob.setAttribute('aria-label', 'Volume; drag to adjust, tap to mute or unmute');
                deckAKnob = mkControlKnob('radio-visual-deck-a-knob', 'Deck A');
                deckAKnob.classList.add('radio-visual-knob--deck-a');
                deckBKnob = mkControlKnob('radio-visual-deck-b-knob', 'Deck B');
                deckBKnob.classList.add('radio-visual-knob--deck-b');
                crossKnob = mkControlKnob('radio-visual-cross-knob', 'Cross-fade between decks');
                autoFadeKnob = mkControlKnob('radio-visual-autofade-knob', 'Auto-fade');
                autoMixKnob = mkControlKnob('radio-visual-automix-knob', 'Auto-mix max interval; click to toggle');
                autoMixKnob.setAttribute('role', 'slider');
                autoFadeReadout = mkReadout(`${(this._readAutoFadeDurationMs() / 1000).toFixed(1)}s`);
                autoMixReadout = mkReadout(String(this._readAutoMixMaxMin()));
                knobs.appendChild(this._mkKnobBlock('Volume', volKnob));
                knobs.appendChild(this._mkKnobBlock('Deck A', deckAKnob));
                knobs.appendChild(this._mkKnobBlock('Deck B', deckBKnob));
                knobs.appendChild(this._mkKnobBlock('Crossfade', crossKnob));
                knobs.appendChild(this._mkKnobBlock('Auto-Fade', autoFadeKnob, autoFadeReadout, true));
                knobs.appendChild(this._mkKnobBlock('Auto-Mix', autoMixKnob, autoMixReadout, true));
                analogBtns = document.createElement('div');
                analogBtns.className = 'radio-visual-analog-actions';
                analogBtns.id = 'radio-visual-analog-btns';
                stageA.appendChild(knobs);
                stageA.appendChild(tunerShell);
                stageA.appendChild(analogBtns);
                }

                let stageD = null;
                if (showDigital) {
                stageD = document.createElement('section');
                stageD.className = 'radio-visual-stage radio-visual-skin--digital is-active';
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
                const clockRow = document.createElement('div');
                clockRow.className = 'radio-visual-digital-clock-row';
                dClk = mkLine('radio-visual-digital-line--clock', 'radio-visual-digital-clock', '—');
                digitalAutoMixTimer = document.createElement('span');
                digitalAutoMixTimer.className = 'radio-visual-digital-automix-timer display-none';
                digitalAutoMixTimer.id = 'radio-visual-digital-automix-timer';
                digitalAutoMixTimer.setAttribute('aria-live', 'polite');
                digitalAutoMixTimer.setAttribute('aria-hidden', 'true');
                clockRow.appendChild(dClk);
                clockRow.appendChild(digitalAutoMixTimer);
                digitalAutoMixPanel = document.createElement('div');
                digitalAutoMixPanel.className = 'radio-visual-digital-automix-panel';
                digitalAutoMixPanel.id = 'radio-visual-digital-automix-panel';
                digitalAutoMixPanel.setAttribute('aria-hidden', 'true');
                const autoMixPanelTitle = document.createElement('div');
                autoMixPanelTitle.className = 'radio-visual-digital-automix-title';
                autoMixPanelTitle.textContent = 'Auto-mix max interval';
                const autoMixPanelRow = document.createElement('div');
                autoMixPanelRow.className = 'radio-visual-digital-automix-row';
                digitalAutoMixSlider = document.createElement('input');
                digitalAutoMixSlider.type = 'range';
                digitalAutoMixSlider.className = 'radio-visual-digital-automix-range';
                digitalAutoMixSlider.min = String(RadioVisualEngine.AUTOMIX_MIN_MIN);
                digitalAutoMixSlider.max = String(RadioVisualEngine.AUTOMIX_MAX_MIN);
                digitalAutoMixSlider.step = '1';
                digitalAutoMixSlider.value = String(this._readAutoMixMaxMin());
                digitalAutoMixSlider.setAttribute('aria-label', 'Auto-mix maximum interval in minutes');
                digitalAutoMixReadout = document.createElement('span');
                digitalAutoMixReadout.className = 'radio-visual-digital-automix-readout';
                digitalAutoMixReadout.textContent = `${this._readAutoMixMaxMin()}m`;
                autoMixPanelRow.appendChild(digitalAutoMixSlider);
                autoMixPanelRow.appendChild(digitalAutoMixReadout);
                digitalAutoMixPanel.appendChild(autoMixPanelTitle);
                digitalAutoMixPanel.appendChild(autoMixPanelRow);
                digitalAutoFadePanel = document.createElement('div');
                digitalAutoFadePanel.className = 'radio-visual-digital-autofade-panel';
                digitalAutoFadePanel.id = 'radio-visual-digital-autofade-panel';
                digitalAutoFadePanel.setAttribute('aria-hidden', 'true');
                const autoFadePanelTitle = document.createElement('div');
                autoFadePanelTitle.className = 'radio-visual-digital-autofade-title';
                autoFadePanelTitle.textContent = 'Auto-fade duration';
                const autoFadePanelRow = document.createElement('div');
                autoFadePanelRow.className = 'radio-visual-digital-autofade-row';
                digitalAutoFadeSlider = document.createElement('input');
                digitalAutoFadeSlider.type = 'range';
                digitalAutoFadeSlider.className = 'radio-visual-digital-autofade-range';
                digitalAutoFadeSlider.min = String(RadioVisualEngine.AUTOFADE_MIN_MS / 1000);
                digitalAutoFadeSlider.max = String(RadioVisualEngine.AUTOFADE_MAX_MS / 1000);
                digitalAutoFadeSlider.step = '0.5';
                digitalAutoFadeSlider.value = String(this._readAutoFadeDurationMs() / 1000);
                digitalAutoFadeSlider.setAttribute('aria-label', 'Auto-fade duration in seconds');
                digitalAutoFadeReadout = document.createElement('span');
                digitalAutoFadeReadout.className = 'radio-visual-digital-autofade-readout';
                digitalAutoFadeReadout.textContent = `${(this._readAutoFadeDurationMs() / 1000).toFixed(1)}s`;
                autoFadePanelRow.appendChild(digitalAutoFadeSlider);
                autoFadePanelRow.appendChild(digitalAutoFadeReadout);
                digitalAutoFadePanel.appendChild(autoFadePanelTitle);
                digitalAutoFadePanel.appendChild(autoFadePanelRow);
                digBtns = document.createElement('div');
                digBtns.className = 'radio-visual-btn-grid radio-visual-digital-feature-btns';
                digBtns.id = 'radio-visual-digital-btns';
                digitalCenter = document.createElement('div');
                digitalCenter.className = 'radio-visual-digital-center';
                digitalCenterSpectrum = document.createElement('div');
                digitalCenterSpectrum.className = 'radio-visual-digital-center-pane is-active is-hub-equaliser';
                digitalCenterSpectrum.dataset.hubMode = 'equaliser';
                spectrumBg = document.createElement('div');
                spectrumBg.className = 'radio-visual-digital-spectrum-bg';
                spectrumBg.setAttribute('aria-hidden', 'true');
                digitalSpectrumRow = document.createElement('div');
                digitalSpectrumRow.className = 'radio-visual-digital-spectrum-row';
                spectrumSideL = document.createElement('div');
                spectrumSideL.className = 'radio-visual-digital-spectrum-side radio-visual-digital-spectrum-side--left';
                digitalSpectrumCanvasL = document.createElement('canvas');
                digitalSpectrumCanvasL.className = 'radio-visual-digital-spectrum-canvas';
                digitalSpectrumCanvasL.id = 'radio-visual-digital-spectrum-l';
                spectrumSideL.appendChild(digitalSpectrumCanvasL);
                dashStack = document.createElement('div');
                dashStack.className = 'radio-visual-digital-dash-stack';
                dashStack.setAttribute('aria-label', 'Digital radio dash');
                const centerInfo = document.createElement('div');
                centerInfo.className = 'radio-visual-digital-center-info';
                centerInfo.setAttribute('aria-live', 'polite');
                centerInfo.appendChild(clockRow);
                digitalCarDisplay = document.createElement('div');
                digitalCarDisplay.className = 'radio-visual-digital-car-display';
                digitalCarDashCanvas = document.createElement('canvas');
                digitalCarDashCanvas.className = 'radio-visual-digital-car-dash-canvas';
                digitalCarDashCanvas.id = 'radio-visual-digital-car-dash';
                digitalHubPanel = this._buildDigitalHubPanel();
                dashXfade = document.createElement('div');
                dashXfade.className = 'radio-visual-digital-dash-xfade';
                const xfLblA = document.createElement('span');
                xfLblA.className = 'radio-visual-digital-dash-xfade-end radio-visual-digital-dash-xfade-end--a';
                xfLblA.textContent = 'A';
                const dashXfadeWrap = document.createElement('span');
                dashXfadeWrap.className = 'radio-visual-digital-dash-xfade-wrap';
                crossDig = document.createElement('input');
                crossDig.type = 'range';
                crossDig.className = 'radio-visual-digital-dash-xfade-range';
                crossDig.id = 'radio-visual-cross-digital';
                crossDig.min = '0';
                crossDig.max = '1';
                crossDig.step = '0.01';
                const crossInit = String(this._getCrossfadeX());
                crossDig.value = crossInit;
                try { dashXfadeWrap.style.setProperty('--cross-x', crossInit); } catch (_) {}
                crossDig.setAttribute('aria-label', 'Crossfade between deck A and deck B');
                const xfLblB = document.createElement('span');
                xfLblB.className = 'radio-visual-digital-dash-xfade-end radio-visual-digital-dash-xfade-end--b';
                xfLblB.textContent = 'B';
                dashXfadeWrap.appendChild(crossDig);
                dashXfade.appendChild(xfLblA);
                dashXfade.appendChild(dashXfadeWrap);
                dashXfade.appendChild(xfLblB);
                const carMain = document.createElement('div');
                carMain.className = 'radio-visual-digital-car-main';
                carMain.appendChild(digitalCarDashCanvas);
                carMain.appendChild(digitalHubPanel);
                digitalCarDisplay.appendChild(carMain);
                digitalCarDisplay.appendChild(dashXfade);
                dashStack.appendChild(centerInfo);
                dashStack.appendChild(digitalCarDisplay);
                spectrumSideR = document.createElement('div');
                spectrumSideR.className = 'radio-visual-digital-spectrum-side radio-visual-digital-spectrum-side--right';
                digitalSpectrumCanvasR = document.createElement('canvas');
                digitalSpectrumCanvasR.className = 'radio-visual-digital-spectrum-canvas';
                digitalSpectrumCanvasR.id = 'radio-visual-digital-spectrum-r';
                spectrumSideR.appendChild(digitalSpectrumCanvasR);
                spectrumStagingSlider = document.createElement('div');
                spectrumStagingSlider.className = 'radio-visual-digital-spectrum-staging-slider';
                spectrumStagingSlider.setAttribute('aria-hidden', 'true');
                spectrumScaleSlider = document.createElement('input');
                spectrumScaleSlider.type = 'range';
                spectrumScaleSlider.className = 'radio-visual-digital-spectrum-scale-range radio-visual-digital-range';
                spectrumScaleSlider.min = '35';
                spectrumScaleSlider.max = '280';
                spectrumScaleSlider.step = '1';
                spectrumScaleSlider.value = '100';
                spectrumScaleSlider.setAttribute('aria-label', 'Spectrum flower size');
                spectrumPanSlider = document.createElement('input');
                spectrumPanSlider.type = 'range';
                spectrumPanSlider.className = 'radio-visual-digital-spectrum-pan-range radio-visual-digital-range';
                spectrumPanSlider.min = '0';
                spectrumPanSlider.max = '100';
                spectrumPanSlider.step = '1';
                spectrumPanSlider.value = '0';
                spectrumPanSlider.setAttribute('aria-label', 'Spread spectrum flowers apart');
                const spectrumScaleWrap = document.createElement('div');
                spectrumScaleWrap.className = 'radio-visual-digital-spectrum-staging-scale-wrap';
                const spectrumPanWrap = document.createElement('div');
                spectrumPanWrap.className = 'radio-visual-digital-spectrum-staging-pan-wrap';
                spectrumScaleWrap.appendChild(spectrumScaleSlider);
                spectrumPanWrap.appendChild(spectrumPanSlider);
                spectrumStagingSlider.appendChild(spectrumScaleWrap);
                spectrumStagingSlider.appendChild(spectrumPanWrap);
                digitalSpectrumRow.appendChild(spectrumSideL);
                digitalSpectrumRow.appendChild(dashStack);
                digitalSpectrumRow.appendChild(spectrumSideR);
                digitalSpectrumRow.appendChild(spectrumStagingSlider);
                digitalStagingMount = document.createElement('div');
                digitalStagingMount.className = 'radio-visual-digital-staging-mount';
                digitalStagingMount.setAttribute('aria-hidden', 'true');
                digitalLocalQueuePanel = this._buildDigitalLocalQueuePanel();
                digitalStationsPanel = this._buildDigitalStationsPanel();
                digitalCenterSpectrum.appendChild(spectrumBg);
                digitalCenterSpectrum.appendChild(digitalSpectrumRow);
                digitalCenterSpectrum.appendChild(digitalStagingMount);
                digitalCenterSpectrum.appendChild(digitalLocalQueuePanel);
                digitalCenterSpectrum.appendChild(digitalStationsPanel);
                digitalCenterDeckB = document.createElement('div');
                digitalCenterDeckB.className = 'radio-visual-digital-center-pane';
                digitalDeckBMount = document.createElement('div');
                digitalDeckBMount.className = 'radio-visual-digital-deck-b-mount';
                digitalDeckBContent = document.createElement('div');
                digitalDeckBContent.className = 'radio-visual-digital-deck-b-content';
                digitalDeckBVideo = document.createElement('video');
                digitalDeckBVideo.className = 'radio-visual-digital-deck-b-video';
                digitalDeckBVideo.playsInline = true;
                digitalDeckBVideo.muted = true;
                digitalDeckBMount.appendChild(digitalDeckBContent);
                digitalDeckBMount.appendChild(digitalDeckBVideo);
                try { this._wireDigitalDeckBVideoFullscreen(digitalDeckBVideo, digitalDeckBMount, sig); } catch (_) {}
                digitalCenterDeckB.appendChild(digitalDeckBMount);
                digitalCenter.appendChild(digitalCenterSpectrum);
                digitalCenter.appendChild(digitalCenterDeckB);
                digitalToolbar = document.createElement('div');
                digitalToolbar.className = 'radio-visual-digital-toolbar';
                digitalToolbar.id = 'radio-visual-digital-toolbar';
                const rvToolbarTextBtnClass = 'radio-visual-btn radio-visual-digital-toolbar-text-btn';
                btnDigitalSpectrum = document.createElement('button');
                btnDigitalSpectrum.type = 'button';
                btnDigitalSpectrum.className = rvToolbarTextBtnClass;
                this._appendRvButtonLabel(btnDigitalSpectrum, 'EQUALISER');
                btnDigitalVideo = document.createElement('button');
                btnDigitalVideo.type = 'button';
                btnDigitalVideo.className = rvToolbarTextBtnClass;
                btnDigitalVideo.setAttribute('aria-label', 'Show playing video in equaliser');
                this._appendRvButtonLabel(btnDigitalVideo, 'VIDEO');
                btnVis = document.createElement('button');
                btnVis.type = 'button';
                btnVis.className = 'radio-visual-btn radio-visual-digital-toolbar-icon-btn radio-visual-digital-vis-btn';
                btnVis.textContent = ' 🔆 ';
                btnVis.setAttribute('aria-label', 'Background visual');
                btnVis.setAttribute('aria-pressed', 'false');
                volGroup = document.createElement('div');
                volGroup.className = 'radio-visual-digital-toolbar-vol radio-visual-digital-toolbar-center';
                volGroup.dataset.centerMode = 'clock';
                volGroup.setAttribute('aria-label', 'Centre display');
                volDown = document.createElement('button');
                volDown.type = 'button';
                volDown.className = 'radio-visual-btn radio-visual-digital-step-btn';
                this._appendRvButtonLabel(volDown, '−');
                volDown.setAttribute('aria-label', 'Volume down');
                volDigitalReadout = document.createElement('button');
                volDigitalReadout.type = 'button';
                volDigitalReadout.className = 'radio-visual-digital-vol-readout radio-visual-digital-center-readout';
                volDigitalReadout.id = 'radio-visual-vol-readout';
                volDigitalReadout.setAttribute('aria-label', 'Date and time — tap to change centre display');
                volReadoutLabel = document.createElement('span');
                volReadoutLabel.className = 'radio-visual-center-readout-text';
                volReadoutLabel.textContent = '—';
                volDigitalReadout.appendChild(volReadoutLabel);
                digitalToolbarCenterSlot = document.createElement('div');
                digitalToolbarCenterSlot.className = 'radio-visual-digital-center-slot';
                digitalToolbarCenterXfadeWrap = document.createElement('div');
                digitalToolbarCenterXfadeWrap.className = 'radio-visual-digital-toolbar-xfade-wrap';
                digitalToolbarCenterXfadeCycleHit = document.createElement('button');
                digitalToolbarCenterXfadeCycleHit.type = 'button';
                digitalToolbarCenterXfadeCycleHit.className = 'radio-visual-digital-toolbar-xfade-cycle-hit';
                digitalToolbarCenterXfadeCycleHit.setAttribute('aria-label', 'Tap to change centre display');
                digitalToolbarCenterXfade = document.createElement('input');
                digitalToolbarCenterXfade.type = 'range';
                digitalToolbarCenterXfade.className = 'radio-visual-digital-toolbar-xfade-range';
                digitalToolbarCenterXfade.id = 'radio-visual-toolbar-xfade';
                digitalToolbarCenterXfade.min = '0';
                digitalToolbarCenterXfade.max = '1';
                digitalToolbarCenterXfade.step = '0.01';
                digitalToolbarCenterXfade.value = String(this._getCrossfadeX());
                digitalToolbarCenterXfade.setAttribute('aria-label', 'Crossfade between deck A and deck B');
                try {
                    digitalToolbarCenterXfadeWrap.style.setProperty('--cross-x', digitalToolbarCenterXfade.value);
                } catch (_) {}
                digitalToolbarCenterXfadeWrap.appendChild(digitalToolbarCenterXfadeCycleHit);
                digitalToolbarCenterXfadeWrap.appendChild(digitalToolbarCenterXfade);
                digitalToolbarCenterSlot.appendChild(volDigitalReadout);
                digitalToolbarCenterSlot.appendChild(digitalToolbarCenterXfadeWrap);
                volUp = document.createElement('button');
                volUp.type = 'button';
                volUp.className = 'radio-visual-btn radio-visual-digital-step-btn';
                this._appendRvButtonLabel(volUp, '+');
                volUp.setAttribute('aria-label', 'Volume up');
                volGroup.appendChild(volDown);
                volGroup.appendChild(digitalToolbarCenterSlot);
                volGroup.appendChild(volUp);
                const toolbarMain = document.createElement('div');
                toolbarMain.className = 'radio-visual-digital-toolbar-main';
                toolbarMain.setAttribute('role', 'group');
                toolbarMain.setAttribute('aria-label', 'Deck controls');
                const mkRvDigitalBtn = (act, lab) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = rvToolbarTextBtnClass;
                    b.dataset.rvDigital = act;
                    this._appendRvButtonLabel(b, lab);
                    if (act === 'ai') {
                        btnDigitalAi = b;
                        b.title = 'Tap: toggle auto-mix + AI video · Hold: max interval';
                    }
                    if (act === 'fade') {
                        btnDigitalFade = b;
                        b.title = 'Tap: auto-fade · Hold: fade duration';
                    }
                    return b;
                };
                const mkRvStationBtn = (lab, deck) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = rvToolbarTextBtnClass;
                    b.dataset.rvDeckTransport = deck;
                    this._appendRvButtonLabel(b, lab);
                    return b;
                };
                const btnDeckATransport = mkRvStationBtn('A >', 'a');
                const btnAi = mkRvDigitalBtn('ai', 'Ai');
                const btnFade = mkRvDigitalBtn('fade', 'Fade');
                const btnDeckBTransport = mkRvStationBtn('B >', 'b');
                btnXfadeStation = document.createElement('button');
                btnXfadeStation.type = 'button';
                btnXfadeStation.className = 'radio-visual-btn radio-visual-digital-toolbar-icon-btn radio-visual-digital-xfade-station-btn';
                btnXfadeStation.dataset.rvDigital = 'xfade-station';
                btnXfadeStation.textContent = ' 🔀 ';
                btnXfadeStation.title = 'Change station when auto-fading (toggle)';
                btnXfadeStation.setAttribute('aria-label', 'Change station when auto-fading');
                toolbarMain.appendChild(btnDigitalSpectrum);
                toolbarMain.appendChild(btnAi);
                toolbarMain.appendChild(btnDeckATransport);
                toolbarMain.appendChild(volGroup);
                toolbarMain.appendChild(btnDeckBTransport);
                toolbarMain.appendChild(btnFade);
                toolbarMain.appendChild(btnDigitalVideo);
                digitalToolbar.appendChild(btnVis);
                digitalToolbar.appendChild(toolbarMain);
                digitalToolbar.appendChild(btnXfadeStation);
                dPanel.appendChild(digitalCenter);
                dPanel.appendChild(digitalToolbar);
                dPanel.appendChild(digitalAutoMixPanel);
                dPanel.appendChild(digitalAutoFadePanel);
                dPanel.appendChild(digBtns);
                stageD.appendChild(dPanel);
                }

                try {
                    if (typeof globalThis.reapplyDigitalThemeImageLayers === 'function') {
                        globalThis.reapplyDigitalThemeImageLayers();
                    }
                } catch (_) {}

                if (stageA) root.appendChild(stageA);
                if (stageD) root.appendChild(stageD);

                this.els = {
                    btnSkinAnalog: btnA,
                    btnSkinDigital: btnD,
                    stageAnalog: stageA,
                    stageDigital: stageD,
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
                    btnDigitalVideo,
                    btnVis,
                    btnDigitalAi,
                    btnDigitalFade,
                    btnXfadeStation,
                    digitalStagingMount,
                    digitalAutoMixPanel,
                    digitalAutoMixSlider,
                    digitalAutoMixReadout,
                    digitalAutoFadePanel,
                    digitalAutoFadeSlider,
                    digitalAutoFadeReadout,
                    spectrumBg,
                    digitalDashStack: dashStack,
                    digitalCenterSpectrum,
                    digitalLocalQueuePanel,
                    digitalStationsPanel,
                    digitalCenterDeckB,
                    spectrumSideL,
                    spectrumSideR,
                    digitalSpectrumCanvasL,
                    digitalSpectrumCanvasR,
                    digitalCarDashCanvas,
                    digitalHubPanel,
                    digitalHubEffects: digitalHubPanel?.querySelector('.radio-visual-digital-hub-effects'),
                    digitalHubMixFx: digitalHubPanel?.querySelector('.radio-visual-digital-hub-fx-center'),
                    digitalSpectrumRow,
                    digitalSpectrumStagingSlider: spectrumStagingSlider,
                    digitalSpectrumScaleSlider: spectrumScaleSlider,
                    digitalSpectrumPanSlider: spectrumPanSlider,
                    digitalHubAiVideoA: digitalHubPanel?.querySelector('.radio-visual-digital-hub-ai-video--a'),
                    digitalHubAiVideoB: digitalHubPanel?.querySelector('.radio-visual-digital-hub-ai-video--b'),
                    digitalHubAiVideo: digitalHubPanel?.querySelector('.radio-visual-digital-hub-ai-video--a'),
                    digitalHubAi: digitalHubPanel?.querySelector('.radio-visual-digital-hub-ai'),
                    digitalHubAiLoader: digitalHubPanel?.querySelector('.radio-visual-digital-hub-ai-loader'),
                    digitalHubDeckVideoWrap: digitalHubPanel?.querySelector('.radio-visual-digital-hub-deck-video'),
                    digitalHubDeckVideo: digitalHubPanel?.querySelector('.radio-visual-digital-hub-deck-video-el'),
                    digitalCarDisplay,
                    digitalDeckBVideo,
                    digitalDashXfade: dashXfade,
                    volDigitalReadout,
                    volDown,
                    volUp,
                    digitalToolbarCenterSlot,
                    digitalToolbarCenterXfadeWrap,
                    digitalToolbarCenterXfadeCycleHit,
                    digitalToolbarCenterXfade,
                    digitalToolbarCenterReadoutLabel: volReadoutLabel,
                    digitalToolbarCenter: volGroup,
                    digitalToolbar: digitalToolbar,
                    digitalClock: dClk,
                    digitalAutoMixTimer,
                    digitalCenter,
                    analogBtns,
                    digitalBtns: digBtns
                };

                if (analogBtns) this._buildFeatureButtons(analogBtns);
                if (digBtns) this._buildFeatureButtons(digBtns, { deckBInPanel: true });
                if (digitalToolbar) this._wireDigitalToolbarLabelFit(digitalToolbar);
                if (digitalCenter) this._bindDigitalStageInteractions(digitalCenter);
                try { this._wireDigitalHubAiVideoInteractions(this.abortCtrl.signal); } catch (_) {}
                try { this._wireDigitalHubDeckVideoInteractions(this.abortCtrl.signal); } catch (_) {}
                try { this._applySkinUi(); } catch (_) {}
                try { this._syncVolumeFromGlobal(); } catch (_) {}
                try { this._syncDigitalToolbarCenterMode(); } catch (_) {}
                try { this._wireGlobalVolumeSliderSync(sig); } catch (_) {}
                if (showAnalogue) {
                    try { this._setAutoFadeDurationNorm(this._autoFadeDurationNorm()); } catch (_) {}
                    try { this._setAutoMixMaxNorm(this._autoMixMaxNorm()); } catch (_) {}
                }
                try { this._syncDeckSwitches(); } catch (_) {}
                try { this._syncAutoMixKnob(); } catch (_) {}
                try { this._syncAutoFadeChangeStationKnob(); } catch (_) {}
                try { this._syncDonutCoreHues(); } catch (_) {}
                if (showDigital) {
                    this._digitalStagingView = null;
                    try { this._tearDownDigitalStagingView(); } catch (_) {}
                    try { this._wireDigitalHubPanel(this.abortCtrl.signal); } catch (_) {}
                    try { this._wireDigitalSpectrumStagingSlider(this.abortCtrl.signal); } catch (_) {}
                    try { this.applySpectrumSettings(RadioVisualEngine.loadSpectrumSettingsFromStorage(), { skipSave: true }); } catch (_) {}
                    try { this._finalizeDigitalRadioInit(); } catch (_) {}
                    try { this._initDigitalSpectrumBg(); } catch (_) {}
                    try { this._refreshDigitalAiVideoList(); } catch (_) {}
                    try {
                        if (localStorage.getItem(RadioVisualEngine.AUTOMIX_ENABLED_KEY) === '1') {
                            state.autoMixEnabled = true;
                        }
                    } catch (_) {}
                    try {
                        if (this._isAutoMixEnabled()) {
                            this._scheduleRadioAutoMix();
                            this._showDigitalToolbarAutomixCountdown();
                        }
                    } catch (_) {}
                    try { this._scheduleDigitalAiButtonIntroHint(); } catch (_) {}
                }
                try { this._updateStationUi(); } catch (_) {}
                try { this._tickClock(); } catch (_) {}
                try { this.onResize(); } catch (_) {}

                if (btnA) {
                    btnA.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('analogue'); }, sig);
                }
                if (btnD) {
                    btnD.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('digital'); }, sig);
                }
                if (showDigital) {
                    if (volDown) this._wireDigitalToolbarCenterStepButton(volDown, -1, sig);
                    if (volUp) this._wireDigitalToolbarCenterStepButton(volUp, 1, sig);
                    if (digitalToolbarCenterXfade) {
                        this._wireDigitalToolbarCenterCrossfadeSlider(digitalToolbarCenterXfade, sig);
                    }
                    if (digitalToolbarCenterXfadeCycleHit) {
                        this._wireDigitalToolbarCenterSlotCycle(digitalToolbarCenterXfadeCycleHit, sig);
                    }
                    if (volDigitalReadout) {
                        volDigitalReadout.addEventListener('click', (ev) => {
                            this._stopClick(ev);
                            this._cycleDigitalToolbarCenterMode();
                        }, sig);
                    }
                    if (btnVis) this._wireDigitalVisBgButton(btnVis, sig);
                    try { this._wireDigitalKeyboardShortcuts(sig); } catch (_) {}
                    try {
                        this._wireDigitalSpectrumLocalDrop(
                            [spectrumSideL, digitalSpectrumCanvasL],
                            'a',
                            sig
                        );
                        this._wireDigitalSpectrumLocalDrop(
                            [spectrumSideR, digitalSpectrumCanvasR],
                            'b',
                            sig
                        );
                    } catch (_) {}
                    try { this._wireDigitalLocalQueuePanel(sig); } catch (_) {}
                    try {
                        const engine = this;
                        const prevRefresh = window.__refreshDjQueueUi;
                        window.__refreshDjQueueUi = () => {
                            try {
                                if (typeof prevRefresh === 'function') prevRefresh();
                            } catch (_) {}
                            try { engine._refreshDigitalLocalQueueUi(); } catch (_) {}
                        };
                        window.__refreshDigitalStationsUi = () => {
                            try { engine._refreshDigitalStationsUi(); } catch (_) {}
                        };
                        window.__syncDigitalStationsActiveHighlight = () => {
                            try { engine._syncDigitalStationsActiveHighlight(); } catch (_) {}
                        };
                    } catch (_) {}
                    if (btnDigitalSpectrum) {
                        btnDigitalSpectrum.setAttribute('aria-label', 'Centre display mode');
                        try { this._syncDigitalSpectrumButtonState(); } catch (_) {}
                        btnDigitalSpectrum.addEventListener('contextmenu', (ev) => {
                            this._stopClick(ev);
                            try { ev.preventDefault(); } catch (_) {}
                            this._setDigitalHubModeOff();
                        }, sig);
                        btnDigitalSpectrum.addEventListener('click', (ev) => {
                            this._stopClick(ev);
                            if (this.digitalCenterMode !== 'spectrum' || this._digitalStagingView) {
                                this._returnToDefaultDigitalSpectrumView();
                            } else {
                                this._cycleDigitalSpectrumLayout();
                            }
                        }, sig);
                    }
                    if (btnDigitalVideo) {
                        btnDigitalVideo.addEventListener('click', (ev) => {
                            this._stopClick(ev);
                            this._toggleDigitalStagingFeature('video');
                        }, sig);
                    }
                    if (crossDig) {
                        const stopXfadePointerBubble = (ev) => {
                            try { ev.stopPropagation(); } catch (_) {}
                            try { window.__suppressNextClick = true; } catch (_) {}
                        };
                        crossDig.addEventListener('input', () => {
                            this._setCrossfadeX(crossDig.value);
                            try { this._resumeDecksForCrossfade(); } catch (_) {}
                        }, sig);
                        crossDig.addEventListener('pointerdown', stopXfadePointerBubble, sig);
                        crossDig.addEventListener('pointerup', stopXfadePointerBubble, sig);
                        crossDig.addEventListener('pointercancel', stopXfadePointerBubble, sig);
                        crossDig.addEventListener('change', stopXfadePointerBubble, sig);
                        this._wireDigitalCrossfadeCutHold(crossDig, sig);
                    }
                    try { this._syncDigitalCrossfadeLabels(); } catch (_) {}
                    if (btnDigitalAi) this._wireDigitalAiButton(btnDigitalAi, sig);
                    if (btnDigitalFade) this._wireDigitalFadeButton(btnDigitalFade, sig);
                    if (digitalAutoMixSlider) {
                        const applyAutoMixMax = () => {
                            const mins = this._writeAutoMixMaxMin(Number(digitalAutoMixSlider.value));
                            if (digitalAutoMixReadout) digitalAutoMixReadout.textContent = `${mins}m`;
                            this._setAutoMixMaxNorm(this._autoMixMaxNorm());
                            if (this._isAutoMixEnabled()) this._scheduleRadioAutoMix();
                        };
                        digitalAutoMixSlider.addEventListener('input', applyAutoMixMax, sig);
                        digitalAutoMixSlider.addEventListener('change', () => {
                            applyAutoMixMax();
                            this._closeDigitalAutoMixPanel();
                        }, sig);
                    }
                    if (digitalAutoFadeSlider) {
                        const applyAutoFadeDuration = () => {
                            const ms = this._writeAutoFadeDurationMs(Number(digitalAutoFadeSlider.value) * 1000);
                            if (digitalAutoFadeReadout) {
                                digitalAutoFadeReadout.textContent = `${(ms / 1000).toFixed(1)}s`;
                            }
                            try {
                                this._setKnobRotation(this.els.autoFadeKnob, (this._autoFadeDurationNorm() * 270) - 135);
                            } catch (_) {}
                        };
                        digitalAutoFadeSlider.addEventListener('input', applyAutoFadeDuration, sig);
                        digitalAutoFadeSlider.addEventListener('change', () => {
                            applyAutoFadeDuration();
                            this._closeDigitalAutoFadePanel();
                        }, sig);
                    }
                    this._wireDigitalAutoMixPanelDismiss(sig);
                    if (digitalToolbar) {
                        digitalToolbar.querySelectorAll('[data-rv-digital]').forEach((b) => {
                            b.addEventListener('click', (ev) => {
                                this._stopClick(ev);
                                const act = b.dataset.rvDigital;
                                if (act === 'ai' || act === 'fade') return;
                                if (act === 'xfade-station') this._toggleAutoFadeChangeStation();
                                this._syncDeckSwitches();
                            }, sig);
                        });
                        digitalToolbar.querySelectorAll('[data-rv-deck-transport]').forEach((b) => {
                            this._wireDigitalDeckTransportBtn(b, b.dataset.rvDeckTransport, sig);
                        });
                    }
                }
                if (tunerRail) {
                    tunerRail.addEventListener('click', (ev) => {
                        this._stopClick(ev);
                        if (!Array.isArray(stations) || stations.length < 2) return;
                        const r = tunerRail.getBoundingClientRect();
                        const t = Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
                        const idx = Math.round(t * (stations.length - 1));
                        if (typeof setStation === 'function') setStation(idx);
                    }, sig);
                }
                if (volKnob) {
                    this._wirePointerKnob(volKnob, {
                        get: () => Number(document.getElementById('volume-slider')?.value || 0.5),
                        set: (v) => this._applyVolume(v)
                    }, { onTap: () => this._toggleVolumeMute() });
                }
                if (deckAKnob) this._wireDeckKnob(deckAKnob, 'a');
                if (deckBKnob) this._wireDeckKnob(deckBKnob, 'b');
                if (crossKnob) this._wireCrossfadeKnob(crossKnob);
                if (autoFadeKnob) this._wireAutoFadeKnob(autoFadeKnob);
                if (autoMixKnob) this._wireAutoMixKnob(autoMixKnob);

                const stopRv = (ev) => {
                    if (this._shouldBypassRootGestureSuppression(ev)) return;
                    this._stopInteraction(ev);
                };
                root.addEventListener('click', stopRv, sig);
                root.addEventListener('pointerdown', stopRv, sig);
                root.addEventListener('pointerup', stopRv, sig);

                window.addEventListener('resize', this.resizeHandler, sig);
                try {
                    if (window.visualViewport) {
                        window.visualViewport.addEventListener('resize', this.resizeHandler, sig);
                        window.visualViewport.addEventListener('scroll', this.resizeHandler, sig);
                    }
                } catch (_) {}
                const onRvFsResize = () => {
                    requestAnimationFrame(() => {
                        requestAnimationFrame(() => {
                            try { this.onResize(); } catch (_) {}
                        });
                    });
                };
                document.addEventListener('fullscreenchange', onRvFsResize, sig);
                document.addEventListener('webkitfullscreenchange', onRvFsResize, sig);
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
                        if (RadioVisualEngine.digitalHubShowsAiVideo(this._digitalHubMode)) {
                            try { this._tickDigitalHubAiVideo(); } catch (_) {}
                        }
                        if (RadioVisualEngine.digitalHubShowsDeckVideo(this._digitalHubMode)) {
                            const now = performance.now();
                            if (!this._hubDeckVideoSyncAt || (now - this._hubDeckVideoSyncAt) > 800) {
                                this._hubDeckVideoSyncAt = now;
                                try { this._syncDigitalHubDeckVideo(); } catch (_) {}
                            }
                        }
                        if (this._digitalStagingView === 'video') {
                            const now = performance.now();
                            if (this._digitalStagingVideoMode === 'deck-dual') {
                                this._applyDigitalStagingVideoCrossfadeOpacities();
                            }
                            const syncDeck = this._digitalStagingVideoMode === 'deck-sync'
                                || this._digitalStagingVideoMode === 'deck-dual';
                            const interval = syncDeck ? 800 : 4000;
                            if (!this._deckBVideoSyncAt || (now - this._deckBVideoSyncAt) > interval) {
                                this._deckBVideoSyncAt = now;
                                this._syncDigitalStagingVideo();
                            }
                        } else if (this.digitalCenterMode === 'deckB' && this._digitalDeckBView === 'video') {
                            const now = performance.now();
                            if (!this._deckBVideoSyncAt || (now - this._deckBVideoSyncAt) > 800) {
                                this._deckBVideoSyncAt = now;
                                this._syncDigitalDeckBVideo();
                            }
                        }
                        if (this._digitalStagingView && this._rvDigitalPmResize) {
                            try { this._rvDigitalPmResize(); } catch (_) {}
                        } else if (this.digitalCenterMode === 'deckB' && this._rvDigitalPmResize) {
                            try { this._rvDigitalPmResize(); } catch (_) {}
                        }
                    }
                } catch (_) {}
            }

            _isDigitalPageFullscreen() {
                if (this.skin !== 'digital') return false;
                try {
                    const fs = document.fullscreenElement || document.webkitFullscreenElement;
                    if (!fs) return false;
                    return fs === document.documentElement || fs === document.body;
                } catch (_) {}
                return false;
            }

            _clearDigitalFullscreenLayout() {
                const root = this.root;
                const stage = this.els && this.els.stageDigital;
                if (root) root.classList.remove('radio-visual-root--digital-fs');
                if (!stage) return;
                stage.style.removeProperty('width');
                stage.style.removeProperty('max-width');
                stage.style.removeProperty('max-height');
                stage.style.removeProperty('transform');
                stage.style.removeProperty('transform-origin');
                stage.style.removeProperty('margin-top');
                stage.style.removeProperty('margin-bottom');
                stage.style.removeProperty('margin-left');
                stage.style.removeProperty('margin-right');
            }

            /** Page fullscreen: fit stage to available width; shrink uniformly only when height overflows. */
            _syncDigitalFullscreenLayout() {
                const stage = this.els && this.els.stageDigital;
                const root = this.root;
                if (!stage) return;

                if (!this._isDigitalPageFullscreen()) {
                    this._clearDigitalFullscreenLayout();
                    return;
                }

                if (root) root.classList.add('radio-visual-root--digital-fs');

                const host = root || document.getElementById('radio-visual-root');
                const availW = Math.max(1, host ? host.clientWidth : window.innerWidth);
                const availH = Math.max(1, host ? host.clientHeight : window.innerHeight);

                stage.style.removeProperty('transform');
                stage.style.removeProperty('transform-origin');
                stage.style.removeProperty('margin-top');
                stage.style.removeProperty('margin-bottom');
                stage.style.removeProperty('margin-left');
                stage.style.removeProperty('margin-right');
                stage.style.removeProperty('max-height');

                const fitW = Math.max(1, availW);
                stage.style.width = `${fitW}px`;
                stage.style.maxWidth = '100%';

                const measuredH = Math.max(1, stage.offsetHeight);
                const measuredW = Math.max(1, stage.offsetWidth);
                if (measuredH <= availH) return;

                const scale = Math.max(0.05, availH / measuredH);
                stage.style.transform = `scale(${scale})`;
                stage.style.transformOrigin = 'center center';
                const excessH = measuredH * (1 - scale);
                const excessW = measuredW * (1 - scale);
                stage.style.marginTop = `${-excessH / 2}px`;
                stage.style.marginBottom = `${-excessH / 2}px`;
                stage.style.marginLeft = `${-excessW / 2}px`;
                stage.style.marginRight = `${-excessW / 2}px`;
            }

            onResize() {
                try { this._syncDigitalFullscreenLayout(); } catch (_) {}
                try { this._resizeCanvases(); } catch (_) {}
                if (this.skin === 'digital' && this._digitalHubMode === 'spectrum') {
                    try { this._syncSpectrumStagingScale(); } catch (_) {}
                }
                if (this._rvDigitalPmResize) {
                    try { this._rvDigitalPmResize(); } catch (_) {}
                }
                try {
                    if (this.els.digitalBtns) this._fitDigitalFeatureButtonLabels(this.els.digitalBtns);
                } catch (_) {}
                try {
                    const toolbar = document.getElementById('radio-visual-digital-toolbar');
                    if (toolbar) this._fitDigitalToolbarButtonLabels(toolbar);
                } catch (_) {}
            }

            destroy() {
                this._clearRadioAutoMixTimer();
                this._rvAutoMixCyclePending = false;
                try { this._closeDigitalAutoMixPanel(); } catch (_) {}
                this._digitalStagingView = null;
                try { this._tearDownDigitalStagingView(); } catch (_) {}
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
                if (this._spectrumSliderHideTimer) {
                    try { clearTimeout(this._spectrumSliderHideTimer); } catch (_) {}
                    this._spectrumSliderHideTimer = null;
                }
                this._spectrumRingSmooth = { low: null, mid: null, high: null };
                try { if (this.animId) cancelAnimationFrame(this.animId); } catch (_) {}
                this.animId = null;
                try { this._cancelDigitalAiButtonIntroHint(); } catch (_) {}
                try { this._cancelDigitalToolbarVolumePeek(); } catch (_) {}
                try { this._clearDigitalHubDeckVideoTapTimer(); } catch (_) {}
                if (this.clockTimerId) {
                    try { clearInterval(this.clockTimerId); } catch (_) {}
                    this.clockTimerId = null;
                }
                try { if (this.abortCtrl) this.abortCtrl.abort(); } catch (_) {}
                this.abortCtrl = null;
                try {
                    window.removeEventListener('resize', this.resizeHandler);
                } catch (_) {}
                try { this._clearDigitalFullscreenLayout(); } catch (_) {}
                this.root = null;
                this.els = {};
                try { container.innerHTML = ''; } catch (_) {}
            }
        }
