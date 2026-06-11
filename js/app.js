        import * as THREE from 'three';
        import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
        import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
        import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
        import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
        import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
        import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';
        import butterchurn from 'butterchurn';
        import butterchurnPresets from 'butterchurn-presets';
        // --- MOBILE DETECTION & OPTIMIZATION FLAGS ---
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;

// Config: Desktop gets max quality, Mobile gets safe limits
const QUALITY = {
    sphereSegs: isMobile ? 32 : 64,       // High poly for desktop, Low for mobile
    orbSegs: isMobile ? 32 : 96,          // 96 is overkill for mobile screens
    fernCount: isMobile ? 6000 : 30000,   // 30k loops kills mobile CPU
    galaxyCount: isMobile ? 1500 : 4000,  // Reduce particle counts
    pixelRatioCap: isMobile ? 1.5 : 3.0   // Cap mobile resolution to save battery
};
        // --- GLOBAL STATE ---
        const state = {
            audioCtx: null,
            sourceNode: null,
            analyserNode: null,
            isPlaying: false,
            currentModeIdx: 0,
            activeVisualizer: null,
            idleTimer: null,
            mediaStream: null,
            /** Deck A/B playback source: affects queue advance vs radio */
            deckSourceMode: { a: 'radio', b: 'radio' },
            /** Human-readable filename for the current local track (DJ head titles). */
            deckLocalDisplayName: { a: '', b: '' },
            /** AUTO-MIX runs only after the user enables it this session (never resumed on page load). */
            autoMixEnabled: false
        };

        function escapeHtml(str) {
            return String(str ?? '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }

        /** Waiting local tracks per deck ({ name, url } blob URLs). Current track is not in queue. */
        const deckFileQueues = { a: [], b: [] };
        const AUTOMIX_ENABLED_STORAGE_KEY = 'dj.automix.enabled.v1';
        const AUTOFADE_CHANGE_STATION_STORAGE_KEY = 'dj.autofade.changeStation.enabled.v1';
        function ensureAutoMixDeferLocalState() {
            if (!state.autoMixDeferLocal) {
                state.autoMixDeferLocal = {
                    a: { active: false, userReleased: false, seekPos: null },
                    b: { active: false, userReleased: false, seekPos: null }
                };
            }
            return state.autoMixDeferLocal;
        }
        function isAutoMixEnabledGlobal() {
            try {
                if (state && typeof state.autoMixEnabled === 'boolean') return state.autoMixEnabled;
            } catch (_) {}
            return false;
        }
        function isAutoFadeChangeStationEnabledGlobal() {
            try {
                const raw = localStorage.getItem(AUTOFADE_CHANGE_STATION_STORAGE_KEY);
                if (raw === '0') return false;
                if (raw === '1') return true;
            } catch (_) {}
            return true;
        }
        /** Opposite deck from current crossfader (low = incoming B, high = incoming A). */
        function getCrossfaderIncomingDeckKey() {
            try {
                const dc = document.getElementById('dj-crossfader');
                const mc = document.getElementById('mix-crossfader');
                const raw = (dc && dc.value !== undefined && dc.value !== '') ? dc.value : (mc && mc.value);
                const x = Math.max(0, Math.min(1, Number(raw) || 0));
                return x < 0.5 ? 'b' : 'a';
            } catch (_) {
                return 'b';
            }
        }
        /** Incoming deck for the next scheduled AUTO-MIX crossfade (opposite side of the fader). */
        function getAutoMixIncomingDeckKey() {
            try {
                if (!isAutoMixEnabledGlobal()) return null;
                return getCrossfaderIncomingDeckKey();
            } catch (_) {
                return null;
            }
        }
        function isAutoMixDeferredLocalArmed(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const slot = ensureAutoMixDeferLocalState()[dk];
            return !!(slot && slot.active && !slot.userReleased);
        }
        function releaseAutoMixDeferredLocal(deckKey, reason) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const slot = ensureAutoMixDeferLocalState()[dk];
            if (!slot) return;
            slot.userReleased = true;
            slot.active = false;
            try {
                if (reason === 'seek' || reason === 'scrub') {
                    const el = dk === 'b' ? audioElB : audioEl;
                    if (el && el.src) slot.seekPos = Number(el.currentTime) || 0;
                }
            } catch (_) {}
        }
        function markAutoMixDeferredLocal(deckKey, active) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const slot = ensureAutoMixDeferLocalState()[dk];
            slot.active = !!active;
            if (active) {
                slot.userReleased = false;
                slot.seekPos = null;
            }
        }
        function shouldDeferLocalPlayForAutoMix(deckKey, opts) {
            if (opts && opts.forceImmediate) return false;
            const dk = deckKey === 'b' ? 'b' : 'a';
            const slot = ensureAutoMixDeferLocalState()[dk];
            if (slot && slot.userReleased) return false;
            if (!isAutoMixEnabledGlobal()) return false;
            if (!isAutoFadeChangeStationEnabledGlobal()) return false;
            return getAutoMixIncomingDeckKey() === dk;
        }
        function clearAutoMixDeferForNonIncoming() {
            const incoming = getAutoMixIncomingDeckKey();
            ['a', 'b'].forEach((dk) => {
                if (incoming === dk) return;
                const slot = ensureAutoMixDeferLocalState()[dk];
                if (slot && slot.active) {
                    slot.active = false;
                    slot.userReleased = false;
                    slot.seekPos = null;
                }
            });
        }
        function clearAllAutoMixDeferLocal() {
            ['a', 'b'].forEach((dk) => {
                const slot = ensureAutoMixDeferLocalState()[dk];
                slot.active = false;
                slot.userReleased = false;
                slot.seekPos = null;
            });
        }
        /** Start a cued local track when an AUTO-MIX crossfade begins (returns true if playback started). */
        function tryStartAutoMixDeferredLocal(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const slot = ensureAutoMixDeferLocalState()[dk];
            if (!slot || !slot.active || slot.userReleased) return false;
            if (!state.deckSourceMode || state.deckSourceMode[dk] !== 'local') return false;
            const el = dk === 'b' ? audioElB : audioEl;
            const src = el ? String(el.currentSrc || el.src || '') : '';
            if (!el || !src || src === 'about:blank') return false;
            slot.active = false;
            const pos = slot.seekPos;
            const afterPlay = () => {
                try { connectDeckMediaToEq(dk); } catch (_) {}
                try {
                    if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                        state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                    }
                } catch (_) {}
                if (!state.isPlaying) startGame();
                if (dk === 'b') {
                    try { if (typeof updateMixBStatus === 'function') updateMixBStatus(); } catch (_) {}
                } else {
                    try { showStationBanner(state.deckLocalDisplayName.a || 'Local track'); } catch (_) {}
                }
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            };
            try {
                if (typeof pos === 'number' && pos > 0) el.currentTime = Math.min(pos, Math.max(0, (Number(el.duration) || pos) - 0.05));
                else el.currentTime = 0;
            } catch (_) {}
            el.play().then(afterPlay).catch((e) => {
                console.warn('Deferred AUTO-MIX local play failed:', e);
                if (dk === 'b') {
                    try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                } else {
                    try { if (typeof playRadio === 'function') playRadio(); } catch (_) {}
                }
            });
            return true;
        }
        const USER_RADIO_STATIONS_KEY = 'djUserRadioStations.v1';
        const LAST_STATION_A_STORAGE_KEY = 'dj.lastStation.a.v1';
        const LAST_STATION_B_STORAGE_KEY = 'dj.lastStation.b.v1';
        let userRadioStations = [];
        const deckVideoFeeds = { a: null, b: null };
        const deckVideoHistory = { a: [], b: [] };
        const mediaVideoQueue = [];
        let deckBVideoLoopEnabled = false;
        /** When true, advance through the full video queue on each end; when false, optional single-clip loop via LOOP. */
        let deckBVideoPlayAllEnabled = true;
        let deckBVideoShuffleEnabled = false;
        function deckBVideoSingleLoopActive() {
            return !!deckBVideoLoopEnabled && !deckBVideoPlayAllEnabled;
        }
        /** When true, Deck B video shows idle: looping logo.mp4 if present, else blank (after Stop). */
        let deckBVideoUserIdle = false;
        const DECK_B_IDLE_LOGO_URL = 'assets/video/logo.mp4';
        const KARAOKE_NERDS_BASE_URL = 'https://www.karaokenerds.com/';
        /** Hash scrolls the iframe to the search field (below the large logo on first paint). */
        const KARAOKE_NERDS_EMBED_URL = 'https://www.karaokenerds.com/#query';

        function normalizeKaraokeNerdsEmbedUrl(url) {
            let href = (url && String(url).trim()) || '';
            if (!href) return KARAOKE_NERDS_EMBED_URL;
            if (!/^https?:\/\//i.test(href)) href = 'https://' + href.replace(/^\/+/, '');
            try {
                const u = new URL(href);
                const host = (u.hostname || '').replace(/^www\./i, '').toLowerCase();
                if (host === 'karaokenerds.com' && !u.hash) {
                    u.hash = 'query';
                    return u.href;
                }
            } catch (_) {}
            return href;
        }

        const container = document.getElementById('canvas-container');
        const statusEl = document.getElementById('loading-status');
        const uiLayer = document.getElementById('ui-layer');
        const radioQuickBtn = document.getElementById('radio-quick');
        const radioPanel = document.getElementById('radio-panel');
        const radioListEl = document.getElementById('radio-list');
        const radioListElB = document.getElementById('radio-list-b');
        const topMenuStationsWrap = document.getElementById('top-menu-stations-wrap');
        let topMenuStationsBSplitManual = false;
        const radioInputEl = document.getElementById('radio-url') || document.getElementById('station-url');
        const audioEl = document.getElementById('radio-element');
        const audioElRadioAAlt = document.getElementById('radio-element-a-alt');
        const audioElB = document.getElementById('radio-element-b');
        const audioElRadioBAlt = document.getElementById('radio-element-b-alt');
        /** Abort in-flight Deck A/B radio crossfade when switching stations again quickly */
        let radioAHandoffAbortCtrl = null;
        let radioBHandoffAbortCtrl = null;
        const RADIO_STATION_XFADE_KEY = 'dj.radio.stationCrossfade.v1';
        let radioStationCrossfadeEnabled = false;
        let radioStationCrossfadeSec = 0;
        (function loadRadioStationCrossfadeFromStorage() {
            try {
                const raw = localStorage.getItem(RADIO_STATION_XFADE_KEY);
                if (!raw) return;
                const o = JSON.parse(raw);
                if (!o || typeof o !== 'object') return;
                if (typeof o.enabled === 'boolean') radioStationCrossfadeEnabled = o.enabled;
                const s = Number(o.sec);
                if (Number.isFinite(s) && s >= 0 && s <= 15) radioStationCrossfadeSec = s;
            } catch (_) {}
        })();
        function saveRadioStationCrossfadeToStorage() {
            try {
                localStorage.setItem(
                    RADIO_STATION_XFADE_KEY,
                    JSON.stringify({
                        enabled: !!radioStationCrossfadeEnabled,
                        sec: Math.max(0, Math.min(15, Number(radioStationCrossfadeSec) || 0))
                    })
                );
            } catch (_) {}
        }
        /** Deck A gapless station handoff uses ~60 ms; optional longer crossfade when enabled in Auto-Fade panel. */
        function getDeckARadioCrossfadeRampSec() {
            const gaplessMicro = 0.06;
            if (!radioStationCrossfadeEnabled) return gaplessMicro;
            const s = Number(radioStationCrossfadeSec) || 0;
            if (s <= 0) return gaplessMicro;
            return Math.min(15, Math.max(0.05, s));
        }
        const stationBanner = document.getElementById('station-banner');
        const stationBannerNameEl = document.getElementById('station-banner-name');
        const stationBannerMetaWrap = document.getElementById('station-banner-meta-wrap');
        const stationBannerNowplayingEl = document.getElementById('station-banner-nowplaying');
        const btnBannerShazam = document.getElementById('btn-banner-shazam');
        const settingsPanel = document.getElementById('settings-panel');
        const settingsBtn = document.getElementById('btn-settings');
        const settingsApplyBtn = document.getElementById('btn-settings-apply');
        const settingsCloseBtn = document.getElementById('btn-settings-close');
        // Text-In refs/state
        const textInPanel = document.getElementById('textin-panel');
        const textOverlayLayer = document.getElementById('text-overlay-layer');
        const tiText = document.getElementById('ti-text');
        const tiFont = document.getElementById('ti-font');
        const tiFontRand = document.getElementById('ti-font-rand');
        const tiColor = document.getElementById('ti-color');
        const tiColorRandBtn = document.getElementById('ti-color-rand');
        const tiColorRandCheck = document.getElementById('ti-color-rand-check');
        const tiSize = document.getElementById('ti-size');
        const tiSizeRand = document.getElementById('ti-size-rand');
        const tiSizeRMin = document.getElementById('ti-size-rmin');
        const tiSizeRMax = document.getElementById('ti-size-rmax');
        const tiX = document.getElementById('ti-x');
        const tiXRand = document.getElementById('ti-x-rand');
        const tiXRMin = document.getElementById('ti-x-rmin');
        const tiXRMax = document.getElementById('ti-x-rmax');
        const tiBorder = document.getElementById('ti-border');
        const tiBorderColor = document.getElementById('ti-border-color');
        const tiBorderColorRandBtn = document.getElementById('ti-border-color-rand');
        const tiBorderColorRandCheck = document.getElementById('ti-border-color-rand-check');
        const tiBorderRand = document.getElementById('ti-border-rand');
        const tiBorderRMin = document.getElementById('ti-border-rmin');
        const tiBorderRMax = document.getElementById('ti-border-rmax');
        const tiGlowRand = document.getElementById('ti-glow-rand');
        const tiGlowColor = document.getElementById('ti-glow-color');
        const tiGlowColorRandBtn = document.getElementById('ti-glow-color-rand');
        const tiGlowColorRandCheck = document.getElementById('ti-glow-color-rand-check');
        const tiGlowRMin = document.getElementById('ti-glow-rmin');
        const tiGlowRMax = document.getElementById('ti-glow-rmax');
        const tiFlashSpeedRand = document.getElementById('ti-flashspeed-rand');
        const tiFlashSpeedRMin = document.getElementById('ti-flashspeed-rmin');
        const tiFlashSpeedRMax = document.getElementById('ti-flashspeed-rmax');
        const tiSpeedRand = document.getElementById('ti-speed-rand');
        const tiSpeedRMin = document.getElementById('ti-speed-rmin');
        const tiSpeedRMax = document.getElementById('ti-speed-rmax');
        const tiGlow = document.getElementById('ti-glow');
        const tiFlash = document.getElementById('ti-flash');
        const tiFlashSpeed = document.getElementById('ti-flash-speed');
        const tiSpeed = document.getElementById('ti-speed');
        const tiPreview = document.getElementById('textin-preview');
        const tiSend = document.getElementById('ti-send');
        const tiAutoBtn = document.getElementById('ti-auto');
        const tiAutoInterval = document.getElementById('ti-auto-interval');
        const tiClose = document.getElementById('btn-textin-close');
        let textOverlayAnimId = null;
        const activeTextOverlays = [];
        let shortcutsLocked = false;
        let textAutoTimer = null;
        let textAutoOn = false;
        // UI Lock state and refs
        let uiLocked = false;
        const uiLockShield = document.getElementById('ui-lock-shield');
        const uiLockToggle = document.getElementById('ui-lock-toggle');
        function applyUiLockState() {
            if (!uiLockShield || !uiLockToggle) return;
            if (uiLocked) {
                uiLockShield.classList.add('active');
                uiLockToggle.classList.add('on');
                uiLockToggle.setAttribute('aria-pressed', 'true');
                // Close any open panels/menus
                try { hideTextInPanel({ forceCloseDeckB: true }); } catch(_) {}
                try { if (typeof isTopMenuOpen === 'function' && isTopMenuOpen()) closeTopMenuPanel(); } catch(_) {}
                try { hideWebmSettingsPanel(); } catch(_) {}
                try { closeBottomMenuPanel(); } catch(_) {}
                try { closeKeyboardShortcutsPanel(); } catch (_) {}
                try { closeOptionsPanel(); } catch (_) {}
                try {
                    if (settingsPanel && !settingsPanel.classList.contains('display-none')) {
                        settingsPanel.classList.add('display-none');
                        settingsPanel.style.opacity = '';
                        settingsPanel.style.pointerEvents = '';
                    }
                } catch(_) {}
            } else {
                uiLockShield.classList.remove('active');
                uiLockToggle.classList.remove('on');
                uiLockToggle.setAttribute('aria-pressed', 'false');
            }
        }
        if (uiLockToggle) {
            uiLockToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                uiLocked = !uiLocked;
                applyUiLockState();
                resetIdleTimer();
            });
        }
        // Mix panel refs
        const micConfirm = document.getElementById('mic-confirm');
        const micOk = document.getElementById('btn-mic-ok');
        const micCancel = document.getElementById('btn-mic-cancel');
        const midiConfirm = document.getElementById('midi-confirm');
        const midiOk = document.getElementById('btn-midi-ok');
        const midiCancel = document.getElementById('btn-midi-cancel');
        const mixPanel = document.getElementById('mix-panel');
        const mixClose = document.getElementById('btn-mix-close');
        const keyboardShortcutsPanel = document.getElementById('keyboard-shortcuts-panel');
        const btnKeyboardShortcutsClose = document.getElementById('btn-keyboard-shortcuts-close');
        function isKeyboardShortcutsPanelOpen() {
            return !!(keyboardShortcutsPanel && !keyboardShortcutsPanel.classList.contains('display-none') && keyboardShortcutsPanel.classList.contains('open'));
        }
        function openKeyboardShortcutsPanel() {
            if (!keyboardShortcutsPanel || uiLocked) return;
            keyboardShortcutsPanel.classList.remove('display-none');
            requestAnimationFrame(() => { keyboardShortcutsPanel.classList.add('open'); });
        }
        function closeKeyboardShortcutsPanel() {
            if (!keyboardShortcutsPanel) return;
            keyboardShortcutsPanel.classList.remove('open');
            setTimeout(() => { keyboardShortcutsPanel.classList.add('display-none'); }, 350);
        }
        function toggleKeyboardShortcutsPanel() {
            if (isKeyboardShortcutsPanelOpen()) closeKeyboardShortcutsPanel();
            else openKeyboardShortcutsPanel();
        }
        if (btnKeyboardShortcutsClose) {
            btnKeyboardShortcutsClose.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                closeKeyboardShortcutsPanel();
                try { resetIdleTimer(); } catch (_) {}
            });
        }
        const fxLow = document.getElementById('fx-lowpass');
        const fxHigh = document.getElementById('fx-highpass');
        const fxBass = document.getElementById('fx-bass');
        const fxTreble = document.getElementById('fx-treble');
        const fxEcho = document.getElementById('fx-echo');
        const fxDist = document.getElementById('fx-distort');
        const fxArp = document.getElementById('fx-arp');
        const fxTk = document.getElementById('fx-tk');
        const fxCut = document.getElementById('fx-cut');
        const fxReverb = document.getElementById('fx-reverb');
        const fxFlanger = document.getElementById('fx-flanger');
        const fxNoVocal = document.getElementById('fx-novocal');
        // BPM scope refs/state
        const bpmBtnA = document.getElementById('mix-bpm-a');
        const bpmBtnB = document.getElementById('mix-bpm-b');
        const bpmBtnGrid = document.getElementById('mix-bpm-grid');
        const bpmOutA = document.getElementById('mix-bpm-out-a');
        const bpmOutB = document.getElementById('mix-bpm-out-b');
        const scopeCanvas = document.getElementById('mix-scope');
        const scopeCtx = scopeCanvas ? scopeCanvas.getContext('2d') : null;
        let scopeAnimId = null;
        let bpmOnA = false, bpmOnB = false;
        let envA = [], envB = [];
        let envFluxA = [], envFluxB = [];
        const maxEnvSamples = 8 * 60; // ~8 seconds at 60Hz sampling
        // Onset detection (spectral flux)
        let lastSpecA = null, lastSpecB = null;
        let fluxHistA = [], fluxHistB = [];
        const fluxHistLen = 60;
        let lastBeatTsA = 0, lastBeatTsB = 0;
        let bpmSmoothA = null, bpmSmoothB = null;
        let nextBeatTsA = null, nextBeatTsB = null;
        let showBeatGrid = false;
        // Tap/Nudge
        const tapBtnA = document.getElementById('mix-bpm-tap-a');
        const tapBtnB = document.getElementById('mix-bpm-tap-b');
        const nudgeBm = document.getElementById('mix-bpm-nudge-bm');
        const nudgeBp = document.getElementById('mix-bpm-nudge-bp');
        let tapTimesA = [], tapTimesB = [];
        let phaseOffsetBms = 0; // apply to B grid ticks
        // --- OPTIONS PANEL HELPERS ---
        const DIGITAL_THEME_STORAGE_KEY = 'radioVisual.digitalTheme.v1';
        const OPTIONS_AUTO_CLOSE_MS = 30000;
        const DIGITAL_BG_GIF_MANIFEST_URL = 'assets/gifs/digital/manifest.json';
        const DIGITAL_BG_GIF_STORAGE_KEY = 'radioVisual.digitalBgGif.v1';
        const DIGITAL_BG_GIF_ENABLED_KEY = 'radioVisual.digitalBgGif.enabled.v1';
        const DEFAULT_DIGITAL_THEME = {
            presetId: 'midnight-blue',
            bgA: '#0a1628',
            bgB: '#061018',
            bgC: '#0c1020',
            bgOuterA: '#0a1628',
            bgOuterB: '#061018',
            bgOuterC: '#0c1020',
            bgGradientAngle: 165,
            accent: '#ffd246',
            font: "'Orbitron', 'Share Tech Mono', ui-monospace, monospace",
            btnBlueTop: '#123048',
            btnBlueBase: '#081820',
            btnBlueAccent: '#00dcff',
            btnPurpleTop: '#5a3488',
            btnPurpleBase: '#1a0c30',
            btnPurpleLabel: '#5cff9e',
            btnPurpleActive: '#ff5ce8',
            btnBlueFontScale: 1,
            btnPurpleFontScale: 1,
            clockFont: "'Orbitron', 'Share Tech Mono', ui-monospace, monospace",
            clockColor: '#fff566',
            clockFontScale: 1,
            clockFormat: 'weekday-time'
        };
        const DIGITAL_CLOCK_FORMATS = {
            'weekday-time': 'Weekday + 24h time',
            'time-24': '24-hour time only',
            'time-12': '12-hour time',
            'date-time': 'Date + time',
            'date-short': 'Short date + time'
        };
        const DIGITAL_THEME_PRESETS = {
            'midnight-blue': {
                label: 'Midnight Blue',
                bgA: '#0a1628',
                bgB: '#061018',
                bgC: '#0c1020',
                accent: '#ffd246'
            },
            'cyber-cyan': {
                label: 'Cyber Cyan',
                bgA: '#041824',
                bgB: '#062838',
                bgC: '#031018',
                accent: '#00e8ff'
            },
            'sunset-amber': {
                label: 'Sunset Amber',
                bgA: '#1a0c08',
                bgB: '#281408',
                bgC: '#100810',
                accent: '#ff9944'
            },
            'violet-pulse': {
                label: 'Violet Pulse',
                bgA: '#140818',
                bgB: '#1e0a28',
                bgC: '#0a0614',
                accent: '#c86bff'
            },
            'matrix-green': {
                label: 'Matrix Green',
                bgA: '#041408',
                bgB: '#062010',
                bgC: '#020a06',
                accent: '#44ff88'
            },
            'rose-neon': {
                label: 'Rose Neon',
                bgA: '#180810',
                bgB: '#240818',
                bgC: '#100408',
                accent: '#ff4d8a'
            },
            'arctic-steel': {
                label: 'Arctic Steel',
                bgA: '#0c1420',
                bgB: '#142030',
                bgC: '#081018',
                accent: '#8ec8ff'
            },
            'deep-ocean': {
                label: 'Deep Ocean',
                bgA: '#040818',
                bgB: '#081828',
                bgC: '#020610',
                accent: '#3aa8ff'
            }
        };
        const optionsPanel = document.getElementById('options-panel');
        const btnOptions = document.getElementById('btn-options');
        const btnOptionsClose = document.getElementById('btn-options-close');
        const optDigitalThemePreset = document.getElementById('opt-digital-theme-preset');
        const optDigitalBgA = document.getElementById('opt-digital-bg-a');
        const optDigitalBgB = document.getElementById('opt-digital-bg-b');
        const optDigitalBgC = document.getElementById('opt-digital-bg-c');
        const optDigitalBgOuterA = document.getElementById('opt-digital-bg-outer-a');
        const optDigitalBgOuterB = document.getElementById('opt-digital-bg-outer-b');
        const optDigitalBgOuterC = document.getElementById('opt-digital-bg-outer-c');
        const optDigitalBgGradientAngle = document.getElementById('opt-digital-bg-gradient-angle');
        const optDigitalBgGradientAngleReadout = document.getElementById('opt-digital-bg-gradient-angle-readout');
        const optDigitalBgGif = document.getElementById('opt-digital-bg-gif');
        const optDigitalAccent = document.getElementById('opt-digital-accent');
        const optDigitalFont = document.getElementById('opt-digital-font');
        const optDigitalBtnBlueTop = document.getElementById('opt-digital-btn-blue-top');
        const optDigitalBtnBlueBase = document.getElementById('opt-digital-btn-blue-base');
        const optDigitalBtnBlueAccent = document.getElementById('opt-digital-btn-blue-accent');
        const optDigitalBtnPurpleTop = document.getElementById('opt-digital-btn-purple-top');
        const optDigitalBtnPurpleBase = document.getElementById('opt-digital-btn-purple-base');
        const optDigitalBtnPurpleLabel = document.getElementById('opt-digital-btn-purple-label');
        const optDigitalBtnPurpleActive = document.getElementById('opt-digital-btn-purple-active');
        const optDigitalBtnBlueFontScale = document.getElementById('opt-digital-btn-blue-font-scale');
        const optDigitalBtnBlueFontScaleReadout = document.getElementById('opt-digital-btn-blue-font-scale-readout');
        const optDigitalBtnPurpleFontScale = document.getElementById('opt-digital-btn-purple-font-scale');
        const optDigitalBtnPurpleFontScaleReadout = document.getElementById('opt-digital-btn-purple-font-scale-readout');
        const optDigitalClockFormat = document.getElementById('opt-digital-clock-format');
        const optDigitalClockFont = document.getElementById('opt-digital-clock-font');
        const optDigitalClockColor = document.getElementById('opt-digital-clock-color');
        const optDigitalClockFontScale = document.getElementById('opt-digital-clock-font-scale');
        const optDigitalClockFontScaleReadout = document.getElementById('opt-digital-clock-font-scale-readout');
        const optDigitalThemeReset = document.getElementById('opt-digital-theme-reset');
        const optAutomixEnabled = document.getElementById('opt-automix-enabled');
        const optAutomixMax = document.getElementById('opt-automix-max');
        const optAutomixMaxReadout = document.getElementById('opt-automix-max-readout');
        const optAutofadeDuration = document.getElementById('opt-autofade-duration');
        const optAutofadeDurationReadout = document.getElementById('opt-autofade-duration-readout');
        const optAutofadeChangeStation = document.getElementById('opt-autofade-change-station');
        const optSpectrumColorStream = document.getElementById('opt-spectrum-color-stream');
        const optSpectrumSize = document.getElementById('opt-spectrum-size');
        const optSpectrumSizeReadout = document.getElementById('opt-spectrum-size-readout');
        const optSpectrumOpacity = document.getElementById('opt-spectrum-opacity');
        const optSpectrumOpacityReadout = document.getElementById('opt-spectrum-opacity-readout');
        const optSpectrumAudioStrength = document.getElementById('opt-spectrum-audio-strength');
        const optSpectrumAudioStrengthReadout = document.getElementById('opt-spectrum-audio-strength-readout');
        const optSpectrumColorFlow = document.getElementById('opt-spectrum-color-flow');
        const optSpectrumColorFlowReadout = document.getElementById('opt-spectrum-color-flow-readout');
        const optSpectrumReset = document.getElementById('opt-spectrum-reset');
        let spectrumColorStreamSelectReady = false;
        function getSpectrumEngineClass() {
            return globalThis.RadioVisualEngine || null;
        }
        function loadSpectrumSettingsFromStorage() {
            const RVE = getSpectrumEngineClass();
            if (RVE && typeof RVE.loadSpectrumSettingsFromStorage === 'function') {
                return RVE.loadSpectrumSettingsFromStorage();
            }
            try {
                const raw = localStorage.getItem('radioVisual.digitalSpectrum.v1');
                const parsed = raw ? JSON.parse(raw) : null;
                if (!parsed || typeof parsed !== 'object') {
                    return {
                        colorStreamId: 'aurora',
                        scale: 1,
                        opacity: 1,
                        audioStrength: 1,
                        colorFlow: 1
                    };
                }
                return parsed;
            } catch (_) {
                return {
                    colorStreamId: 'aurora',
                    scale: 1,
                    opacity: 1,
                    audioStrength: 1,
                    colorFlow: 1
                };
            }
        }
        function saveSpectrumSettingsToStorage(settings) {
            const RVE = getSpectrumEngineClass();
            if (RVE && typeof RVE.saveSpectrumSettingsToStorage === 'function') {
                return RVE.saveSpectrumSettingsToStorage(settings);
            }
            try { localStorage.setItem('radioVisual.digitalSpectrum.v1', JSON.stringify(settings)); } catch (_) {}
            return settings;
        }
        function applyDigitalSpectrumSettings(settings) {
            const RVE = getSpectrumEngineClass();
            const next = (RVE && typeof RVE.clampSpectrumSettings === 'function')
                ? RVE.clampSpectrumSettings(settings)
                : settings;
            saveSpectrumSettingsToStorage(next);
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv.applySpectrumSettings === 'function') {
                    rv.applySpectrumSettings(next, { skipSave: true });
                }
            } catch (_) {}
            return next;
        }
        function populateSpectrumColorStreamSelect() {
            if (spectrumColorStreamSelectReady || !optSpectrumColorStream) return;
            const RVE = getSpectrumEngineClass();
            if (!RVE || !RVE.SPECTRUM_COLOR_STREAM_PRESETS) return;
            optSpectrumColorStream.innerHTML = '';
            Object.entries(RVE.SPECTRUM_COLOR_STREAM_PRESETS).forEach(([id, preset]) => {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = preset.label || id;
                optSpectrumColorStream.appendChild(opt);
            });
            spectrumColorStreamSelectReady = true;
        }
        function applySpectrumSettingsToControls(settings) {
            const s = settings || loadSpectrumSettingsFromStorage();
            if (optSpectrumColorStream) optSpectrumColorStream.value = s.colorStreamId || 'aurora';
            const sizePct = Math.round((Number(s.scale) || 1) * 100);
            if (optSpectrumSize) optSpectrumSize.value = String(Math.max(35, Math.min(280, sizePct)));
            if (optSpectrumSizeReadout) optSpectrumSizeReadout.textContent = `${sizePct}%`;
            const opacityPct = Math.round((Number(s.opacity) || 1) * 100);
            if (optSpectrumOpacity) optSpectrumOpacity.value = String(Math.max(15, Math.min(100, opacityPct)));
            if (optSpectrumOpacityReadout) optSpectrumOpacityReadout.textContent = `${opacityPct}%`;
            const strengthPct = Math.round((Number(s.audioStrength) || 1) * 100);
            if (optSpectrumAudioStrength) optSpectrumAudioStrength.value = String(Math.max(25, Math.min(300, strengthPct)));
            if (optSpectrumAudioStrengthReadout) optSpectrumAudioStrengthReadout.textContent = `${strengthPct}%`;
            const flowPct = Math.round((Number(s.colorFlow) || 1) * 100);
            if (optSpectrumColorFlow) optSpectrumColorFlow.value = String(Math.max(25, Math.min(300, flowPct)));
            if (optSpectrumColorFlowReadout) optSpectrumColorFlowReadout.textContent = `${flowPct}%`;
        }
        function collectSpectrumSettingsFromControls() {
            const current = loadSpectrumSettingsFromStorage();
            const scale = Math.max(0.35, Math.min(2.8, (Number(optSpectrumSize && optSpectrumSize.value) || 100) / 100));
            const opacity = Math.max(0.15, Math.min(1, (Number(optSpectrumOpacity && optSpectrumOpacity.value) || 100) / 100));
            const audioStrength = Math.max(0.25, Math.min(3, (Number(optSpectrumAudioStrength && optSpectrumAudioStrength.value) || 100) / 100));
            const colorFlow = Math.max(0.25, Math.min(3, (Number(optSpectrumColorFlow && optSpectrumColorFlow.value) || 100) / 100));
            return {
                colorStreamId: (optSpectrumColorStream && optSpectrumColorStream.value) || current.colorStreamId || 'aurora',
                scale,
                opacity,
                audioStrength,
                colorFlow
            };
        }
        function syncSpectrumOptionsControlsFromStorage() {
            populateSpectrumColorStreamSelect();
            applySpectrumSettingsToControls(loadSpectrumSettingsFromStorage());
        }
        function clampThemeGradientAngle(raw, fallback = 165) {
            const v = Number(raw);
            if (!Number.isFinite(v)) return fallback;
            return Math.max(0, Math.min(360, Math.round(v)));
        }
        function sortDigitalBgGifNames(names) {
            const list = (names || [])
                .map((f) => String(f || '').trim())
                .filter((f) => /\.gif$/i.test(f));
            const first = list.filter((f) => f.toLowerCase() === 'dig.gif');
            const rest = list.filter((f) => f.toLowerCase() !== 'dig.gif').sort((a, b) =>
                a.localeCompare(b, undefined, { sensitivity: 'base' })
            );
            return [...first, ...rest];
        }
        async function populateDigitalBgGifSelect() {
            if (!optDigitalBgGif) return;
            let files = [];
            try {
                const res = await fetch(`${DIGITAL_BG_GIF_MANIFEST_URL}?t=${Date.now()}`, { cache: 'no-store' });
                if (res.ok) {
                    const data = await res.json();
                    const raw = Array.isArray(data) ? data : (data && (data.gifs || data.files)) || [];
                    files = sortDigitalBgGifNames(raw);
                }
            } catch (_) {}
            const prev = optDigitalBgGif.value;
            optDigitalBgGif.innerHTML = '';
            const noneOpt = document.createElement('option');
            noneOpt.value = '';
            noneOpt.textContent = 'Off — no background GIF';
            optDigitalBgGif.appendChild(noneOpt);
            files.forEach((file) => {
                const opt = document.createElement('option');
                opt.value = file;
                opt.textContent = file.replace(/\.gif$/i, '');
                optDigitalBgGif.appendChild(opt);
            });
            let selected = '';
            try {
                const enabled = localStorage.getItem(DIGITAL_BG_GIF_ENABLED_KEY);
                const stored = localStorage.getItem(DIGITAL_BG_GIF_STORAGE_KEY);
                if (enabled !== '0' && stored) selected = stored;
            } catch (_) {}
            if (selected && files.includes(selected)) optDigitalBgGif.value = selected;
            else if (prev && Array.from(optDigitalBgGif.options).some((o) => o.value === prev)) optDigitalBgGif.value = prev;
            else optDigitalBgGif.value = '';
        }
        function applyDigitalBgGifFromOptions(filename) {
            const name = String(filename || '').trim();
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv.setDigitalBgGifFromOptions === 'function') {
                    rv.setDigitalBgGifFromOptions(name);
                    return;
                }
            } catch (_) {}
            try {
                if (!name) {
                    localStorage.setItem(DIGITAL_BG_GIF_ENABLED_KEY, '0');
                } else {
                    localStorage.setItem(DIGITAL_BG_GIF_STORAGE_KEY, name);
                    localStorage.setItem(DIGITAL_BG_GIF_ENABLED_KEY, '1');
                }
            } catch (_) {}
        function clampThemeFontScale(raw, fallback = 1) {
            const v = Number(raw);
            if (!Number.isFinite(v)) return fallback;
            return Math.max(0.65, Math.min(1.6, Math.round(v * 100) / 100));
        }
        function normalizeDigitalTheme(parsed) {
            const d = DEFAULT_DIGITAL_THEME;
            const src = (parsed && typeof parsed === 'object') ? parsed : {};
            const clockFormat = (src.clockFormat && DIGITAL_CLOCK_FORMATS[src.clockFormat])
                ? src.clockFormat
                : d.clockFormat;
            return {
                presetId: src.presetId || d.presetId,
                bgA: src.bgA || d.bgA,
                bgB: src.bgB || d.bgB,
                bgC: src.bgC || d.bgC,
                bgOuterA: src.bgOuterA || src.bgA || d.bgOuterA,
                bgOuterB: src.bgOuterB || src.bgB || d.bgOuterB,
                bgOuterC: src.bgOuterC || src.bgC || d.bgOuterC,
                bgGradientAngle: clampThemeGradientAngle(src.bgGradientAngle, d.bgGradientAngle),
                accent: src.accent || d.accent,
                font: src.font || d.font,
                btnBlueTop: src.btnBlueTop || d.btnBlueTop,
                btnBlueBase: src.btnBlueBase || d.btnBlueBase,
                btnBlueAccent: src.btnBlueAccent || d.btnBlueAccent,
                btnPurpleTop: src.btnPurpleTop || d.btnPurpleTop,
                btnPurpleBase: src.btnPurpleBase || d.btnPurpleBase,
                btnPurpleLabel: src.btnPurpleLabel || d.btnPurpleLabel,
                btnPurpleActive: src.btnPurpleActive || d.btnPurpleActive,
                btnBlueFontScale: clampThemeFontScale(src.btnBlueFontScale, d.btnBlueFontScale),
                btnPurpleFontScale: clampThemeFontScale(src.btnPurpleFontScale, d.btnPurpleFontScale),
                clockFont: src.clockFont || d.clockFont,
                clockColor: src.clockColor || d.clockColor,
                clockFontScale: clampThemeFontScale(src.clockFontScale, d.clockFontScale),
                clockFormat
            };
        }
        function themeFromPresetId(presetId, fontFallback, themeFallback) {
            const preset = DIGITAL_THEME_PRESETS[presetId];
            if (!preset) return null;
            const prev = normalizeDigitalTheme(themeFallback);
            return normalizeDigitalTheme({
                ...prev,
                presetId,
                bgA: preset.bgA,
                bgB: preset.bgB,
                bgC: preset.bgC,
                bgOuterA: preset.bgA,
                bgOuterB: preset.bgB,
                bgOuterC: preset.bgC,
                accent: preset.accent,
                font: preset.font || fontFallback || prev.font
            });
        }
        function detectDigitalThemePresetId(theme) {
            if (!theme) return 'custom';
            for (const [id, preset] of Object.entries(DIGITAL_THEME_PRESETS)) {
                if (theme.bgA === preset.bgA
                    && theme.bgB === preset.bgB
                    && theme.bgC === preset.bgC
                    && theme.accent === preset.accent) {
                    return id;
                }
            }
            return 'custom';
        }
        function populateDigitalThemePresetSelect() {
            if (!optDigitalThemePreset) return;
            optDigitalThemePreset.innerHTML = '';
            const customOpt = document.createElement('option');
            customOpt.value = 'custom';
            customOpt.textContent = 'Custom';
            optDigitalThemePreset.appendChild(customOpt);
            Object.entries(DIGITAL_THEME_PRESETS).forEach(([id, preset]) => {
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = preset.label;
                optDigitalThemePreset.appendChild(opt);
            });
        }
        function loadDigitalThemeFromStorage() {
            try {
                const raw = localStorage.getItem(DIGITAL_THEME_STORAGE_KEY);
                const parsed = raw ? JSON.parse(raw) : null;
                const theme = normalizeDigitalTheme(parsed);
                theme.presetId = detectDigitalThemePresetId(theme);
                return theme;
            } catch (_) {
                return { ...DEFAULT_DIGITAL_THEME };
            }
        }
        function saveDigitalThemeToStorage(theme) {
            try { localStorage.setItem(DIGITAL_THEME_STORAGE_KEY, JSON.stringify(normalizeDigitalTheme(theme))); } catch (_) {}
        }
        function applyDigitalThemeCssVars(target, t) {
            if (!target) return;
            target.style.setProperty('--rv-digital-bg-a', t.bgA);
            target.style.setProperty('--rv-digital-bg-b', t.bgB);
            target.style.setProperty('--rv-digital-bg-c', t.bgC);
            target.style.setProperty('--rv-digital-bg-outer-a', t.bgOuterA);
            target.style.setProperty('--rv-digital-bg-outer-b', t.bgOuterB);
            target.style.setProperty('--rv-digital-bg-outer-c', t.bgOuterC);
            target.style.setProperty('--rv-digital-bg-gradient-angle', String(t.bgGradientAngle));
            target.style.setProperty('--rv-digital-accent-color', t.accent);
            target.style.setProperty('--rv-digital-ui-font', t.font);
            target.style.setProperty('--rv-digital-btn-blue-top', t.btnBlueTop);
            target.style.setProperty('--rv-digital-btn-blue-base', t.btnBlueBase);
            target.style.setProperty('--rv-digital-btn-blue-accent', t.btnBlueAccent);
            target.style.setProperty('--rv-digital-btn-purple-top', t.btnPurpleTop);
            target.style.setProperty('--rv-digital-btn-purple-base', t.btnPurpleBase);
            target.style.setProperty('--rv-digital-btn-purple-label', t.btnPurpleLabel);
            target.style.setProperty('--rv-digital-btn-purple-active', t.btnPurpleActive);
            target.style.setProperty('--rv-digital-btn-blue-font-scale', String(t.btnBlueFontScale));
            target.style.setProperty('--rv-digital-btn-purple-font-scale', String(t.btnPurpleFontScale));
            target.style.setProperty('--rv-digital-clock-font', t.clockFont);
            target.style.setProperty('--rv-digital-clock-color', t.clockColor);
            target.style.setProperty('--rv-digital-clock-font-scale', String(t.clockFontScale));
            if (target.dataset) target.dataset.rvClockFormat = t.clockFormat;
        }
        function reflowDigitalRadioThemeUi() {
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv._reflowDigitalThemeUi === 'function') rv._reflowDigitalThemeUi();
            } catch (_) {}
        }
        function applyDigitalRadioTheme(theme) {
            const t = normalizeDigitalTheme(theme || loadDigitalThemeFromStorage());
            const root = document.documentElement;
            root.style.setProperty('--global-rv-digital-bg-a', t.bgA);
            root.style.setProperty('--global-rv-digital-bg-b', t.bgB);
            root.style.setProperty('--global-rv-digital-bg-c', t.bgC);
            root.style.setProperty('--global-rv-digital-bg-outer-a', t.bgOuterA);
            root.style.setProperty('--global-rv-digital-bg-outer-b', t.bgOuterB);
            root.style.setProperty('--global-rv-digital-bg-outer-c', t.bgOuterC);
            root.style.setProperty('--global-rv-digital-bg-gradient-angle', String(t.bgGradientAngle));
            root.style.setProperty('--global-rv-digital-accent-color', t.accent);
            root.style.setProperty('--global-rv-digital-ui-font', t.font);
            root.style.setProperty('--global-rv-digital-btn-blue-top', t.btnBlueTop);
            root.style.setProperty('--global-rv-digital-btn-blue-base', t.btnBlueBase);
            root.style.setProperty('--global-rv-digital-btn-blue-accent', t.btnBlueAccent);
            root.style.setProperty('--global-rv-digital-btn-purple-top', t.btnPurpleTop);
            root.style.setProperty('--global-rv-digital-btn-purple-base', t.btnPurpleBase);
            root.style.setProperty('--global-rv-digital-btn-purple-label', t.btnPurpleLabel);
            root.style.setProperty('--global-rv-digital-btn-purple-active', t.btnPurpleActive);
            root.style.setProperty('--global-rv-digital-btn-blue-font-scale', String(t.btnBlueFontScale));
            root.style.setProperty('--global-rv-digital-btn-purple-font-scale', String(t.btnPurpleFontScale));
            root.style.setProperty('--global-rv-digital-clock-font', t.clockFont);
            root.style.setProperty('--global-rv-digital-clock-color', t.clockColor);
            root.style.setProperty('--global-rv-digital-clock-font-scale', String(t.clockFontScale));
            applyDigitalThemeCssVars(document.getElementById('radio-visual-root'), t);
            reflowDigitalRadioThemeUi();
        }
        function readAutoMixMaxMinFromStorage() {
            try {
                const v = Number(localStorage.getItem('dj.automix.max.min.v1'));
                if (Number.isFinite(v)) return Math.max(1, Math.min(20, Math.round(v)));
            } catch (_) {}
            return 20;
        }
        function writeAutoMixMaxMinToStorage(mins) {
            const v = Math.max(1, Math.min(20, Math.round(Number(mins) || 20)));
            try { localStorage.setItem('dj.automix.max.min.v1', String(v)); } catch (_) {}
            return v;
        }
        function readAutoFadeDurationMsFromStorage() {
            try {
                const v = Number(localStorage.getItem('dj.autofade.duration.ms.v1'));
                if (Number.isFinite(v)) return Math.max(2000, Math.min(15000, v));
            } catch (_) {}
            return 5000;
        }
        function writeAutoFadeDurationMsToStorage(ms) {
            const v = Math.max(2000, Math.min(15000, Math.round(Number(ms) || 5000)));
            try { localStorage.setItem('dj.autofade.duration.ms.v1', String(v)); } catch (_) {}
            return v;
        }
        function syncOptionsPanelControlsFromStorage() {
            const theme = loadDigitalThemeFromStorage();
            if (optDigitalThemePreset) optDigitalThemePreset.value = theme.presetId || detectDigitalThemePresetId(theme);
            if (optDigitalBgA) optDigitalBgA.value = theme.bgA;
            if (optDigitalBgB) optDigitalBgB.value = theme.bgB;
            if (optDigitalBgC) optDigitalBgC.value = theme.bgC;
            if (optDigitalBgOuterA) optDigitalBgOuterA.value = theme.bgOuterA;
            if (optDigitalBgOuterB) optDigitalBgOuterB.value = theme.bgOuterB;
            if (optDigitalBgOuterC) optDigitalBgOuterC.value = theme.bgOuterC;
            if (optDigitalBgGradientAngle) optDigitalBgGradientAngle.value = String(theme.bgGradientAngle);
            if (optDigitalBgGradientAngleReadout) optDigitalBgGradientAngleReadout.textContent = `${theme.bgGradientAngle}°`;
            if (optDigitalAccent) optDigitalAccent.value = theme.accent;
            if (optDigitalFont) optDigitalFont.value = theme.font;
            if (optDigitalBtnBlueTop) optDigitalBtnBlueTop.value = theme.btnBlueTop;
            if (optDigitalBtnBlueBase) optDigitalBtnBlueBase.value = theme.btnBlueBase;
            if (optDigitalBtnBlueAccent) optDigitalBtnBlueAccent.value = theme.btnBlueAccent;
            if (optDigitalBtnPurpleTop) optDigitalBtnPurpleTop.value = theme.btnPurpleTop;
            if (optDigitalBtnPurpleBase) optDigitalBtnPurpleBase.value = theme.btnPurpleBase;
            if (optDigitalBtnPurpleLabel) optDigitalBtnPurpleLabel.value = theme.btnPurpleLabel;
            if (optDigitalBtnPurpleActive) optDigitalBtnPurpleActive.value = theme.btnPurpleActive;
            if (optDigitalBtnBlueFontScale) optDigitalBtnBlueFontScale.value = String(Math.round(theme.btnBlueFontScale * 100));
            if (optDigitalBtnBlueFontScaleReadout) optDigitalBtnBlueFontScaleReadout.textContent = `${Math.round(theme.btnBlueFontScale * 100)}%`;
            if (optDigitalBtnPurpleFontScale) optDigitalBtnPurpleFontScale.value = String(Math.round(theme.btnPurpleFontScale * 100));
            if (optDigitalBtnPurpleFontScaleReadout) optDigitalBtnPurpleFontScaleReadout.textContent = `${Math.round(theme.btnPurpleFontScale * 100)}%`;
            if (optDigitalClockFormat) optDigitalClockFormat.value = theme.clockFormat;
            if (optDigitalClockFont) optDigitalClockFont.value = theme.clockFont;
            if (optDigitalClockColor) optDigitalClockColor.value = theme.clockColor;
            if (optDigitalClockFontScale) optDigitalClockFontScale.value = String(Math.round(theme.clockFontScale * 100));
            if (optDigitalClockFontScaleReadout) optDigitalClockFontScaleReadout.textContent = `${Math.round(theme.clockFontScale * 100)}%`;
            const maxMin = readAutoMixMaxMinFromStorage();
            if (optAutomixMax) optAutomixMax.value = String(maxMin);
            if (optAutomixMaxReadout) optAutomixMaxReadout.textContent = `${maxMin}m`;
            let automixOn = false;
            try { automixOn = localStorage.getItem(AUTOMIX_ENABLED_STORAGE_KEY) === '1'; } catch (_) {}
            try { automixOn = automixOn || !!(state && state.autoMixEnabled); } catch (_) {}
            if (optAutomixEnabled) optAutomixEnabled.checked = automixOn;
            const fadeMs = readAutoFadeDurationMsFromStorage();
            if (optAutofadeDuration) optAutofadeDuration.value = String(fadeMs / 1000);
            if (optAutofadeDurationReadout) optAutofadeDurationReadout.textContent = `${(fadeMs / 1000).toFixed(1)}s`;
            let changeStation = true;
            try {
                const raw = localStorage.getItem(AUTOFADE_CHANGE_STATION_STORAGE_KEY);
                if (raw != null) changeStation = raw === '1';
            } catch (_) {}
            if (optAutofadeChangeStation) optAutofadeChangeStation.checked = changeStation;
            try { syncSpectrumOptionsControlsFromStorage(); } catch (_) {}
            try { populateDigitalBgGifSelect(); } catch (_) {}
        }
        function syncRadioVisualMixPanelsFromOptions() {
            try {
                const rv = getActiveRadioVisualEngine();
                if (!rv) return;
                if (typeof rv._syncDigitalAutoMixPanelUi === 'function') rv._syncDigitalAutoMixPanelUi();
                if (typeof rv._syncDigitalAutoFadePanelUi === 'function') rv._syncDigitalAutoFadePanelUi();
                if (typeof rv._syncAutoMixKnob === 'function') rv._syncAutoMixKnob();
                if (typeof rv._syncAutoFadeChangeStationKnob === 'function') rv._syncAutoFadeChangeStationKnob();
                if (typeof rv._autoFadeDurationNorm === 'function' && typeof rv._setAutoFadeDurationNorm === 'function') {
                    rv._setAutoFadeDurationNorm(rv._autoFadeDurationNorm());
                }
            } catch (_) {}
        }
        function collectDigitalThemeFromControls() {
            const theme = {
                bgA: optDigitalBgA ? optDigitalBgA.value : DEFAULT_DIGITAL_THEME.bgA,
                bgB: optDigitalBgB ? optDigitalBgB.value : DEFAULT_DIGITAL_THEME.bgB,
                bgC: optDigitalBgC ? optDigitalBgC.value : DEFAULT_DIGITAL_THEME.bgC,
                bgOuterA: optDigitalBgOuterA ? optDigitalBgOuterA.value : DEFAULT_DIGITAL_THEME.bgOuterA,
                bgOuterB: optDigitalBgOuterB ? optDigitalBgOuterB.value : DEFAULT_DIGITAL_THEME.bgOuterB,
                bgOuterC: optDigitalBgOuterC ? optDigitalBgOuterC.value : DEFAULT_DIGITAL_THEME.bgOuterC,
                bgGradientAngle: clampThemeGradientAngle(
                    optDigitalBgGradientAngle ? optDigitalBgGradientAngle.value : DEFAULT_DIGITAL_THEME.bgGradientAngle,
                    DEFAULT_DIGITAL_THEME.bgGradientAngle
                ),
                accent: optDigitalAccent ? optDigitalAccent.value : DEFAULT_DIGITAL_THEME.accent,
                font: optDigitalFont ? optDigitalFont.value : DEFAULT_DIGITAL_THEME.font,
                btnBlueTop: optDigitalBtnBlueTop ? optDigitalBtnBlueTop.value : DEFAULT_DIGITAL_THEME.btnBlueTop,
                btnBlueBase: optDigitalBtnBlueBase ? optDigitalBtnBlueBase.value : DEFAULT_DIGITAL_THEME.btnBlueBase,
                btnBlueAccent: optDigitalBtnBlueAccent ? optDigitalBtnBlueAccent.value : DEFAULT_DIGITAL_THEME.btnBlueAccent,
                btnPurpleTop: optDigitalBtnPurpleTop ? optDigitalBtnPurpleTop.value : DEFAULT_DIGITAL_THEME.btnPurpleTop,
                btnPurpleBase: optDigitalBtnPurpleBase ? optDigitalBtnPurpleBase.value : DEFAULT_DIGITAL_THEME.btnPurpleBase,
                btnPurpleLabel: optDigitalBtnPurpleLabel ? optDigitalBtnPurpleLabel.value : DEFAULT_DIGITAL_THEME.btnPurpleLabel,
                btnPurpleActive: optDigitalBtnPurpleActive ? optDigitalBtnPurpleActive.value : DEFAULT_DIGITAL_THEME.btnPurpleActive,
                btnBlueFontScale: clampThemeFontScale(
                    (Number(optDigitalBtnBlueFontScale && optDigitalBtnBlueFontScale.value) || 100) / 100,
                    DEFAULT_DIGITAL_THEME.btnBlueFontScale
                ),
                btnPurpleFontScale: clampThemeFontScale(
                    (Number(optDigitalBtnPurpleFontScale && optDigitalBtnPurpleFontScale.value) || 100) / 100,
                    DEFAULT_DIGITAL_THEME.btnPurpleFontScale
                ),
                clockFont: optDigitalClockFont ? optDigitalClockFont.value : DEFAULT_DIGITAL_THEME.clockFont,
                clockColor: optDigitalClockColor ? optDigitalClockColor.value : DEFAULT_DIGITAL_THEME.clockColor,
                clockFontScale: clampThemeFontScale(
                    (Number(optDigitalClockFontScale && optDigitalClockFontScale.value) || 100) / 100,
                    DEFAULT_DIGITAL_THEME.clockFontScale
                ),
                clockFormat: (optDigitalClockFormat && DIGITAL_CLOCK_FORMATS[optDigitalClockFormat.value])
                    ? optDigitalClockFormat.value
                    : DEFAULT_DIGITAL_THEME.clockFormat
            };
            theme.presetId = detectDigitalThemePresetId(theme);
            if (optDigitalThemePreset) optDigitalThemePreset.value = theme.presetId;
            return normalizeDigitalTheme(theme);
        }
        function applyDigitalThemeToControls(theme) {
            const t = normalizeDigitalTheme(theme);
            if (optDigitalBgA) optDigitalBgA.value = t.bgA;
            if (optDigitalBgB) optDigitalBgB.value = t.bgB;
            if (optDigitalBgC) optDigitalBgC.value = t.bgC;
            if (optDigitalBgOuterA) optDigitalBgOuterA.value = t.bgOuterA;
            if (optDigitalBgOuterB) optDigitalBgOuterB.value = t.bgOuterB;
            if (optDigitalBgOuterC) optDigitalBgOuterC.value = t.bgOuterC;
            if (optDigitalBgGradientAngle) optDigitalBgGradientAngle.value = String(t.bgGradientAngle);
            if (optDigitalBgGradientAngleReadout) optDigitalBgGradientAngleReadout.textContent = `${t.bgGradientAngle}°`;
            if (optDigitalAccent) optDigitalAccent.value = t.accent;
            if (optDigitalFont && t.font) optDigitalFont.value = t.font;
            if (optDigitalBtnBlueTop) optDigitalBtnBlueTop.value = t.btnBlueTop;
            if (optDigitalBtnBlueBase) optDigitalBtnBlueBase.value = t.btnBlueBase;
            if (optDigitalBtnBlueAccent) optDigitalBtnBlueAccent.value = t.btnBlueAccent;
            if (optDigitalBtnPurpleTop) optDigitalBtnPurpleTop.value = t.btnPurpleTop;
            if (optDigitalBtnPurpleBase) optDigitalBtnPurpleBase.value = t.btnPurpleBase;
            if (optDigitalBtnPurpleLabel) optDigitalBtnPurpleLabel.value = t.btnPurpleLabel;
            if (optDigitalBtnPurpleActive) optDigitalBtnPurpleActive.value = t.btnPurpleActive;
            if (optDigitalBtnBlueFontScale) optDigitalBtnBlueFontScale.value = String(Math.round(t.btnBlueFontScale * 100));
            if (optDigitalBtnBlueFontScaleReadout) optDigitalBtnBlueFontScaleReadout.textContent = `${Math.round(t.btnBlueFontScale * 100)}%`;
            if (optDigitalBtnPurpleFontScale) optDigitalBtnPurpleFontScale.value = String(Math.round(t.btnPurpleFontScale * 100));
            if (optDigitalBtnPurpleFontScaleReadout) optDigitalBtnPurpleFontScaleReadout.textContent = `${Math.round(t.btnPurpleFontScale * 100)}%`;
            if (optDigitalClockFormat) optDigitalClockFormat.value = t.clockFormat;
            if (optDigitalClockFont) optDigitalClockFont.value = t.clockFont;
            if (optDigitalClockColor) optDigitalClockColor.value = t.clockColor;
            if (optDigitalClockFontScale) optDigitalClockFontScale.value = String(Math.round(t.clockFontScale * 100));
            if (optDigitalClockFontScaleReadout) optDigitalClockFontScaleReadout.textContent = `${Math.round(t.clockFontScale * 100)}%`;
            if (optDigitalThemePreset) optDigitalThemePreset.value = t.presetId || detectDigitalThemePresetId(t);
        }
        function syncDigitalThemeScaleReadoutsFromControls() {
            if (optDigitalBtnBlueFontScale && optDigitalBtnBlueFontScaleReadout) {
                optDigitalBtnBlueFontScaleReadout.textContent = `${optDigitalBtnBlueFontScale.value}%`;
            }
            if (optDigitalBtnPurpleFontScale && optDigitalBtnPurpleFontScaleReadout) {
                optDigitalBtnPurpleFontScaleReadout.textContent = `${optDigitalBtnPurpleFontScale.value}%`;
            }
            if (optDigitalClockFontScale && optDigitalClockFontScaleReadout) {
                optDigitalClockFontScaleReadout.textContent = `${optDigitalClockFontScale.value}%`;
            }
            if (optDigitalBgGradientAngle && optDigitalBgGradientAngleReadout) {
                optDigitalBgGradientAngleReadout.textContent = `${optDigitalBgGradientAngle.value}°`;
            }
        }
        function isOptionsOpen() {
            return !!(optionsPanel && !optionsPanel.classList.contains('display-none') && optionsPanel.classList.contains('show'));
        }
        let optionsAutoCloseId = null;
        function armOptionsAutoClose() {
            if (optionsAutoCloseId) { clearTimeout(optionsAutoCloseId); optionsAutoCloseId = null; }
            optionsAutoCloseId = setTimeout(() => { closeOptionsPanel(); }, OPTIONS_AUTO_CLOSE_MS);
        }
        function openOptionsPanel() {
            if (uiLocked) return;
            if (!optionsPanel) return;
            try { syncOptionsPanelControlsFromStorage(); } catch (_) {}
            optionsPanel.classList.remove('display-none');
            requestAnimationFrame(() => optionsPanel.classList.add('show'));
            armOptionsAutoClose();
        }
        function closeOptionsPanel() {
            if (!optionsPanel) return;
            if (optionsAutoCloseId) { clearTimeout(optionsAutoCloseId); optionsAutoCloseId = null; }
            optionsPanel.classList.remove('show');
            setTimeout(() => optionsPanel.classList.add('display-none'), 350);
        }
        function toggleOptionsPanel() {
            if (isOptionsOpen()) closeOptionsPanel(); else openOptionsPanel();
        }
        if (btnOptions) {
            btnOptions.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleOptionsPanel();
            });
        }
        if (btnOptionsClose) btnOptionsClose.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); closeOptionsPanel(); });
        if (optionsPanel) {
            optionsPanel.addEventListener('pointerdown', () => { if (isOptionsOpen()) armOptionsAutoClose(); });
            optionsPanel.addEventListener('input', () => { if (isOptionsOpen()) armOptionsAutoClose(); });
            optionsPanel.addEventListener('mousemove', () => { if (isOptionsOpen()) armOptionsAutoClose(); });
        }
        function wireOptionsPanelControls() {
            populateDigitalThemePresetSelect();
            const onThemeChange = () => {
                syncDigitalThemeScaleReadoutsFromControls();
                const theme = collectDigitalThemeFromControls();
                saveDigitalThemeToStorage(theme);
                applyDigitalRadioTheme(theme);
            };
            if (optDigitalThemePreset) {
                optDigitalThemePreset.addEventListener('change', () => {
                    const presetId = optDigitalThemePreset.value;
                    if (presetId === 'custom') return;
                    const currentFont = optDigitalFont ? optDigitalFont.value : DEFAULT_DIGITAL_THEME.font;
                    const currentButtons = loadDigitalThemeFromStorage();
                    const theme = themeFromPresetId(presetId, currentFont, currentButtons);
                    if (!theme) return;
                    applyDigitalThemeToControls(theme);
                    saveDigitalThemeToStorage(theme);
                    applyDigitalRadioTheme(theme);
                });
            }
            [optDigitalBgA, optDigitalBgB, optDigitalBgC, optDigitalAccent,
                optDigitalBgOuterA, optDigitalBgOuterB, optDigitalBgOuterC, optDigitalBgGradientAngle,
                optDigitalBtnBlueTop, optDigitalBtnBlueBase, optDigitalBtnBlueAccent,
                optDigitalBtnPurpleTop, optDigitalBtnPurpleBase, optDigitalBtnPurpleLabel, optDigitalBtnPurpleActive,
                optDigitalBtnBlueFontScale, optDigitalBtnPurpleFontScale,
                optDigitalClockColor, optDigitalClockFontScale
            ].forEach((el) => {
                if (!el) return;
                el.addEventListener('input', onThemeChange);
                el.addEventListener('change', onThemeChange);
            });
            if (optDigitalClockFormat) optDigitalClockFormat.addEventListener('change', onThemeChange);
            if (optDigitalClockFont) optDigitalClockFont.addEventListener('change', onThemeChange);
            if (optDigitalFont) {
                optDigitalFont.addEventListener('change', onThemeChange);
            }
            if (optDigitalBgGif) {
                optDigitalBgGif.addEventListener('change', () => {
                    applyDigitalBgGifFromOptions(optDigitalBgGif.value);
                    if (isOptionsOpen()) armOptionsAutoClose();
                });
            }
            if (optDigitalThemeReset) {
                optDigitalThemeReset.addEventListener('click', (e) => {
                    e.preventDefault();
                    saveDigitalThemeToStorage(DEFAULT_DIGITAL_THEME);
                    applyDigitalRadioTheme(DEFAULT_DIGITAL_THEME);
                    syncOptionsPanelControlsFromStorage();
                });
            }
            const onSpectrumSettingsChange = () => {
                const settings = collectSpectrumSettingsFromControls();
                applySpectrumSettingsToControls(settings);
                applyDigitalSpectrumSettings(settings);
            };
            if (optSpectrumColorStream) {
                optSpectrumColorStream.addEventListener('change', onSpectrumSettingsChange);
            }
            if (optSpectrumSize) {
                optSpectrumSize.addEventListener('input', onSpectrumSettingsChange);
                optSpectrumSize.addEventListener('change', onSpectrumSettingsChange);
            }
            if (optSpectrumOpacity) {
                optSpectrumOpacity.addEventListener('input', onSpectrumSettingsChange);
                optSpectrumOpacity.addEventListener('change', onSpectrumSettingsChange);
            }
            if (optSpectrumAudioStrength) {
                optSpectrumAudioStrength.addEventListener('input', onSpectrumSettingsChange);
                optSpectrumAudioStrength.addEventListener('change', onSpectrumSettingsChange);
            }
            if (optSpectrumColorFlow) {
                optSpectrumColorFlow.addEventListener('input', onSpectrumSettingsChange);
                optSpectrumColorFlow.addEventListener('change', onSpectrumSettingsChange);
            }
            if (optSpectrumReset) {
                optSpectrumReset.addEventListener('click', (e) => {
                    e.preventDefault();
                    const RVE = getSpectrumEngineClass();
                    const defaults = (RVE && RVE.DEFAULT_SPECTRUM_SETTINGS)
                        ? { ...RVE.DEFAULT_SPECTRUM_SETTINGS }
                        : {
                            colorStreamId: 'aurora',
                            scale: 1,
                            opacity: 1,
                            audioStrength: 1,
                            colorFlow: 1
                        };
                    applySpectrumSettingsToControls(defaults);
                    applyDigitalSpectrumSettings(defaults);
                });
            }
            if (optAutomixMax) {
                const syncAutomixMax = () => {
                    const mins = writeAutoMixMaxMinToStorage(optAutomixMax.value);
                    if (optAutomixMaxReadout) optAutomixMaxReadout.textContent = `${mins}m`;
                    syncRadioVisualMixPanelsFromOptions();
                };
                optAutomixMax.addEventListener('input', syncAutomixMax);
                optAutomixMax.addEventListener('change', syncAutomixMax);
            }
            if (optAutomixEnabled) {
                optAutomixEnabled.addEventListener('change', () => {
                    const next = !!optAutomixEnabled.checked;
                    let currentlyOn = false;
                    try { currentlyOn = localStorage.getItem(AUTOMIX_ENABLED_STORAGE_KEY) === '1'; } catch (_) {}
                    try { currentlyOn = currentlyOn || !!(state && state.autoMixEnabled); } catch (_) {}
                    if (currentlyOn === next) return;
                    const rv = getActiveRadioVisualEngine();
                    if (rv && typeof rv._toggleAutoMix === 'function') {
                        rv._toggleAutoMix();
                    } else {
                        const btn = document.getElementById('mix-automix') || document.getElementById('dj-automix');
                        if (btn) btn.click();
                        else {
                            try { localStorage.setItem(AUTOMIX_ENABLED_STORAGE_KEY, next ? '1' : '0'); } catch (_) {}
                            try { state.autoMixEnabled = next; } catch (_) {}
                        }
                    }
                    syncRadioVisualMixPanelsFromOptions();
                });
            }
            if (optAutofadeDuration) {
                const syncFade = () => {
                    const ms = writeAutoFadeDurationMsToStorage(Number(optAutofadeDuration.value) * 1000);
                    if (optAutofadeDurationReadout) optAutofadeDurationReadout.textContent = `${(ms / 1000).toFixed(1)}s`;
                    syncRadioVisualMixPanelsFromOptions();
                };
                optAutofadeDuration.addEventListener('input', syncFade);
                optAutofadeDuration.addEventListener('change', syncFade);
            }
            if (optAutofadeChangeStation) {
                optAutofadeChangeStation.addEventListener('change', () => {
                    const next = !!optAutofadeChangeStation.checked;
                    let currently = true;
                    try {
                        const raw = localStorage.getItem(AUTOFADE_CHANGE_STATION_STORAGE_KEY);
                        if (raw != null) currently = raw === '1';
                    } catch (_) {}
                    if (currently === next) return;
                    const rv = getActiveRadioVisualEngine();
                    if (rv && typeof rv._toggleAutoFadeChangeStation === 'function') {
                        rv._toggleAutoFadeChangeStation();
                    } else {
                        try { localStorage.setItem(AUTOFADE_CHANGE_STATION_STORAGE_KEY, next ? '1' : '0'); } catch (_) {}
                        const cb = document.getElementById('dj-autofade-change-station');
                        if (cb) {
                            cb.checked = next;
                            try { cb.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
                        }
                    }
                    syncRadioVisualMixPanelsFromOptions();
                });
            }
        }
        applyDigitalRadioTheme(loadDigitalThemeFromStorage());
        wireOptionsPanelControls();

        const UI_HUD_POSITION_KEY = 'ui.hud.position.v1';

        function isDigitalRadioHudTapActive() {
            try {
                const av = state && state.activeVisualizer;
                return !!(av && av.name === 'Digital Radio' && av.skin === 'digital');
            } catch (_) {
                return false;
            }
        }

        function syncModeInfoHudHintForDigitalRadio() {
            const info = document.getElementById('mode-info');
            if (!info) return;
            if (!isDigitalRadioHudTapActive()) {
                info.title = 'Hold for options';
                info.setAttribute('aria-label', 'Visual mode and station. Hold for options.');
                return;
            }
            const atTop = document.documentElement.classList.contains('ui-hud-at-top');
            const hint = atTop
                ? 'Tap title: move HUD to bottom · Hold: options'
                : 'Tap title: move HUD to top · Hold: options';
            info.title = hint;
            info.setAttribute('aria-label', hint);
        }

        function applyUiHudPosition(atTop) {
            document.documentElement.classList.toggle('ui-hud-at-top', !!atTop);
            try { localStorage.setItem(UI_HUD_POSITION_KEY, atTop ? 'top' : 'bottom'); } catch (_) {}
            syncModeInfoHudHintForDigitalRadio();
        }

        function toggleUiHudPosition() {
            applyUiHudPosition(!document.documentElement.classList.contains('ui-hud-at-top'));
            try { resetIdleTimer(); } catch (_) {}
        }

        function initUiHudPosition() {
            let atTop = false;
            try { atTop = localStorage.getItem(UI_HUD_POSITION_KEY) === 'top'; } catch (_) {}
            document.documentElement.classList.toggle('ui-hud-at-top', atTop);
            syncModeInfoHudHintForDigitalRadio();
        }

        function wireModeInfoOptionsLongPress() {
            const info = document.getElementById('mode-info');
            if (!info || info.dataset.modeInfoHoldWired === '1') return;
            info.dataset.modeInfoHoldWired = '1';
            const HOLD_MS = 500;
            let holdTimer = null;
            let pointerDown = false;
            let holdFired = false;
            let downAt = 0;
            const clearHold = () => {
                if (holdTimer) {
                    clearTimeout(holdTimer);
                    holdTimer = null;
                }
            };
            info.addEventListener('pointerdown', (e) => {
                if (uiLocked) return;
                if (e.button !== 0) return;
                try { e.stopPropagation(); } catch (_) {}
                pointerDown = true;
                holdFired = false;
                downAt = performance.now();
                clearHold();
                holdTimer = setTimeout(() => {
                    holdTimer = null;
                    if (!pointerDown) return;
                    holdFired = true;
                    try { toggleOptionsPanel(); } catch (_) {}
                    try { resetIdleTimer(); } catch (_) {}
                }, HOLD_MS);
            });
            info.addEventListener('pointerup', (e) => {
                try { e.stopPropagation(); } catch (_) {}
                const held = downAt ? (performance.now() - downAt) : HOLD_MS;
                pointerDown = false;
                clearHold();
                if (!holdFired && held < HOLD_MS && isDigitalRadioHudTapActive()) {
                    try { toggleUiHudPosition(); } catch (_) {}
                }
            });
            info.addEventListener('pointercancel', () => {
                pointerDown = false;
                holdFired = false;
                downAt = 0;
                clearHold();
            });
            info.addEventListener('click', (e) => {
                try {
                    e.preventDefault();
                    e.stopPropagation();
                } catch (_) {}
            });
        }
        initUiHudPosition();
        wireModeInfoOptionsLongPress();
        // Update Avatar play/stop button label helper
        function updateAvatarPlayButton() {
            try {
                const btn = document.getElementById('avatar-webm-open');
                if (!btn) return;
                btn.textContent = (typeof webmOn !== 'undefined' && webmOn) ? '■' : '▶';
                btn.title = (webmOn ? 'Stop WebM' : 'Play WebM');
            } catch(_) {}
        }
        // Beat worklet helpers (lazy load/create)
        function ensureBeatModuleLoaded(cb) {
            try {
                if (!state || !state.audioCtx) return;
                if (state._beatModuleLoaded) { cb && cb(); return; }
                state.audioCtx.audioWorklet.addModule('beat-worklet.js').then(()=>{ state._beatModuleLoaded = true; cb && cb(); }).catch(()=>{});
            } catch(_) {}
        }
        function startBeatA() {
            try {
                if (!state || !state.audioCtx || state.beatNodeA) return;
                const make = () => {
                    if (state.beatNodeA) return;
                    state.beatNodeA = new AudioWorkletNode(state.audioCtx, 'beat-detector', { numberOfInputs: 1, numberOfOutputs: 0, processorOptions: { channelId: 'A' } });
                    state.trimA && state.trimA.connect(state.beatNodeA);
                    state.beatNodeA.port.onmessage = (ev) => {
                        const d = ev.data || {};
                        if (d.type === 'onset') {
                            try { lastBeatTsA = d.t * 1000; } catch(_) {}
                        } else if (d.type === 'tempo') {
                            bpmSmoothA = d.bpm;
                            nextBeatTsA = (d.next || 0) * 1000;
                            if (bpmOutA) bpmOutA.textContent = bpmSmoothA ? String(Math.round(bpmSmoothA)) : '--';
                            try { applyRhythmCutGateRate(); } catch (_) {}
                        }
                    };
                };
                if (!state._beatModuleLoaded) ensureBeatModuleLoaded(make); else make();
            } catch(_) {}
        }
        function stopBeatA() {
            try {
                if (state && state.beatNodeA) {
                    try { state.trimA && state.trimA.disconnect(state.beatNodeA); } catch(_) {}
                    try { state.beatNodeA.port.onmessage = null; } catch(_) {}
                    state.beatNodeA = null;
                }
            } catch(_) {}
        }
        function startBeatB() {
            try {
                if (!state || !state.audioCtx || state.beatNodeB) return;
                const make = () => {
                    if (state.beatNodeB) return;
                    state.beatNodeB = new AudioWorkletNode(state.audioCtx, 'beat-detector', { numberOfInputs: 1, numberOfOutputs: 0, processorOptions: { channelId: 'B' } });
                    state.trimB && state.trimB.connect(state.beatNodeB);
                    state.beatNodeB.port.onmessage = (ev) => {
                        const d = ev.data || {};
                        if (d.type === 'onset') {
                            try { lastBeatTsB = d.t * 1000; } catch(_) {}
                        } else if (d.type === 'tempo') {
                            bpmSmoothB = d.bpm;
                            nextBeatTsB = (d.next || 0) * 1000;
                            if (bpmOutB) bpmOutB.textContent = bpmSmoothB ? String(Math.round(bpmSmoothB)) : '--';
                            try { applyRhythmCutGateRate(); } catch (_) {}
                        }
                    };
                };
                if (!state._beatModuleLoaded) ensureBeatModuleLoaded(make); else make();
            } catch(_) {}
        }
        function stopBeatB() {
            try {
                if (state && state.beatNodeB) {
                    try { state.trimB && state.trimB.disconnect(state.beatNodeB); } catch(_) {}
                    try { state.beatNodeB.port.onmessage = null; } catch(_) {}
                    state.beatNodeB = null;
                }
            } catch(_) {}
        }
        const mixCross = document.getElementById('mix-crossfader');
        // Sample player
        const audioElSample = new Audio();
        audioElSample.preload = 'auto';
        audioElSample.src = 'assets/audio/wav.mp3';
        const audioElSample1 = new Audio();
        audioElSample1.preload = 'auto';
        audioElSample1.src = 'assets/audio/wav1.mp3';
        const audioElSample2 = new Audio();
        audioElSample2.preload = 'auto';
        audioElSample2.src = 'assets/audio/wav2.mp3';
        const audioElSample3 = new Audio();
        audioElSample3.preload = 'auto';
        audioElSample3.src = 'assets/audio/wav3.mp3';
        const audioElSample4 = new Audio();
        audioElSample4.preload = 'auto';
        audioElSample4.src = 'assets/audio/wav4.mp3';
        const audioElSample5 = new Audio();
        audioElSample5.preload = 'auto';
        audioElSample5.src = 'assets/audio/wav5.mp3';
        const audioElSample6 = new Audio();
        audioElSample6.preload = 'auto';
        audioElSample6.src = 'assets/audio/wav6.mp3';
        let sampleLoop = false, sampleLoop1 = false, sampleLoop2 = false, sampleLoop3 = false;
        try {
            window.sampleLoop4 = false;
            window.sampleLoop5 = false;
            window.sampleLoop6 = false;
        } catch (_) {}
        // Ensure media elements are unmuted; mixing is via Web Audio gain nodes (streamA/B + crossfader).
        try { audioEl.muted = false; } catch(_) {}
        try { if (audioElRadioAAlt) audioElRadioAAlt.muted = false; } catch (_) {}
        try { audioElB.muted = false; } catch(_) {}
        try { audioElSample.muted = false; } catch(_) {}
        try { audioElSample1.muted = false; } catch(_) {}
        try { audioElSample2.muted = false; } catch(_) {}
        try { audioElSample3.muted = false; } catch(_) {}
        try { audioElSample4.muted = false; } catch(_) {}
        try { audioElSample5.muted = false; } catch(_) {}
        try { audioElSample6.muted = false; } catch(_) {}
        try { audioEl.volume = 1; } catch(_) {}
        try { if (audioElRadioAAlt) audioElRadioAAlt.volume = 1; } catch (_) {}
        try { audioElB.volume = 1; } catch(_) {}
        try { audioElSample.volume = 1; } catch(_) {}
        try { audioElSample1.volume = 1; } catch(_) {}
        try { audioElSample2.volume = 1; } catch(_) {}
        try { audioElSample3.volume = 1; } catch(_) {}
        try { audioElSample4.volume = 1; } catch(_) {}
        try { audioElSample5.volume = 1; } catch(_) {}
        try { audioElSample6.volume = 1; } catch(_) {}
        function updateMixBStatus() {
            // Check the active dynamic audio object first, fallback to the static element
            const activeB = state.audioElB || audioElB;
            
            const playing = !!(
                (activeB && !activeB.paused && !activeB.ended) ||
                (audioElSample && !audioElSample.paused && !audioElSample.ended) ||
                (audioElSample1 && !audioElSample1.paused && !audioElSample1.ended) ||
                (audioElSample2 && !audioElSample2.paused && !audioElSample2.ended) ||
                (audioElSample3 && !audioElSample3.paused && !audioElSample3.ended) ||
                (audioElSample4 && !audioElSample4.paused && !audioElSample4.ended) ||
                (audioElSample5 && !audioElSample5.paused && !audioElSample5.ended) ||
                (audioElSample6 && !audioElSample6.paused && !audioElSample6.ended)
            );
            try { syncTopMenuStationsLayout(); } catch (_) {}
        }
        const inpShuffleMin = document.getElementById('inp-shuffle-min');
        const inpShuffleMax = document.getElementById('inp-shuffle-max');
        const inpTransition = document.getElementById('inp-transition');
        const inpPixelRatio = document.getElementById('inp-pixelratio');
        const webmOverlayEl = document.getElementById('webm-overlay');
        const webmVideoEl = document.getElementById('webm-video');
        const webmVideoLeftEl = document.getElementById('webm-video-left');
        const webmVideoRightEl = document.getElementById('webm-video-right');
        const webmBtn = document.getElementById('btn-webm');
        const webmPrevBtn = document.getElementById('webm-prev');
        const webmNextBtn = document.getElementById('webm-next');
        const webmSettingsPanel = document.getElementById('webm-settings-panel');
        const webmCloseBtn = document.getElementById('btn-webm-close');
        const inpWebmScale = document.getElementById('inp-webm-scale');
        const inpWebmX = document.getElementById('inp-webm-x');
        const inpWebmY = document.getElementById('inp-webm-y');
        const inpWebmRot = document.getElementById('inp-webm-rot');
        const inpWebmSpeed = document.getElementById('inp-webm-speed');
        const inpWebmOpacity = document.getElementById('inp-webm-opacity');
        const inpWebmDupSpacing = document.getElementById('inp-webm-dup-spacing');
        const stations = [];
        const stationCycleEnabledByUrl = new Map();
        const sanitizeUrlForAudio = (v) => String(v || '').trim();
        function deriveNameFromUrl(url) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean) return 'URL source';
            try {
                const u = new URL(clean);
                const bits = (u.pathname || '').split('/').filter(Boolean);
                const last = bits.length ? decodeURIComponent(bits[bits.length - 1]) : '';
                return last || (u.hostname || clean);
            } catch (_) {
                const cut = clean.split('?')[0];
                const bits = cut.split('/').filter(Boolean);
                return bits.length ? bits[bits.length - 1] : clean;
            }
        }
        /** Heuristic: streaming radio / Icecast / playlist endpoints vs direct media pages. */
        function isLikelyRadioStreamUrl(url) {
            const v = sanitizeUrlForAudio(url).toLowerCase();
            if (!v) return false;
            if (/youtube\.com|youtu\.be|vimeo\.com|twitch\.tv|facebook\.com\/watch|tiktok\.com/i.test(v)) return false;
            if (/\.(mp4|webm|mkv|mov|m4v|avi)(\?|#|$)/i.test(v)) return false;
            if (/\.(pls|m3u)(\?|#|$)/i.test(v)) return true;
            if (/\.m3u8(\?|#|$)/i.test(v)) return false;
            if (/\/stream|\/live\/|\/listen|icecast|shoutcast|\/;\s*stream|(^|\/)radio\.|broadcast\.|tune\.in/i.test(v)) return true;
            if (/:(8000|8010|8080|8443|8888)\b/.test(v)) return true;
            return false;
        }
        function loadUserRadioStations() {
            try {
                const raw = localStorage.getItem(USER_RADIO_STATIONS_KEY);
                const parsed = raw ? JSON.parse(raw) : [];
                if (Array.isArray(parsed)) {
                    userRadioStations = parsed
                        .filter((it) => it && typeof it.url === 'string' && it.url.trim())
                        .map((it) => ({ name: String(it.name || deriveNameFromUrl(it.url)).trim(), url: it.url.trim() }))
                        .slice(0, 200);
                }
            } catch (_) {}
        }
        function saveUserRadioStations() {
            try { localStorage.setItem(USER_RADIO_STATIONS_KEY, JSON.stringify(userRadioStations.slice(0, 200))); } catch (_) {}
        }
        function resolveStationIndexFromSaved(saved) {
            if (!Array.isArray(stations) || !stations.length) return -1;
            if (!saved || typeof saved !== 'object') return -1;
            const url = saved.url ? sanitizeUrlForAudio(String(saved.url)) : '';
            if (url) {
                const byUrl = stations.findIndex((s) => s && sanitizeUrlForAudio(String(s.url || '')) === url);
                if (byUrl >= 0) return byUrl;
            }
            const idx = Number(saved.index);
            if (Number.isFinite(idx) && idx >= 0 && idx < stations.length) return idx;
            return -1;
        }
        function saveLastStationSelection(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const idx = dk === 'b' ? currentStationBIndex : currentStationIndex;
            if (typeof idx !== 'number' || idx < 0 || !Array.isArray(stations) || idx >= stations.length || !stations[idx]) return;
            const key = dk === 'b' ? LAST_STATION_B_STORAGE_KEY : LAST_STATION_A_STORAGE_KEY;
            try {
                localStorage.setItem(key, JSON.stringify({
                    index: idx,
                    url: sanitizeUrlForAudio(String(stations[idx].url || ''))
                }));
            } catch (_) {}
        }
        function loadLastStationSelections() {
            let idxA = -1;
            let idxB = -1;
            try {
                const rawA = localStorage.getItem(LAST_STATION_A_STORAGE_KEY);
                if (rawA) idxA = resolveStationIndexFromSaved(JSON.parse(rawA));
            } catch (_) {}
            try {
                const rawB = localStorage.getItem(LAST_STATION_B_STORAGE_KEY);
                if (rawB) idxB = resolveStationIndexFromSaved(JSON.parse(rawB));
            } catch (_) {}
            return { a: idxA, b: idxB };
        }
        function applySavedStationSelections() {
            if (!Array.isArray(stations) || !stations.length) return;
            const saved = loadLastStationSelections();
            if (saved.a >= 0) {
                currentStationIndex = saved.a;
                if (radioInputEl) radioInputEl.value = stations[saved.a].url;
            } else if (typeof currentStationIndex !== 'number' || currentStationIndex < 0) {
                currentStationIndex = 0;
                if (radioInputEl) radioInputEl.value = stations[0].url;
            }
            if (saved.b >= 0) {
                currentStationBIndex = saved.b;
            } else if (typeof currentStationBIndex !== 'number' || currentStationBIndex < 0) {
                currentStationBIndex = 0;
            } else if (currentStationBIndex >= stations.length) {
                currentStationBIndex = stations.length - 1;
            }
        }
        /** Add or enable a user station in the Radio station cycle list (from pasted stream URLs). */
        function addUserRadioStation(url, name) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean) return false;
            if (!stations.some((s) => s && s.url === clean)) {
                const entry = { name: String(name || deriveNameFromUrl(clean)), url: clean };
                stations.push(entry);
                userRadioStations.push(entry);
                saveUserRadioStations();
            }
            stationCycleEnabledByUrl.set(clean, true);
            syncStationCycleSelection();
            try { renderStationList(); } catch (_) {}
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            return true;
        }
        /** After removing station at removedIdx, map old list index to new index (or -1 if empty). */
        function reindexStationCursor(prevIdx, removedIdx) {
            if (!Array.isArray(stations) || stations.length === 0) return -1;
            if (typeof prevIdx !== 'number' || !Number.isFinite(prevIdx)) return 0;
            if (prevIdx < removedIdx) return prevIdx;
            if (prevIdx === removedIdx) return Math.min(removedIdx, stations.length - 1);
            return prevIdx - 1;
        }
        /** Remove a station from the in-memory list, user-saved list, cycle map, top menu, and mix dropdown. */
        function removeRadioStationAtIndex(idx) {
            if (!Array.isArray(stations) || idx < 0 || idx >= stations.length) return;
            const removed = stations[idx];
            if (!removed || !removed.url) return;
            const rawUrl = removed.url;
            const clean = sanitizeUrlForAudio(String(rawUrl));
            const prevA = (typeof currentStationIndex === 'number') ? currentStationIndex : -1;
            const prevB = (typeof currentStationBIndex === 'number') ? currentStationBIndex : 0;
            const oldBUrl = stations[prevB] ? sanitizeUrlForAudio(String(stations[prevB].url || '')) : '';

            stations.splice(idx, 1);
            try {
                stationCycleEnabledByUrl.delete(rawUrl);
                stationCycleEnabledByUrl.delete(clean);
            } catch (_) {}

            userRadioStations = userRadioStations.filter((u) => {
                if (!u || !u.url) return false;
                return sanitizeUrlForAudio(u.url) !== clean;
            });
            saveUserRadioStations();

            deckPlaybackHistory.a = deckPlaybackHistory.a
                .filter((entry) => {
                    if (!entry || entry.kind !== 'station') return true;
                    return entry.index !== idx;
                })
                .map((entry) => {
                    if (entry && entry.kind === 'station' && entry.index > idx) {
                        return { kind: 'station', index: entry.index - 1 };
                    }
                    return entry;
                });
            deckPlaybackHistory.b = deckPlaybackHistory.b
                .filter((entry) => {
                    if (!entry || entry.kind !== 'station') return true;
                    return entry.index !== idx;
                })
                .map((entry) => {
                    if (entry && entry.kind === 'station' && entry.index > idx) {
                        return { kind: 'station', index: entry.index - 1 };
                    }
                    return entry;
                });
            syncLegacyStationHistoryFromDeck();

            if (!stations.length) {
                currentStationIndex = -1;
                currentStationBIndex = 0;
                try { if (radioInputEl) radioInputEl.value = ''; } catch (_) {}
                try { resetRadioADualStreamHandoff(); } catch (_) {}
                try { resetRadioBDualStreamHandoff(); } catch (_) {}
                try { if (audioEl) { audioEl.pause(); audioEl.removeAttribute('src'); audioEl.load(); } } catch (_) {}
                try { if (audioElB) { audioElB.pause(); audioElB.removeAttribute('src'); audioElB.load(); } } catch (_) {}
            } else {
                currentStationIndex = reindexStationCursor(prevA, idx);
                currentStationBIndex = Math.max(0, Math.min(stations.length - 1, reindexStationCursor(prevB, idx)));

                if (prevA === idx && currentStationIndex >= 0 && stations[currentStationIndex]) {
                    suppressHistoryPush = true;
                    try {
                        const s = stations[currentStationIndex];
                        if (radioInputEl) radioInputEl.value = s.url;
                        if (typeof showStationBanner === 'function') showStationBanner(s.name);
                        if (typeof playRadio === 'function') playRadio();
                        try { if (typeof updateStationActiveHighlight === 'function') updateStationActiveHighlight(); } catch (_) {}
                    } finally {
                        suppressHistoryPush = false;
                    }
                } else {
                    try { if (typeof updateStationActiveHighlight === 'function') updateStationActiveHighlight(); } catch (_) {}
                }

                const newBUrl = stations[currentStationBIndex]
                    ? sanitizeUrlForAudio(String(stations[currentStationBIndex].url || ''))
                    : '';
                if (oldBUrl !== newBUrl) {
                    try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                }
            }

            try { renderStationList(); } catch (_) {}
            try { refreshMixStationB(); } catch (_) {}
            try { updateStationActiveHighlight(); } catch (_) {}
            try { saveLastStationSelection('a'); saveLastStationSelection('b'); } catch (_) {}
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        function upsertMediaVideoQueueEntry(url, label) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean || isLikelyRadioStreamUrl(clean) || !isVideoQueueEligibleUrl(clean)) return null;
            const lab = String(label || deriveNameFromUrl(clean) || 'Video');
            const mqIdx = mediaVideoQueue.findIndex((it) => it && it.url === clean);
            if (mqIdx >= 0) mediaVideoQueue.splice(mqIdx, 1);
            mediaVideoQueue.unshift({ id: String(Date.now()) + '-' + Math.random().toString(36).slice(2, 8), url: clean, label: lab });
            if (mediaVideoQueue.length > 120) mediaVideoQueue.length = 120;
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            return mediaVideoQueue[0];
        }
        function isLikelyVideoUrl(url) {
            const v = sanitizeUrlForAudio(url).toLowerCase();
            if (!v) return false;
            /* blob: is ambiguous (Deck A local audio vs queued video) — do not treat as video by URL alone. */
            if (v.startsWith('blob:')) return false;
            if (v.startsWith('data:')) return /^data:video\//i.test(v);
            return /\.(mp4|webm|mkv|mov|m4v|ogv|ogg|avi)(\?|#|$)/i.test(v);
        }
        /** URLs eligible for the Video queue list (files, extensions, common embed hosts). */
        function isVideoQueueEligibleUrl(url) {
            const v = sanitizeUrlForAudio(url).toLowerCase();
            if (!v) return false;
            if (v.startsWith('blob:')) return true;
            if (isLikelyVideoUrl(url)) return true;
            return /youtube\.com|youtu\.be|vimeo\.com|twitch\.tv\/videos/i.test(v);
        }
        function registerDeckVideoFeed(deckKey, url, label, forceVideo) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean) return;
            if (!forceVideo && !isLikelyVideoUrl(clean) && !isVideoQueueEligibleUrl(clean)) return;
            const k = deckKey === 'b' ? 'b' : 'a';
            const videoFile = !!forceVideo || isLikelyVideoUrl(clean);
            deckVideoFeeds[k] = { url: clean, label: String(label || deriveNameFromUrl(clean) || 'Video source'), videoFile };
            const arr = deckVideoHistory[k];
            const idx = arr.findIndex((it) => it && it.url === clean);
            if (idx >= 0) arr.splice(idx, 1);
            arr.unshift({ url: clean, label: deckVideoFeeds[k].label });
            if (arr.length > 50) arr.length = 50;
            upsertMediaVideoQueueEntry(clean, deckVideoFeeds[k].label);
            try { refreshActiveDeckVideoDisplays(); } catch (_) {}
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        /** Stop mirroring a deck's local MP4 in the video panel; restore media queue or idle logo. */
        function releaseDeckVideoFeed(deckKey, urlHint) {
            const k = deckKey === 'b' ? 'b' : 'a';
            const feed = deckVideoFeeds[k];
            const url = sanitizeUrlForAudio(urlHint || (feed && feed.url) || '');
            deckVideoFeeds[k] = null;
            if (url) {
                for (let i = mediaVideoQueue.length - 1; i >= 0; i--) {
                    const it = mediaVideoQueue[i];
                    if (it && urlsMediaMatch(it.url, url)) mediaVideoQueue.splice(i, 1);
                }
                const hist = deckVideoHistory[k];
                for (let i = hist.length - 1; i >= 0; i--) {
                    if (hist[i] && urlsMediaMatch(hist[i].url, url)) hist.splice(i, 1);
                }
            }
            try { refreshActiveDeckVideoDisplays(); } catch (_) {}
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        function refreshActiveDeckVideoDisplays() {
            try {
                const av = state && state.activeVisualizer;
                if (!av) return;
                if (av.name === 'DJ Decks' && typeof av.refreshDeckBVideoSource === 'function') {
                    av.refreshDeckBVideoSource();
                    return;
                }
                const n = av.name;
                const onRadio = n === 'Digital Radio' || n === 'Analogue radio' || n === 'Radio' || n === 'Radio Visual';
                if (onRadio && typeof av._refreshDigitalDeckVideoMirrors === 'function') {
                    av._refreshDigitalDeckVideoMirrors();
                }
            } catch (_) {}
        }
        /** Deck B / Digital Radio video elements: active feed, media queue, or idle logo. */
        function applyDeckVideoMirrorToElement(vid) {
            if (!vid) return;
            try {
                const metaB = getDeckActiveVideoMeta('b');
                const metaA = getDeckActiveVideoMeta('a');
                const meta = metaB || metaA;
                if (meta && meta.url) {
                    applyDeckBVideoPayloadToElement(vid, {
                        url: meta.url,
                        label: meta.label,
                        syncFrom: meta.syncFrom || meta.media
                    }, null);
                    return;
                }
            } catch (_) {}
            try {
                if (mediaVideoQueue.length) {
                    const q = mediaVideoQueue[0];
                    if (q && q.url) {
                        applyDeckBVideoPayloadToElement(vid, { url: q.url, label: q.label }, null);
                        return;
                    }
                }
            } catch (_) {}
            const idleUrl = DECK_B_IDLE_LOGO_URL;
            const had = String(vid.currentSrc || vid.src || '');
            if (urlsMediaMatch(idleUrl, had)) {
                try { vid.loop = true; } catch (_) {}
                try {
                    if (vid.paused) vid.play().catch(() => {});
                } catch (_) {}
                return;
            }
            try {
                vid.pause();
                vid.removeAttribute('src');
                vid.load();
            } catch (_) {}
            try {
                vid.loop = true;
                vid.src = DECK_B_IDLE_LOGO_URL;
                vid.play().catch(() => {});
            } catch (_) {}
        }
        function removeMediaVideoQueueItem(id) {
            const idx = mediaVideoQueue.findIndex((it) => it && String(it.id) === String(id));
            if (idx < 0) return;
            const removed = mediaVideoQueue[idx];
            try {
                if (removed && removed.url && String(removed.url).startsWith('blob:')) {
                    URL.revokeObjectURL(removed.url);
                }
            } catch (_) {}
            mediaVideoQueue.splice(idx, 1);
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        function getDeckBVideoCandidates() {
            const out = [];
            const pushUnique = (it) => {
                if (!it || !it.url) return;
                if (out.some((x) => x.url === it.url)) return;
                out.push({ url: it.url, label: it.label || deriveNameFromUrl(it.url) || 'Video source' });
            };
            pushUnique(deckVideoFeeds.b);
            pushUnique(deckVideoFeeds.a);
            (deckVideoHistory.b || []).forEach(pushUnique);
            (deckVideoHistory.a || []).forEach(pushUnique);
            return out;
        }
        function isDeckAVideoActivelyPlaying() {
            return !!getDeckActiveVideoMeta('a');
        }
        /** Active video on a deck: HTMLVideo on A, or audio+video feed matching deckVideoFeeds. */
        function getDeckActiveVideoMeta(deckKey) {
            const k = deckKey === 'b' ? 'b' : 'a';
            try {
                if (k === 'a') {
                    const el = getDeckAMediaForPlaybackState();
                    if (el && !(el instanceof HTMLAudioElement)) {
                        const src = String(el.currentSrc || el.src || '').trim();
                        if (src && !el.paused && !el.ended && (isLikelyVideoUrl(src) || isVideoQueueEligibleUrl(src))) {
                            let label = 'Deck A video';
                            try {
                                const fa = deckVideoFeeds.a;
                                if (fa && fa.url && urlsMediaMatch(src, fa.url)) label = fa.label || label;
                                else label = deriveNameFromUrl(src) || label;
                            } catch (_) {}
                            return { deckKey: 'a', url: src, label, syncFrom: el, media: el };
                        }
                    }
                }
                const feed = deckVideoFeeds[k];
                const media = k === 'b' ? audioElB : audioEl;
                if (!feed || !feed.url || !media) return null;
                const vf = feed.videoFile === undefined ? isLikelyVideoUrl(feed.url) : !!feed.videoFile;
                if (!vf) return null;
                const src = String(media.currentSrc || media.src || '').trim();
                if (!src || media.paused || media.ended) return null;
                if (!urlsMediaMatch(src, feed.url)) return null;
                return {
                    deckKey: k,
                    media,
                    url: src,
                    label: feed.label || (typeof deriveNameFromUrl === 'function' ? deriveNameFromUrl(src) : '') || 'Deck ' + String(k).toUpperCase() + ' video',
                    syncFrom: media
                };
            } catch (_) {
                return null;
            }
        }
        function urlsMediaMatch(a, b) {
            if (!a || !b) return false;
            if (a === b) return true;
            try {
                const sa = sanitizeUrlForAudio(String(a));
                const sb = sanitizeUrlForAudio(String(b));
                if (sa === sb) return true;
                const pathKey = (u) => {
                    const s = sanitizeUrlForAudio(String(u));
                    if (!s) return '';
                    if (s.startsWith('blob:') || s.startsWith('data:')) return s;
                    return new URL(s, globalThis.location?.href || 'http://localhost/').pathname;
                };
                return pathKey(sa) === pathKey(sb);
            } catch (_) {
                return false;
            }
        }
        function getDjCrossfade01() {
            try {
                const dc = document.getElementById('dj-crossfader');
                const mc = document.getElementById('mix-crossfader');
                const raw = dc && dc.value !== undefined && dc.value !== '' ? dc.value : mc && mc.value;
                return Math.max(0, Math.min(1, Number(raw) || 0));
            } catch (_) {
                return 0;
            }
        }
        /** Crossfader 0–1 from Digital Radio dash, DJ, or Mix (same order as readCrossfadePosition). */
        function getAppCrossfade01() {
            try {
                const dig = document.getElementById('radio-visual-cross-digital');
                const mc = document.getElementById('mix-crossfader');
                const dj = document.getElementById('dj-crossfader');
                const raw = (dig && dig.value !== undefined && dig.value !== '')
                    ? dig.value
                    : (mc && mc.value !== undefined && mc.value !== '')
                        ? mc.value
                        : (dj && dj.value);
                return Math.max(0, Math.min(1, Number(raw) || 0));
            } catch (_) {
                return 0;
            }
        }
        /** Digital Radio VIDEO staging: blend deck A/B mirrors by crossfader (no queue layer). */
        function computeDigitalStagingVideoCrossfadePlan() {
            let x = getAppCrossfade01();
            const EDGE_SNAP = 0.03;
            if (x <= EDGE_SNAP) x = 0;
            else if (x >= 1 - EDGE_SNAP) x = 1;
            const ga = 1 - x;
            const gb = x;
            const metaA = getDeckActiveVideoMeta('a');
            const metaB = getDeckActiveVideoMeta('b');
            const layerA = metaA ? { url: metaA.url, label: metaA.label, syncFrom: metaA.syncFrom || metaA.media } : null;
            const layerB = metaB ? { url: metaB.url, label: metaB.label, syncFrom: metaB.syncFrom || metaB.media } : null;
            let opA = 0;
            let opB = 0;
            if (layerA && layerB) {
                opA = ga;
                opB = gb;
            } else if (layerA) {
                opA = 1;
            } else if (layerB) {
                opB = 1;
            }
            let label = 'Video';
            if (opB >= opA && layerB) label = layerB.label;
            else if (layerA) label = layerA.label;
            return { layerA, layerB, opA, opB, label, dual: !!(layerA && layerB) };
        }
        /**
         * MP4 etc. play through <audio> into the mix; VIDEO mirror follows the deck winning the DJ crossfader
         * (same 0.5 threshold as pickRandomStationForCrossfadedDeck).
         */
        function getDeckAudioVjsMirrorMeta() {
            const xf = getDjCrossfade01();
            const preferB = xf >= 0.5;
            return getDeckActiveVideoMeta(preferB ? 'b' : 'a') || getDeckActiveVideoMeta(preferB ? 'a' : 'b');
        }
        function computeDeckBVideoCrossfadePlan(viz) {
            let x = getDjCrossfade01();
            const EDGE_SNAP = 0.03;
            if (x <= EDGE_SNAP) x = 0;
            else if (x >= 1 - EDGE_SNAP) x = 1;
            const ga = 1 - x;
            const gb = x;
            const metaA = getDeckActiveVideoMeta('a');
            const metaB = getDeckActiveVideoMeta('b');
            const layerA = metaA ? { url: metaA.url, label: metaA.label, syncFrom: metaA.syncFrom || metaA.media } : null;
            const layerB = metaB ? { url: metaB.url, label: metaB.label, syncFrom: metaB.syncFrom || metaB.media } : null;
            let layerQ = null;
            if (mediaVideoQueue.length && !deckBVideoUserIdle) {
                let qi = 0;
                try {
                    if (viz && typeof viz.deckBVideoIndex === 'number') qi = viz.deckBVideoIndex;
                } catch (_) {}
                qi = Math.max(0, Math.min(qi, mediaVideoQueue.length - 1));
                const q = mediaVideoQueue[qi];
                if (q && q.url) {
                    layerQ = { url: q.url, label: q.label || deriveNameFromUrl(q.url) || 'Queue video', deckKey: 'queue' };
                }
            }
            let opA = 0;
            let opB = 0;
            let opQ = 0;
            if (layerA && layerB) {
                opA = ga;
                opB = gb;
            } else if (layerA && layerQ) {
                opA = ga;
                opQ = gb;
            } else if (layerB && layerQ) {
                opB = gb;
                opQ = ga;
            } else if (layerA) {
                opA = 1;
            } else if (layerB) {
                opB = 1;
            } else if (layerQ) {
                opQ = 1;
            }
            let label = 'Video';
            if (opB >= opA && opB >= opQ && layerB) label = layerB.label;
            else if (opA >= opQ && layerA) label = layerA.label;
            else if (layerQ) label = layerQ.label;
            return { layerA, layerB, layerQ, opA, opB, opQ, label };
        }
        function applyDeckBVideoCrossfadeLayers(viz) {
            if (!viz || viz.deckBVizMode !== 'video') return;
            const vA = viz.deckBVideoElA;
            const vB = viz.deckBVideoElB;
            const vQ = viz.deckBVideoElQ;
            if (!vA || !vB || !vQ) return;
            const plan = computeDeckBVideoCrossfadePlan(viz);
            const setLayer = (vid, layer, op) => {
                const o = Math.max(0, Math.min(1, Number(op) || 0));
                try {
                    vid.style.opacity = String(o);
                    vid.style.pointerEvents = o > 0.35 ? 'auto' : 'none';
                } catch (_) {}
                if (o <= 0.001) {
                    try {
                        if (!vid.paused) vid.pause();
                    } catch (_) {}
                    return;
                }
                if (layer && layer.url) applyDeckBVideoPayloadToElement(vid, layer, null);
            };
            setLayer(vA, plan.layerA, plan.opA);
            setLayer(vB, plan.layerB, plan.opB);
            setLayer(vQ, plan.layerQ, plan.opQ);
            try {
                if (plan.opB >= plan.opA && plan.opB >= plan.opQ) viz.deckBVideoEl = vB;
                else if (plan.opA >= plan.opQ) viz.deckBVideoEl = vA;
                else viz.deckBVideoEl = vQ;
            } catch (_) {}
            try {
                viz.showDeckBVisualBackButton(plan.label || 'Video');
            } catch (_) {}
        }
        function getDeckBJogSeekMedia(viz) {
            try {
                if (viz && viz.deckBVizMode === 'video' && !deckBVideoUserIdle) {
                    const meta = getDeckActiveVideoMeta('b');
                    if (meta && meta.media) return meta.media;
                    const plan = computeDeckBVideoCrossfadePlan(viz);
                    if (plan.layerB && plan.opB > 0.02 && viz.deckBVideoElB) return viz.deckBVideoElB;
                    if (plan.layerQ && plan.opQ > 0.02 && viz.deckBVideoElQ) return viz.deckBVideoElQ;
                }
            } catch (_) {}
            return audioElB;
        }
        function applyDeckBVideoPayloadToElement(vid, cur, hooks) {
            if (!vid || !cur || !cur.url) return;
            try {
                vid.loop = !!deckBVideoSingleLoopActive();
            } catch (_) {}
            const setLbl = hooks && typeof hooks.setPlayingLabel === 'function' ? hooks.setPlayingLabel : null;
            if (cur.syncFrom) {
                const sf = cur.syncFrom;
                const apply = () => {
                    try {
                        let t = Number(sf.currentTime) || 0;
                        const md = sf.duration;
                        if (Number.isFinite(md) && md > 0) t = Math.min(Math.max(0, t), md - 0.05);
                        const vd = vid.duration;
                        if (Number.isFinite(vd) && vd > 0) t = Math.min(t, vd - 0.05);
                        vid.currentTime = t;
                    } catch (_) {}
                    try {
                        if (sf.paused || sf.ended) {
                            vid.pause();
                            if (setLbl) setLbl(false);
                        } else {
                            vid.play().catch(() => {
                                if (setLbl) setLbl(false);
                            });
                            if (setLbl) setLbl(true);
                        }
                    } catch (_) {}
                };
                const want = String(cur.url);
                const had = String(vid.currentSrc || vid.src || '');
                const same = urlsMediaMatch(want, had);
                if (!same) {
                    vid.src = want;
                    vid.addEventListener('loadedmetadata', apply, { once: true });
                    vid.addEventListener('loadeddata', apply, { once: true });
                } else {
                    apply();
                }
                return;
            }
            const samePlain = urlsMediaMatch(String(cur.url), String(vid.currentSrc || vid.src || ''));
            if (!samePlain) vid.src = cur.url;
            vid.play()
                .then(() => {
                    if (setLbl) setLbl(true);
                })
                .catch(() => {
                    if (setLbl) setLbl(false);
                });
        }
        /** Deck B video panel playlist: active deck feeds first; queue when no deck video is live. */
        function getDeckBVideoPlaybackSources() {
            const out = [];
            const push = (it) => {
                if (!it || !it.url) return;
                if (out.some((x) => urlsMediaMatch(x.url, it.url))) return;
                out.push(it);
            };
            const metaA = getDeckActiveVideoMeta('a');
            const metaB = getDeckActiveVideoMeta('b');
            if (metaA) push({ url: metaA.url, label: metaA.label, syncFrom: metaA.syncFrom || metaA.media });
            if (metaB) push({ url: metaB.url, label: metaB.label, syncFrom: metaB.syncFrom || metaB.media });
            if (out.length) return out;
            if (mediaVideoQueue.length) {
                return mediaVideoQueue.map((m) => ({ url: m.url, label: m.label }));
            }
            return getDeckBVideoCandidates();
        }
        // Volume slider
        const volumeSlider = document.getElementById('volume-slider');
        const topBar = document.getElementById('top-bar');
        const shuffleToggle = document.getElementById('shuffle-toggle');
        let currentStationIndex = -1;
        let currentStationBIndex = 0;
        // Per-deck playback history (radio station index or local file snapshot).
        const DECK_PLAYBACK_HISTORY_MAX = 12;
        const deckPlaybackHistory = { a: [], b: [] };
        // Legacy Deck A station index stack (mirrors radio entries in deckPlaybackHistory.a).
        let stationHistory = [];
        // Guard to avoid pushing into history when navigating back
        let suppressHistoryPush = false;

        function deckPlaybackEntriesEqual(a, b) {
            if (!a || !b || a.kind !== b.kind) return false;
            if (a.kind === 'station') return a.index === b.index;
            return a.url === b.url;
        }

        function captureDeckPlaybackSnapshot(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            try {
                if (state.deckSourceMode && state.deckSourceMode[dk] === 'local') {
                    const el = dk === 'b'
                        ? audioElB
                        : ((typeof getDeckAMediaForPlaybackState === 'function')
                            ? getDeckAMediaForPlaybackState()
                            : audioEl);
                    const url = el ? sanitizeUrlForAudio(String(el.currentSrc || el.src || '')) : '';
                    if (!url || url === 'about:blank') return null;
                    const name = (state.deckLocalDisplayName && state.deckLocalDisplayName[dk]) || '';
                    const isVideo = !!(typeof isLikelyVideoUrl === 'function' && isLikelyVideoUrl(url));
                    return { kind: 'local', url, name: String(name || ''), isVideo: !!isVideo };
                }
                const idx = dk === 'b' ? currentStationBIndex : currentStationIndex;
                if (typeof idx !== 'number' || idx < 0) return null;
                if (!Array.isArray(stations) || idx >= stations.length) return null;
                return { kind: 'station', index: idx };
            } catch (_) {
                return null;
            }
        }

        function syncLegacyStationHistoryFromDeck() {
            stationHistory = deckPlaybackHistory.a
                .filter((e) => e && e.kind === 'station' && typeof e.index === 'number')
                .map((e) => e.index);
        }

        function pushDeckPlaybackHistory(deckKey, snapshot) {
            if (suppressHistoryPush) return;
            const dk = deckKey === 'b' ? 'b' : 'a';
            const snap = snapshot || captureDeckPlaybackSnapshot(dk);
            if (!snap) return;
            if (snap.kind === 'station') {
                if (snap.index < 0 || !Array.isArray(stations) || snap.index >= stations.length) return;
            } else if (snap.kind === 'local') {
                if (!snap.url) return;
            } else {
                return;
            }
            const hist = deckPlaybackHistory[dk];
            const last = hist[hist.length - 1];
            if (last && deckPlaybackEntriesEqual(last, snap)) return;
            hist.push(snap);
            while (hist.length > DECK_PLAYBACK_HISTORY_MAX) hist.shift();
            if (dk === 'a') syncLegacyStationHistoryFromDeck();
        }

        function restoreDeckPlaybackSnapshot(deckKey, entry) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            if (!entry) return false;
            if (entry.kind === 'station') {
                const idx = entry.index;
                if (typeof idx !== 'number' || idx < 0 || !Array.isArray(stations) || idx >= stations.length) {
                    return false;
                }
                suppressHistoryPush = true;
                try {
                    if (dk === 'b') {
                        currentStationBIndex = idx;
                        try { state.deckSourceMode.b = 'radio'; } catch (_) {}
                        try { state.deckLocalDisplayName.b = ''; } catch (_) {}
                        try { if (typeof refreshMixStationB === 'function') refreshMixStationB(); } catch (_) {}
                        try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                    } else {
                        setStation(idx);
                    }
                } finally {
                    suppressHistoryPush = false;
                }
                return true;
            }
            if (entry.kind === 'local' && entry.url) {
                suppressHistoryPush = true;
                try {
                    try { initAudio(); } catch (_) {}
                    const item = {
                        url: entry.url,
                        name: entry.name || 'Local track',
                        isVideo: !!entry.isVideo
                    };
                    if (dk === 'b') {
                        try { if (typeof prepareDeckBLocalPlayback === 'function') prepareDeckBLocalPlayback(); } catch (_) {}
                        try { if (typeof revokeBlobSrc === 'function') revokeBlobSrc(audioElB); } catch (_) {}
                        state.deckSourceMode.b = 'local';
                        state.deckLocalDisplayName.b = item.name;
                        try { audioElB.crossOrigin = 'anonymous'; } catch (_) {}
                        audioElB.src = item.url;
                        if (item.isVideo && typeof registerDeckVideoFeed === 'function') {
                            registerDeckVideoFeed('b', item.url, item.name, true);
                        } else {
                            try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('b'); } catch (_) {}
                        }
                        try { connectDeckMediaToEq('b'); } catch (_) {}
                        audioElB.play().catch(() => {});
                    } else {
                        try { if (typeof abortRadioAHandoff === 'function') abortRadioAHandoff(); } catch (_) {}
                        try { if (typeof resetRadioADualStreamHandoff === 'function') resetRadioADualStreamHandoff(); } catch (_) {}
                        try { if (typeof revokeBlobSrc === 'function') revokeBlobSrc(audioEl); } catch (_) {}
                        state.deckSourceMode.a = 'local';
                        state.deckLocalDisplayName.a = item.name;
                        try { audioEl.crossOrigin = 'anonymous'; } catch (_) {}
                        audioEl.src = item.url;
                        if (item.isVideo && typeof registerDeckVideoFeed === 'function') {
                            registerDeckVideoFeed('a', item.url, item.name, true);
                        } else {
                            try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('a'); } catch (_) {}
                        }
                        try { connectDeckMediaToEq('a'); } catch (_) {}
                        audioEl.play().catch(() => {});
                    }
                    try { if (typeof showStationBanner === 'function') showStationBanner(item.name); } catch (_) {}
                } finally {
                    suppressHistoryPush = false;
                }
                return true;
            }
            return false;
        }

        function goPreviousDeckPlayback(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const hist = deckPlaybackHistory[dk];
            while (hist.length) {
                const prev = hist.pop();
                if (dk === 'a') syncLegacyStationHistoryFromDeck();
                if (restoreDeckPlaybackSnapshot(dk, prev)) return;
            }
            if (dk === 'a') pickRandomStation();
            else pickRandomStationB();
        }
        let panelIdleTimer = null;
        let webmList = [];
        let webmIndex = 0;
        let webmOn = false;
        /** When true, WebM overlay is laid out fixed to the center of the Deck B column (see applyWebmSettings), including the crossfader strip so the anchor matches the visible purple deck. */
        let webmAnchorDeckB = false;
        let webmDeckBLayoutBound = false;
        let webmDeckBResizeObserver = null;
        let webmDeckBLayoutRaf = null;
        function __webmDeckBOnLayout() {
            if (webmDeckBLayoutRaf) return;
            webmDeckBLayoutRaf = requestAnimationFrame(() => {
                webmDeckBLayoutRaf = null;
                try {
                    if (webmOn && webmAnchorDeckB) applyWebmSettings();
                } catch (_) {}
            });
        }
        function bindWebmDeckBLayoutWatchers() {
            if (webmDeckBLayoutBound) return;
            try {
                window.addEventListener('resize', __webmDeckBOnLayout, { passive: true });
            } catch (_) {
                try { window.addEventListener('resize', __webmDeckBOnLayout); } catch (_) {}
            }
            try {
                if (typeof ResizeObserver !== 'undefined') {
                    webmDeckBResizeObserver = new ResizeObserver(() => __webmDeckBOnLayout());
                    const mount = document.getElementById('dj-deck-b-viz-mount');
                    const vizLayer = document.getElementById('dj-deck-b-viz-layer');
                    if (mount) webmDeckBResizeObserver.observe(mount);
                    if (vizLayer) webmDeckBResizeObserver.observe(vizLayer);
                }
            } catch (_) {}
            webmDeckBLayoutBound = true;
        }
        function unbindWebmDeckBLayoutWatchers() {
            if (!webmDeckBLayoutBound) return;
            try { window.removeEventListener('resize', __webmDeckBOnLayout); } catch (_) {}
            try {
                if (webmDeckBResizeObserver) {
                    webmDeckBResizeObserver.disconnect();
                    webmDeckBResizeObserver = null;
                }
            } catch (_) {}
            webmDeckBLayoutBound = false;
        }
        let radioPanelTimer = null;
        let settingsPanelTimer = null;
        let webmSettingsTimer = null;
        let stationBannerTimer = null;
        let nowPlayingPollTimer = null;
        let nowPlayingPollUrl = '';
        let icyPollBusy = false;
        let currentNowPlayingICY = '';
        // Hold the radio button visible after right-click random selection
        let radioQuickHoldUntil = 0;
        let radioQuickHoldTimeout = null;
        // Guard to suppress unintended overlay-start when opening local file via "O"
        let suppressNextOverlayStartUntil = 0;
        function syncStationCycleSelection() {
            const seen = new Set();
            stations.forEach((st) => {
                if (!st || !st.url) return;
                seen.add(st.url);
                if (!stationCycleEnabledByUrl.has(st.url)) stationCycleEnabledByUrl.set(st.url, true);
            });
            Array.from(stationCycleEnabledByUrl.keys()).forEach((url) => {
                if (!seen.has(url)) stationCycleEnabledByUrl.delete(url);
            });
        }
        function getCycleEligibleStationIndexes(excludeIndex) {
            if (!Array.isArray(stations) || stations.length === 0) return [];
            syncStationCycleSelection();
            const enabled = [];
            stations.forEach((st, idx) => {
                if (!st || !st.url) return;
                if (stationCycleEnabledByUrl.get(st.url) !== false) enabled.push(idx);
            });
            if (!enabled.length) return [];
            if (enabled.length > 1 && typeof excludeIndex === 'number') {
                return enabled.filter((idx) => idx !== excludeIndex);
            }
            return enabled;
        }
        const WEBM_SETTINGS_STORAGE_KEY = 'webm.avatar.settings.v1';
        const WEBM_DEFAULT_SCALE_VW = 42;
        const WEBM_DEFAULT_DUP_SPACING = 0.6;
        const webmSettings = {
            scaleVw: WEBM_DEFAULT_SCALE_VW,
            posXvw: 50,
            posYvh: 50,
            rotationDeg: 0,
            playbackRate: 1.0,
            opacity: 0.82,
            duplicates: 0,
            duplicateSpacing: WEBM_DEFAULT_DUP_SPACING
        };
        function loadWebmSettingsFromStorage() {
            try {
                const raw = localStorage.getItem(WEBM_SETTINGS_STORAGE_KEY);
                if (!raw) return;
                const saved = JSON.parse(raw);
                if (!saved || typeof saved !== 'object') return;
                const num = (v, fallback, min, max) => {
                    const n = Number(v);
                    if (!Number.isFinite(n)) return fallback;
                    if (min != null && n < min) return min;
                    if (max != null && n > max) return max;
                    return n;
                };
                webmSettings.scaleVw = num(saved.scaleVw, WEBM_DEFAULT_SCALE_VW, 10, 200);
                webmSettings.posXvw = num(saved.posXvw, 50, 0, 100);
                webmSettings.posYvh = num(saved.posYvh, 50, 0, 100);
                webmSettings.rotationDeg = num(saved.rotationDeg, 0, -180, 180);
                webmSettings.playbackRate = num(saved.playbackRate, 1.0, 0.1, 4);
                webmSettings.opacity = num(saved.opacity, 0.82, 0, 1);
                webmSettings.duplicates = Math.max(0, Math.min(2, Math.round(num(saved.duplicates, 0, 0, 2))));
                webmSettings.duplicateSpacing = num(saved.duplicateSpacing, WEBM_DEFAULT_DUP_SPACING, 0.15, 1);
            } catch (_) {}
        }
        function saveWebmSettingsToStorage() {
            try {
                localStorage.setItem(WEBM_SETTINGS_STORAGE_KEY, JSON.stringify({
                    scaleVw: webmSettings.scaleVw,
                    posXvw: webmSettings.posXvw,
                    posYvh: webmSettings.posYvh,
                    rotationDeg: webmSettings.rotationDeg,
                    playbackRate: webmSettings.playbackRate,
                    opacity: webmSettings.opacity,
                    duplicates: webmSettings.duplicates,
                    duplicateSpacing: webmSettings.duplicateSpacing
                }));
            } catch (_) {}
        }
        loadWebmSettingsFromStorage();
        // EQ State Storage
        const eqState = {
            a: { gain: 1.0, high: 0, mid: 0, low: 0 },
            b: { gain: 1.0, high: 0, mid: 0, low: 0 }
        };
        // --- BPM scope + detection ---
        function drawScopeAndUpdateBpm() {
            if (!scopeCanvas || !scopeCtx) return;
            const w = scopeCanvas.width, h = scopeCanvas.height;
            // Scroll left by 1px
            try { scopeCtx.drawImage(scopeCanvas, -1, 0); } catch(_) {}
            // Clear rightmost column slightly to create fresh lane
            scopeCtx.fillStyle = 'rgba(0,0,0,0.35)';
            scopeCtx.fillRect(w - 1, 0, 1, h);
            // Compute envelope (RMS) for BPM and spectral flux for beat onsets
            const calcRms = (analyser) => {
                if (!analyser) return null;
                const size = analyser.fftSize || 1024;
                const buf = new Uint8Array(size);
                try { analyser.getByteTimeDomainData(buf); } catch(_) { return null; }
                let sum = 0; for (let i=0;i<buf.length;i++){ const v=(buf[i]-128)/128; sum += v*v; }
                return Math.sqrt(sum / buf.length);
            };
            const calcFlux = (analyser, lastSpec, hist) => {
                if (!analyser) return { flux: null, last: lastSpec };
                const bins = analyser.frequencyBinCount || 512;
                const cur = new Uint8Array(bins);
                try { analyser.getByteFrequencyData(cur); } catch(_) { return { flux: null, last: lastSpec }; }
                if (!lastSpec || lastSpec.length !== cur.length) lastSpec = cur.slice(0);
                // 3-band weighted spectral flux
                const sr = (state && state.audioCtx && state.audioCtx.sampleRate) ? state.audioCtx.sampleRate : 44100;
                const fftSize = analyser.fftSize || (bins * 2);
                const binHz = sr / fftSize;
                const lowMax = 180;    // Hz: kick/low percussion emphasis (tuned)
                const midMax = 1500;   // Hz: snare/claps/body (tuned)
                // weights favoring low (tuned)
                const wL = 0.7, wM = 0.25, wH = 0.05;
                let fluxL = 0, fluxM = 0, fluxH = 0;
                let nL = 0, nM = 0, nH = 0;
                for (let i = 0; i < bins; i++) {
                    const f = i * binHz;
                    const d = cur[i] - lastSpec[i];
                    if (d <= 0) continue;
                    if (f < lowMax) { fluxL += d; nL++; }
                    else if (f < midMax) { fluxM += d; nM++; }
                    else { fluxH += d; nH++; }
                }
                // Normalize per-band by bin count and scale to [0,1]
                const norm = (sum, n) => (n > 0 ? (sum / (n * 255)) : 0);
                fluxL = norm(fluxL, nL);
                fluxM = norm(fluxM, nM);
                fluxH = norm(fluxH, nH);
                const flux = (wL * fluxL + wM * fluxM + wH * fluxH) / (wL + wM + wH);
                hist.push(flux);
                if (hist.length > fluxHistLen) hist.shift();
                const m = hist.reduce((a,b)=>a+b,0)/hist.length;
                const v = hist.reduce((a,b)=>a + (b-m)*(b-m), 0) / hist.length;
                const sd = Math.sqrt(Math.max(0, v));
                return { flux, mean: m, sd, last: cur };
            };
            // A channel
            let rmsA = null, rmsB = null;
            if (bpmOnA) {
                rmsA = calcRms(state.analyserNodeA);
                const fa = calcFlux(state.analyserNodeA, lastSpecA, fluxHistA);
                lastSpecA = fa.last;
                const now = performance.now();
                const isBeatA = (typeof fa.flux === 'number') && (fa.flux > (fa.mean + 1.5 * (fa.sd || 0.0001))) && ((now - lastBeatTsA) > 200);
                if (isBeatA) lastBeatTsA = now;
                if (isBeatA) { scopeCtx.fillStyle = '#00ffff'; scopeCtx.fillRect(w-1, 2, 1, Math.max(2, Math.floor(h*0.4)-3)); }
            }
            // B channel
            if (bpmOnB) {
                rmsB = calcRms(state.analyserNodeB);
                const fb = calcFlux(state.analyserNodeB, lastSpecB, fluxHistB);
                lastSpecB = fb.last;
                const now = performance.now();
                const isBeatB = (typeof fb.flux === 'number') && (fb.flux > (fb.mean + 1.5 * (fb.sd || 0.0001))) && ((now - lastBeatTsB) > 200);
                if (isBeatB) lastBeatTsB = now;
                if (isBeatB) { scopeCtx.fillStyle = '#ff00ff'; scopeCtx.fillRect(w-1, Math.floor(h*0.6), 1, Math.max(2, Math.floor(h*0.4)-2)); }
            }
            // Center marker (now line)
            scopeCtx.strokeStyle = 'rgba(255,255,255,0.25)'; scopeCtx.lineWidth = 1;
            const cx = Math.floor(w/2) + 0.5; scopeCtx.beginPath(); scopeCtx.moveTo(cx, 0); scopeCtx.lineTo(cx, h); scopeCtx.stroke();
            const pushEnv = (arr, v) => { if (typeof v === 'number') { arr.push(v); if (arr.length > maxEnvSamples) arr.shift(); } };
            pushEnv(envA, rmsA); pushEnv(envB, rmsB);
            // Push flux envelope for comb BPM
            if (bpmOnA && typeof lastSpecA !== 'undefined' && fluxHistA.length > 0) {
                const latestFluxA = fluxHistA[fluxHistA.length - 1];
                if (typeof latestFluxA === 'number') { envFluxA.push(latestFluxA); if (envFluxA.length > maxEnvSamples) envFluxA.shift(); }
            }
            if (bpmOnB && typeof lastSpecB !== 'undefined' && fluxHistB.length > 0) {
                const latestFluxB = fluxHistB[fluxHistB.length - 1];
                if (typeof latestFluxB === 'number') { envFluxB.push(latestFluxB); if (envFluxB.length > maxEnvSamples) envFluxB.shift(); }
            }
            // Comb-filter BPM estimate about once per 15 frames
            if (!drawScopeAndUpdateBpm._ctr) drawScopeAndUpdateBpm._ctr = 0;
            drawScopeAndUpdateBpm._ctr++;
            if (drawScopeAndUpdateBpm._ctr % 15 === 0) {
                const estComb = (env) => {
                    if (!env || env.length < 120) return null;
                    const minBpm = 60, maxBpm = 240, fps = 60;
                    // Normalize envelope
                    const mean = env.reduce((a,b)=>a+b,0)/env.length;
                    const std = Math.sqrt(Math.max(1e-6, env.reduce((a,b)=>a+(b-mean)*(b-mean),0)/env.length));
                    const e = env.map(v => (v-mean)/std);
                    let bestBpm = null, bestScore = -Infinity;
                    for (let bpm = minBpm; bpm <= maxBpm; bpm++) {
                        const period = fps * 60 / bpm; // frames
                        let score = 0;
                        // Sum over last window using ~4 beats
                        const beats = Math.max(3, Math.min(8, Math.floor(env.length / period)));
                        for (let k = 0; k < beats; k++) {
                            const idx = Math.round(env.length - 1 - k * period);
                            const idxH = Math.round(env.length - 1 - (k + 0.5) * period);
                            if (idx >= 0) score += e[idx];
                            if (idxH >= 0) score += 0.5 * e[idxH]; // include half-beat
                        }
                        if (score > bestScore) { bestScore = score; bestBpm = bpm; }
                    }
                    return bestBpm;
                };
                const newA = bpmOnA ? estComb(envFluxA) : null;
                const newB = bpmOnB ? estComb(envFluxB) : null;
                const smooth = (prev, cur) => {
                    if (!cur) return prev;
                    if (!prev) return cur;
                    const alpha = 0.2;
                    return (1 - alpha) * prev + alpha * cur;
                };
                bpmSmoothA = smooth(bpmSmoothA, newA);
                bpmSmoothB = smooth(bpmSmoothB, newB);
                if (bpmOutA) bpmOutA.textContent = bpmSmoothA ? String(Math.round(bpmSmoothA)) : '--';
                if (bpmOutB) bpmOutB.textContent = bpmSmoothB ? String(Math.round(bpmSmoothB)) : '--';
                // Update next beat predictions for grid
                const nowTs = performance.now();
                if (showBeatGrid && bpmSmoothA && lastBeatTsA) {
                    const intervalA = 60000 / bpmSmoothA;
                    if (!nextBeatTsA || nextBeatTsA < nowTs - 2*intervalA || nextBeatTsA < lastBeatTsA) {
                        nextBeatTsA = lastBeatTsA + intervalA;
                    }
                }
                if (showBeatGrid && bpmSmoothB && lastBeatTsB) {
                    const intervalB = 60000 / bpmSmoothB;
                    if (!nextBeatTsB || nextBeatTsB < nowTs - 2*intervalB || nextBeatTsB < lastBeatTsB) {
                        nextBeatTsB = lastBeatTsB + intervalB;
                    }
                }
            }
            // Draw beat grid markers at right edge when a predicted beat occurs (they will scroll left)
            if (showBeatGrid) {
                const nowTs = performance.now();
                const drawGridTick = (tsNext, color, bpmVal, setNext) => {
                    if (!tsNext || !bpmVal) return tsNext;
                    const interval = 60000 / bpmVal;
                    if (nowTs + 8 >= tsNext) { // small lookahead to avoid misses
                        scopeCtx.fillStyle = color;
                        scopeCtx.globalAlpha = 0.25;
                        scopeCtx.fillRect(w - 1, 0, 1, h);
                        scopeCtx.globalAlpha = 1.0;
                        tsNext = tsNext + interval;
                    }
                    return tsNext;
                };
                nextBeatTsA = drawGridTick(nextBeatTsA, '#00ffff', bpmSmoothA, (v)=>nextBeatTsA=v);
                // Apply B phase offset
                const adjBpm = bpmSmoothB;
                if (nextBeatTsB && typeof phaseOffsetBms === 'number') {
                    nextBeatTsB += 0; // keep accumulator; visual offset applied by early-fire logic
                }
                nextBeatTsB = drawGridTick(nextBeatTsB ? nextBeatTsB + phaseOffsetBms : nextBeatTsB, '#ff00ff', adjBpm, (v)=>nextBeatTsB=v);
            }
            if (bpmOnA || bpmOnB) scopeAnimId = requestAnimationFrame(drawScopeAndUpdateBpm);
            else { scopeAnimId = null; if (scopeCtx) scopeCtx.clearRect(0,0,w,h); }
        }
        function updateBpmRunLoop() {
            if ((bpmOnA || bpmOnB) && !scopeAnimId) {
                try { cancelAnimationFrame(scopeAnimId); } catch(_) {}
                scopeAnimId = requestAnimationFrame(drawScopeAndUpdateBpm);
            }
            if (!bpmOnA && !bpmOnB && scopeAnimId) {
                try { cancelAnimationFrame(scopeAnimId); } catch(_) {}
                scopeAnimId = null;
                if (scopeCtx) scopeCtx.clearRect(0,0,scopeCanvas.width, scopeCanvas.height);
            }
        }
        if (bpmBtnA) bpmBtnA.addEventListener('click', (e) => {
            e.preventDefault();
            bpmOnA = !bpmOnA;
            bpmBtnA.classList.toggle('on', bpmOnA);
            if (bpmOnA) {
                startBeatA();
                if (lastBeatTsA && bpmSmoothA) { nextBeatTsA = lastBeatTsA + 60000 / bpmSmoothA; }
            } else {
                envA.length = 0; envFluxA.length = 0; if (bpmOutA) bpmOutA.textContent = '--'; bpmSmoothA = null; nextBeatTsA = null;
                stopBeatA();
            }
            updateBpmRunLoop();
        });
        if (bpmBtnB) bpmBtnB.addEventListener('click', (e) => {
            e.preventDefault();
            bpmOnB = !bpmOnB;
            bpmBtnB.classList.toggle('on', bpmOnB);
            if (bpmOnB) {
                startBeatB();
                if (lastBeatTsB && bpmSmoothB) { nextBeatTsB = lastBeatTsB + 60000 / bpmSmoothB; }
            } else {
                envB.length = 0; envFluxB.length = 0; if (bpmOutB) bpmOutB.textContent = '--'; bpmSmoothB = null; nextBeatTsB = null;
                stopBeatB();
            }
            updateBpmRunLoop();
        });
        if (bpmBtnGrid) bpmBtnGrid.addEventListener('click', (e) => {
            e.preventDefault();
            showBeatGrid = !showBeatGrid;
            bpmBtnGrid.classList.toggle('on', showBeatGrid);
            if (showBeatGrid) {
                if (bpmOnA && lastBeatTsA && bpmSmoothA) nextBeatTsA = lastBeatTsA + 60000 / bpmSmoothA;
                if (bpmOnB && lastBeatTsB && bpmSmoothB) nextBeatTsB = lastBeatTsB + 60000 / bpmSmoothB;
            } else {
                nextBeatTsA = nextBeatTsB = null;
            }
        });
        function handleTap(arr, setBpm, setLastTs, setNextTs) {
            const now = performance.now();
            arr.push(now);
            if (arr.length > 6) arr.shift();
            if (arr.length >= 2) {
                const diffs = [];
                for (let i=1;i<arr.length;i++) diffs.push(arr[i]-arr[i-1]);
                diffs.sort((a,b)=>a-b);
                const m = diffs.length>>1;
                const med = diffs.length%2 ? diffs[m] : 0.5*(diffs[m-1]+diffs[m]);
                const bpm = 60000 / Math.max(250, Math.min(2000, med)); // constrain 30–240 BPM range
                setBpm(bpm);
                setLastTs(now);
                setNextTs(now + 60000 / bpm);
            }
        }
        if (tapBtnA) tapBtnA.addEventListener('click', (e)=>{ e.preventDefault(); handleTap(tapTimesA, (v)=>{ bpmSmoothA=v; if (bpmOutA) bpmOutA.textContent=String(Math.round(v)); }, (t)=>{ lastBeatTsA=t; }, (n)=>{ nextBeatTsA=n; }); updateBpmRunLoop(); try { applyRhythmCutGateRate(); } catch (_) {} });
        if (tapBtnB) tapBtnB.addEventListener('click', (e)=>{ e.preventDefault(); handleTap(tapTimesB, (v)=>{ bpmSmoothB=v; if (bpmOutB) bpmOutB.textContent=String(Math.round(v)); }, (t)=>{ lastBeatTsB=t; }, (n)=>{ nextBeatTsB=n; }); updateBpmRunLoop(); try { applyRhythmCutGateRate(); } catch (_) {} });
        if (nudgeBm) nudgeBm.addEventListener('click', (e)=>{ e.preventDefault(); phaseOffsetBms -= 10; });
        if (nudgeBp) nudgeBp.addEventListener('click', (e)=>{ e.preventDefault(); phaseOffsetBms += 10; });
        // Helper to bind a visual knob to a value and callback
        function setKnobUi(el, min, max, val) {
            if (!el) return;
            const indicator = el.querySelector('.knob-indicator');
            const valTooltip = el.querySelector('.knob-value');
            const pct = (val - min) / (max - min);
            const deg = 45 + (pct * 270);
            if (indicator) indicator.style.transform = `translate(-50%, 0) rotate(${deg}deg)`;
            if (valTooltip) valTooltip.textContent = String(Math.round(val * 10) / 10);
        }

        function bindKnob(id, min, max, initial, stateObj, key, callback) {
            const el = document.getElementById(id);
            if (!el) return;
            try {
                if (!window.__mixKnobConfig) window.__mixKnobConfig = Object.create(null);
                window.__mixKnobConfig[id] = { min, max, stateObj, key, callback };
            } catch (_) {}

            stateObj[key] = initial;

            const updateVisuals = (val) => {
                setKnobUi(el, min, max, val);
                try {
                    const djM = window.djKnobMirrors && window.djKnobMirrors[id];
                    if (djM) setKnobUi(djM, min, max, val);
                } catch (_) {}
            };

            updateVisuals(stateObj[key]);

            let startY = 0;
            let startVal = 0;

            const onMove = (e) => {
                const y = e.clientY || (e.touches ? e.touches[0].clientY : 0);
                const dy = startY - y;
                const range = max - min;
                const delta = (dy / 200) * range;
                let newVal = Math.max(min, Math.min(max, startVal + delta));

                stateObj[key] = newVal;
                updateVisuals(newVal);
                if (callback) callback(newVal);
            };

            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onMove);
                window.removeEventListener('mouseup', onUp);
                window.removeEventListener('touchend', onUp);
            };

            const onDown = (e) => {
                e.preventDefault();
                startY = e.clientY || (e.touches ? e.touches[0].clientY : 0);
                startVal = stateObj[key];
                window.addEventListener('mousemove', onMove);
                window.addEventListener('touchmove', onMove, { passive: false });
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchend', onUp);
            };

            el.addEventListener('mousedown', onDown);
            el.addEventListener('touchstart', onDown, { passive: false });

            el.addEventListener('dblclick', (e) => {
                try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                const def = (key === 'gain') ? 1.0 : 0;
                stateObj[key] = def;
                updateVisuals(def);
                if (callback) callback(def);
            });
        }

        /** Link a DJ Deck mirror knob to an existing Mixer knob id (same eqState / Web Audio). */
        function wireDjKnobMirror(mirrorEl, knobId, signal) {
            if (!mirrorEl || !knobId) return;
            const cfg = window.__mixKnobConfig && window.__mixKnobConfig[knobId];
            if (!cfg) return;
            const { min, max, stateObj, key, callback } = cfg;
            try {
                if (!window.djKnobMirrors) window.djKnobMirrors = Object.create(null);
                window.djKnobMirrors[knobId] = mirrorEl;
            } catch (_) {}

            const updateAll = (val) => {
                stateObj[key] = val;
                const mixerEl = document.getElementById(knobId);
                if (mixerEl) setKnobUi(mixerEl, min, max, val);
                setKnobUi(mirrorEl, min, max, val);
                if (callback) callback(val);
            };

            updateAll(stateObj[key]);

            let startY = 0;
            let startVal = 0;

            const onMove = (e) => {
                const y = e.clientY || (e.touches ? e.touches[0].clientY : 0);
                const dy = startY - y;
                const range = max - min;
                const delta = (dy / 200) * range;
                let newVal = Math.max(min, Math.min(max, startVal + delta));
                updateAll(newVal);
            };

            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onMove);
                window.removeEventListener('mouseup', onUp);
                window.removeEventListener('touchend', onUp);
            };

            const onDown = (e) => {
                e.preventDefault();
                startY = e.clientY || (e.touches ? e.touches[0].clientY : 0);
                startVal = stateObj[key];
                window.addEventListener('mousemove', onMove);
                window.addEventListener('touchmove', onMove, { passive: false });
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchend', onUp);
            };

            const opts = signal ? { signal } : {};
            mirrorEl.addEventListener('mousedown', onDown, opts);
            mirrorEl.addEventListener('touchstart', onDown, Object.assign({ passive: false }, opts));
            mirrorEl.addEventListener('dblclick', (e) => {
                try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                const def = (key === 'gain') ? 1.0 : 0;
                updateAll(def);
            }, opts);
        }
        // Radio retry policy
        let radioRetryAttempts = 0;
        let radioBRetryAttempts = 0;
        const MAX_RADIO_RETRIES = 6;
        // WebM auto-random toggle
        let webmAutoOn = false;
        let webmAutoTimer = null;
		// Bottom text color history (avoid immediate repeats)
		let recentBottomColors = [];
		// Overlay glow color cycler
		let overlayGlowCycleCount = 0;
		let overlayGlowListenerBound = false;
		let overlayGlowColorTimer = null;
		const overlayGlowDurationMs = 2400; // must match CSS animation duration
		let tapGifLoadPromise = null;
		/** tap.gif is used by #border-frame border-image; glow must wait until it is ready */
		function ensureTapGifLoaded() {
			if (tapGifLoadPromise) return tapGifLoadPromise;
			tapGifLoadPromise = new Promise((resolve) => {
				const img = new Image();
				const finish = () => {
					try { resolve(); } catch (_) {}
				};
				img.onload = () => {
					try {
						const d = img.decode && img.decode();
						if (d && typeof d.then === 'function') {
							d.then(finish).catch(finish);
							return;
						}
					} catch (_) {}
					finish();
				};
				img.onerror = finish;
				img.src = 'assets/gifs/tap.gif';
			});
			return tapGifLoadPromise;
		}
		let ptaGifLoadPromise = null;
		function ensurePtaGifLoaded() {
			if (ptaGifLoadPromise) return ptaGifLoadPromise;
			ptaGifLoadPromise = new Promise((resolve) => {
				const img = new Image();
				const finish = () => resolve();
				img.onload = () => {
					try {
						const d = img.decode && img.decode();
						if (d && typeof d.then === 'function') {
							d.then(finish).catch(finish);
							return;
						}
					} catch (_) {}
					finish();
				};
				img.onerror = finish;
				img.src = 'assets/gifs/pta.gif';
			});
			return ptaGifLoadPromise;
		}
		let patGifLoadPromise = null;
		/** pat.gif fills the OMNI letters via background-clip: text; #logo-omni
		 *  stays at opacity 0 until this promise resolves, then a CSS class
		 *  triggers one 3s opacity transition for the entire "OMNI>" group.
		 *  Mirrors ensureTapGifLoaded / ensurePtaGifLoaded. */
		function ensurePatGifLoaded() {
			if (patGifLoadPromise) return patGifLoadPromise;
			patGifLoadPromise = new Promise((resolve) => {
				const img = new Image();
				const finish = () => resolve();
				img.onload = () => {
					try {
						const d = img.decode && img.decode();
						if (d && typeof d.then === 'function') {
							d.then(finish).catch(finish);
							return;
						}
					} catch (_) {}
					finish();
				};
				img.onerror = finish;
				img.src = 'assets/gifs/pat.gif';
			});
			return patGifLoadPromise;
		}
		let startScreenRevealPromise = null;
		function resetStartScreenReveal() {
			startScreenRevealPromise = null;
			try {
				const pta = document.getElementById('pta-start-bg');
				if (pta) pta.classList.remove('pta-visible');
			} catch (_) {}
			try {
				const logo = document.getElementById('logo-omni');
				if (logo) logo.classList.remove('pat-revealed');
			} catch (_) {}
			try {
				const border = document.getElementById('border-frame');
				if (border) border.classList.remove('visible');
			} catch (_) {}
		}
		/** Fade in pta/tap/pat start visuals together once all three GIFs have decoded. */
		function revealStartScreenAfterAssets() {
			if (startScreenRevealPromise) return startScreenRevealPromise;
			startScreenRevealPromise = Promise.all([
				ensureTapGifLoaded(),
				ensurePtaGifLoaded(),
				ensurePatGifLoaded()
			]).then(() => {
				requestAnimationFrame(() => {
					if (!isStartOverlayShowing()) return;
					try {
						const pta = document.getElementById('pta-start-bg');
						if (pta) pta.classList.add('pta-visible');
					} catch (_) {}
					try {
						const logo = document.getElementById('logo-omni');
						if (logo) logo.classList.add('pat-revealed');
					} catch (_) {}
					try {
						const border = document.getElementById('border-frame');
						if (border) border.classList.add('visible');
					} catch (_) {}
					try {
						const glow = globalThis.applyOverlayGlowFx;
						if (typeof glow === 'function') glow();
					} catch (_) {}
				});
			});
			return startScreenRevealPromise;
		}
		function fadeInPtaStartBg() {
			return revealStartScreenAfterAssets();
		}
		function hidePtaStartBg() {
			try {
				const el = document.getElementById('pta-start-bg');
				if (el) el.classList.remove('pta-visible');
			} catch (_) {}
		}
		function isStartOverlayShowing() {
			try {
				const overlay = document.getElementById('overlay');
				if (!overlay || overlay.classList.contains('hidden')) return false;
				const cs = window.getComputedStyle(overlay);
				return cs.display !== 'none';
			} catch (_) {
				return false;
			}
		}
		function revealStartScreenEdgeFx() {
			return revealStartScreenAfterAssets();
		}
		function randomHexColor() {
			return '#' + Math.floor(Math.random()*0xFFFFFF).toString(16).padStart(6,'0');
		}
		// Typed status text effect
		let statusTypeTimer = null;
		function typeStatus(text, onDone) {
			const el = document.getElementById('loading-status');
			if (!el) return;
			if (statusTypeTimer) { try { clearTimeout(statusTypeTimer); } catch(e) {} statusTypeTimer = null; }
			const full = String(text || '');
			el.innerText = '';
			let i = 0;
			const step = () => {
				el.innerText = full.slice(0, i);
				try { if (typeof layoutOverlayElements === 'function') layoutOverlayElements(); } catch(e) {}
				i++;
				if (i <= full.length) {
					statusTypeTimer = setTimeout(step, 60);
				} else {
					try { if (typeof onDone === 'function') onDone(); } catch(e) {}
				}
			};
			step();
		}
function randomGlowColor() {
    const h = Math.floor(Math.random() * 360);
    return `hsl(${h}, 90%, 60%)`;
}
		// Generic typewriter for any element id
		let shortcutsTypeTimer = null;
		function typeStatusTo(elId, text, speed) {
			const el = document.getElementById(elId);
			if (!el) return;
			const useSpeed = typeof speed === 'number' ? speed : 50;
			// separate timer for shortcuts line to avoid interference
			if (elId === 'shortcuts-status' && shortcutsTypeTimer) { try { clearTimeout(shortcutsTypeTimer); } catch(e) {} shortcutsTypeTimer = null; }
			const full = String(text || '');
			el.innerText = '';
			let i = 0;
			const step = () => {
				el.innerText = full.slice(0, i);
				try { if (typeof layoutOverlayElements === 'function') layoutOverlayElements(); } catch(e) {}
				i++;
				if (i <= full.length) {
					if (elId === 'shortcuts-status') shortcutsTypeTimer = setTimeout(step, useSpeed);
					else setTimeout(step, useSpeed);
				}
			};
			step();
		}
		// Layout helper to keep URL and shortcuts a fixed distance from title/status
		function layoutOverlayElements() {
			const logo = document.getElementById('logo-omni');
			const urlFly = document.getElementById('url-flyover');
			const shortcuts = document.getElementById('shortcuts-status');
			const status = document.getElementById('loading-status');
			if (!logo) return;
			const logoRect = logo.getBoundingClientRect();
			// Tight, consistent gaps to avoid large spacing
			const gapAbove = 50; // px above title (URL input offset)
			const gapBelow = 10; // px below status/title
			// Position URL flyover above title
			if (urlFly) {
				const y = Math.max(0, Math.round(logoRect.top - gapAbove));
				urlFly.style.top = y + 'px';
			}
			// Position shortcuts below (under status if present, else under title)
			if (shortcuts) {
				let anchorBottom = logoRect.bottom;
				if (status) {
					const stRect = status.getBoundingClientRect();
					anchorBottom = Math.max(anchorBottom, stRect.bottom);
				}
				const top = Math.round(anchorBottom + gapBelow);
				shortcuts.style.top = top + 'px';
				// Ensure up to 5 lines visible
				const lineHeight = 1.35 * 12; // matches CSS font-size:12px; line-height:1.35
				shortcuts.style.maxHeight = Math.round(lineHeight * 5) + 'px';
			}
		}
		function setBottomTextRandomColor() {
			const titleEl = document.getElementById('mode-title');
			const subEl = document.getElementById('mode-sub');
			if (!titleEl && !subEl) return;
			const pickHudHsl = (hue, sat, lit) => `hsl(${hue} ${sat}% ${lit}%)`;
			let titleHue = Math.floor(Math.random() * 360);
			let subHue = (titleHue + 132 + Math.floor(Math.random() * 96)) % 360;
			let pairKey = '';
			let attempts = 0;
			do {
				titleHue = Math.floor(Math.random() * 360);
				subHue = (titleHue + 132 + Math.floor(Math.random() * 96)) % 360;
				pairKey = `${titleHue}|${subHue}`;
				attempts++;
			} while (recentBottomColors.includes(pairKey) && attempts < 12);
			const titleColor = pickHudHsl(titleHue, 78 + Math.floor(Math.random() * 14), 68 + Math.floor(Math.random() * 12));
			const subColor = pickHudHsl(subHue, 72 + Math.floor(Math.random() * 18), 62 + Math.floor(Math.random() * 14));
			if (titleEl) titleEl.style.color = titleColor;
			if (subEl) subEl.style.color = subColor;
			recentBottomColors.push(pairKey);
			if (recentBottomColors.length > 16) recentBottomColors.shift();
		}
        let modeShuffleOn = false;
        let modeShuffleTimer = null;
        
        // Settings state (for ProjectM v2)
        const visualSettings = {
            shuffleMinSec: 30,
            shuffleMaxSec: 60,
            transitionSec: 2.7,
            pixelRatio: QUALITY.pixelRatioCap // FIXED: Uses 1.5 on mobile, 3.0 on desktop
        };

        // --- UI UTILS ---
        function resetIdleTimer() {
            uiLayer.style.opacity = '1';
			if (radioQuickBtn) {
				radioQuickBtn.style.opacity = '1';
				radioQuickBtn.style.pointerEvents = 'auto';
			}
            if (btnOptions && state.isPlaying) {
                btnOptions.style.opacity = '1';
                btnOptions.style.pointerEvents = 'auto';
            }
            // Show volume slider on interaction
            const vs = document.getElementById('volume-slider-container');
            if (vs && state.isPlaying) { vs.style.opacity = '1'; vs.style.pointerEvents = 'auto'; }
            // Show top bar on interaction
            if (topBar && state.isPlaying) { topBar.style.opacity = '1'; topBar.style.pointerEvents = 'auto'; }
			if(radioPanel && !radioPanel.classList.contains('display-none')) {
				radioPanel.style.opacity = '1';
				radioPanel.style.pointerEvents = 'auto';
			}
            // Show webm nav when overlay is active
            if(webmOn && !webmOverlayEl.classList.contains('display-none')) {
                webmPrevBtn.style.opacity = '1';
                webmNextBtn.style.opacity = '1';
                webmPrevBtn.style.pointerEvents = 'auto';
                webmNextBtn.style.pointerEvents = 'auto';
            }
            document.body.style.cursor = 'default';
            if (state.idleTimer) clearTimeout(state.idleTimer);
            if (panelIdleTimer) clearTimeout(panelIdleTimer);
            if (radioPanelTimer) clearTimeout(radioPanelTimer);
            if (settingsPanelTimer) clearTimeout(settingsPanelTimer);
            if (webmSettingsTimer) clearTimeout(webmSettingsTimer);
            if(state.isPlaying) {
                state.idleTimer = setTimeout(() => {
                    uiLayer.style.opacity = '0';
					// Honor hold window after right-click
					if (Date.now() >= radioQuickHoldUntil && radioQuickBtn) {
						radioQuickBtn.style.opacity = '0';
						radioQuickBtn.style.pointerEvents = 'none';
					}
                    // Hide volume slider on idle (but keep visible if top menu is open)
                    const vs = document.getElementById('volume-slider-container');
                    const menu = document.getElementById('top-menu-panel');
                    const menuOpen = !!(menu && !menu.classList.contains('display-none') && menu.classList.contains('open'));
                    if (vs) {
                        if (menuOpen) {
                            vs.style.opacity = '1';
                            vs.style.pointerEvents = 'auto';
                        } else {
                            vs.style.opacity = '0';
                            vs.style.pointerEvents = 'none';
                        }
                    }
                    // Hide top bar on idle
                    if (topBar) { topBar.style.opacity = '0'; topBar.style.pointerEvents = 'none'; }
                    if (btnOptions) {
                        btnOptions.style.opacity = '0';
                        btnOptions.style.pointerEvents = 'none';
                    }
                    // Hide webm nav
                    webmPrevBtn.style.opacity = '0';
                    webmNextBtn.style.opacity = '0';
                    webmPrevBtn.style.pointerEvents = 'none';
                    webmNextBtn.style.pointerEvents = 'none';
                    document.body.style.cursor = 'none';
                }, 3000);
				// Panels fade out and close after 30s if open
				if(radioPanel && !radioPanel.classList.contains('display-none')) scheduleRadioPanelClose();
                if(!settingsPanel.classList.contains('display-none')) scheduleSettingsPanelClose();
                if(!webmSettingsPanel.classList.contains('display-none')) scheduleWebmSettingsClose();
            }
        }
        window.addEventListener('mousemove', resetIdleTimer);
        window.addEventListener('click', resetIdleTimer);
        window.addEventListener('resize', () => { try { layoutOverlayElements(); } catch(e) {} });

        // Updated Fullscreen Logic with forced Resize Trigger
        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().then(() => {
                    setTimeout(() => forceResize(), 200);
                }).catch(err => console.log(err));
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen().then(() => {
                        setTimeout(() => forceResize(), 200);
                    });
                }
            }
        }

        function getDeckBVizMountEl() {
            return document.getElementById('dj-deck-b-viz-mount');
        }
        function getDeckBVizFullscreenEl() {
            const mount = getDeckBVizMountEl();
            if (!mount) return null;
            const fs = document.fullscreenElement || document.webkitFullscreenElement;
            if (!fs) return null;
            if (fs === mount || mount.contains(fs)) return fs;
            return null;
        }
        function afterDeckBVizFullscreenChange() {
            setTimeout(() => {
                try { forceResize(); } catch (_) {}
                try { applyWebmSettings(); } catch (_) {}
            }, 200);
        }
        function exitDeckBVizFullscreen() {
            try {
                if (document.exitFullscreen) {
                    document.exitFullscreen().then(afterDeckBVizFullscreenChange).catch(afterDeckBVizFullscreenChange);
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                    afterDeckBVizFullscreenChange();
                }
            } catch (_) {}
        }
        /** Fullscreen Deck B visuals (`#dj-deck-b-viz-mount` or nested `.dj-video-shell`), not the whole DJ canvas. */
        function toggleVideoSurfaceFullscreen(videoEl, containerEl) {
            if (!videoEl) return false;
            const container = containerEl || videoEl.parentElement;
            if (!container) return false;
            const now = Date.now();
            if (now - (window.__deckBVizFsToggleAt || 0) < 450) return false;
            window.__deckBVizFsToggleAt = now;
            try {
                const fs = document.fullscreenElement || document.webkitFullscreenElement;
                if (fs && (fs === videoEl || fs === container || container.contains(fs))) {
                    exitDeckBVizFullscreen();
                    return true;
                }
                const target = container;
                const req = target.requestFullscreen || target.webkitRequestFullscreen;
                if (req) {
                    req.call(target).then(afterDeckBVizFullscreenChange).catch(() => {});
                    return true;
                }
            } catch (_) {}
            return false;
        }
        function toggleDeckBVizMountFullscreen() {
            const mount = getDeckBVizMountEl();
            if (!mount) return false;
            const now = Date.now();
            if (now - (window.__deckBVizFsToggleAt || 0) < 450) return false;
            window.__deckBVizFsToggleAt = now;
            try {
                if (getDeckBVizFullscreenEl()) {
                    exitDeckBVizFullscreen();
                    return true;
                }
                let target = mount;
                try {
                    const av = state && state.activeVisualizer;
                    if (av && av.name === 'DJ Decks' && av.deckBVizMode === 'video') {
                        const shell = mount.querySelector('.dj-video-shell');
                        if (shell) target = shell;
                    }
                } catch (_) {}
                const req = target.requestFullscreen || target.webkitRequestFullscreen;
                if (req) {
                    req.call(target).then(afterDeckBVizFullscreenChange).catch(() => {});
                }
            } catch (_) {}
            return true;
        }
        (function wireDeckBVizFullscreenChange() {
            if (window.__deckBVizFsChangeWired) return;
            window.__deckBVizFsChangeWired = true;
            const onFs = () => {
                try { afterDeckBVizFullscreenChange(); } catch (_) {}
            };
            document.addEventListener('fullscreenchange', onFs);
            document.addEventListener('webkitfullscreenchange', onFs);
        })();

        /** Reliable hit-test for Deck B ProjectM / AUDIO:BAR canvas — WebGL targets sometimes skip closest() from container. */
        function isEventOnDeckBVizMount(e) {
            try {
                if (e && e.target && typeof e.target.closest === 'function' && e.target.closest('#dj-deck-b-viz-mount')) return true;
                if (e && typeof e.composedPath === 'function') {
                    const path = e.composedPath();
                    for (let i = 0; i < path.length; i++) {
                        const n = path[i];
                        if (n && n.nodeType === 1 && n.id === 'dj-deck-b-viz-mount') return true;
                    }
                }
            } catch (_) {}
            return false;
        }

        /** Deck B viz mount uses `pointer-events: none` so ProjectM / AUDIO:BAR often miss DOM hit-testing; use mode + coordinates. */
        function isDeckBVizBarsOrProjectMActive() {
            try {
                const av = state && state.activeVisualizer;
                if (!av || av.name !== 'DJ Decks') return false;
                const m = av.deckBVizMode;
                if (m !== 'bars' && m !== 'projectm') return false;
                const stage = document.querySelector('#dj-visual-root .dj-deck-b-stage');
                return !!(stage && stage.classList.contains('dj-deck-b-visual-mode'));
            } catch (_) {}
            return false;
        }

        function isPointerInDeckBVizMountRect(clientX, clientY) {
            try {
                const mount = document.getElementById('dj-deck-b-viz-mount');
                if (!mount) return false;
                const r = mount.getBoundingClientRect();
                if (r.width < 4 || r.height < 4) return false;
                const x = Number(clientX);
                const y = Number(clientY);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
                return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
            } catch (_) {}
            return false;
        }

        function shouldToggleDeckBVizMountFullscreenFromPointer(e) {
            try {
                if (isEventOnDeckBVizMount(e)) return true;
            } catch (_) {}
            if (!isDeckBVizBarsOrProjectMActive()) return false;
            try {
                const x = e.clientX != null ? e.clientX : (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientX : NaN);
                const y = e.clientY != null ? e.clientY : (e.changedTouches && e.changedTouches[0] ? e.changedTouches[0].clientY : NaN);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
                const topEl = document.elementFromPoint(x, y);
                if (topEl && typeof topEl.closest === 'function') {
                    if (topEl.closest('.knob-wrap') || topEl.closest('#dj-deck-splitter') || topEl.closest('.dj-jog-wrap')) return false;
                    if (topEl.closest('#dj-deck-b-viz-mount')) return true;
                }
                return isPointerInDeckBVizMountRect(x, y);
            } catch (_) {}
            return false;
        }

        function forceResize() {
            if(state.activeVisualizer && state.activeVisualizer.onResize) {
                state.activeVisualizer.onResize();
            }
			try { layoutOverlayElements(); } catch(e) {}
		}
		// Start screen text loop (fade out after 30s, retype after 30s) until playing
		let startTextLoopTimer = null;
		function cancelStartTextLoop() {
			if (startTextLoopTimer) { try { clearTimeout(startTextLoopTimer); } catch(e) {} startTextLoopTimer = null; }
		}
		function scheduleStartTextLoop() {
			try { cancelStartTextLoop(); } catch(e) {}
			if (state.isPlaying) return;
			const sh = document.getElementById('shortcuts-status');
			if (!sh) return;
			sh.style.opacity = '1';
			const shortcuts = [
				'F Fullscreen  •  C Next Visual  •  ,/. Visual',
				'V Play/Pause A  •  B Play/Pause B  •  N Next Station  •  L Lock',
				'T Text-In Panel  •  Y Text-In Auto  •  U Send Text',
                'P Radio Stations  •  R Avatar Settings  •  W Toggle Avatar  •  Arrows Move Avatar',
				'H QUEUE  •  J VIDEO  •  K KARAOKE (Deck B)',
				'+/− Size  •  Q/E Speed  •  Z/X Opacity  •  Space Auto-Fade  •  Esc Back'
			].join('\n');
			let phase = 'visible';
			const loop = () => {
				if (state.isPlaying) { cancelStartTextLoop(); return; }
				if (phase === 'visible') {
					startTextLoopTimer = setTimeout(() => {
						sh.style.opacity = '0';
						phase = 'hidden';
						loop();
					}, 30000);
				} else {
					startTextLoopTimer = setTimeout(() => {
                        try { sh.innerText = ''; } catch(_) {}
						sh.style.opacity = '1';
						try { typeStatusTo('shortcuts-status', shortcuts, 30); } catch(e) {}
						phase = 'visible';
						loop();
					}, 30000);
				}
			};
			loop();
        }
		// ">" starts radio; "M" opens MIDI; "I" opens microphone (confirm)
		(() => {
			try {
				const logo = document.getElementById('logo-omni');
				if (!logo) return;
				const randomOmniLetterBorder = () =>
					'#' + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0');
				logo.querySelectorAll('.logo-letter').forEach((el) => {
					el.addEventListener('mouseenter', () => {
						try {
							el.style.setProperty('--omni-letter-border', randomOmniLetterBorder());
							el.classList.add('omni-letter-hover-active');
						} catch (_) {}
					});
					el.addEventListener('mouseleave', () => {
						try {
							el.classList.remove('omni-letter-hover-active');
						} catch (_) {}
					});
				});
				const starts = logo.querySelectorAll('.logo-start');
				starts.forEach((el) => {
					el.style.cursor = 'pointer';
					el.title = 'Start Radio';
					el.addEventListener('click', (e) => {
						e.stopPropagation();
						const btn = document.getElementById('btn-radio');
						if (btn) btn.click();
						else if (typeof playRadio === 'function') playRadio();
					});
				});
				const midiLetter = logo.querySelector('.logo-m');
				if (midiLetter) {
					midiLetter.style.cursor = 'pointer';
					midiLetter.title = 'Enable MIDI Input';
					midiLetter.addEventListener('click', (e) => {
						e.stopPropagation();
						try { showMidiConfirm(); } catch (_) {
							try { initWebMidi(); } catch (_) {}
						}
					});
				}
				// Make "N" toggle the URL input flyover (reserve space via is-hidden)
				const nEl = logo.querySelector('.logo-letter:nth-child(3)');
				if (nEl) {
					nEl.style.cursor = 'pointer';
					nEl.title = 'Toggle Stream URL';
					nEl.addEventListener('click', (e) => {
						e.stopPropagation();
						const fly = document.getElementById('url-flyover');
						if (!fly) return;
						const hidden = fly.classList.contains('is-hidden');
						if (hidden) fly.classList.remove('is-hidden');
						else fly.classList.add('is-hidden');
						// focus/select input when showing
						if (!hidden) return;
                    const inp = document.getElementById('radio-url') || document.getElementById('station-url');
						if (inp) {
							try { inp.focus(); inp.select(); } catch(e) {}
						}
					});
				}
				// Make "O" load local audio
				const oEl = logo.querySelector('.logo-o');
				if (oEl) {
					oEl.style.cursor = 'pointer';
					oEl.title = 'Load Local Audio';
					oEl.addEventListener('click', (e) => {
						e.stopPropagation();
                        // Suppress overlay click-to-start shortly after opening file dialog
                        try { suppressNextOverlayStartUntil = Date.now() + 1200; } catch(_) {}
						const btn = document.getElementById('btn-file');
						if (btn) btn.click();
						else {
							const input = document.getElementById('file-input');
							if (input) input.click();
						}
					});
				}
				// Make "I" use mic (with confirm)
				const micLetter = logo.querySelector('.logo-midi');
				if (micLetter) {
					micLetter.style.cursor = 'pointer';
					micLetter.title = 'Use Microphone';
					micLetter.addEventListener('click', (e) => {
						e.stopPropagation();
						try { showMicConfirm(); } catch(_) {
							const btn = document.getElementById('btn-mic');
							if (btn) btn.click();
							else if (typeof useMic === 'function') useMic();
						}
					});
				}
			} catch(e) {}
		})();
        
        // Start radio when clicking anywhere on the start screen background (excluding title/inputs)
        (function bindOverlayClickToStart() {
            try {
                const overlay = document.getElementById('overlay');
                if (!overlay) return;
                overlay.addEventListener('click', (e) => {
                    if (state && state.isPlaying) return;
                    // If a recent "O" triggered file dialog, do not start radio
                    try { if (Date.now() < suppressNextOverlayStartUntil) return; } catch(_) {}
                    const logo = document.getElementById('logo-omni');
                    const urlFly = document.getElementById('url-flyover');
                    // Ignore clicks on title (OMNI>) and URL input area
                    if ((logo && logo.contains(e.target)) || (urlFly && urlFly.contains(e.target))) {
                        return;
                    }
                    try { e.stopPropagation(); } catch(_) {}
                    if (typeof playRadio === 'function') playRadio();
                });
            } catch(e) {}
        })();
        // --- Bottom Avatar Menu helpers ---
        function isBottomMenuOpen() {
            const p = document.getElementById('bottom-avatar-panel');
            return !!(p && !p.classList.contains('display-none') && p.classList.contains('open'));
        }
        function openBottomMenuPanel() {
            if (uiLocked) return;
            // Prevent opening if a recent panel close just happened
            try { if (window.__panelGuardUntilMs && Date.now() < window.__panelGuardUntilMs) return; } catch(_) {}
            const p = document.getElementById('bottom-avatar-panel');
            if (!p) return;
            try {
                // Populate WebM select
                const sel = document.getElementById('avatar-webm-select');
                if (sel) {
                    sel.innerHTML = '';
                    if (Array.isArray(webmList) && webmList.length > 0) {
                        webmList.forEach((pth, idx) => {
                            const opt = document.createElement('option');
                            opt.value = String(idx);
                            opt.textContent = (pth || '').split('/').pop() || pth || ('Item ' + (idx+1));
                            if (typeof webmIndex === 'number' && idx === webmIndex) opt.selected = true;
                            sel.appendChild(opt);
                        });
                    } else {
                        const opt = document.createElement('option');
                        opt.value = '0';
                        opt.textContent = '(load list)';
                        sel.appendChild(opt);
                    }
                }
                // Sync inputs from current settings
                const aScale = document.getElementById('avatar-inp-scale');
                const aX = document.getElementById('avatar-inp-x');
                const aY = document.getElementById('avatar-inp-y');
                const aRot = document.getElementById('avatar-inp-rot');
                const aSpeed = document.getElementById('avatar-inp-speed');
                const aOpacity = document.getElementById('avatar-inp-opacity');
                const aDupSpacing = document.getElementById('avatar-inp-dup-spacing');
                if (aScale) aScale.value = String(webmSettings.scaleVw);
                if (aX) aX.value = String(webmSettings.posXvw);
                if (aY) aY.value = String(webmSettings.posYvh);
                if (aRot) aRot.value = String(webmSettings.rotationDeg);
                if (aSpeed) aSpeed.value = String(webmSettings.playbackRate);
                if (aOpacity) aOpacity.value = String(webmSettings.opacity);
                if (aDupSpacing) aDupSpacing.value = String(Math.round((webmSettings.duplicateSpacing || WEBM_DEFAULT_DUP_SPACING) * 100));
                try { syncAllWebmDupKnobs(); } catch (_) {}
                const autoBtn = document.getElementById('avatar-btn-auto');
                if (autoBtn) { autoBtn.textContent = 'Auto'; autoBtn.classList.toggle('on', webmAutoOn); }
                // Sync play/stop label
                try { updateAvatarPlayButton(); } catch(_) {}
            } catch(_) {}
            p.classList.remove('display-none');
            requestAnimationFrame(() => { p.classList.add('open'); });
        }
        function closeBottomMenuPanel() {
            const p = document.getElementById('bottom-avatar-panel');
            if (!p) return;
            p.classList.remove('open');
            try { window.__panelGuardUntilMs = Date.now() + 1200; } catch(_) {}
            setTimeout(() => { p.classList.add('display-none'); }, 350);
        }
        function toggleBottomMenuPanel() {
            if (uiLocked) return;
            if (isBottomMenuOpen()) closeBottomMenuPanel(); else openBottomMenuPanel();
        }
        // Bind bottom menu controls
        (function bindBottomMenuControls(){
            try {
                const btnClose = document.getElementById('btn-bottommenu-close');
                if (btnClose) btnClose.addEventListener('click', (e)=>{ e.stopPropagation(); closeBottomMenuPanel(); });
                const selWebm = document.getElementById('avatar-webm-select');
                const btnOpenWebm = document.getElementById('avatar-webm-open');
                const btnPrevWebm = document.getElementById('avatar-webm-prev');
                const btnNextWebm = document.getElementById('avatar-webm-next');
                if (btnPrevWebm) btnPrevWebm.addEventListener('click', (e)=>{ e.stopPropagation(); prevWebm(); });
                if (btnNextWebm) btnNextWebm.addEventListener('click', (e)=>{ e.stopPropagation(); nextWebm(); });
                if (btnOpenWebm) btnOpenWebm.addEventListener('click', (e)=>{ 
                    e.stopPropagation();
                    // Toggle behavior: if open, close; otherwise open selected
                    if (typeof webmOn !== 'undefined' && webmOn) {
                        try { hideWebm(); } catch(_) {}
                        try { updateAvatarPlayButton(); } catch(_) {}
                        return;
                    }
                    if (typeof loadWebmList === 'function' && webmList.length === 0) {
                        loadWebmList().finally(()=>{ try { const idx = Math.max(0, Math.min(webmList.length-1, parseInt((selWebm?.value)||'0',10)||0)); setWebm(idx); showWebm(); updateAvatarPlayButton(); } catch(_){} });
                    } else {
                        try { const idx = Math.max(0, Math.min(webmList.length-1, parseInt((selWebm?.value)||'0',10)||0)); setWebm(idx); showWebm(); updateAvatarPlayButton(); } catch(_){}
                    }
                });
                const aScale = document.getElementById('avatar-inp-scale');
                const aX = document.getElementById('avatar-inp-x');
                const aY = document.getElementById('avatar-inp-y');
                const aRot = document.getElementById('avatar-inp-rot');
                const aSpeed = document.getElementById('avatar-inp-speed');
                const aOpacity = document.getElementById('avatar-inp-opacity');
                const aDupSpacing = document.getElementById('avatar-inp-dup-spacing');
                const updateFromAvatarInputs = () => {
                    webmSettings.scaleVw = Number(aScale?.value) || WEBM_DEFAULT_SCALE_VW;
                    webmSettings.posXvw = Number(aX?.value) || 50;
                    webmSettings.posYvh = Number(aY?.value) || 50;
                    webmSettings.rotationDeg = Number(aRot?.value) || 0;
                    webmSettings.playbackRate = Math.max(0.1, Math.min(4, Number(aSpeed?.value) || 1));
                    webmSettings.opacity = Math.max(0, Math.min(1, Number(aOpacity?.value) || 1));
                    webmSettings.duplicateSpacing = Math.max(0.15, Math.min(1, (Number(aDupSpacing?.value) || 60) / 100));
                    try { syncWebmDupSpacingInputs(); } catch (_) {}
                    applyWebmSettings();
                };
                [aScale, aX, aY, aRot, aSpeed, aOpacity, aDupSpacing].forEach(el => {
                    if (el) el.addEventListener('input', updateFromAvatarInputs);
                });
                const btnAuto = document.getElementById('avatar-btn-auto');
                if (btnAuto) btnAuto.addEventListener('click', (e)=>{ 
                    e.preventDefault(); 
                    e.stopPropagation(); 
                    setWebmAuto(!webmAutoOn); 
                    const autoBtn = document.getElementById('avatar-btn-auto'); 
                    if (autoBtn) { autoBtn.textContent = 'Auto'; autoBtn.classList.toggle('on', webmAutoOn); }
                    // If auto just turned on and no WebM is playing, start the first WebM
                    try {
                        if (webmAutoOn && !webmOn) {
                            if (webmList.length === 0 && typeof loadWebmList === 'function') {
                                loadWebmList().finally(() => {
                                    if (webmList.length > 0) { setWebm(0); showWebm(); }
                                });
                            } else if (webmList.length > 0) {
                                setWebm(0); showWebm();
                            }
                        }
                    } catch(_) {}
                });
                const btnReset = document.getElementById('avatar-btn-reset');
                if (btnReset) btnReset.addEventListener('click', (e)=>{ 
                    e.preventDefault(); e.stopPropagation();
                    webmSettings.scaleVw = WEBM_DEFAULT_SCALE_VW;
                    webmSettings.posXvw = 50;
                    webmSettings.posYvh = 50;
                    webmSettings.rotationDeg = 0;
                    webmSettings.playbackRate = 1.0;
                    webmSettings.opacity = 0.82;
                    webmSettings.duplicates = 0;
                    webmSettings.duplicateSpacing = WEBM_DEFAULT_DUP_SPACING;
                    if (aScale) aScale.value = String(WEBM_DEFAULT_SCALE_VW);
                    if (aX) aX.value = '50';
                    if (aY) aY.value = '50';
                    if (aRot) aRot.value = '0';
                    if (aSpeed) aSpeed.value = '1.0';
                    if (aOpacity) aOpacity.value = '0.82';
                    if (aDupSpacing) aDupSpacing.value = String(Math.round(WEBM_DEFAULT_DUP_SPACING * 100));
                    try { syncAllWebmDupKnobs(); } catch (_) {}
                    try { syncWebmDupSpacingInputs(); } catch (_) {}
                    applyWebmSettings();
                });
            } catch(e) {}
        })();

        // Gesture handling: swipe to change visual, double-tap to open WebM; WebM overlay swipes to navigate videos
        (function bindGestures() {
            try {
                const canvasEl = document.getElementById('canvas-container');
                const webmOverlay = document.getElementById('webm-overlay');
                if (!canvasEl) return;
                const SWIPE_PX = 40;
                /** Horizontal vs vertical: require dominant axis to beat the other by this factor (reduces diagonal mis-classification). */
                const SWIPE_AXIS_DOMINANCE = 1.85;
                /** After any canvas swipe changes visuals or opens/closes menus, ignore competing gestures (esp. trackpad wheel). */
                const CANVAS_GESTURE_COOLDOWN_MS = 500;
                const TAP_MAX_MOVE = 8;
                const DOUBLE_TAP_MS = 300;
                let pDownX = 0, pDownY = 0, pDownTime = 0;
                let pDownTarget = null;
                let lastTapTime = 0;
                let handledSwipe = false;
                // Suppress random-station click when we just handled a swipe/double-tap
                window.__suppressNextClick = false;
                const canvasGestureLocked = () => Date.now() < (window.__canvasGestureLockUntil || 0);
                const lockCanvasGestures = () => {
                    window.__canvasGestureLockUntil = Date.now() + CANVAS_GESTURE_COOLDOWN_MS;
                };
                const onPointerDown = (e) => {
                    const t = e.target || null;
                    if (t && typeof t.closest === 'function'
                        && (t.closest('#radio-visual-root') || t.closest('#dj-visual-root'))) {
                        pDownTarget = null;
                        return;
                    }
                    pDownTarget = t;
                    pDownX = e.clientX || (e.touches && e.touches[0]?.clientX) || 0;
                    pDownY = e.clientY || (e.touches && e.touches[0]?.clientY) || 0;
                    pDownTime = Date.now();
                    handledSwipe = false;
                };
                const onPointerUpCanvas = (e) => {
                    if (!state || !state.isPlaying) return;
                    if (!pDownTarget) return;
                    // ignore if UI overlays present
                    const overlay = document.getElementById('overlay');
                    if (overlay && !overlay.classList.contains('hidden') && overlay.style.display !== 'none') return;
                    const x = e.clientX || (e.changedTouches && e.changedTouches[0]?.clientX) || 0;
                    const y = e.clientY || (e.changedTouches && e.changedTouches[0]?.clientY) || 0;
                    const dx = x - pDownX;
                    const dy = y - pDownY;
                    const adx = Math.abs(dx), ady = Math.abs(dy);
                    const swipeVerticalIntent = ady > SWIPE_PX && ady > adx * SWIPE_AXIS_DOMINANCE;
                    const swipeHorizontalIntent = adx > SWIPE_PX && adx > ady * SWIPE_AXIS_DOMINANCE;
                    // Vertical swipe: down/up closes open panel first; only opens target if none open
                    if (swipeVerticalIntent) {
                        if (canvasGestureLocked()) return;
                        handledSwipe = true;
                        window.__suppressNextClick = true;
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        
                        // Guard against follow-up events
                        const guardUntil = (typeof window !== 'undefined' && window.__panelGuardUntilMs) ? window.__panelGuardUntilMs : 0;
                        if (Date.now() < guardUntil) return;

                        // Check which panels are currently open
                        const topOpen = (typeof isTopMenuOpen === 'function') && isTopMenuOpen();
                        const avatarOpen = (typeof isBottomMenuOpen === 'function') && isBottomMenuOpen(); // Now the side panel
                        // Check if Mix Panel is open (manually since it doesn't have an isXOpen helper)
                        const mixOpen = mixPanel && !mixPanel.classList.contains('display-none') && mixPanel.classList.contains('open');
                        const kbdSheetOpen = (typeof isKeyboardShortcutsPanelOpen === 'function') && isKeyboardShortcutsPanelOpen();

                        // PRIORITY: Close any open panel before opening a new one (top menu → Text-In → Avatar)
                        if (topOpen) {
                            if (typeof closeTopMenuPanel === 'function') closeTopMenuPanel();
                            lockCanvasGestures();
                            return;
                        }
                        const tipEl = document.getElementById('textin-panel');
                        const textInOpenVert = !!(tipEl && !tipEl.classList.contains('display-none') && tipEl.classList.contains('open'));
                        if (textInOpenVert) {
                            if (typeof hideTextInPanel === 'function') hideTextInPanel();
                            lockCanvasGestures();
                            return;
                        }
                        if (avatarOpen) {
                            if (typeof closeBottomMenuPanel === 'function') closeBottomMenuPanel();
                            lockCanvasGestures();
                            return;
                        }
                        
                        const vertFromDj = pDownTarget && typeof pDownTarget.closest === 'function' && pDownTarget.closest('#dj-visual-root');
                        const vertFromRv = pDownTarget && typeof pDownTarget.closest === 'function' && pDownTarget.closest('#radio-visual-root');

                        if (dy > 0) { // SWIPE DOWN
                            if (kbdSheetOpen) {
                                try { closeKeyboardShortcutsPanel(); } catch (_) {}
                                lockCanvasGestures();
                                return;
                            }
                            // If Mix Panel is open at the bottom, swipe down should close it
                            if (mixOpen) {
                                toggleMixPanel();
                                lockCanvasGestures();
                                return;
                            }
                            // Do not open Top Menu from vertical drags that started on DJ deck (e.g. TK filter pad)
                            if (vertFromDj || vertFromRv) return;
                            if (typeof openTopMenuPanel === 'function') openTopMenuPanel();
                            lockCanvasGestures();
                        } else { // SWIPE UP
                            // Open Mix Settings (instead of Avatar/WebM)
                            if (kbdSheetOpen) {
                                try { closeKeyboardShortcutsPanel(); } catch (_) {}
                                lockCanvasGestures();
                                return;
                            }
                            if (!mixOpen && !vertFromDj && !vertFromRv) {
                                toggleMixPanel();
                                lockCanvasGestures();
                            }
                        }
                        return;
                    }
                    // Horizontal swipe: close floating/side panels first, then change visual
                    if (swipeHorizontalIntent) {
                        try {
                            if (e.target && e.target.closest && e.target.closest('#dj-deck-splitter')) return;
                            if (pDownTarget && pDownTarget.closest && pDownTarget.closest('#dj-deck-splitter')) return;
                        } catch (_) {}
                        const gestureFromDeckOrRadioUi = (el) => {
                            if (!el || typeof el.closest !== 'function') return false;
                            return !!(el.closest('#dj-visual-root') || el.closest('#radio-visual-root'));
                        };
                        if (gestureFromDeckOrRadioUi(pDownTarget) || gestureFromDeckOrRadioUi(e.target)) {
                            return;
                        }
                        if (canvasGestureLocked()) return;
                        handledSwipe = true;
                        window.__suppressNextClick = true;
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        const guardH = (typeof window !== 'undefined' && window.__panelGuardUntilMs) ? window.__panelGuardUntilMs : 0;
                        if (Date.now() < guardH) return;

                        const topOpenH = (typeof isTopMenuOpen === 'function') && isTopMenuOpen();
                        const tipH = document.getElementById('textin-panel');
                        const textInOpenH = !!(tipH && !tipH.classList.contains('display-none') && tipH.classList.contains('open'));
                        const avatarOpenH = (typeof isBottomMenuOpen === 'function') && isBottomMenuOpen();

                        if (topOpenH) {
                            if (typeof closeTopMenuPanel === 'function') closeTopMenuPanel();
                            lockCanvasGestures();
                            return;
                        }
                        if ((typeof isKeyboardShortcutsPanelOpen === 'function') && isKeyboardShortcutsPanelOpen()) {
                            try { closeKeyboardShortcutsPanel(); } catch (_) {}
                            lockCanvasGestures();
                            return;
                        }
                        if (textInOpenH) {
                            if (typeof hideTextInPanel === 'function') hideTextInPanel();
                            lockCanvasGestures();
                            return;
                        }
                        if (avatarOpenH) {
                            if (typeof closeBottomMenuPanel === 'function') closeBottomMenuPanel();
                            lockCanvasGestures();
                            return;
                        }

                        if (dx < 0) loadMode(state.currentModeIdx + 1);
                        else loadMode(state.currentModeIdx - 1);
                        lockCanvasGestures();
                        return;
                    }
                    // Double tap toggles Fullscreen (no station change, no WebM)
                    const moved = (adx > TAP_MAX_MOVE || ady > TAP_MAX_MOVE);
                    const now = Date.now();
                    if (!moved && (now - lastTapTime) < DOUBLE_TAP_MS) {
                        try {
                            const elTap = e.target;
                            if (elTap && elTap.closest && elTap.closest('.knob-wrap')) {
                                lastTapTime = now;
                                return;
                            }
                            if (elTap && elTap.closest && elTap.closest('#dj-deck-splitter')) {
                                lastTapTime = now;
                                return;
                            }
                            if (pDownTarget && pDownTarget.closest && pDownTarget.closest('.knob-wrap')) {
                                lastTapTime = now;
                                return;
                            }
                            if (pDownTarget && pDownTarget.closest && pDownTarget.closest('#dj-deck-splitter')) {
                                lastTapTime = now;
                                return;
                            }
                            if (elTap && elTap.closest && elTap.closest('.dj-jog-wrap')) {
                                lastTapTime = 0;
                                return;
                            }
                            if (pDownTarget && pDownTarget.closest && pDownTarget.closest('.dj-jog-wrap')) {
                                lastTapTime = 0;
                                return;
                            }
                        } catch (_) {}
                        window.__suppressNextClick = true;
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        try { if (window.__randomClickTimer) { clearTimeout(window.__randomClickTimer); window.__randomClickTimer = null; } } catch(_) {}
                        try {
                            if (shouldToggleDeckBVizMountFullscreenFromPointer(e)) {
                                toggleDeckBVizMountFullscreen();
                            } else {
                                toggleFullscreen();
                            }
                        } catch (_) {}
                        lastTapTime = 0; // reset
                        return;
                    }
                    // Record tap time for double-tap detection
                    if (!moved) lastTapTime = now;
                };
                const onPointerUpWebm = (e) => {
                    if (!webmOn) return;
                    const x = e.clientX || (e.changedTouches && e.changedTouches[0]?.clientX) || 0;
                    const y = e.clientY || (e.changedTouches && e.changedTouches[0]?.clientY) || 0;
                    const dx = x - pDownX;
                    const dy = y - pDownY;
                    const adx = Math.abs(dx), ady = Math.abs(dy);
                    const webmHorizontalIntent = adx > SWIPE_PX && adx > ady * SWIPE_AXIS_DOMINANCE;
                    if (webmHorizontalIntent) {
                        if (canvasGestureLocked()) return;
                        handledSwipe = true;
                        window.__suppressNextClick = true;
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        const guardW = (typeof window !== 'undefined' && window.__panelGuardUntilMs) ? window.__panelGuardUntilMs : 0;
                        if (Date.now() < guardW) return;

                        const topOpenW = (typeof isTopMenuOpen === 'function') && isTopMenuOpen();
                        const tipW = document.getElementById('textin-panel');
                        const textInOpenW = !!(tipW && !tipW.classList.contains('display-none') && tipW.classList.contains('open'));
                        const avatarOpenW = (typeof isBottomMenuOpen === 'function') && isBottomMenuOpen();

                        if (topOpenW) { if (typeof closeTopMenuPanel === 'function') closeTopMenuPanel(); return; }
                        if ((typeof isKeyboardShortcutsPanelOpen === 'function') && isKeyboardShortcutsPanelOpen()) { try { closeKeyboardShortcutsPanel(); } catch (_) {} return; }
                        if (textInOpenW) { if (typeof hideTextInPanel === 'function') hideTextInPanel(); return; }
                        if (avatarOpenW) { if (typeof closeBottomMenuPanel === 'function') closeBottomMenuPanel(); return; }

                        // Only within overlay area: navigate webms
                        if (dx < 0) nextWebm();
                        else prevWebm();
                        lockCanvasGestures();
                    }
                };
                // Pointer listeners (mouse + touch via pointer events).
                // Capture pointerdown so pDownTarget gets the real target (e.g. #dj-deck-splitter) before a child’s
                // stopPropagation() — otherwise stale canvas targets caused horizontal drags on the splitter to change visuals.
                canvasEl.addEventListener('pointerdown', onPointerDown, { passive: true, capture: true });
                canvasEl.addEventListener('pointerup', onPointerUpCanvas, { passive: false });
                if (webmOverlay) {
                    webmOverlay.addEventListener('pointerdown', onPointerDown, { passive: true });
                    webmOverlay.addEventListener('pointerup', onPointerUpWebm, { passive: false });
                }
                // Double click: Deck B viz mount → fullscreen mount only; elsewhere → fullscreen DJ canvas
                canvasEl.addEventListener('dblclick', (e) => {
                    if (!state || !state.isPlaying) return;
                    try {
                        if (e.target && typeof e.target.closest === 'function' && e.target.closest('.radio-visual-digital-hub-ai')) return;
                    } catch (_) {}
                    try {
                        if (e.target && typeof e.target.closest === 'function' && e.target.closest('.dj-jog-wrap')) return;
                    } catch (_) {}
                    if (shouldToggleDeckBVizMountFullscreenFromPointer(e)) {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { if (window.__randomClickTimer) { clearTimeout(window.__randomClickTimer); window.__randomClickTimer = null; } } catch (_) {}
                        try { toggleDeckBVizMountFullscreen(); } catch (_) {}
                        return;
                    }
                    try {
                        if (e.target && typeof e.target.closest === 'function' && e.target.closest('#dj-deck-splitter')) return;
                    } catch (_) {}
                    try {
                        if (e.target && typeof e.target.closest === 'function' && e.target.closest('.knob-wrap')) return;
                    } catch (_) {}
                    try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                    try { if (window.__randomClickTimer) { clearTimeout(window.__randomClickTimer); window.__randomClickTimer = null; } } catch(_) {}
                    try { toggleFullscreen(); } catch(_) {}
                }, { passive: false });
                // Double click on WebM overlay: toggle Fullscreen
                if (webmOverlay) {
                    webmOverlay.addEventListener('dblclick', (e) => {
                        try {
                            if (e.target && typeof e.target.closest === 'function' && e.target.closest('#dj-deck-splitter')) return;
                        } catch (_) {}
                        try {
                            if (e.target && typeof e.target.closest === 'function' && e.target.closest('.dj-jog-wrap')) return;
                        } catch (_) {}
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        try { if (window.__randomClickTimer) { clearTimeout(window.__randomClickTimer); window.__randomClickTimer = null; } } catch(_) {}
                        try { toggleFullscreen(); } catch(_) {}
                    }, { passive: false });
                }
                // Wheel: scroll UP opens top menu; scroll DOWN opens bottom menu; never open the other when closing (trackpads/mice)
                // Wheel: scroll UP opens top menu; scroll DOWN opens Mix Settings (bottom); 
                let lastWheelAt = 0;
                canvasEl.addEventListener('wheel', (e) => {
                    if (!state || !state.isPlaying) return;
                    try {
                        if (document.getElementById('radio-visual-root')) return;
                        const t = e.target;
                        if (t && t.closest && t.closest('#radio-visual-root')) return;
                        if (t && t.closest && t.closest('#dj-visual-root')) return;
                        if (t && t.closest && t.closest('.dj-jog-wrap')) return;
                        if (t && t.closest && t.closest('.dj-deck-b-queue-panel')) return;
                        if (t && t.closest && t.closest('.dj-deck-b-media-panel')) return;
                    } catch (_) {}
                    const guardUntil = (typeof window !== 'undefined' && window.__panelGuardUntilMs) ? window.__panelGuardUntilMs : 0;
                    const now = Date.now();
                    if (now < guardUntil) return;
                    if (canvasGestureLocked()) return;
                    if (now - lastWheelAt < 250) return;
                    lastWheelAt = now;
                    
                    const dy = e.deltaY || 0;
                    
                    // Check if Mix Panel is open (manual check)
                    const mixOpen = mixPanel && !mixPanel.classList.contains('display-none') && mixPanel.classList.contains('open');
                    const kbdShortcutsOpen = (typeof isKeyboardShortcutsPanelOpen === 'function') && isKeyboardShortcutsPanelOpen();
                    const sideOpen = typeof isBottomMenuOpen === 'function' && isBottomMenuOpen();
                    const topOpenWh = typeof isTopMenuOpen === 'function' && isTopMenuOpen();
                    const tipWh = document.getElementById('textin-panel');
                    const textInOpenWh = !!(tipWh && !tipWh.classList.contains('display-none') && tipWh.classList.contains('open'));

                    if (dy < 0) {
                        // SCROLL UP: Open Top Menu (or close others)
                        try {
                            if (kbdShortcutsOpen) { closeKeyboardShortcutsPanel(); return; }
                            if (mixOpen) { toggleMixPanel(); return; }
                            if (textInOpenWh) { if (typeof hideTextInPanel === 'function') hideTextInPanel(); return; }
                            if (sideOpen) { closeBottomMenuPanel(); return; }
                            if (!topOpenWh) { openTopMenuPanel(); }
                        } catch(_) {}
                    } else if (dy > 0) {
                        // SCROLL DOWN: Open Mix Settings at Bottom (or close others)
                        try {
                            if (kbdShortcutsOpen) { closeKeyboardShortcutsPanel(); return; }
                            if (topOpenWh) { closeTopMenuPanel(); return; }
                            if (textInOpenWh) { if (typeof hideTextInPanel === 'function') hideTextInPanel(); return; }
                            if (sideOpen) { closeBottomMenuPanel(); return; }
                            if (!mixOpen) { toggleMixPanel(); }
                        } catch(_) {}
                    }
                }, { passive: true });
            } catch(e) {}
        })();

        // When playing, clicking anywhere (except UI controls) picks a random station
        (function bindGlobalRandomClickWhilePlaying() {
            try {
                const area = document.getElementById('canvas-container');
                if (!area) return;
                area.addEventListener('click', (e) => {
                    try {
                        if (!state || !state.isPlaying) return;
                        // Delay station change to detect potential double-click; cancel if dblclick fires
                        if (window.__suppressNextClick) { window.__suppressNextClick = false; return; }
                        const delayMs = 280; // align with DOUBLE_TAP_MS window
                        try { if (window.__randomClickTimer) { clearTimeout(window.__randomClickTimer); window.__randomClickTimer = null; } } catch(_) {}
                        window.__randomClickTimer = setTimeout(() => {
                            try {
                                if (!state || !state.isPlaying) return;
                                // If a panel is open or overlay visible, ignore
                                const overlay = document.getElementById('overlay');
                                const menu = document.getElementById('top-menu-panel');
                                const webm = document.getElementById('webm-overlay');
                                if (overlay && !overlay.classList.contains('hidden') && overlay.style.display !== 'none') return;
                                if (menu && !menu.classList.contains('display-none') && menu.classList.contains('open')) return;
                                // If WebM overlay is visible, allow clicks outside of it to change station
                                if (webm && !webm.classList.contains('display-none')) {
                                    if (e.target && e.target.closest && e.target.closest('#webm-overlay')) return;
                                }
                                // DJ Decks visual manages its own station transport.
                                // When it's active, prevent global "click screen => next/random station".
                                if (document.getElementById('dj-visual-root')) return;
                                if (document.getElementById('radio-visual-root')) return;
                                if (e.target && e.target.closest && e.target.closest('#radio-visual-root')) return;
                                if (typeof pickRandomStation === 'function') pickRandomStation();
                            } catch(_) {} 
                        }, delayMs);
                        // If start overlay or top menu is visible, ignore
                        const overlay = document.getElementById('overlay');
                        const menu = document.getElementById('top-menu-panel');
                        const webm = document.getElementById('webm-overlay');
                    } catch(_) {}
                }, false);
            } catch(e) {}
        })();

        // WebM speed helpers
        function setWebmSpeed(newRate) {
            const clamped = Math.max(0.1, Math.min(4, Number(newRate) || 1));
            webmSettings.playbackRate = clamped;
            if (typeof inpWebmSpeed !== 'undefined' && inpWebmSpeed) {
                inpWebmSpeed.value = String(clamped);
            }
            applyWebmSettings();
        }
        function adjustWebmSpeed(delta) {
            setWebmSpeed((Number(webmSettings.playbackRate) || 1) + (delta || 0));
        }
        // WebM position/size/opacity helpers
        function adjustWebmPosition(dxVw, dyVh) {
            if (!webmOn) return;
            var x = Number(webmSettings.posXvw) || 50;
            var y = Number(webmSettings.posYvh) || 50;
            var nx = Math.max(0, Math.min(100, x + (dxVw || 0)));
            var ny = Math.max(0, Math.min(100, y + (dyVh || 0)));
            webmSettings.posXvw = nx;
            webmSettings.posYvh = ny;
            if (typeof inpWebmX !== 'undefined' && inpWebmX) inpWebmX.value = String(nx);
            if (typeof inpWebmY !== 'undefined' && inpWebmY) inpWebmY.value = String(ny);
            applyWebmSettings();
        }
        function adjustWebmScale(deltaVw) {
            if (!webmOn) return;
            var min = Number((typeof inpWebmScale !== 'undefined' && inpWebmScale && inpWebmScale.min) ? inpWebmScale.min : 10);
            var max = Number((typeof inpWebmScale !== 'undefined' && inpWebmScale && inpWebmScale.max) ? inpWebmScale.max : 100);
            var cur = Number(webmSettings.scaleVw) || min;
            var nv = Math.max(min, Math.min(max, cur + (deltaVw || 0)));
            webmSettings.scaleVw = nv;
            if (typeof inpWebmScale !== 'undefined' && inpWebmScale) inpWebmScale.value = String(nv);
            applyWebmSettings();
        }
        function adjustWebmOpacity(delta) {
            if (!webmOn) return;
            var cur = Number(webmSettings.opacity) || 0;
            var nv = Math.max(0, Math.min(1, cur + (delta || 0)));
            webmSettings.opacity = nv;
            if (typeof inpWebmOpacity !== 'undefined' && inpWebmOpacity) inpWebmOpacity.value = String(nv);
            applyWebmSettings();
        }
        // Volume control handling (50% = 0 dB; below attenuate, above amplify)
        function setVolume(value) {
            const v = Math.max(0, Math.min(1, Number(value) || 0));
            if (state && state.gainNode && state.gainNode.gain) {
                // Map slider to dB curve:
                // v = 0.5 -> 0 dB (1.0 linear)
                // v < 0.5 -> -40 dB .. 0 dB
                // v > 0.5 -> 0 dB .. +12 dB
                let gainLinear = 1.0;
                if (v === 0) {
                    gainLinear = 0.0;
                } else if (v < 0.5) {
                    const t = v / 0.5;           // 0..1
                    const dB = -40 * (1 - t);    // -40..0
                    gainLinear = Math.pow(10, dB / 20);
                } else if (v > 0.5) {
                    const t = (v - 0.5) / 0.5;   // 0..1
                    const dB = 12 * t;           // 0..+12
                    gainLinear = Math.pow(10, dB / 20);
                } else {
                    gainLinear = 1.0;
                }
                state.gainNode.gain.value = gainLinear;
            }
        }


function exposeAppBindingsToGlobal() {
    const g = globalThis;
    try { g.AUTOFADE_CHANGE_STATION_STORAGE_KEY = AUTOFADE_CHANGE_STATION_STORAGE_KEY; } catch (_) {}
    try { g.AUTOMIX_ENABLED_STORAGE_KEY = AUTOMIX_ENABLED_STORAGE_KEY; } catch (_) {}
    try { g.CANVAS_GESTURE_COOLDOWN_MS = CANVAS_GESTURE_COOLDOWN_MS; } catch (_) {}
    try { g.DECK_B_IDLE_LOGO_URL = DECK_B_IDLE_LOGO_URL; } catch (_) {}
    try { g.KARAOKE_NERDS_BASE_URL = KARAOKE_NERDS_BASE_URL; } catch (_) {}
    try { g.KARAOKE_NERDS_EMBED_URL = KARAOKE_NERDS_EMBED_URL; } catch (_) {}
    try { g.normalizeKaraokeNerdsEmbedUrl = normalizeKaraokeNerdsEmbedUrl; } catch (_) {}
    try { g.DOUBLE_TAP_MS = DOUBLE_TAP_MS; } catch (_) {}
    try { g.EDGE_SNAP = EDGE_SNAP; } catch (_) {}
    try { g.MAX_RADIO_RETRIES = MAX_RADIO_RETRIES; } catch (_) {}
    try { g.QUALITY = QUALITY; } catch (_) {}
    try { g.RADIO_STATION_XFADE_KEY = RADIO_STATION_XFADE_KEY; } catch (_) {}
    try { g.SWIPE_AXIS_DOMINANCE = SWIPE_AXIS_DOMINANCE; } catch (_) {}
    try { g.SWIPE_PX = SWIPE_PX; } catch (_) {}
    try { g.TAP_MAX_MOVE = TAP_MAX_MOVE; } catch (_) {}
    try { g.USER_RADIO_STATIONS_KEY = USER_RADIO_STATIONS_KEY; } catch (_) {}
    try { g.__webmDeckBOnLayout = __webmDeckBOnLayout; } catch (_) {}
    try { g.aDupSpacing = document.getElementById('avatar-inp-dup-spacing'); } catch (_) {}
    try { g.aOpacity = aOpacity; } catch (_) {}
    try { g.aRot = aRot; } catch (_) {}
    try { g.aScale = aScale; } catch (_) {}
    try { g.aSpeed = aSpeed; } catch (_) {}
    try { g.aX = aX; } catch (_) {}
    try { g.aY = aY; } catch (_) {}
    try { g.activeB = activeB; } catch (_) {}
    try { g.activeTextOverlays = activeTextOverlays; } catch (_) {}
    try { g.addUserRadioStation = addUserRadioStation; } catch (_) {}
    try { g.adjBpm = adjBpm; } catch (_) {}
    try { g.adjustWebmOpacity = adjustWebmOpacity; } catch (_) {}
    try { g.adjustWebmPosition = adjustWebmPosition; } catch (_) {}
    try { g.adjustWebmScale = adjustWebmScale; } catch (_) {}
    try { g.adjustWebmSpeed = adjustWebmSpeed; } catch (_) {}
    try { g.adx = adx; } catch (_) {}
    try { g.afterDeckBVizFullscreenChange = afterDeckBVizFullscreenChange; } catch (_) {}
    try { g.afterPlay = afterPlay; } catch (_) {}
    try { g.alpha = alpha; } catch (_) {}
    try { g.anchorBottom = anchorBottom; } catch (_) {}
    try { g.apply = apply; } catch (_) {}
    try { g.applyDeckBVideoCrossfadeLayers = applyDeckBVideoCrossfadeLayers; } catch (_) {}
    try { g.applyDeckBVideoPayloadToElement = applyDeckBVideoPayloadToElement; } catch (_) {}
    try { g.applyUiLockState = applyUiLockState; } catch (_) {}
    try { g.applyCrossfade = applyCrossfade; } catch (_) {}
    try { g.area = area; } catch (_) {}
    try { g.arr = arr; } catch (_) {}
    try { g.attempts = attempts; } catch (_) {}
    try { g.audioEl = audioEl; } catch (_) {}
    try { g.audioElB = audioElB; } catch (_) {}
    try { g.audioElRadioAAlt = audioElRadioAAlt; } catch (_) {}
    try { g.audioElRadioBAlt = audioElRadioBAlt; } catch (_) {}
    try { g.audioElSample = audioElSample; } catch (_) {}
    try { g.audioElSample1 = audioElSample1; } catch (_) {}
    try { g.audioElSample2 = audioElSample2; } catch (_) {}
    try { g.audioElSample3 = audioElSample3; } catch (_) {}
    try { g.audioElSample4 = audioElSample4; } catch (_) {}
    try { g.audioElSample5 = audioElSample5; } catch (_) {}
    try { g.audioElSample6 = audioElSample6; } catch (_) {}
    try { g.autoBtn = autoBtn; } catch (_) {}
    try { g.av = av; } catch (_) {}
    try { g.avatarOpen = avatarOpen; } catch (_) {}
    try { g.avatarOpenH = avatarOpenH; } catch (_) {}
    try { g.avatarOpenW = avatarOpenW; } catch (_) {}
    try { g.beats = beats; } catch (_) {}
    try { g.bestBpm = bestBpm; } catch (_) {}
    try { g.binHz = binHz; } catch (_) {}
    try { g.bindKnob = bindKnob; } catch (_) {}
    try { g.bindWebmDeckBLayoutWatchers = bindWebmDeckBLayoutWatchers; } catch (_) {}
    try { g.bins = bins; } catch (_) {}
    try { g.bits = bits; } catch (_) {}
    try { g.blob = blob; } catch (_) {}
    try { g.border = border; } catch (_) {}
    try { g.bpm = bpm; } catch (_) {}
    try { g.bpmBtnA = bpmBtnA; } catch (_) {}
    try { g.bpmBtnB = bpmBtnB; } catch (_) {}
    try { g.bpmBtnGrid = bpmBtnGrid; } catch (_) {}
    try { g.bpmOnA = bpmOnA; } catch (_) {}
    try { g.bpmOutA = bpmOutA; } catch (_) {}
    try { g.bpmOutB = bpmOutB; } catch (_) {}
    try { g.bpmSmoothA = bpmSmoothA; } catch (_) {}
    try { g.btn = btn; } catch (_) {}
    try { g.btnAuto = btnAuto; } catch (_) {}
    try { g.btnBannerShazam = btnBannerShazam; } catch (_) {}
    try { g.btnClose = btnClose; } catch (_) {}
    try { g.btnKeyboardShortcutsClose = btnKeyboardShortcutsClose; } catch (_) {}
    try { g.btnNextWebm = btnNextWebm; } catch (_) {}
    try { g.btnOpenWebm = btnOpenWebm; } catch (_) {}
    try { g.btnOptions = btnOptions; } catch (_) {}
    try { g.btnOptionsClose = btnOptionsClose; } catch (_) {}
    try { g.btnPrevWebm = btnPrevWebm; } catch (_) {}
    try { g.btnReset = btnReset; } catch (_) {}
    try { g.buf = buf; } catch (_) {}
    try { g.calcFlux = calcFlux; } catch (_) {}
    try { g.calcRms = calcRms; } catch (_) {}
    try { g.cancelStartTextLoop = cancelStartTextLoop; } catch (_) {}
    try { g.canvasEl = canvasEl; } catch (_) {}
    try { g.canvasGestureLocked = canvasGestureLocked; } catch (_) {}
    try { g.cfg = cfg; } catch (_) {}
    try { g.chunk = chunk; } catch (_) {}
    try { g.clamped = clamped; } catch (_) {}
    try { g.clean = clean; } catch (_) {}
    try { g.clearAllAutoMixDeferLocal = clearAllAutoMixDeferLocal; } catch (_) {}
    try { g.clearAutoMixDeferForNonIncoming = clearAutoMixDeferForNonIncoming; } catch (_) {}
    try { g.closeBottomMenuPanel = closeBottomMenuPanel; } catch (_) {}
    try { g.closeKeyboardShortcutsPanel = closeKeyboardShortcutsPanel; } catch (_) {}
    try { g.applyDigitalBgGifFromOptions = applyDigitalBgGifFromOptions; } catch (_) {}
    try { g.applyDigitalRadioTheme = applyDigitalRadioTheme; } catch (_) {}
    try { g.applyDigitalSpectrumSettings = applyDigitalSpectrumSettings; } catch (_) {}
    try { g.closeOptionsPanel = closeOptionsPanel; } catch (_) {}
    try { g.code = code; } catch (_) {}
    try { g.computeDeckBVideoCrossfadePlan = computeDeckBVideoCrossfadePlan; } catch (_) {}
    try { g.computeDigitalStagingVideoCrossfadePlan = computeDigitalStagingVideoCrossfadePlan; } catch (_) {}
    try { g.container = container; } catch (_) {}
    try { g.cs = cs; } catch (_) {}
    try { g.cur = cur; } catch (_) {}
    try { g.current = current; } catch (_) {}
    try { g.currentNowPlayingICY = currentNowPlayingICY; } catch (_) {}
    try {
        Object.defineProperty(g, 'currentStationBIndex', {
            get: () => currentStationBIndex,
            set: (v) => {
                currentStationBIndex = v;
                try { saveLastStationSelection('b'); } catch (_) {}
            },
            enumerable: true,
            configurable: true,
        });
    } catch (_) {}
    try {
        Object.defineProperty(g, 'currentStationIndex', {
            get: () => currentStationIndex,
            set: (v) => {
                currentStationIndex = v;
                try { saveLastStationSelection('a'); } catch (_) {}
            },
            enumerable: true,
            configurable: true,
        });
    } catch (_) {}
    try { g.cut = cut; } catch (_) {}
    try { g.cx = cx; } catch (_) {}
    try { g.d = d; } catch (_) {}
    try { g.dB = dB; } catch (_) {}
    try { g.dc = dc; } catch (_) {}
    try { g.deckBVideoLoopEnabled = deckBVideoLoopEnabled; } catch (_) {}
    try { g.deckBVideoPlayAllEnabled = deckBVideoPlayAllEnabled; } catch (_) {}
    try { g.deckBVideoShuffleEnabled = deckBVideoShuffleEnabled; } catch (_) {}
    try { g.deckBVideoSingleLoopActive = deckBVideoSingleLoopActive; } catch (_) {}
    try { g.deckBVideoUserIdle = deckBVideoUserIdle; } catch (_) {}
    try { g.deckFileQueues = deckFileQueues; } catch (_) {}
    try { g.deckVideoFeeds = deckVideoFeeds; } catch (_) {}
    try { g.deckVideoHistory = deckVideoHistory; } catch (_) {}
    try { g.def = def; } catch (_) {}
    try { g.deg = deg; } catch (_) {}
    try { g.delayMs = delayMs; } catch (_) {}
    try { g.delta = delta; } catch (_) {}
    try { g.deriveNameFromUrl = deriveNameFromUrl; } catch (_) {}
    try { g.deriveTitleFromUrl = deriveTitleFromUrl; } catch (_) {}
    try { g.clearNowPlayingICYBanner = clearNowPlayingICYBanner; } catch (_) {}
    try { g.showStationBanner = showStationBanner; } catch (_) {}
    try { g.updateModeSubStationLine = updateModeSubStationLine; } catch (_) {}
    try { g.syncModeInfoHudHintForDigitalRadio = syncModeInfoHudHintForDigitalRadio; } catch (_) {}
    try { g.toggleUiHudPosition = toggleUiHudPosition; } catch (_) {}
    try { g.getCrossfaderAudibleDeckKey = getCrossfaderAudibleDeckKey; } catch (_) {}
    try { g.getDeckStationDisplayName = getDeckStationDisplayName; } catch (_) {}
    try { g.hideStationBannerPermanently = hideStationBannerPermanently; } catch (_) {}
    try { g.diffs = diffs; } catch (_) {}
    try { g.djM = djM; } catch (_) {}
    try { g.dk = dk; } catch (_) {}
    try { g.drawGridTick = drawGridTick; } catch (_) {}
    try { g.drawScopeAndUpdateBpm = drawScopeAndUpdateBpm; } catch (_) {}
    try { g.dx = dx; } catch (_) {}
    try { g.dy = dy; } catch (_) {}
    try { g.e = e; } catch (_) {}
    try { g.el = el; } catch (_) {}
    try { g.elTap = elTap; } catch (_) {}
    try { g.eligible = eligible; } catch (_) {}
    try { g.enabled = enabled; } catch (_) {}
    try { g.ensureAutoMixDeferLocalState = ensureAutoMixDeferLocalState; } catch (_) {}
    try { g.ensureBeatModuleLoaded = ensureBeatModuleLoaded; } catch (_) {}
    try { g.ensurePatGifLoaded = ensurePatGifLoaded; } catch (_) {}
    try { g.ensurePtaGifLoaded = ensurePtaGifLoaded; } catch (_) {}
    try { g.ensureTapGifLoaded = ensureTapGifLoaded; } catch (_) {}
    try { g.entry = entry; } catch (_) {}
    try { g.envA = envA; } catch (_) {}
    try { g.envFluxA = envFluxA; } catch (_) {}
    try { g.eqState = eqState; } catch (_) {}
    try { g.escapeHtml = escapeHtml; } catch (_) {}
    try { g.estComb = estComb; } catch (_) {}
    try { g.exitDeckBVizFullscreen = exitDeckBVizFullscreen; } catch (_) {}
    try { g.f = f; } catch (_) {}
    try { g.fa = fa; } catch (_) {}
    try { g.fadeInPtaStartBg = fadeInPtaStartBg; } catch (_) {}
    try { g.revealStartScreenAfterAssets = revealStartScreenAfterAssets; } catch (_) {}
    try { g.resetStartScreenReveal = resetStartScreenReveal; } catch (_) {}
    try { g.fb = fb; } catch (_) {}
    try { g.feed = feed; } catch (_) {}
    try { g.fftSize = fftSize; } catch (_) {}
    try { g.finish = finish; } catch (_) {}
    try { g.flux = flux; } catch (_) {}
    try { g.fluxHistA = fluxHistA; } catch (_) {}
    try { g.fluxHistLen = fluxHistLen; } catch (_) {}
    try { g.fluxL = fluxL; } catch (_) {}
    try { g.fly = fly; } catch (_) {}
    try { g.forceResize = forceResize; } catch (_) {}
    try { g.fs = fs; } catch (_) {}
    try { g.full = full; } catch (_) {}
    try { g.fxArp = fxArp; } catch (_) {}
    try { g.fxBass = fxBass; } catch (_) {}
    try { g.fxCut = fxCut; } catch (_) {}
    try { g.fxDist = fxDist; } catch (_) {}
    try { g.fxEcho = fxEcho; } catch (_) {}
    try { g.fxFlanger = fxFlanger; } catch (_) {}
    try { g.fxHigh = fxHigh; } catch (_) {}
    try { g.fxLow = fxLow; } catch (_) {}
    try { g.fxNoVocal = fxNoVocal; } catch (_) {}
    try { g.fxReverb = fxReverb; } catch (_) {}
    try { g.fxTk = fxTk; } catch (_) {}
    try { g.fxTreble = fxTreble; } catch (_) {}
    try { g.ga = ga; } catch (_) {}
    try { g.gainLinear = gainLinear; } catch (_) {}
    try { g.gapAbove = gapAbove; } catch (_) {}
    try { g.gapBelow = gapBelow; } catch (_) {}
    try { g.gaplessMicro = gaplessMicro; } catch (_) {}
    try { g.gb = gb; } catch (_) {}
    try { g.getAutoMixIncomingDeckKey = getAutoMixIncomingDeckKey; } catch (_) {}
    try { g.getCrossfaderIncomingDeckKey = getCrossfaderIncomingDeckKey; } catch (_) {}
    try { g.getCycleEligibleStationIndexes = getCycleEligibleStationIndexes; } catch (_) {}
    try { g.goPreviousStation = goPreviousStation; } catch (_) {}
    try { g.getDeckARadioCrossfadeRampSec = getDeckARadioCrossfadeRampSec; } catch (_) {}
    try { g.getDeckActiveVideoMeta = getDeckActiveVideoMeta; } catch (_) {}
    try { g.getDeckAudioVjsMirrorMeta = getDeckAudioVjsMirrorMeta; } catch (_) {}
    try { g.getDeckBJogSeekMedia = getDeckBJogSeekMedia; } catch (_) {}
    try { g.getDeckBVideoCandidates = getDeckBVideoCandidates; } catch (_) {}
    try { g.getDeckBVideoPlaybackSources = getDeckBVideoPlaybackSources; } catch (_) {}
    try { g.getDeckBVizFullscreenEl = getDeckBVizFullscreenEl; } catch (_) {}
    try { g.getDeckBVizMountEl = getDeckBVizMountEl; } catch (_) {}
    try { g.getDjCrossfade01 = getDjCrossfade01; } catch (_) {}
    try { g.guardH = guardH; } catch (_) {}
    try { g.guardUntil = guardUntil; } catch (_) {}
    try { g.guardW = guardW; } catch (_) {}
    try { g.h = h; } catch (_) {}
    try { g.had = had; } catch (_) {}
    try { g.handleTap = handleTap; } catch (_) {}
    try { g.handledSwipe = handledSwipe; } catch (_) {}
    try { g.hex = hex; } catch (_) {}
    try { g.hidden = hidden; } catch (_) {}
    try { g.hidePtaStartBg = hidePtaStartBg; } catch (_) {}
    try { g.i = i; } catch (_) {}
    try { g.icyPollBusy = icyPollBusy; } catch (_) {}
    try { g.idx = idx; } catch (_) {}
    try { g.idxH = idxH; } catch (_) {}
    try { g.img = img; } catch (_) {}
    try { g.incoming = incoming; } catch (_) {}
    try { g.indicator = indicator; } catch (_) {}
    try { g.inp = inp; } catch (_) {}
    try { g.inpPixelRatio = inpPixelRatio; } catch (_) {}
    try { g.inpShuffleMax = inpShuffleMax; } catch (_) {}
    try { g.inpShuffleMin = inpShuffleMin; } catch (_) {}
    try { g.inpTransition = inpTransition; } catch (_) {}
    try { g.inpWebmDupSpacing = inpWebmDupSpacing; } catch (_) {}
    try { g.inpWebmOpacity = inpWebmOpacity; } catch (_) {}
    try { g.inpWebmRot = inpWebmRot; } catch (_) {}
    try { g.inpWebmScale = inpWebmScale; } catch (_) {}
    try { g.inpWebmSpeed = inpWebmSpeed; } catch (_) {}
    try { g.inpWebmX = inpWebmX; } catch (_) {}
    try { g.inpWebmY = inpWebmY; } catch (_) {}
    try { g.input = input; } catch (_) {}
    try { g.interval = interval; } catch (_) {}
    try { g.intervalA = intervalA; } catch (_) {}
    try { g.intervalB = intervalB; } catch (_) {}
    try { g.isAutoFadeChangeStationEnabledGlobal = isAutoFadeChangeStationEnabledGlobal; } catch (_) {}
    try { g.isAutoMixDeferredLocalArmed = isAutoMixDeferredLocalArmed; } catch (_) {}
    try { g.isAutoMixEnabledGlobal = isAutoMixEnabledGlobal; } catch (_) {}
    try { g.isBeatA = isBeatA; } catch (_) {}
    try { g.isBeatB = isBeatB; } catch (_) {}
    try { g.isBottomMenuOpen = isBottomMenuOpen; } catch (_) {}
    try { g.isDeckAVideoActivelyPlaying = isDeckAVideoActivelyPlaying; } catch (_) {}
    try { g.isDeckBVizBarsOrProjectMActive = isDeckBVizBarsOrProjectMActive; } catch (_) {}
    try { g.isEventOnDeckBVizMount = isEventOnDeckBVizMount; } catch (_) {}
    try { g.isKeyboardShortcutsPanelOpen = isKeyboardShortcutsPanelOpen; } catch (_) {}
    try { g.isLikelyRadioStreamUrl = isLikelyRadioStreamUrl; } catch (_) {}
    try { g.isLikelyVideoUrl = isLikelyVideoUrl; } catch (_) {}
    try { g.isMobile = isMobile; } catch (_) {}
    try { g.isOptionsOpen = isOptionsOpen; } catch (_) {}
    try { g.isPointerInDeckBVizMountRect = isPointerInDeckBVizMountRect; } catch (_) {}
    try { g.isStartOverlayShowing = isStartOverlayShowing; } catch (_) {}
    try { g.isVideoQueueEligibleUrl = isVideoQueueEligibleUrl; } catch (_) {}
    try { g.k = k; } catch (_) {}
    try { g.kbdSheetOpen = kbdSheetOpen; } catch (_) {}
    try { g.kbdShortcutsOpen = kbdShortcutsOpen; } catch (_) {}
    try { g.keyboardShortcutsPanel = keyboardShortcutsPanel; } catch (_) {}
    try { g.l = l; } catch (_) {}
    try { g.lab = lab; } catch (_) {}
    try { g.label = label; } catch (_) {}
    try { g.last = last; } catch (_) {}
    try { g.lastBeatTsA = lastBeatTsA; } catch (_) {}
    try { g.lastSpecA = lastSpecA; } catch (_) {}
    try { g.lastTapTime = lastTapTime; } catch (_) {}
    try { g.lastWheelAt = lastWheelAt; } catch (_) {}
    try { g.latestFluxA = latestFluxA; } catch (_) {}
    try { g.latestFluxB = latestFluxB; } catch (_) {}
    try { g.layerA = layerA; } catch (_) {}
    try { g.layerB = layerB; } catch (_) {}
    try { g.layerQ = layerQ; } catch (_) {}
    try { g.layoutOverlayElements = layoutOverlayElements; } catch (_) {}
    try { g.lineHeight = lineHeight; } catch (_) {}
    try { g.loadUserRadioStations = loadUserRadioStations; } catch (_) {}
    try { g.lockCanvasGestures = lockCanvasGestures; } catch (_) {}
    try { g.logo = logo; } catch (_) {}
    try { g.logoRect = logoRect; } catch (_) {}
    try { g.loop = loop; } catch (_) {}
    try { g.lowMax = lowMax; } catch (_) {}
    try { g.m = m; } catch (_) {}
    try { g.make = make; } catch (_) {}
    try { g.markAutoMixDeferredLocal = markAutoMixDeferredLocal; } catch (_) {}
    try { g.max = max; } catch (_) {}
    try { g.maxEnvSamples = maxEnvSamples; } catch (_) {}
    try { g.mc = mc; } catch (_) {}
    try { g.md = md; } catch (_) {}
    try { g.mean = mean; } catch (_) {}
    try { g.med = med; } catch (_) {}
    try { g.media = media; } catch (_) {}
    try { g.mediaVideoQueue = mediaVideoQueue; } catch (_) {}
    try { g.menu = menu; } catch (_) {}
    try { g.menuOpen = menuOpen; } catch (_) {}
    try { g.meta = meta; } catch (_) {}
    try { g.metaA = metaA; } catch (_) {}
    try { g.metaB = metaB; } catch (_) {}
    try { g.micCancel = micCancel; } catch (_) {}
    try { g.micConfirm = micConfirm; } catch (_) {}
    try { g.micLetter = micLetter; } catch (_) {}
    try { g.micOk = micOk; } catch (_) {}
    try { g.midMax = midMax; } catch (_) {}
    try { g.midiCancel = midiCancel; } catch (_) {}
    try { g.midiConfirm = midiConfirm; } catch (_) {}
    try { g.midiLetter = midiLetter; } catch (_) {}
    try { g.midiOk = midiOk; } catch (_) {}
    try { g.min = min; } catch (_) {}
    try { g.minBpm = minBpm; } catch (_) {}
    try { g.mixBStatusDot = mixBStatusDot; } catch (_) {}
    try { g.mixBStatusText = mixBStatusText; } catch (_) {}
    try { g.mixClose = mixClose; } catch (_) {}
    try { g.mixCross = mixCross; } catch (_) {}
    try { g.mixOpen = mixOpen; } catch (_) {}
    try { g.mixPanel = mixPanel; } catch (_) {}
    try { g.mixerEl = mixerEl; } catch (_) {}
    try { g.modeShuffleOn = modeShuffleOn; } catch (_) {}
    try { g.modeShuffleTimer = modeShuffleTimer; } catch (_) {}
    try { g.mount = mount; } catch (_) {}
    try { g.moved = moved; } catch (_) {}
    try { g.mqIdx = mqIdx; } catch (_) {}
    try { g.n = n; } catch (_) {}
    try { g.nEl = nEl; } catch (_) {}
    try { g.nL = nL; } catch (_) {}
    try { g.newA = newA; } catch (_) {}
    try { g.newB = newB; } catch (_) {}
    try { g.newBUrl = newBUrl; } catch (_) {}
    try { g.newVal = newVal; } catch (_) {}
    try { g.nextBeatTsA = nextBeatTsA; } catch (_) {}
    try { g.norm = norm; } catch (_) {}
    try { g.now = now; } catch (_) {}
    try { g.nowPlayingPollTimer = nowPlayingPollTimer; } catch (_) {}
    try { g.nowPlayingPollUrl = nowPlayingPollUrl; } catch (_) {}
    try { g.nowTs = nowTs; } catch (_) {}
    try { g.nudgeBm = nudgeBm; } catch (_) {}
    try { g.nudgeBp = nudgeBp; } catch (_) {}
    try { g.nv = nv; } catch (_) {}
    try { g.nx = nx; } catch (_) {}
    try { g.ny = ny; } catch (_) {}
    try { g.o = o; } catch (_) {}
    try { g.oEl = oEl; } catch (_) {}
    try { g.oldBUrl = oldBUrl; } catch (_) {}
    try { g.onDown = onDown; } catch (_) {}
    try { g.onFs = onFs; } catch (_) {}
    try { g.onMove = onMove; } catch (_) {}
    try { g.onPointerDown = onPointerDown; } catch (_) {}
    try { g.onPointerUpCanvas = onPointerUpCanvas; } catch (_) {}
    try { g.onPointerUpWebm = onPointerUpWebm; } catch (_) {}
    try { g.onUp = onUp; } catch (_) {}
    try { g.opA = opA; } catch (_) {}
    try { g.opB = opB; } catch (_) {}
    try { g.opQ = opQ; } catch (_) {}
    try { g.openBottomMenuPanel = openBottomMenuPanel; } catch (_) {}
    try { g.openKeyboardShortcutsPanel = openKeyboardShortcutsPanel; } catch (_) {}
    try { g.openOptionsPanel = openOptionsPanel; } catch (_) {}
    try { g.opt = opt; } catch (_) {}
    try { g.optBottom = optBottom; } catch (_) {}
    try { g.optLeft = optLeft; } catch (_) {}
    try { g.optRight = optRight; } catch (_) {}
    try { g.optTop = optTop; } catch (_) {}
    try { g.optionsAutoCloseId = optionsAutoCloseId; } catch (_) {}
    try { g.optionsPanel = optionsPanel; } catch (_) {}
    try { g.opts = opts; } catch (_) {}
    try { g.out = out; } catch (_) {}
    try { g.overlay = overlay; } catch (_) {}
    try { g.overlayGlowColorTimer = overlayGlowColorTimer; } catch (_) {}
    try { g.overlayGlowCycleCount = overlayGlowCycleCount; } catch (_) {}
    try { g.overlayGlowDurationMs = overlayGlowDurationMs; } catch (_) {}
    try { g.overlayGlowListenerBound = overlayGlowListenerBound; } catch (_) {}
    try { g.p = p; } catch (_) {}
    try { g.pDownTarget = pDownTarget; } catch (_) {}
    try { g.pDownX = pDownX; } catch (_) {}
    try { g.panelIdleTimer = panelIdleTimer; } catch (_) {}
    try { g.parsed = parsed; } catch (_) {}
    try { g.patGifLoadPromise = patGifLoadPromise; } catch (_) {}
    try { g.path = path; } catch (_) {}
    try { g.paths = paths; } catch (_) {}
    try { g.pct = pct; } catch (_) {}
    try { g.period = period; } catch (_) {}
    try { g.phase = phase; } catch (_) {}
    try { g.phaseOffsetBms = phaseOffsetBms; } catch (_) {}
    try { g.plan = plan; } catch (_) {}
    try { g.playing = playing; } catch (_) {}
    try { g.pickRandomStation = pickRandomStation; } catch (_) {}
    try { g.pickRandomStationB = pickRandomStationB; } catch (_) {}
    try { g.pickRandomStationForCrossfadedDeck = pickRandomStationForCrossfadedDeck; } catch (_) {}
    try { g.playRadioB = playRadioB; } catch (_) {}
    try { g.pos = pos; } catch (_) {}
    try { g.preferB = preferB; } catch (_) {}
    try { g.prevA = prevA; } catch (_) {}
    try { g.prevB = prevB; } catch (_) {}
    try { g.ptaGifLoadPromise = ptaGifLoadPromise; } catch (_) {}
    try { g.push = push; } catch (_) {}
    try { g.pushEnv = pushEnv; } catch (_) {}
    try { g.pushUnique = pushUnique; } catch (_) {}
    try { g.q = q; } catch (_) {}
    try { g.qi = qi; } catch (_) {}
    try { g.r = r; } catch (_) {}
    try { g.radioAHandoffAbortCtrl = radioAHandoffAbortCtrl; } catch (_) {}
    try { g.radioBHandoffAbortCtrl = radioBHandoffAbortCtrl; } catch (_) {}
    try { g.radioBRetryAttempts = radioBRetryAttempts; } catch (_) {}
    try { g.radioInputEl = radioInputEl; } catch (_) {}
    try { g.refreshMixStationB = refreshMixStationB; } catch (_) {}
    try { g.radioListEl = radioListEl; } catch (_) {}
    try { g.radioPanel = radioPanel; } catch (_) {}
    try { g.radioPanelTimer = radioPanelTimer; } catch (_) {}
    try { g.radioQuickBtn = radioQuickBtn; } catch (_) {}
    try { g.radioQuickHoldTimeout = radioQuickHoldTimeout; } catch (_) {}
    try { g.radioQuickHoldUntil = radioQuickHoldUntil; } catch (_) {}
    try { g.radioRetryAttempts = radioRetryAttempts; } catch (_) {}
    try { g.radioStationCrossfadeEnabled = radioStationCrossfadeEnabled; } catch (_) {}
    try { g.radioStationCrossfadeSec = radioStationCrossfadeSec; } catch (_) {}
    try { g.randomGlowColor = randomGlowColor; } catch (_) {}
    try { g.randomHexColor = randomHexColor; } catch (_) {}
    try { g.randomOmniLetterBorder = randomOmniLetterBorder; } catch (_) {}
    try { g.range = range; } catch (_) {}
    try { g.raw = raw; } catch (_) {}
    try { g.rawUrl = rawUrl; } catch (_) {}
    try { g.recentBottomColors = recentBottomColors; } catch (_) {}
    try { g.registerDeckVideoFeed = registerDeckVideoFeed; } catch (_) {}
    try { g.releaseDeckVideoFeed = releaseDeckVideoFeed; } catch (_) {}
    try { g.refreshActiveDeckVideoDisplays = refreshActiveDeckVideoDisplays; } catch (_) {}
    try { g.applyDeckVideoMirrorToElement = applyDeckVideoMirrorToElement; } catch (_) {}
    try { g.reindexStationCursor = reindexStationCursor; } catch (_) {}
    try { g.releaseAutoMixDeferredLocal = releaseAutoMixDeferredLocal; } catch (_) {}
    try { g.removeMediaVideoQueueItem = removeMediaVideoQueueItem; } catch (_) {}
    try { g.removeRadioStationAtIndex = removeRadioStationAtIndex; } catch (_) {}
    try { g.removed = removed; } catch (_) {}
    try { g.req = req; } catch (_) {}
    try { g.res = res; } catch (_) {}
    try { g.resetIdleTimer = resetIdleTimer; } catch (_) {}
    try { g.revealStartScreenEdgeFx = revealStartScreenEdgeFx; } catch (_) {}
    try { g.rmsA = rmsA; } catch (_) {}
    try { g.s = s; } catch (_) {}
    try { g.same = same; } catch (_) {}
    try { g.samePlain = samePlain; } catch (_) {}
    try { g.sampleLoop = sampleLoop; } catch (_) {}
    try { g.sanitizeUrlForAudio = sanitizeUrlForAudio; } catch (_) {}
    try { g.saveRadioStationCrossfadeToStorage = saveRadioStationCrossfadeToStorage; } catch (_) {}
    try { g.saveUserRadioStations = saveUserRadioStations; } catch (_) {}
    try { g.scheduleStartTextLoop = scheduleStartTextLoop; } catch (_) {}
    try { g.scopeAnimId = scopeAnimId; } catch (_) {}
    try { g.scopeCanvas = scopeCanvas; } catch (_) {}
    try { g.scopeCtx = scopeCtx; } catch (_) {}
    try { g.score = score; } catch (_) {}
    try { g.sd = sd; } catch (_) {}
    try { g.seen = seen; } catch (_) {}
    try { g.sel = sel; } catch (_) {}
    try { g.selWebm = selWebm; } catch (_) {}
    try { g.setBottomTextRandomColor = setBottomTextRandomColor; } catch (_) {}
    try { g.setKnobUi = setKnobUi; } catch (_) {}
    try { g.setLayer = setLayer; } catch (_) {}
    try { g.setLbl = setLbl; } catch (_) {}
    try { g.setStation = setStation; } catch (_) {}
    try { g.setStationB = setStationB; } catch (_) {}
    try { g.syncTopMenuStationsLayout = syncTopMenuStationsLayout; } catch (_) {}
    try { g.deckBHasLoadedContent = deckBHasLoadedContent; } catch (_) {}
    try { g.setVolume = setVolume; } catch (_) {}
    try { g.setWebmSpeed = setWebmSpeed; } catch (_) {}
    try { g.settingsApplyBtn = settingsApplyBtn; } catch (_) {}
    try { g.settingsBtn = settingsBtn; } catch (_) {}
    try { g.settingsCloseBtn = settingsCloseBtn; } catch (_) {}
    try { g.settingsPanel = settingsPanel; } catch (_) {}
    try { g.settingsPanelTimer = settingsPanelTimer; } catch (_) {}
    try { g.sf = sf; } catch (_) {}
    try { g.sh = sh; } catch (_) {}
    try { g.shell = shell; } catch (_) {}
    try { g.shortcuts = shortcuts; } catch (_) {}
    try { g.shortcutsLocked = shortcutsLocked; } catch (_) {}
    try { g.shortcutsTypeTimer = shortcutsTypeTimer; } catch (_) {}
    try { g.shouldDeferLocalPlayForAutoMix = shouldDeferLocalPlayForAutoMix; } catch (_) {}
    try { g.shouldToggleDeckBVizMountFullscreenFromPointer = shouldToggleDeckBVizMountFullscreenFromPointer; } catch (_) {}
    try { g.showBeatGrid = showBeatGrid; } catch (_) {}
    try { g.shuffleToggle = shuffleToggle; } catch (_) {}
    try { g.sideOpen = sideOpen; } catch (_) {}
    try { g.size = size; } catch (_) {}
    try { g.slot = slot; } catch (_) {}
    try { g.smooth = smooth; } catch (_) {}
    try { g.sr = sr; } catch (_) {}
    try { g.src = src; } catch (_) {}
    try { g.stRect = stRect; } catch (_) {}
    try { g.stage = stage; } catch (_) {}
    try { g.startBeatA = startBeatA; } catch (_) {}
    try { g.startBeatB = startBeatB; } catch (_) {}
    try { g.startTextLoopTimer = startTextLoopTimer; } catch (_) {}
    try { g.startVal = startVal; } catch (_) {}
    try { g.startY = startY; } catch (_) {}
    try { g.starts = starts; } catch (_) {}
    try { g.state = state; } catch (_) {}
    try { g.stationBanner = stationBanner; } catch (_) {}
    try { g.stationBannerMetaWrap = stationBannerMetaWrap; } catch (_) {}
    try { g.stationBannerNameEl = stationBannerNameEl; } catch (_) {}
    try { g.stationBannerNowplayingEl = stationBannerNowplayingEl; } catch (_) {}
    try { g.stationBannerTimer = stationBannerTimer; } catch (_) {}
    try { g.stationCycleEnabledByUrl = stationCycleEnabledByUrl; } catch (_) {}
    try { g.deckPlaybackHistory = deckPlaybackHistory; } catch (_) {}
    try { g.pushDeckPlaybackHistory = pushDeckPlaybackHistory; } catch (_) {}
    try { g.goPreviousDeckPlayback = goPreviousDeckPlayback; } catch (_) {}
    try { g.captureDeckPlaybackSnapshot = captureDeckPlaybackSnapshot; } catch (_) {}
    try { g.stationHistory = stationHistory; } catch (_) {}
    try { g.stations = stations; } catch (_) {}
    try { g.status = status; } catch (_) {}
    try { g.statusEl = statusEl; } catch (_) {}
    try { g.statusTypeTimer = statusTypeTimer; } catch (_) {}
    try { g.std = std; } catch (_) {}
    try { g.step = step; } catch (_) {}
    try { g.stopBeatA = stopBeatA; } catch (_) {}
    try { g.stopBeatB = stopBeatB; } catch (_) {}
    try { g.subEl = subEl; } catch (_) {}
    try { g.sum = sum; } catch (_) {}
    try { g.suppressHistoryPush = suppressHistoryPush; } catch (_) {}
    try { g.suppressNextOverlayStartUntil = suppressNextOverlayStartUntil; } catch (_) {}
    try { g.swipeHorizontalIntent = swipeHorizontalIntent; } catch (_) {}
    try { g.swipeVerticalIntent = swipeVerticalIntent; } catch (_) {}
    try { g.syncStationCycleSelection = syncStationCycleSelection; } catch (_) {}
    try { g.t = t; } catch (_) {}
    try { g.tapBtnA = tapBtnA; } catch (_) {}
    try { g.tapBtnB = tapBtnB; } catch (_) {}
    try { g.tapGifLoadPromise = tapGifLoadPromise; } catch (_) {}
    try { g.tapTimesA = tapTimesA; } catch (_) {}
    try { g.target = target; } catch (_) {}
    try { g.textAutoOn = textAutoOn; } catch (_) {}
    try { g.textAutoTimer = textAutoTimer; } catch (_) {}
    try { g.textInOpenH = textInOpenH; } catch (_) {}
    try { g.textInOpenVert = textInOpenVert; } catch (_) {}
    try { g.textInOpenW = textInOpenW; } catch (_) {}
    try { g.textInOpenWh = textInOpenWh; } catch (_) {}
    try { g.textInPanel = textInPanel; } catch (_) {}
    try { g.textOverlayAnimId = textOverlayAnimId; } catch (_) {}
    try { g.textOverlayLayer = textOverlayLayer; } catch (_) {}
    try { g.tiAutoBtn = tiAutoBtn; } catch (_) {}
    try { g.tiAutoInterval = tiAutoInterval; } catch (_) {}
    try { g.tiBorder = tiBorder; } catch (_) {}
    try { g.tiBorderColor = tiBorderColor; } catch (_) {}
    try { g.tiBorderColorRandBtn = tiBorderColorRandBtn; } catch (_) {}
    try { g.tiBorderColorRandCheck = tiBorderColorRandCheck; } catch (_) {}
    try { g.tiBorderRMax = tiBorderRMax; } catch (_) {}
    try { g.tiBorderRMin = tiBorderRMin; } catch (_) {}
    try { g.tiBorderRand = tiBorderRand; } catch (_) {}
    try { g.tiClose = tiClose; } catch (_) {}
    try { g.tiColor = tiColor; } catch (_) {}
    try { g.tiColorRandBtn = tiColorRandBtn; } catch (_) {}
    try { g.tiColorRandCheck = tiColorRandCheck; } catch (_) {}
    try { g.tiFlash = tiFlash; } catch (_) {}
    try { g.tiFlashSpeed = tiFlashSpeed; } catch (_) {}
    try { g.tiFlashSpeedRMax = tiFlashSpeedRMax; } catch (_) {}
    try { g.tiFlashSpeedRMin = tiFlashSpeedRMin; } catch (_) {}
    try { g.tiFlashSpeedRand = tiFlashSpeedRand; } catch (_) {}
    try { g.tiFont = tiFont; } catch (_) {}
    try { g.tiFontRand = tiFontRand; } catch (_) {}
    try { g.tiGlow = tiGlow; } catch (_) {}
    try { g.tiGlowColor = tiGlowColor; } catch (_) {}
    try { g.tiGlowColorRandBtn = tiGlowColorRandBtn; } catch (_) {}
    try { g.tiGlowColorRandCheck = tiGlowColorRandCheck; } catch (_) {}
    try { g.tiGlowRMax = tiGlowRMax; } catch (_) {}
    try { g.tiGlowRMin = tiGlowRMin; } catch (_) {}
    try { g.tiGlowRand = tiGlowRand; } catch (_) {}
    try { g.tiPreview = tiPreview; } catch (_) {}
    try { g.tiSend = tiSend; } catch (_) {}
    try { g.tiSize = tiSize; } catch (_) {}
    try { g.tiSizeRMax = tiSizeRMax; } catch (_) {}
    try { g.tiSizeRMin = tiSizeRMin; } catch (_) {}
    try { g.tiSizeRand = tiSizeRand; } catch (_) {}
    try { g.tiSpeed = tiSpeed; } catch (_) {}
    try { g.tiSpeedRMax = tiSpeedRMax; } catch (_) {}
    try { g.tiSpeedRMin = tiSpeedRMin; } catch (_) {}
    try { g.tiSpeedRand = tiSpeedRand; } catch (_) {}
    try { g.tiText = tiText; } catch (_) {}
    try { g.tiX = tiX; } catch (_) {}
    try { g.tiXRMax = tiXRMax; } catch (_) {}
    try { g.tiXRMin = tiXRMin; } catch (_) {}
    try { g.tiXRand = tiXRand; } catch (_) {}
    try { g.tipEl = tipEl; } catch (_) {}
    try { g.tipH = tipH; } catch (_) {}
    try { g.tipW = tipW; } catch (_) {}
    try { g.tipWh = tipWh; } catch (_) {}
    try { g.titleEl = titleEl; } catch (_) {}
    try { g.toggleBottomMenuPanel = toggleBottomMenuPanel; } catch (_) {}
    try { g.toggleDeckBVizMountFullscreen = toggleDeckBVizMountFullscreen; } catch (_) {}
    try { g.toggleVideoSurfaceFullscreen = toggleVideoSurfaceFullscreen; } catch (_) {}
    try { g.toggleFullscreen = toggleFullscreen; } catch (_) {}
    try { g.toggleKeyboardShortcutsPanel = toggleKeyboardShortcutsPanel; } catch (_) {}
    try { g.toggleMixPanel = toggleMixPanel; } catch (_) {}
    try { g.toggleOptionsPanel = toggleOptionsPanel; } catch (_) {}
    try { g.toggleRadioPanel = toggleRadioPanel; } catch (_) {}
    try { g.toggleTopMenuPanel = toggleTopMenuPanel; } catch (_) {}
    try { g.toggleTextInPanel = toggleTextInPanel; } catch (_) {}
    try { g.openTextInForTarget = openTextInForTarget; } catch (_) {}
    try { g.showWebm = showWebm; } catch (_) {}
    try { g.hideWebm = hideWebm; } catch (_) {}
    try { g.loadWebmList = loadWebmList; } catch (_) {}
    try { g.isWebmOverlayVisible = isWebmOverlayVisible; } catch (_) {}
    try { g.toggleWebmOverlay = toggleWebmOverlay; } catch (_) {}
    try { g.top = top; } catch (_) {}
    try { g.topBar = topBar; } catch (_) {}
    try { g.topEl = topEl; } catch (_) {}
    try { g.topOpen = topOpen; } catch (_) {}
    try { g.topOpenH = topOpenH; } catch (_) {}
    try { g.topOpenW = topOpenW; } catch (_) {}
    try { g.topOpenWh = topOpenWh; } catch (_) {}
    try { g.tryStartAutoMixDeferredLocal = tryStartAutoMixDeferredLocal; } catch (_) {}
    try { g.typeStatus = typeStatus; } catch (_) {}
    try { g.typeStatusTo = typeStatusTo; } catch (_) {}
    try { g.u = u; } catch (_) {}
    try { g.uiLayer = uiLayer; } catch (_) {}
    try { g.uiLockShield = uiLockShield; } catch (_) {}
    try { g.uiLockToggle = uiLockToggle; } catch (_) {}
    try { g.uiLocked = uiLocked; } catch (_) {}
    try { g.unbindWebmDeckBLayoutWatchers = unbindWebmDeckBLayoutWatchers; } catch (_) {}
    try { g.updateAll = updateAll; } catch (_) {}
    try { g.updateAvatarPlayButton = updateAvatarPlayButton; } catch (_) {}
    try { g.updateBpmRunLoop = updateBpmRunLoop; } catch (_) {}
    try { g.updateFromAvatarInputs = updateFromAvatarInputs; } catch (_) {}
    try { g.updateMixBStatus = updateMixBStatus; } catch (_) {}
    try { g.updateVisuals = updateVisuals; } catch (_) {}
    try { g.upsertMediaVideoQueueEntry = upsertMediaVideoQueueEntry; } catch (_) {}
    try { g.url = url; } catch (_) {}
    try { g.urlFly = urlFly; } catch (_) {}
    try { g.urlsMediaMatch = urlsMediaMatch; } catch (_) {}
    try { g.useSpeed = useSpeed; } catch (_) {}
    try { g.userRadioStations = userRadioStations; } catch (_) {}
    try { g.v = v; } catch (_) {}
    try { g.vA = vA; } catch (_) {}
    try { g.vB = vB; } catch (_) {}
    try { g.vQ = vQ; } catch (_) {}
    try { g.valTooltip = valTooltip; } catch (_) {}
    try { g.vd = vd; } catch (_) {}
    try { g.vertFromDj = vertFromDj; } catch (_) {}
    try { g.vf = vf; } catch (_) {}
    try { g.videoFile = videoFile; } catch (_) {}
    try { g.visualSettings = visualSettings; } catch (_) {}
    try { g.vizLayer = vizLayer; } catch (_) {}
    try { g.volumeSlider = volumeSlider; } catch (_) {}
    try { g.vs = vs; } catch (_) {}
    try { g.w = w; } catch (_) {}
    try { g.wL = wL; } catch (_) {}
    try { g.want = want; } catch (_) {}
    try { g.webm = webm; } catch (_) {}
    try { g.webmAnchorDeckB = webmAnchorDeckB; } catch (_) {}
    try { g.webmAutoOn = webmAutoOn; } catch (_) {}
    try { g.webmAutoTimer = webmAutoTimer; } catch (_) {}
    try { g.webmBtn = webmBtn; } catch (_) {}
    try { g.webmCloseBtn = webmCloseBtn; } catch (_) {}
    try { g.webmDeckBLayoutBound = webmDeckBLayoutBound; } catch (_) {}
    try { g.webmDeckBLayoutRaf = webmDeckBLayoutRaf; } catch (_) {}
    try { g.webmDeckBResizeObserver = webmDeckBResizeObserver; } catch (_) {}
    try { g.webmHorizontalIntent = webmHorizontalIntent; } catch (_) {}
    try { g.webmIndex = webmIndex; } catch (_) {}
    try {
        Object.defineProperty(g, 'webmList', { get: () => webmList, enumerable: true, configurable: true });
    } catch (_) {}
    try { g.webmNextBtn = webmNextBtn; } catch (_) {}
    try {
        Object.defineProperty(g, 'webmOn', { get: () => webmOn, enumerable: true, configurable: true });
    } catch (_) {}
    try { g.webmOverlay = webmOverlay; } catch (_) {}
    try { g.webmOverlayEl = webmOverlayEl; } catch (_) {}
    try { g.webmPrevBtn = webmPrevBtn; } catch (_) {}
    try { g.webmSettings = webmSettings; } catch (_) {}
    try { g.webmSettingsPanel = webmSettingsPanel; } catch (_) {}
    try { g.webmSettingsTimer = webmSettingsTimer; } catch (_) {}
    try { g.webmVideoEl = webmVideoEl; } catch (_) {}
    try { g.webmVideoLeftEl = webmVideoLeftEl; } catch (_) {}
    try { g.webmVideoRightEl = webmVideoRightEl; } catch (_) {}
    try { g.syncSpectrumOptionsControlsFromStorage = syncSpectrumOptionsControlsFromStorage; } catch (_) {}
    try { g.wireDjKnobMirror = wireDjKnobMirror; } catch (_) {}
    try { g.x = x; } catch (_) {}
    try { g.xf = xf; } catch (_) {}
    try { g.y = y; } catch (_) {}
}



/** Load extracted chunks in original source order (classic script = correct function hoisting). */
async function loadExtractedChunks() {
    exposeAppBindingsToGlobal();
    globalThis.THREE = THREE;
    globalThis.EffectComposer = EffectComposer;
    globalThis.RenderPass = RenderPass;
    globalThis.ShaderPass = ShaderPass;
    globalThis.AfterimagePass = AfterimagePass;
    globalThis.UnrealBloomPass = UnrealBloomPass;
    globalThis.butterchurn = butterchurn;
    globalThis.butterchurnPresets = butterchurnPresets;
    globalThis.SimplexNoise = SimplexNoise;
    globalThis.isMobile = isMobile;
    globalThis.QUALITY = QUALITY;
    async function fetchChunk(path) {
        const res = await fetch(path, { cache: 'no-store' });
        if (!res.ok) throw new Error('Failed to load ' + path);
        let chunk = await res.text();
        return chunk.replace(/^\/\*[\s\S]*?\*\/\s*/, '');
    }
    const visualMarker = '// --- MASTER CONTROL ---';
    const visualFull = await fetchChunk('js/visual-modes.js');
    const splitAt = visualFull.indexOf(visualMarker);
    if (splitAt < 0) throw new Error('visual-modes.js missing MASTER CONTROL marker');
    const code = [
        await fetchChunk('js/audio-engine.js'),
        visualFull.slice(0, splitAt),
        await fetchChunk('js/dj-decks.js'),
        await fetchChunk('js/radio-visual.js'),
        visualFull.slice(splitAt),
    ].join('\n');
    await new Promise((resolve, reject) => {
        const blob = new Blob([code], { type: 'text/javascript' });
        const url = URL.createObjectURL(blob);
        const s = document.createElement('script');
        s.src = url;
        s.onload = () => {
            URL.revokeObjectURL(url);
            resolve();
        };
        s.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error('Failed to execute extracted script'));
        };
        document.head.appendChild(s);
    });
}

await loadExtractedChunks();
try { globalThis.syncSpectrumOptionsControlsFromStorage?.(); } catch (_) {}

// Bridge extracted globals into this module (same bindings as former single-file scope).
const DjDecksEngine = globalThis.DjDecksEngine;
const EMOJIS = globalThis.EMOJIS;
const KaleidoShader = globalThis.KaleidoShader;
const MilkdropEngine = globalThis.MilkdropEngine;
const MilkdropEngineV2 = globalThis.MilkdropEngineV2;
const MilkdropEngineV3 = globalThis.MilkdropEngineV3;
const MilkdropPresetEngine = globalThis.MilkdropPresetEngine;
const RadialZoomShader = globalThis.RadialZoomShader;
const RadioVisualEngine = globalThis.RadioVisualEngine;
const ThreeEngine = globalThis.ThreeEngine;
const abortRadioAHandoff = globalThis.abortRadioAHandoff;
const addVideoFilesToMediaQueue = globalThis.addVideoFilesToMediaQueue;
const applyDeckBNoVocalPeakingGains = globalThis.applyDeckBNoVocalPeakingGains;
const applyDjBeatFxArp = globalThis.applyDjBeatFxArp;
const applyDjBeatFxDeckBCutRatePct = globalThis.applyDjBeatFxDeckBCutRatePct;
const applyDjBeatFxDeckBFlangerPct = globalThis.applyDjBeatFxDeckBFlangerPct;
const applyDjBeatFxDeckBNoVocalPct = globalThis.applyDjBeatFxDeckBNoVocalPct;
const applyDjBeatFxDeckBReverbPct = globalThis.applyDjBeatFxDeckBReverbPct;
const applyDjBeatFxLoop = globalThis.applyDjBeatFxLoop;
const applyDjBeatFxTk = globalThis.applyDjBeatFxTk;
const applyDjBeatPadLoopVisual = globalThis.applyDjBeatPadLoopVisual;
const applyRhythmCutGateRate = globalThis.applyRhythmCutGateRate;
const beginDeckAPingPongMediaHandoff = globalThis.beginDeckAPingPongMediaHandoff;
const bindDeckBMixerFxKnob = globalThis.bindDeckBMixerFxKnob;
const bindDjBeatFxKnobPair = globalThis.bindDjBeatFxKnobPair;
const buildDeckLocalItemsFromFiles = globalThis.buildDeckLocalItemsFromFiles;
const buildKaleidoSource = globalThis.buildKaleidoSource;
const closeDjBeatPadLoopMenu = globalThis.closeDjBeatPadLoopMenu;
const connectDeckMediaToEq = globalThis.connectDeckMediaToEq;
const createEmojiTexture = globalThis.createEmojiTexture;
const crossfadeValueFromPointerOnRange = globalThis.crossfadeValueFromPointerOnRange;
const enqueueDeckLocalFiles = globalThis.enqueueDeckLocalFiles;
const enqueueDeckUrl = globalThis.enqueueDeckUrl;
const ensureRadioAAltWired = globalThis.ensureRadioAAltWired;
const getDeckAMediaForPlaybackState = globalThis.getDeckAMediaForPlaybackState;
const getDeckARadioAudibleEl = globalThis.getDeckARadioAudibleEl;
const hideOverlay = globalThis.hideOverlay;
const inferLocalMediaKind = globalThis.inferLocalMediaKind;
const ingestLocalFilesToDeckAndPlay = globalThis.ingestLocalFilesToDeckAndPlay;
const initAudio = globalThis.initAudio;
const isDeckARadioOutputFromAlt = globalThis.isDeckARadioOutputFromAlt;
const isVideoFileForMediaQueue = globalThis.isVideoFileForMediaQueue;
const loadMode = globalThis.loadMode;
const makeReverbImpulseBuffer = globalThis.makeReverbImpulseBuffer;
const modes = globalThis.modes;
const moveQueuedTrackToOtherDeck = globalThis.moveQueuedTrackToOtherDeck;
const onDeckAEndedForQueue = globalThis.onDeckAEndedForQueue;
const onDeckBEndedForQueue = globalThis.onDeckBEndedForQueue;
const openDjBeatPadLoopMenu = globalThis.openDjBeatPadLoopMenu;
const playDeckATrackFromQueue = globalThis.playDeckATrackFromQueue;
const playDeckBTrackFromQueue = globalThis.playDeckBTrackFromQueue;
const playDeckUrlNow = globalThis.playDeckUrlNow;
const playQueuedTrackNow = globalThis.playQueuedTrackNow;
const playRadio = globalThis.playRadio;
const prependDeckLocalItems = globalThis.prependDeckLocalItems;
const promptAddUrlForDeck = globalThis.promptAddUrlForDeck;
const rebuildEffectsChain = globalThis.rebuildEffectsChain;
const reconnectDjDecksDeckBProjectMIfActive = globalThis.reconnectDjDecksDeckBProjectMIfActive;
const refreshDjBeatArpKnobVisuals = globalThis.refreshDjBeatArpKnobVisuals;
const refreshDjBeatLoopKnobVisuals = globalThis.refreshDjBeatLoopKnobVisuals;
const removeQueuedTrack = globalThis.removeQueuedTrack;
const resetDeckFileQueuesAndRevoke = globalThis.resetDeckFileQueuesAndRevoke;
const resetRadioADualStreamHandoff = globalThis.resetRadioADualStreamHandoff;
const abortRadioBHandoff = globalThis.abortRadioBHandoff;
const resetRadioBDualStreamHandoff = globalThis.resetRadioBDualStreamHandoff;
const ensureRadioBAltWired = globalThis.ensureRadioBAltWired;
const getDeckBRadioAudibleEl = globalThis.getDeckBRadioAudibleEl;
const isDeckBRadioOutputFromAlt = globalThis.isDeckBRadioOutputFromAlt;
const revokeBlobSrc = globalThis.revokeBlobSrc;
const sceneBarnsleyFern = globalThis.sceneBarnsleyFern;
const sceneBars = globalThis.sceneBars;
const sceneBars3D = globalThis.sceneBars3D;
const sceneBarsCircle = globalThis.sceneBarsCircle;
const sceneBarsVerticalMirror = globalThis.sceneBarsVerticalMirror;
const sceneBarsVortex = globalThis.sceneBarsVortex;
const sceneBlank = globalThis.sceneBlank;
const sceneElectroSphere = globalThis.sceneElectroSphere;
const sceneEmojiSwarm = globalThis.sceneEmojiSwarm;
const sceneGalaxy = globalThis.sceneGalaxy;
const sceneGameOfLife = globalThis.sceneGameOfLife;
const sceneHexGrid = globalThis.sceneHexGrid;
const sceneInfinityMirror = globalThis.sceneInfinityMirror;
const sceneInfinityTunnel = globalThis.sceneInfinityTunnel;
const sceneJulia = globalThis.sceneJulia;
const sceneKaleido = globalThis.sceneKaleido;
const sceneKaleidoLayered = globalThis.sceneKaleidoLayered;
const sceneKaleidoRings = globalThis.sceneKaleidoRings;
const sceneKaleidoSpiral = globalThis.sceneKaleidoSpiral;
const sceneKaleidoZoom = globalThis.sceneKaleidoZoom;
const sceneLorenz = globalThis.sceneLorenz;
const sceneMandelbrot = globalThis.sceneMandelbrot;
const sceneMengerSponge = globalThis.sceneMengerSponge;
const sceneNeonTunnel = globalThis.sceneNeonTunnel;
const sceneParticleTunnel = globalThis.sceneParticleTunnel;
const sceneParticles = globalThis.sceneParticles;
const scenePhotonShell = globalThis.scenePhotonShell;
const scenePulseOrb = globalThis.scenePulseOrb;
const sceneRibbons = globalThis.sceneRibbons;
const sceneSierpinskiCarpet = globalThis.sceneSierpinskiCarpet;
const sceneSierpinskiTetra = globalThis.sceneSierpinskiTetra;
const sceneSphere = globalThis.sceneSphere;
const sceneStarfield = globalThis.sceneStarfield;
const sceneTerrain = globalThis.sceneTerrain;
const sceneTunnel = globalThis.sceneTunnel;
const sceneTwistTunnel = globalThis.sceneTwistTunnel;
const sceneWaveGrid = globalThis.sceneWaveGrid;
const scheduleClearDjLocalPickImmediate = globalThis.scheduleClearDjLocalPickImmediate;
const setDjDeckRadioLoadingSpinner = globalThis.setDjDeckRadioLoadingSpinner;
const setRhythmCutModulationActive = globalThis.setRhythmCutModulationActive;
const setStartControlsEnabled = globalThis.setStartControlsEnabled;
const startGame = globalThis.startGame;
const stopAllAndShowStart = globalThis.stopAllAndShowStart;
const syncDjBeatFxKnobActiveDom = globalThis.syncDjBeatFxKnobActiveDom;
const syncDjMusicalLoopDelayTime = globalThis.syncDjMusicalLoopDelayTime;
const toggleDeckBMixerFxFromKnob = globalThis.toggleDeckBMixerFxFromKnob;
const toggleDjBeatFxArpFromKnob = globalThis.toggleDjBeatFxArpFromKnob;
const toggleDjBeatFxLoopFromKnob = globalThis.toggleDjBeatFxLoopFromKnob;
const toggleDjBeatFxTkFromKnob = globalThis.toggleDjBeatFxTkFromKnob;
const toggleRhythmCutFx = globalThis.toggleRhythmCutFx;
const updateDjDecksShortcutVisibility = globalThis.updateDjDecksShortcutVisibility;
const updateSkipPresetButtonVisibility = globalThis.updateSkipPresetButtonVisibility;
const useFile = globalThis.useFile;
const useMic = globalThis.useMic;
const waitForRadioStreamAudible = globalThis.waitForRadioStreamAudible;
const wireDjBeatFxKnobs = globalThis.wireDjBeatFxKnobs;


        // --- BINDINGS ---
        document.getElementById('btn-prev')?.addEventListener('click', () => { loadMode(state.currentModeIdx - 1); resetIdleTimer(); });
        document.getElementById('btn-next')?.addEventListener('click', () => { loadMode(state.currentModeIdx + 1); resetIdleTimer(); });
        const btnRadioEl = document.getElementById('btn-radio');
        if (btnRadioEl) btnRadioEl.addEventListener('click', playRadio);
        const btnMicEl = document.getElementById('btn-mic');
        if (btnMicEl) btnMicEl.addEventListener('click', useMic);
        const fileInputEl = document.getElementById('file-input');
        if (fileInputEl) fileInputEl.addEventListener('change', useFile);
        document.getElementById('btn-fullscreen')?.addEventListener('click', toggleFullscreen);
        const btnDjDecksShortcut = document.getElementById('btn-dj-decks');
        if (btnDjDecksShortcut) {
            btnDjDecksShortcut.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const loadDjDecksMode = () => {
                    const load = (typeof globalThis.loadMode === 'function')
                        ? globalThis.loadMode
                        : (typeof loadMode === 'function' ? loadMode : null);
                    const m = globalThis.modes || modes;
                    if (!Array.isArray(m) || typeof load !== 'function') return;
                    const idx = m.findIndex((x) => x && x.name === 'DJ Decks');
                    if (idx >= 0) load(idx);
                };
                try {
                    const visName = state.activeVisualizer && state.activeVisualizer.name;
                    if (visName && visName !== 'DJ Decks') {
                        loadDjDecksMode();
                        resetIdleTimer();
                        return;
                    }
                    const onDj = !!(state.activeVisualizer && state.activeVisualizer.name === 'DJ Decks');
                    const deckBVisualActive = !!(
                        onDj &&
                        state.activeVisualizer &&
                        (state.activeVisualizer.deckBVizMode === 'bars' ||
                         state.activeVisualizer.deckBVizMode === 'projectm' ||
                         state.activeVisualizer.deckBVizMode === 'blank' ||
                         state.activeVisualizer.deckBVizMode === 'video' ||
                         state.activeVisualizer.deckBVizMode === 'karaoke' ||
                         state.activeVisualizer.deckBVizMode === 'kbop' ||
                         state.activeVisualizer.deckBQueueVisible ||
                         state.activeVisualizer.deckBMediaPanelVisible)
                    );
                    const deckBTextActive = (() => {
                        try {
                            if (typeof getDeckBStageEl !== 'function') return false;
                            const stage = getDeckBStageEl();
                            return !!(stage && stage.classList.contains('dj-deck-b-text-mode'));
                        } catch (_) { return false; }
                    })();
                    if (onDj && deckBTextActive) {
                        try { if (typeof setTextAuto === 'function') setTextAuto(false); } catch (_) {}
                        try { if (typeof setDeckBTextMode === 'function') setDeckBTextMode(false); } catch (_) {}
                        try {
                            const tip = document.getElementById('textin-panel');
                            if (tip) delete tip.dataset.textTarget;
                            const panelOpen = !!(tip && !tip.classList.contains('display-none') && tip.classList.contains('open'));
                            if (panelOpen && typeof hideTextInPanel === 'function') {
                                hideTextInPanel({ forceCloseDeckB: true });
                            }
                        } catch (_) {}
                        try { if (typeof syncDjTextInDeckLights === 'function') syncDjTextInDeckLights(); } catch (_) {}
                        resetIdleTimer();
                        return;
                    }
                    if (deckBVisualActive) {
                        try { if (state.activeVisualizer.deckBQueueVisible && typeof state.activeVisualizer.hideDeckBQueueView === 'function') state.activeVisualizer.hideDeckBQueueView(); } catch (_) {}
                        try { if (state.activeVisualizer.deckBMediaPanelVisible && typeof state.activeVisualizer.hideDeckBMediaView === 'function') state.activeVisualizer.hideDeckBMediaView(); } catch (_) {}
                        if (typeof state.activeVisualizer.tearDownDeckBViz === 'function') {
                            state.activeVisualizer.tearDownDeckBViz();
                        }
                        if (typeof state.activeVisualizer.syncDeckBVisualButtons === 'function') {
                            state.activeVisualizer.syncDeckBVisualButtons();
                        }
                        if (typeof state.activeVisualizer.updateStationTitles === 'function') {
                            state.activeVisualizer.updateStationTitles();
                        }
                        resetIdleTimer();
                        return;
                    }
                } catch (_) {}
                loadDjDecksMode();
                resetIdleTimer();
            });
        }
        const btnReturnRadio = document.getElementById('btn-return-radio');
        if (btnReturnRadio) {
            btnReturnRadio.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                try {
                    if (typeof globalThis.toggleRadioVisualVariant === 'function') {
                        if (globalThis.toggleRadioVisualVariant()) {
                            resetIdleTimer();
                            return;
                        }
                    }
                    if (typeof globalThis.loadRadioVisualMode === 'function') {
                        globalThis.loadRadioVisualMode();
                    } else if (typeof loadRadioVisualMode === 'function') {
                        loadRadioVisualMode();
                    }
                } catch (_) {}
                resetIdleTimer();
            });
        }
        const btnWebmPanelToggle = document.getElementById('btn-webm-panel-toggle');
        if (btnWebmPanelToggle) {
            const toggleAvatarPlayStop = () => {
                try {
                    if (!webmOn) {
                        if (webmList.length === 0 && typeof loadWebmList === 'function') {
                            loadWebmList().finally(() => {
                                if (webmList.length > 0) {
                                    showWebm();
                                    try { updateAvatarPlayButton(); } catch(_) {}
                                }
                            });
                        } else {
                            showWebm();
                            try { updateAvatarPlayButton(); } catch(_) {}
                        }
                    } else {
                        hideWebm();
                        try { updateAvatarPlayButton(); } catch(_) {}
                    }
                } catch(_) {}
            };
            let avatarHoldTimer = null;
            let avatarLongPressHandled = false;
            const AVATAR_LONG_PRESS_MS = 500;
            const clearAvatarHold = () => {
                if (avatarHoldTimer) {
                    clearTimeout(avatarHoldTimer);
                    avatarHoldTimer = null;
                }
            };
            const onAvatarHoldStart = (e) => {
                try { e.preventDefault(); } catch(_) {}
                avatarLongPressHandled = false;
                clearAvatarHold();
                avatarHoldTimer = setTimeout(() => {
                    avatarLongPressHandled = true;
                    toggleAvatarPlayStop();
                }, AVATAR_LONG_PRESS_MS);
            };
            const onAvatarHoldEnd = () => clearAvatarHold();
            btnWebmPanelToggle.addEventListener('pointerdown', onAvatarHoldStart, { passive: false });
            btnWebmPanelToggle.addEventListener('pointerup', onAvatarHoldEnd);
            btnWebmPanelToggle.addEventListener('pointercancel', onAvatarHoldEnd);
            btnWebmPanelToggle.addEventListener('pointerleave', onAvatarHoldEnd);
            btnWebmPanelToggle.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (avatarLongPressHandled) {
                    avatarLongPressHandled = false;
                    return;
                }
                // Toggle the Avatar/WebM Panel (which is now the right-side panel)
                if (typeof toggleBottomMenuPanel === 'function') {
                    toggleBottomMenuPanel();
                }
                resetIdleTimer();
            });
        }
        // HUD ⌨️ opens keyboard shortcuts reference sheet
        if (webmBtn) {
            webmBtn.title = "Keyboard shortcuts";
            webmBtn.addEventListener('click', (e) => { 
                e.preventDefault(); 
                e.stopPropagation(); 
                try { toggleKeyboardShortcutsPanel(); } catch(_) {}
                resetIdleTimer(); 
            });
        }
        // Settings button now opens Mix Settings (right panel)
        function toggleMixPanel() {
            try {
                if (mixPanel.classList.contains('display-none') || !mixPanel.classList.contains('open')) {
                    try { refreshMixStationB(); } catch(_) {}
                    mixPanel.classList.remove('display-none');
                    requestAnimationFrame(() => { mixPanel.classList.add('open'); });
                } else {
                    mixPanel.classList.remove('open');
                    
                    // ADD THIS LINE: Prevent other panels from opening for 1.2 seconds
                    try { window.__panelGuardUntilMs = Date.now() + 1200; } catch(_) {}
                    
                    setTimeout(() => { mixPanel.classList.add('display-none'); }, 350);
                }
            } catch(_) {}
        }
        function showMicConfirm() {
            if (!micConfirm) { if (typeof useMic === 'function') useMic(); return; }
            micConfirm.style.display = 'flex';
        }
        function hideMicConfirm() {
            if (!micConfirm) return;
            micConfirm.style.display = 'none';
        }
        function showMidiConfirm() {
            const titleEl = document.getElementById('midi-confirm-title');
            const descEl = document.getElementById('midi-confirm-desc');
            if (!midiConfirm) {
                try { initWebMidi(); } catch (_) {}
                return;
            }
            if (typeof navigator === 'undefined' || !navigator.requestMIDIAccess) {
                if (titleEl) titleEl.textContent = 'MIDI not available';
                if (descEl) descEl.textContent = 'This browser does not support the Web MIDI API.';
            } else if (window.__midiInitDone) {
                if (titleEl) titleEl.textContent = 'MIDI Input';
                if (descEl) descEl.textContent = 'MIDI is already enabled. Connect or reconnect USB or Bluetooth controllers; connection status appears beside the mix controls.';
            } else {
                if (titleEl) titleEl.textContent = 'Enable MIDI Input?';
                if (descEl) descEl.textContent = 'Your browser will ask permission to access MIDI devices. Controllers can then adjust crossfader, deck gains, and transport (defaults are described in the mix panel MIDI tooltip).';
            }
            midiConfirm.style.display = 'flex';
        }
        function hideMidiConfirm() {
            if (!midiConfirm) return;
            midiConfirm.style.display = 'none';
        }
        if (micOk) micOk.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            hideMicConfirm();
            try { useMic(); } catch(_) {}
        });
        if (micCancel) micCancel.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            hideMicConfirm();
        });
        if (midiOk) midiOk.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            hideMidiConfirm();
            try { initWebMidi(); } catch (_) {}
        });
        if (midiCancel) midiCancel.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            hideMidiConfirm();
        });
        const midiStatusTap = document.getElementById('midi-status');
        if (midiStatusTap) {
            midiStatusTap.addEventListener('click', (e) => {
                e.preventDefault(); e.stopPropagation();
                try { showMidiConfirm(); } catch (_) {}
                try { resetIdleTimer(); } catch (_) {}
            });
        }
        if (settingsBtn) {
            settingsBtn.title = "Mix Settings";
            settingsBtn.addEventListener('click', (e) => {
                e.preventDefault(); e.stopPropagation();
                toggleMixPanel();
                resetIdleTimer();
            });
        }
        if (mixClose) {
            mixClose.addEventListener('click', (e) => {
                e.preventDefault(); e.stopPropagation();
                mixPanel.classList.remove('open');
                
                // ADD THIS LINE: Prevent other panels from opening for 1.2 seconds
                try { window.__panelGuardUntilMs = Date.now() + 1200; } catch(_) {}
                
                setTimeout(() => { mixPanel.classList.add('display-none'); }, 350);
            });
        }
        // Populate second station select (legacy no-op; lists live in top menu)
        function refreshMixStationB() {
            try { updateStationActiveHighlight(); } catch (_) {}
            try { syncTopMenuStationsLayout(); } catch (_) {}
        }
        refreshMixStationB();
        // Crossfader handler (equal-power)
        function applyCrossfade(val) {
            let x = Math.max(0, Math.min(1, Number(val)||0));
            // Add a small dead-zone at both extremes so "fully A/B" is a hard cut.
            // This avoids tiny residual bleed when pointer/touch lands near (but not exactly on) 0 or 1.
            const EDGE_SNAP = 0.03;
            if (x <= EDGE_SNAP) x = 0;
            else if (x >= 1 - EDGE_SNAP) x = 1;
            // Linear crossfade: exact 50/50 amplitude at midpoint
            const ga = 1 - x;
            const gb = x;
            if (state && state.audioCtx) {
                const t = state.audioCtx.currentTime;
                try {
                    if (state.streamAGain) {
                        state.streamAGain.gain.cancelScheduledValues(t);
                        state.streamAGain.gain.setValueAtTime(ga, t);
                    }
                    if (state.streamBGain) {
                        state.streamBGain.gain.cancelScheduledValues(t);
                        state.streamBGain.gain.setValueAtTime(gb, t);
                    }
                } catch (_) {
                    if (state.streamAGain) state.streamAGain.gain.value = ga;
                    if (state.streamBGain) state.streamBGain.gain.value = gb;
                }
            }
            try {
                // Also drive the two-tone gauge fill on each slider rail wrapper.
                // The visible rail is inset by half the 56px thumb width, matching
                // the browser's actual thumb-centre travel. Keeping `--cross-x`
                // in lock-step with `.value` here makes drag, AUTO-FADE ticks,
                // cut-fade hold, MIDI, and the mix-panel mirror all update the
                // corrected rail without each path needing its own hook.
                const cx = String(x);
                if (mixCross) {
                    mixCross.value = cx;
                    try { mixCross.style.setProperty('--cross-x', cx); } catch (_) {}
                    try { if (mixCross.parentElement) mixCross.parentElement.style.setProperty('--cross-x', cx); } catch (_) {}
                }
                const djX = document.getElementById('dj-crossfader');
                if (djX) {
                    djX.value = cx;
                    try { djX.style.setProperty('--cross-x', cx); } catch (_) {}
                    try { if (djX.parentElement) djX.parentElement.style.setProperty('--cross-x', cx); } catch (_) {}
                }
                const rdX = document.getElementById('radio-visual-cross-digital');
                if (rdX) {
                    rdX.value = cx;
                    try { rdX.style.setProperty('--cross-x', cx); } catch (_) {}
                    try { if (rdX.parentElement) rdX.parentElement.style.setProperty('--cross-x', cx); } catch (_) {}
                }
            } catch(_) {}
            try { refreshActiveDeckVideoDisplays(); } catch (_) {}
            try { updateModeSubStationLine(); } catch (_) {}
        }

        const CROSSFADE_KEY_STEP = 0.04;
        function nudgeCrossfade(delta) {
            const d = Number(delta);
            if (!Number.isFinite(d) || d === 0) return;
            const cur = getAppCrossfade01();
            applyCrossfade(Math.max(0, Math.min(1, cur + d)));
            try { resumeDecksForCrossfadeLevels(); } catch (_) {}
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv._syncCrossfadeKnob === 'function') rv._syncCrossfadeKnob();
            } catch (_) {}
            try { resetIdleTimer(); } catch (_) {}
        }

        if (mixCross) {
            mixCross.addEventListener('input', () => applyCrossfade(mixCross.value));
            try { applyCrossfade(mixCross.value); } catch(_) {}
        }

        function resumeDecksForCrossfadeLevels() {
            try {
                const djX = document.getElementById('dj-crossfader');
                const xv = Math.max(0, Math.min(1, Number((djX && djX.value) || (mixCross && mixCross.value) || 0)));
                const ga = 1 - xv;
                const gb = xv;
                const thresh = 0.03;
                if (ga > thresh && typeof audioEl !== 'undefined' && audioEl && audioEl.src && audioEl.paused) audioEl.play().catch(() => {});
                if (gb > thresh && typeof audioElB !== 'undefined' && audioElB && audioElB.src && audioElB.paused) audioElB.play().catch(() => {});
            } catch (_) {}
        }
        function wireMixCrossCutFadeHold() {
            if (!mixCross || mixCross.dataset.cutFadeHold === '1') return;
            mixCross.dataset.cutFadeHold = '1';
            let cutFadeHoldActive = false;
            let cutFadeHoldRestore = 0;
            let cutFadeHoldPointerId = null;
            const restoreCutFadeHold = () => {
                if (!cutFadeHoldActive) return;
                cutFadeHoldActive = false;
                cutFadeHoldPointerId = null;
                try { applyCrossfade(cutFadeHoldRestore); } catch (_) {}
                try { resumeDecksForCrossfadeLevels(); } catch (_) {}
            };
            const onCutFadePointerDown = (ev) => {
                if (ev.button !== 2) return;
                try { ev.preventDefault(); } catch (_) {}
                cutFadeHoldRestore = Math.max(0, Math.min(1, Number(mixCross.value) || 0));
                const cutTo = crossfadeValueFromPointerOnRange(mixCross, ev.clientX);
                cutFadeHoldActive = true;
                cutFadeHoldPointerId = ev.pointerId;
                try { applyCrossfade(cutTo); } catch (_) {}
                try { resumeDecksForCrossfadeLevels(); } catch (_) {}
                try { mixCross.setPointerCapture(ev.pointerId); } catch (_) {}
            };
            const onCutFadePointerEnd = (ev) => {
                if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                try { mixCross.releasePointerCapture(ev.pointerId); } catch (_) {}
                restoreCutFadeHold();
            };
            mixCross.addEventListener('pointerdown', onCutFadePointerDown);
            mixCross.addEventListener('pointerup', onCutFadePointerEnd);
            mixCross.addEventListener('pointercancel', onCutFadePointerEnd);
            mixCross.addEventListener('lostpointercapture', (ev) => {
                if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                restoreCutFadeHold();
            });
            mixCross.addEventListener('contextmenu', (e) => {
                try { e.preventDefault(); } catch (_) {}
            });
        }
        wireMixCrossCutFadeHold();

        function wireMixPanelAutoButtonsToDjDeck() {
            const pairs = [
                ['mix-autofade', 'dj-autofade'],
                ['mix-automix', 'dj-automix']
            ];
            pairs.forEach(([mixId, djId]) => {
                const m = document.getElementById(mixId);
                const d = document.getElementById(djId);
                if (!m || !d || m.dataset.djMirror === '1') return;
                m.dataset.djMirror = '1';
                const syncClass = () => {
                    queueMicrotask(() => {
                        try { m.classList.toggle('on', d.classList.contains('on')); } catch (_) {}
                    });
                };
                ['pointerdown', 'pointerup', 'pointercancel', 'pointerleave'].forEach((t) => {
                    m.addEventListener(t, (ev) => {
                        try {
                            d.dispatchEvent(new PointerEvent(t, {
                                bubbles: true,
                                cancelable: true,
                                composed: true,
                                view: window,
                                detail: ev.detail,
                                pointerId: ev.pointerId,
                                pointerType: ev.pointerType,
                                isPrimary: ev.isPrimary,
                                clientX: ev.clientX,
                                clientY: ev.clientY,
                                screenX: ev.screenX,
                                screenY: ev.screenY,
                                button: ev.button,
                                buttons: ev.buttons,
                                altKey: ev.altKey,
                                ctrlKey: ev.ctrlKey,
                                metaKey: ev.metaKey,
                                shiftKey: ev.shiftKey,
                                pressure: ev.pressure,
                                tangentialPressure: ev.tangentialPressure,
                                tiltX: ev.tiltX,
                                tiltY: ev.tiltY,
                                twist: ev.twist,
                                width: ev.width,
                                height: ev.height
                            }));
                        } catch (_) {}
                        try { ev.preventDefault(); } catch (_) {}
                        try { ev.stopPropagation(); } catch (_) {}
                        syncClass();
                    });
                });
                m.addEventListener('click', (ev) => {
                    try { ev.preventDefault(); } catch (_) {}
                    try { ev.stopPropagation(); } catch (_) {}
                    try { d.click(); } catch (_) {}
                    syncClass();
                });
            });
        }
        wireMixPanelAutoButtonsToDjDeck();

        // --- Web MIDI (USB + Bluetooth class-compliant devices via OS driver) ---
        const MIDI_MAP_DEFAULT = Object.freeze({
            crossfaderCC: 54,
            crossfaderChannel: null,
            gainACC: 7,
            gainAChannel: 0,
            gainBCC: 7,
            gainBChannel: 1,
            playANote: 36,
            playBNote: 37,
            playChannel: null
        });
        function loadMidiMap() {
            try {
                const raw = localStorage.getItem('omniMidiMap');
                if (!raw) return { ...MIDI_MAP_DEFAULT };
                return { ...MIDI_MAP_DEFAULT, ...JSON.parse(raw) };
            } catch (_) {
                return { ...MIDI_MAP_DEFAULT };
            }
        }
        let midiMap = loadMidiMap();
        let midiAccessRef = null;
        const midiNoteDebounceMs = 120;
        const midiLastNoteAt = Object.create(null);

        function midiChannelMatches(cfgCh, msgCh) {
            if (cfgCh == null || cfgCh === '') return true;
            return Number(cfgCh) === Number(msgCh);
        }

        function updateMidiStatus(mode) {
            const el = document.getElementById('midi-status');
            if (!el) return;
            if (mode === 'unsupported') {
                el.textContent = 'MIDI · n/a';
            } else if (mode === 'denied') {
                el.textContent = 'MIDI · blocked';
            } else if (mode === 'ready') {
                el.textContent = 'MIDI · ready';
            } else if (mode === 'ok') {
                let n = 0;
                try {
                    if (midiAccessRef && midiAccessRef.inputs) n = midiAccessRef.inputs.size;
                } catch (_) {}
                el.textContent = n ? ('MIDI · ' + n + ' in') : 'MIDI · —';
            }
        }

        function midiApplyDeckGain(deck, ccValue) {
            const v = Math.max(0, Math.min(127, ccValue)) / 127 * 2;
            try { initAudio(); } catch (_) {}
            try {
                if (deck === 'a') {
                    eqState.a.gain = v;
                    if (state.trimA) state.trimA.gain.value = v;
                    setKnobUi(document.getElementById('knob-a-gain'), 0, 2, v);
                    const djM = window.djKnobMirrors && window.djKnobMirrors['knob-a-gain'];
                    if (djM) setKnobUi(djM, 0, 2, v);
                } else {
                    eqState.b.gain = v;
                    if (state.trimB) state.trimB.gain.value = v;
                    setKnobUi(document.getElementById('knob-b-gain'), 0, 2, v);
                    const djM = window.djKnobMirrors && window.djKnobMirrors['knob-b-gain'];
                    if (djM) setKnobUi(djM, 0, 2, v);
                }
            } catch (_) {}
        }

        async function midiToggleDeckA() {
            try { initAudio(); } catch (_) {}
            try {
                const eng = getDjDecksEngineIfActive();
                try {
                    if (eng && typeof eng.cancelAutoFade === 'function') eng.cancelAutoFade();
                } catch (_) {}
                const media = (typeof getDeckAMediaForPlaybackState === 'function')
                    ? getDeckAMediaForPlaybackState()
                    : audioEl;
                if (!media || !deckHasSource(media)) {
                    try {
                        if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') eng.clearSuppressEnsureCrossfadeDeckPlayback();
                    } catch (_) {}
                    if (typeof playRadio === 'function') playRadio();
                    return;
                }
                if (media.paused) {
                    try {
                        if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') eng.clearSuppressEnsureCrossfadeDeckPlayback();
                    } catch (_) {}
                    await media.play().catch(() => { if (typeof playRadio === 'function') playRadio(); });
                } else {
                    try {
                        if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') eng.clearSuppressEnsureCrossfadeDeckPlayback();
                    } catch (_) {}
                    media.pause();
                }
            } catch (_) {}
        }

        async function midiToggleDeckB() {
            try { initAudio(); } catch (_) {}
            try {
                if (!audioElB || !audioElB.src || audioElB.paused) {
                    try {
                        const eng = getDjDecksEngineIfActive();
                        if (eng && typeof eng.clearSuppressEnsureCrossfadeDeckPlayback === 'function') eng.clearSuppressEnsureCrossfadeDeckPlayback();
                    } catch (_) {}
                    if (typeof playRadioB === 'function') playRadioB();
                } else {
                    audioElB.pause();
                }
            } catch (_) {}
        }

        function handleMidiMessage(data) {
            if (!data || data.length < 2 || typeof state === 'undefined') return;
            midiMap = loadMidiMap();
            const st = data[0];
            const d1 = data[1];
            const d2 = data.length > 2 ? data[2] : 0;
            const type = st & 0xf0;
            const channel = st & 0x0f;

            if (type === 0xB0) {
                const cc = d1;
                const v = d2;
                if (cc === midiMap.crossfaderCC && midiChannelMatches(midiMap.crossfaderChannel, channel)) {
                    if (typeof applyCrossfade === 'function') applyCrossfade(v / 127);
                    return;
                }
                if (cc === midiMap.gainACC && midiChannelMatches(midiMap.gainAChannel, channel)) {
                    midiApplyDeckGain('a', v);
                    return;
                }
                if (cc === midiMap.gainBCC && midiChannelMatches(midiMap.gainBChannel, channel)) {
                    midiApplyDeckGain('b', v);
                    return;
                }
            }

            if (type === 0x90 && d2 > 0) {
                const note = d1;
                if (!midiChannelMatches(midiMap.playChannel, channel)) return;
                const now = performance.now();
                if (note === midiMap.playANote) {
                    const k = 'a' + note;
                    if (midiLastNoteAt[k] && now - midiLastNoteAt[k] < midiNoteDebounceMs) return;
                    midiLastNoteAt[k] = now;
                    midiToggleDeckA();
                    return;
                }
                if (note === midiMap.playBNote) {
                    const k = 'b' + note;
                    if (midiLastNoteAt[k] && now - midiLastNoteAt[k] < midiNoteDebounceMs) return;
                    midiLastNoteAt[k] = now;
                    midiToggleDeckB();
                }
            }
        }

        function wireMidiInputs(access) {
            try {
                for (const inp of access.inputs.values()) {
                    try {
                        inp.onmidimessage = (e) => handleMidiMessage(e.data);
                    } catch (_) {}
                }
            } catch (_) {}
        }

        async function initWebMidi() {
            if (typeof navigator === 'undefined' || !navigator.requestMIDIAccess) {
                updateMidiStatus('unsupported');
                return;
            }
            if (window.__midiInitDone) return;
            try {
                midiAccessRef = await navigator.requestMIDIAccess({ sysex: false });
                window.__midiInitDone = true;
                wireMidiInputs(midiAccessRef);
                midiAccessRef.onstatechange = () => {
                    wireMidiInputs(midiAccessRef);
                    updateMidiStatus('ok');
                };
                updateMidiStatus('ok');
            } catch (e) {
                updateMidiStatus('denied');
                try { console.warn('Web MIDI unavailable:', e); } catch (_) {}
            }
        }

        // Effect toggles
        function toggleFx(key, btn) {
            if (!state || !state.fx || !state.fx[key]) return;
            state.fx[key].on = !state.fx[key].on;
            if (btn) btn.classList.toggle('on', state.fx[key].on);
            rebuildEffectsChain();
            try { if (typeof syncDjBeatFxKnobActiveDom === 'function') syncDjBeatFxKnobActiveDom(); } catch (_) {}
        }
        if (fxLow) fxLow.addEventListener('click', () => toggleFx('low', fxLow));
        if (fxHigh) fxHigh.addEventListener('click', () => toggleFx('high', fxHigh));
        if (fxBass) fxBass.addEventListener('click', () => toggleFx('bass', fxBass));
        if (fxTreble) fxTreble.addEventListener('click', () => toggleFx('treble', fxTreble));
        if (fxEcho) {
            fxEcho.addEventListener('click', () => {
                // echo uses two nodes; we key off delay node's .on flag
                state.fx.echoDelay.on = !state.fx.echoDelay.on;
                state.fx.echoFeedback.on = state.fx.echoDelay.on;
                fxEcho.classList.toggle('on', state.fx.echoDelay.on);
                rebuildEffectsChain();
            });
        }
        if (fxDist) fxDist.addEventListener('click', () => toggleFx('distort', fxDist));
        if (fxArp) {
            fxArp.addEventListener('click', () => {
                state.fx.arp.on = !state.fx.arp.on;
                fxArp.classList.toggle('on', state.fx.arp.on);
                // Enable/disable modulation depth (map to +/- 600 Hz)
                state.fx.arp.lfoGain.gain.value = state.fx.arp.on ? 600 : 0;
                rebuildEffectsChain();
                try { if (typeof syncDjBeatFxKnobActiveDom === 'function') syncDjBeatFxKnobActiveDom(); } catch (_) {}
            });
        }
        if (fxTk) {
            fxTk.addEventListener('click', () => {
                state.fx.tk.on = !state.fx.tk.on;
                fxTk.classList.toggle('on', state.fx.tk.on);
                rebuildEffectsChain();
                try { if (typeof syncDjBeatFxKnobActiveDom === 'function') syncDjBeatFxKnobActiveDom(); } catch (_) {}
            });
        }
        if (fxCut) fxCut.addEventListener('click', () => toggleRhythmCutFx());
        if (fxNoVocal) fxNoVocal.addEventListener('click', () => toggleDeckBMixerFxFromKnob('noVocal'));
        if (fxFlanger) fxFlanger.addEventListener('click', () => toggleFx('flanger', fxFlanger));
        if (fxReverb) fxReverb.addEventListener('click', () => toggleFx('reverb', fxReverb));
        function attachSamplePlayingIndicator(btn, media) {
            if (!btn || !media) return;
            const sync = () => {
                try {
                    const playing = !media.paused && !media.ended;
                    btn.classList.toggle('sample-playing', playing);
                } catch (_) {}
            };
            ['play', 'pause', 'ended'].forEach((ev) => {
                try { media.addEventListener(ev, sync); } catch (_) {}
            });
            sync();
        }

        function bindSample(btn, media, loopFlagKey, sourceKey) {
            if (!btn || !media) return;
            attachSamplePlayingIndicator(btn, media);
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                try { initAudio(); } catch (_) {}
                try { media.loop = !!window[loopFlagKey]; } catch(_) {}
                try { media.currentTime = 0; } catch(_) {}
                
                // Route through main mix (same bus as ProjectM / AUDIO:BAR master analyser)
                try {
                    if (state.audioCtx && state.mixInput) {
                        if (!state[sourceKey]) {
                            const mkSrc = (typeof globalThis.createMediaSourceFromElement === 'function')
                                ? globalThis.createMediaSourceFromElement
                                : (ctx, el) => ctx.createMediaElementSource(el);
                            state[sourceKey] = mkSrc(state.audioCtx, media);
                            if (state[sourceKey]) state[sourceKey].connect(state.mixInput);
                        }
                    }
                } catch(e) { console.warn(e); }
                media.play().catch(()=>{});
            });
            btn.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                const nv = !window[loopFlagKey];
                window[loopFlagKey] = nv;
                try { media.loop = !!nv; } catch(_) {}
                btn.classList.toggle('on', nv);
            });
        }
        // Play/Stop B
        // Play/Stop B
        // Play/Stop B
        function playRadioB() {
            initAudio();
            try { releaseDeckVideoFeed('b'); } catch (_) {}
            state.deckSourceMode.b = 'radio';
            try { state.deckLocalDisplayName.b = ''; } catch (_) {}

            let idx = (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                ? currentStationBIndex : 0;
            idx = Math.max(0, Math.min(stations.length - 1, idx));
            currentStationBIndex = idx;

            const sel = stations[idx];
            if (!sel || !sel.url) return;
            const urlClean = sanitizeUrlForAudio(String(sel.url));

            try { abortRadioBHandoff(); } catch (_) {}
            const audibleEl = (typeof getDeckBRadioAudibleEl === 'function') ? getDeckBRadioAudibleEl() : audioElB;
            const audibleSrc = audibleEl ? sanitizeUrlForAudio(String(audibleEl.currentSrc || audibleEl.src || '')) : '';
            const audiblePlaying = !!(audibleEl && audibleSrc && !audibleEl.paused && !audibleEl.ended);
            const warmCrossfade = !!(
                state.gainRadioBPrimaryPath &&
                state.gainRadioBSecondaryPath &&
                audioElRadioBAlt &&
                audiblePlaying &&
                urlClean &&
                audibleSrc &&
                urlClean !== audibleSrc
            );

            if (!warmCrossfade) {
                if (typeof resetRadioBDualStreamHandoff === 'function') resetRadioBDualStreamHandoff();
                revokeBlobSrc(audioElB);
            }

            try { setDjDeckRadioLoadingSpinner('b', true); } catch (_) {}

            const afterConnect = () => {
                try {
                    if (state.audioCtx && state.audioCtx.state === 'suspended') state.audioCtx.resume();
                } catch (_) {}
                try { connectDeckMediaToEq('b'); } catch (e) {
                    console.warn('Web Audio setup failed:', e);
                }
                radioBRetryAttempts = 0;
                try { setDjDeckRadioLoadingSpinner('b', false); } catch (_) {}
                updateMixBStatus();
                try { updateStationActiveHighlight(); } catch (_) {}
            };

            const onPlayFail = (e) => {
                console.warn('Stream B playback failed:', e);
                try { setDjDeckRadioLoadingSpinner('b', false); } catch (_) {}
                radioBRetryAttempts = (radioBRetryAttempts || 0) + 1;
                if (radioBRetryAttempts <= MAX_RADIO_RETRIES && stations.length > 0) {
                    try { statusEl.innerText = 'Deck B stream failed. Trying another station...'; } catch (_) {}
                    try { if (typeof pickRandomStationB === 'function') pickRandomStationB(); } catch (_) {}
                } else {
                    try { statusEl.innerText = 'Deck B: no playable station found.'; } catch (_) {}
                }
                updateMixBStatus();
            };

            if (!warmCrossfade) {
                try { audioElB.pause(); } catch (_) {}
                audioElB.crossOrigin = 'anonymous';
                audioElB.src = urlClean;
                state.audioElB = audioElB;
                audioElB.play().then(afterConnect).catch(onPlayFail);
                return;
            }

            try { ensureRadioBAltWired(); } catch (_) {}
            if (!state.radioAltBMediaWired || !state.radioElementSourceBAlt) {
                if (typeof resetRadioBDualStreamHandoff === 'function') resetRadioBDualStreamHandoff();
                revokeBlobSrc(audioElB);
                audioElB.crossOrigin = 'anonymous';
                audioElB.src = urlClean;
                audioElB.play().then(afterConnect).catch(onPlayFail);
                return;
            }

            const outputFromSecondary = (typeof isDeckBRadioOutputFromAlt === 'function')
                ? isDeckBRadioOutputFromAlt()
                : false;
            const liveEl = outputFromSecondary ? audioElRadioBAlt : audioElB;
            const prepEl = outputFromSecondary ? audioElB : audioElRadioBAlt;
            const liveGain = outputFromSecondary ? state.gainRadioBSecondaryPath : state.gainRadioBPrimaryPath;
            const prepGain = outputFromSecondary ? state.gainRadioBPrimaryPath : state.gainRadioBSecondaryPath;

            radioBHandoffAbortCtrl = new AbortController();
            const signal = radioBHandoffAbortCtrl.signal;

            beginDeckAPingPongMediaHandoff(
                prepEl, liveEl, liveGain, prepGain,
                urlClean, signal, 14000, afterConnect, onPlayFail
            );
        }
        function stopRadioB() {
            try { abortRadioBHandoff(); } catch (_) {}
            try { if (typeof resetRadioBDualStreamHandoff === 'function') resetRadioBDualStreamHandoff(); } catch (_) {}
            try { state.deckSourceMode.b = 'radio'; } catch (_) {}
            try { state.deckLocalDisplayName.b = ''; } catch (_) {}
            try { setDjDeckRadioLoadingSpinner('b', false); } catch (_) {}
            if (audioElB) {
                try { audioElB.pause(); } catch (_) {}
                try { audioElB.removeAttribute('src'); } catch (_) {}
                try { audioElB.load(); } catch (_) {}
            }
            updateMixBStatus();
            try { syncTopMenuStationsLayout(); } catch (_) {}
        }
        try {
            ['play','pause','ended','stalled','error','suspend','abort'].forEach(ev => {
                audioElB.addEventListener(ev, updateMixBStatus);
            });
        } catch(_) {}
        try {
            audioEl.addEventListener('ended', onDeckAEndedForQueue);
            audioElB.addEventListener('ended', onDeckBEndedForQueue);
            if (audioElRadioAAlt) audioElRadioAAlt.addEventListener('ended', onDeckAEndedForQueue);
            if (audioElRadioBAlt) audioElRadioBAlt.addEventListener('ended', onDeckBEndedForQueue);
        } catch (_) {}
        webmPrevBtn.addEventListener('click', () => { prevWebm(); resetIdleTimer(); });
        webmNextBtn.addEventListener('click', () => { nextWebm(); resetIdleTimer(); });
        // Remove legacy binding that opened the top menu; Mix Settings supersedes it.
		// Horizontal scroll wheel changes visual left/right (ignore when interacting with panels)
		document.addEventListener('wheel', (e) => {
			try {
				const horizIntent = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? Math.abs(e.deltaX) : (e.shiftKey ? Math.abs(e.deltaY) : 0));
				if (horizIntent > 5) {
					const inUi = !!(e.target && (e.target.closest('#top-menu-content') || e.target.closest('#settings-panel') || e.target.closest('#webm-settings-panel') || e.target.closest('#bottom-avatar-content') || e.target.closest('#keyboard-shortcuts-panel') || e.target.closest('#dj-visual-root') || e.target.closest('#radio-visual-root')));
					if (!inUi) {
						// One step per gesture: lock during momentum and unlock after wheel quiets down
						if (!window.__wheelNavLocked) {
							window.__wheelNavLocked = true;
							if ((e.deltaX > 0) || (e.shiftKey && e.deltaY > 0)) loadMode(state.currentModeIdx + 1);
							else loadMode(state.currentModeIdx - 1);
							if (typeof resetIdleTimer === 'function') resetIdleTimer();
						}
						// Reset unlock timer to release lock shortly after wheel stops
						if (window.__wheelNavUnlockId) clearTimeout(window.__wheelNavUnlockId);
						window.__wheelNavUnlockId = setTimeout(() => { window.__wheelNavLocked = false; }, 220);
						e.preventDefault();
					}
				}
			} catch(_) {}
		}, { passive: false });
        settingsCloseBtn.addEventListener('click', () => { 
            if (settingsPanelTimer) { clearTimeout(settingsPanelTimer); settingsPanelTimer = null; }
            settingsPanel.classList.add('display-none');
            settingsPanel.style.opacity = '';
            settingsPanel.style.pointerEvents = '';
        });
        function scheduleSettingsPanelClose() {
            if (settingsPanelTimer) clearTimeout(settingsPanelTimer);
            settingsPanelTimer = setTimeout(() => {
                settingsPanel.style.opacity = '0';
                settingsPanel.style.pointerEvents = 'none';
                setTimeout(() => { 
                    settingsPanel.classList.add('display-none'); 
                    settingsPanel.style.opacity = '';
                    settingsPanel.style.pointerEvents = '';
                }, 1200);
            }, 30000);
        }
        settingsApplyBtn.addEventListener('click', () => {
            // Read and clamp settings
            const minS = Math.max(3, Math.min(120, Number(inpShuffleMin.value) || 12));
            const maxS = Math.max(5, Math.min(180, Number(inpShuffleMax.value) || 25));
            const trans = Math.max(0, Math.min(10, Number(inpTransition.value) || 2.7));
            const px = Math.max(0.5, Math.min(3, Number(inpPixelRatio.value) || 1));
            visualSettings.shuffleMinSec = minS;
            visualSettings.shuffleMaxSec = maxS;
            visualSettings.transitionSec = trans;
            visualSettings.pixelRatio = px;
            try {
                if (typeof setKnobUi === 'function') {
                    setKnobUi(document.getElementById('knob-tm-shuffle-min'), 3, 120, minS);
                    setKnobUi(document.getElementById('knob-tm-shuffle-max'), 5, 180, maxS);
                    setKnobUi(document.getElementById('knob-tm-transition'), 0, 10, trans);
                    setKnobUi(document.getElementById('knob-tm-pixelratio'), 0.5, 3, px);
                }
            } catch (_) {}
            // Apply if active is v2
            if(state.activeVisualizer && state.activeVisualizer instanceof MilkdropEngineV2) {
                state.activeVisualizer.applySettings?.();
            }
        });
        
        // Skip Preset Button Logic (main visual ProjectM / Milkdrop, or Deck B ProjectM)
        document.getElementById('btn-skip-preset').addEventListener('click', () => {
            const av = state.activeVisualizer;
            if (av && av.name === 'DJ Decks' && av.deckBVizMode === 'projectm' && typeof av.nextDeckBProjectMPreset === 'function') {
                try { av.nextDeckBProjectMPreset(); } catch (_) {}
                return;
            }
            if (av && typeof av.nextPreset === 'function') {
                try { av.nextPreset(); } catch (_) {}
            }
        });

        // --- RADIO: Load stations and UI ---
        async function loadStations() {
            try {
                const resp = await fetch('radio.txt');
                const txt = await resp.text();
                stations.length = 0;
                txt.split('\n').forEach((line) => {
                    const raw = line.trim();
                    if(!raw) return;
                    const parts = raw.split('|');
                    if(parts.length >= 2) {
                        const name = parts[0].trim();
                        const url = parts.slice(1).join('|').trim();
                        stations.push({ name, url });
                    }
                });
                loadUserRadioStations();
                userRadioStations.forEach((us) => {
                    if (!us || !us.url) return;
                    const u = String(us.url).trim();
                    if (!u || stations.some((s) => s && s.url === u)) return;
                    stations.push({ name: us.name || deriveNameFromUrl(u), url: u });
                });
                // Restore last played stations when saved, else default Deck A to the first entry.
                applySavedStationSelections();
                syncStationCycleSelection();
                renderStationList();
                try { refreshMixStationB(); } catch(_) {}
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            } catch(e) {
                console.warn('Failed to load radio.txt', e);
            }
        }

        function deckBHasLoadedContent() {
            try {
                if (state.deckSourceMode && state.deckSourceMode.b === 'local') {
                    const q = state.deckLocalQueue && state.deckLocalQueue.b;
                    if (Array.isArray(q) && q.length > 0) return true;
                    const dn = state.deckLocalDisplayName && state.deckLocalDisplayName.b;
                    if (dn && String(dn).trim()) return true;
                }
                const activeB = state.audioElB || audioElB;
                if (typeof deckHasSource === 'function' && deckHasSource(activeB)) return true;
            } catch (_) {}
            return false;
        }

        function syncTopMenuStationsLayout() {
            const wrap = topMenuStationsWrap || document.getElementById('top-menu-stations-wrap');
            if (!wrap) return;
            const loaded = deckBHasLoadedContent();
            if (loaded) topMenuStationsBSplitManual = false;
            const split = loaded || topMenuStationsBSplitManual;
            wrap.classList.toggle('is-split', split);
            wrap.classList.toggle('is-focus-a', !split);
            wrap.classList.toggle('is-deck-b-loaded', loaded);
            const expandBtn = document.getElementById('topmenu-stations-b-expand');
            if (expandBtn) {
                expandBtn.setAttribute('aria-pressed', split ? 'true' : 'false');
                expandBtn.title = loaded
                    ? 'Deck B expanded (active deck)'
                    : (split ? 'Collapse Deck B column' : 'Expand Deck B column');
            }
        }

        function renderStationList() {
            if (!radioListEl) return;
            const listB = radioListElB || document.getElementById('radio-list-b');
            radioListEl.innerHTML = '';
            if (listB) listB.innerHTML = '';
            stations.forEach((s, i) => {
                const mkItem = (deckKey) => {
                    const item = document.createElement('div');
                    item.className = 'radio-item';
                    item.dataset.index = String(i);
                    item.dataset.deck = deckKey;
                    const activeIdx = deckKey === 'b' ? currentStationBIndex : currentStationIndex;
                    if (i === activeIdx) item.classList.add('active');
                    const nameEl = document.createElement('div');
                    nameEl.textContent = s.name;
                    const goEl = document.createElement('div');
                    goEl.textContent = '➤';
                    item.appendChild(nameEl);
                    item.appendChild(goEl);
                    item.addEventListener('click', () => {
                        if (uiLocked) return;
                        if (deckKey === 'b') setStationB(i);
                        else setStation(i);
                    });
                    return item;
                };
                radioListEl.appendChild(mkItem('a'));
                if (listB) listB.appendChild(mkItem('b'));
            });
            try { syncTopMenuStationsLayout(); } catch (_) {}
            try {
                if (typeof window.__refreshDigitalStationsUi === 'function') window.__refreshDigitalStationsUi();
            } catch (_) {}
        }

        function setStation(index, opts) {
            const force = opts && opts.force === true;
            if (uiLocked && !force) return;
            if(index < 0 || index >= stations.length) return;
            if (!suppressHistoryPush && currentStationIndex !== -1 && currentStationIndex !== index) {
                pushDeckPlaybackHistory('a');
            }
            currentStationIndex = index;
            const s = stations[index];
            if (radioInputEl) radioInputEl.value = s.url;
            showStationBanner(s.name);
            updateStationActiveHighlight();
            try { saveLastStationSelection('a'); } catch (_) {}
            playRadio();
        }

        function setStationB(index, opts) {
            const force = opts && opts.force === true;
            if (uiLocked && !force) return;
            if (index < 0 || index >= stations.length) return;
            if (!suppressHistoryPush && currentStationBIndex !== index) {
                pushDeckPlaybackHistory('b');
            }
            currentStationBIndex = index;
            updateStationActiveHighlight();
            try { saveLastStationSelection('b'); } catch (_) {}
            try { syncTopMenuStationsLayout(); } catch (_) {}
            try { playRadioB(); } catch (_) {}
            try { updateModeSubStationLine(); } catch (_) {}
            try { resetIdleTimer(); } catch (_) {}
        }

        function updateStationActiveHighlight() {
            const apply = (root, activeIdx) => {
                if (!root) return;
                Array.from(root.children).forEach((el) => {
                    const idx = Number(el.dataset.index || -1);
                    el.classList.toggle('active', idx === activeIdx);
                });
            };
            apply(radioListEl, currentStationIndex);
            apply(radioListElB || document.getElementById('radio-list-b'), currentStationBIndex);
            try {
                if (typeof window.__syncDigitalStationsActiveHighlight === 'function') {
                    window.__syncDigitalStationsActiveHighlight();
                }
            } catch (_) {}
        }

        function pickRandomStation() {
            if(stations.length === 0) return;
            const eligible = getCycleEligibleStationIndexes(currentStationIndex);
            if (!eligible.length) return;
            const idx = eligible[Math.floor(Math.random() * eligible.length)];
            setStation(idx);
        }
        /**
         * Tunes Deck B to a fresh random "eligible" station (one that's distinct from
         * its current pick, mirroring the rules pickRandomStation() uses for Deck A).
         * Used by the global N shortcut and any other "Rand →" trigger on Deck B.
         */
        function pickRandomStationB() {
            if (!Array.isArray(stations) || stations.length === 0) return;
            if (!suppressHistoryPush) pushDeckPlaybackHistory('b');
            const cur = (typeof currentStationBIndex === 'number' && Number.isFinite(currentStationBIndex)) ? currentStationBIndex : 0;
            const eligible = (typeof getCycleEligibleStationIndexes === 'function') ? getCycleEligibleStationIndexes(cur) : [];
            if (!eligible.length) return;
            const idx = eligible[Math.floor(Math.random() * eligible.length)];
            currentStationBIndex = Math.max(0, Math.min(stations.length - 1, Number(idx) || 0));
            try { saveLastStationSelection('b'); } catch (_) {}
            try { if (typeof refreshMixStationB === 'function') refreshMixStationB(); } catch (_) {}
            try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
        }
        /**
         * Reads the crossfader (DJ Decks panel first, then the Mixer panel fallback)
         * and tunes whichever deck is currently winning the mix to a new random
         * station. Used by the global N shortcut so "next station" always lands on
         * the audible deck.
         */
        function pickRandomStationForCrossfadedDeck() {
            try {
                const dc = document.getElementById('dj-crossfader');
                const mc = document.getElementById('mix-crossfader');
                const raw = (dc && dc.value) || (mc && mc.value) || 0;
                const x = Math.max(0, Math.min(1, Number(raw) || 0));
                // x < 0.5 = Deck A is louder (or fully on the A side), >= 0.5 = Deck B.
                if (x < 0.5) pickRandomStation();
                else pickRandomStationB();
            } catch (_) {
                try { pickRandomStation(); } catch (__) {}
            }
        }

        function showRadioPanel() { 
            if (uiLocked) return;
            // Legacy panel removed: open top menu instead
            try { openTopMenuPanel(); } catch(e) {}
            if (!radioPanel) return;
            radioPanel.classList.remove('display-none'); 
            radioPanel.style.opacity = '1';
            radioPanel.style.pointerEvents = 'auto';
            scheduleRadioPanelClose();
        }
        function goPreviousStation() {
            goPreviousDeckPlayback('a');
        }
        function hideRadioPanel() { 
            if (radioPanelTimer) { clearTimeout(radioPanelTimer); radioPanelTimer = null; }
            try { closeTopMenuPanel(); } catch(e) {}
            if (!radioPanel) return;
            radioPanel.classList.add('display-none'); 
            radioPanel.style.opacity = '';
            radioPanel.style.pointerEvents = '';
        }
        function toggleRadioPanel() {
            if (uiLocked) return;
            // Toggle top menu now
            try { toggleTopMenuPanel(); } catch(e) {}
            if (!radioPanel) return;
            if(radioPanel.classList.contains('display-none')) showRadioPanel();
            else hideRadioPanel();
        }
        function scheduleRadioPanelClose() {
            if (radioPanelTimer) clearTimeout(radioPanelTimer);
            radioPanelTimer = setTimeout(() => {
                if (radioPanel) {
                    radioPanel.style.opacity = '0';
                    radioPanel.style.pointerEvents = 'none';
                }
                setTimeout(() => { hideRadioPanel(); }, 1200);
            }, 30000);
        }

        // --- TEXT-IN PANEL & OVERLAY ---
        /** TEXT-IN routing: #textin-panel[data-text-target] (deck-b vs global). Persists while Deck B display stays on, even if the panel is hidden. */
        function getTextInPanelTarget() {
            try {
                if (!textInPanel) return 'global';
                return textInPanel.dataset.textTarget === 'deck-b' ? 'deck-b' : 'global';
            } catch (_) {
                return 'global';
            }
        }
        function getTextInSpawnTarget() {
            try {
                if (!textInPanel) return 'global';
                if (textInPanel.dataset.textTarget === 'deck-b') {
                    const stage = getDeckBStageEl();
                    if (stage && stage.classList.contains('dj-deck-b-text-mode')) return 'deck-b';
                }
                return 'global';
            } catch (_) {
                return 'global';
            }
        }
        function getDeckBTextStageEl() { return document.getElementById('dj-deck-b-text-stage'); }
        function getDeckBStageEl() {
            const t = getDeckBTextStageEl();
            return t ? t.closest('.dj-deck-b-stage') : null;
        }
        function setDeckBTextMode(on) {
            const stage = getDeckBStageEl();
            const ts = getDeckBTextStageEl();
            const wasOn = !!(stage && stage.classList.contains('dj-deck-b-text-mode'));
            if (stage) stage.classList.toggle('dj-deck-b-text-mode', !!on);
            if (ts) ts.setAttribute('aria-hidden', on ? 'false' : 'true');
            if (!on) {
                if (wasOn) {
                    try { if (typeof setTextAuto === 'function') setTextAuto(false); } catch(_) {}
                }
                const layer = document.getElementById('dj-deck-b-text-overlay-layer');
                if (layer) {
                    while (layer.firstChild) layer.removeChild(layer.firstChild);
                }
                if (Array.isArray(activeTextOverlays)) {
                    for (let i = activeTextOverlays.length - 1; i >= 0; i--) {
                        if (activeTextOverlays[i] && activeTextOverlays[i].target === 'deck-b') {
                            try { activeTextOverlays[i].el.remove(); } catch(_) {}
                            activeTextOverlays.splice(i, 1);
                        }
                    }
                }
            }
            try { syncDjTextInDeckLights(); } catch(_) {}
            try { if (typeof updateDjDecksShortcutVisibility === 'function') updateDjDecksShortcutVisibility(); } catch(_) {}
        }
        function syncDjTextInDeckLights() {
            try {
                const panelOpen = !!(textInPanel && !textInPanel.classList.contains('display-none') && textInPanel.classList.contains('open'));
                const tgt = getTextInPanelTarget();
                const stage = getDeckBStageEl();
                const deckBDisplayOn = !!(stage && stage.classList.contains('dj-deck-b-text-mode'));
                const a = document.getElementById('dj-fx-tk');
                const b = document.getElementById('dj-b-fx-tk');
                if (a) a.classList.toggle('on', (panelOpen && tgt === 'global') || deckBDisplayOn);
                if (b) b.classList.toggle('on', deckBDisplayOn && tgt === 'deck-b');
            } catch(_) {}
        }
        function openTextInForTarget(target) {
            if (uiLocked) return;
            const t = (target === 'deck-b') ? 'deck-b' : 'global';
            const open = !!(textInPanel && !textInPanel.classList.contains('display-none') && textInPanel.classList.contains('open'));
            if (open && getTextInPanelTarget() === t) {
                hideTextInPanel();
                return;
            }
            try {
                if (textInPanel) textInPanel.dataset.textTarget = t;
            } catch (_) {}
            setDeckBTextMode(t === 'deck-b');
            if (!open) showTextInPanel();
            syncDjTextInDeckLights();
        }
        function showTextInPanel() {
            if (uiLocked) return;
            if (!textInPanel) return;
            textInPanel.classList.remove('display-none');
            void textInPanel.offsetWidth;
            textInPanel.classList.add('open');
            try { syncDjTextInDeckLights(); } catch(_) {}
        }
        function hideTextInPanel(opts) {
            if (!textInPanel) return;
            const forceCloseDeckB = !!(opts && opts.forceCloseDeckB);
            textInPanel.classList.remove('open');
            setTimeout(() => { textInPanel.classList.add('display-none'); }, 350);
            const keepDeckB = !forceCloseDeckB && textInPanel.dataset.textTarget === 'deck-b';
            if (!keepDeckB) {
                try { setDeckBTextMode(false); } catch(_) {}
                try { delete textInPanel.dataset.textTarget; } catch (_) {}
            }
            try { syncDjTextInDeckLights(); } catch(_) {}
        }
        function toggleTextInPanel() {
            if (uiLocked) return;
            if (!textInPanel) return;
            if (textInPanel.classList.contains('display-none') || !textInPanel.classList.contains('open')) {
                try {
                    textInPanel.dataset.textTarget = 'global';
                    setDeckBTextMode(false);
                } catch (_) {}
                showTextInPanel();
            } else hideTextInPanel();
        }
        function applyTextStyles(el, cfg) {
            if (!el || !cfg) return;
            el.style.fontFamily = cfg.font;
            el.style.color = cfg.color;
            el.style.fontSize = cfg.size + 'px';
            el.style.webkitTextStroke = (cfg.border > 0 ? (cfg.border + 'px ' + cfg.borderColor) : '0px transparent');
            const glowPx = Math.max(0, cfg.glow|0);
            
            // --- FIX START: Use correct config flag instead of wrong DOM element ---
            const useRandomGlowColor = (typeof cfg.glowColorRandom === 'boolean') 
                                        ? cfg.glowColorRandom 
                                        : (tiGlowColorRandCheck ? tiGlowColorRandCheck.checked : false);

            const glowColor = useRandomGlowColor
                ? randomGlowColor()
                : (
                    cfg.glowColor && /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(cfg.glowColor)
                        ? cfg.glowColor
                        : cfg.color
                  );
            // --- FIX END ---

            const glow = glowPx > 0 ? `0 0 ${glowPx}px ${glowColor}` : '';
            const strokeShadow = (cfg.border > 0) ? `0 0 ${Math.max(1,cfg.border)}px ${cfg.borderColor}` : '';
            el.style.textShadow = [glow, strokeShadow].filter(Boolean).join(', ');
            el.style.filter = '';
        }
        function getTextConfigFromUi() {
            return {
                text: (tiText?.value || '').trim() || ' ',
                font: tiFont?.value || "'Segoe UI', sans-serif",
                color: tiColor?.value || '#ffffff',
                colorRandom: !!(tiColorRandCheck?.checked),
                size: Math.max(16, Math.min(200, Number(tiSize?.value)||64)),
                sizeMin: Math.max(0, Number(tiSizeRMin?.value) || Number(tiSize?.min) || 16),
                sizeMax: Math.max(0, Number(tiSizeRMax?.value) || Number(tiSize?.max) || 200),
                sizeRandom: !!(tiSizeRand?.checked),
                xPercent: Math.max(0, Math.min(100, Number(tiX?.value)||50)),
                xMin: Math.max(0, Number(tiXRMin?.value) || Number(tiX?.min) || 0),
                xMax: Math.max(0, Number(tiXRMax?.value) || Number(tiX?.max) || 100),
                xRandom: !!(tiXRand?.checked),
                border: Math.max(0, Math.min(10, Number(tiBorder?.value)||0)),
                borderMin: Math.max(0, Number(tiBorderRMin?.value) || Number(tiBorder?.min) || 0),
                borderMax: Math.max(0, Number(tiBorderRMax?.value) || Number(tiBorder?.max) || 10),
                borderRandom: !!(tiBorderRand?.checked),
                borderColor: tiBorderColor?.value || '#000000',
                borderColorRandom: !!(tiBorderColorRandCheck?.checked),
                glow: Math.max(0, Math.min(50, Number(tiGlow?.value)||0)),
                glowMin: Math.max(0, Number(tiGlowRMin?.value) || Number(tiGlow?.min) || 0),
                glowMax: Math.max(0, Number(tiGlowRMax?.value) || Number(tiGlow?.max) || 50),
                glowRandom: !!(tiGlowRand?.checked),
                glowColor: tiGlowColor?.value || '',
                // --- FIX: Added this line to capture the checkbox state ---
                glowColorRandom: !!(tiGlowColorRandCheck?.checked),
                // ---------------------------------------------------------
                fontRandom: !!(tiFontRand?.checked),
                flash: !!(tiFlash?.checked),
                flashSpeed: Math.max(0, Math.min(5, Number(tiFlashSpeed?.value)||1)),
                flashSpeedMin: Math.max(0, Number(tiFlashSpeedRMin?.value) || Number(tiFlashSpeed?.min) || 0),
                flashSpeedMax: Math.max(0, Number(tiFlashSpeedRMax?.value) || Number(tiFlashSpeed?.max) || 5),
                flashSpeedRandom: !!(tiFlashSpeedRand?.checked),
                speed: Math.max(25, Math.min(900, Number(tiSpeed?.value)||25)),
                speedMin: Math.max(0, Number(tiSpeedRMin?.value) || Number(tiSpeed?.min) || 25),
                speedMax: Math.max(0, Number(tiSpeedRMax?.value) || Number(tiSpeed?.max) || 900),
                speedRandom: !!(tiSpeedRand?.checked)
            };
        }
        function updateTextInPreview() {
            const cfg = getTextConfigFromUi();
            if (tiPreview) {
                tiPreview.textContent = cfg.text;
                applyTextStyles(tiPreview, cfg);
                try { tiPreview.style.setProperty('font-family', cfg.font, 'important'); } catch(_) {}
                // Reflect selected font in the input so the user types in the chosen style
                try { if (tiText) tiText.style.setProperty('font-family', cfg.font, 'important'); } catch(_) {}
                // Disable inputs when random toggles are on
                if (tiSize && tiSizeRand) tiSize.disabled = !!tiSizeRand.checked;
                if (tiX && tiXRand) tiX.disabled = !!tiXRand.checked;
                if (tiBorder && tiBorderRand) tiBorder.disabled = !!tiBorderRand.checked;
                if (tiBorderColor && tiBorderColorRandCheck) tiBorderColor.disabled = !!tiBorderColorRandCheck.checked;
                if (tiGlow && tiGlowRand) tiGlow.disabled = !!tiGlowRand.checked;
                if (tiFont && tiFontRand) tiFont.disabled = !!tiFontRand.checked;
                if (tiColor && tiColorRandCheck) tiColor.disabled = !!tiColorRandCheck.checked;
                const tiSpeed = document.getElementById('ti-speed');
                if (tiSpeed && tiSpeedRand) tiSpeed.disabled = !!tiSpeedRand.checked;
                if (tiFlashSpeed && tiFlashSpeedRand) tiFlashSpeed.disabled = !!tiFlashSpeedRand.checked;
                // Toggle sub-range visibility based on Random checkboxes
                const toggleSub = (rmin, rmax, on) => {
                    if (!rmin || !rmax) return;
                    const container = rmin.closest('.range-sub');
                    if (container) container.style.display = on ? 'block' : 'none';
                };
                toggleSub(tiSizeRMin, tiSizeRMax, !!(tiSizeRand && tiSizeRand.checked));
                toggleSub(tiXRMin, tiXRMax, !!(tiXRand && tiXRand.checked));
                toggleSub(tiBorderRMin, tiBorderRMax, !!(tiBorderRand && tiBorderRand.checked));
                toggleSub(tiGlowRMin, tiGlowRMax, !!(tiGlowRand && tiGlowRand.checked));
                toggleSub(tiFlashSpeedRMin, tiFlashSpeedRMax, !!(tiFlashSpeedRand && tiFlashSpeedRand.checked));
                toggleSub(tiSpeedRMin, tiSpeedRMax, !!(tiSpeedRand && tiSpeedRand.checked));
            }
        }
        // lock/unlock shortcuts when focusing inside Text-In
        if (textInPanel) {
            textInPanel.addEventListener('focusin', () => { shortcutsLocked = true; }, true);
            textInPanel.addEventListener('focusout', (e) => {
                try {
                    const next = e.relatedTarget;
                    if (!(textInPanel.contains(next))) shortcutsLocked = false;
                } catch(_) { shortcutsLocked = false; }
            }, true);
        }
        // stop bubbling of keydown events from inputs inside panel
        [tiText, tiFont, tiColor, tiSize, tiBorder, tiBorderColor, tiGlow, tiFlashSpeed, tiSpeed].forEach(el => {
            if (el) el.addEventListener('keydown', (ev) => ev.stopPropagation());
        });
        if (tiFont) tiFont.addEventListener('change', updateTextInPreview);
        // Auto-disable random when user adjusts specific controls
        if (tiSize) tiSize.addEventListener('input', () => { if (tiSizeRand && tiSizeRand.checked) { tiSizeRand.checked = false; updateTextInPreview(); } });
        if (tiX) tiX.addEventListener('input', () => { if (tiXRand && tiXRand.checked) { tiXRand.checked = false; updateTextInPreview(); } });
        if (tiBorder) tiBorder.addEventListener('input', () => { if (tiBorderRand && tiBorderRand.checked) { tiBorderRand.checked = false; updateTextInPreview(); } });
        if (tiGlow) tiGlow.addEventListener('input', () => { if (tiGlowRand && tiGlowRand.checked) { tiGlowRand.checked = false; updateTextInPreview(); } });
        if (tiGlowColorRandBtn) tiGlowColorRandBtn.addEventListener('click', (e) => { e.preventDefault(); if (tiGlowColor) { tiGlowColor.value = randomHexColor(); updateTextInPreview(); } });
        if (tiGlowColor) tiGlowColor.addEventListener('input', updateTextInPreview);
        if (tiBorderColor) tiBorderColor.addEventListener('input', () => { if (tiBorderColorRandCheck && tiBorderColorRandCheck.checked) { tiBorderColorRandCheck.checked = false; updateTextInPreview(); } });
        if (tiFlashSpeed) tiFlashSpeed.addEventListener('input', () => { if (tiFlashSpeedRand && tiFlashSpeedRand.checked) { tiFlashSpeedRand.checked = false; updateTextInPreview(); } });
        if (tiFont) tiFont.addEventListener('input', () => { if (tiFontRand && tiFontRand.checked) { tiFontRand.checked = false; updateTextInPreview(); } });
        const tiSpeedEl = document.getElementById('ti-speed');
        if (tiSpeedEl) tiSpeedEl.addEventListener('input', () => { if (tiSpeedRand && tiSpeedRand.checked) { tiSpeedRand.checked = false; updateTextInPreview(); } });
        // Dual-range helpers for random bounds
        function bindDualRange(minEl, maxEl, overallMin, overallMax, step) {
            if (!minEl || !maxEl) return;
            const clamp = () => {
                let a = Number(minEl.value), b = Number(maxEl.value);
                if (isNaN(a)) a = overallMin; if (isNaN(b)) b = overallMax;
                a = Math.max(overallMin, Math.min(overallMax, a));
                b = Math.max(overallMin, Math.min(overallMax, b));
                if (a > b) { const t = a; a = b; b = t; }
                minEl.value = String(a);
                maxEl.value = String(b);
                updateTextInPreview();
            };
            minEl.addEventListener('input', clamp);
            maxEl.addEventListener('input', clamp);
            clamp();
        }
        bindDualRange(tiSizeRMin, tiSizeRMax, 16, 200, 1);
        bindDualRange(tiXRMin, tiXRMax, 0, 100, 1);
        bindDualRange(tiBorderRMin, tiBorderRMax, 0, 10, 1);
        bindDualRange(tiGlowRMin, tiGlowRMax, 0, 50, 1);
        bindDualRange(tiFlashSpeedRMin, tiFlashSpeedRMax, 0, 5, 0.1);
        bindDualRange(tiSpeedRMin, tiSpeedRMax, 25, 900, 1);
        if (tiColor) tiColor.addEventListener('input', () => { if (tiColorRandCheck && tiColorRandCheck.checked) { tiColorRandCheck.checked = false; updateTextInPreview(); } });
        // Enter to send while in text field
        if (tiText) {
            tiText.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const cfg = getTextConfigFromUi();
                    spawnRisingText(cfg);
                }
            });
        }
        // Randomize colors via small buttons
        if (tiColorRandBtn) {
            tiColorRandBtn.addEventListener('click', (e) => {
                e.preventDefault();
                try { tiColor.value = randomHexColor(); } catch(_) {}
                updateTextInPreview();
            });
        }
        if (tiBorderColorRandBtn) {
            tiBorderColorRandBtn.addEventListener('click', (e) => {
                e.preventDefault();
                try { tiBorderColor.value = randomHexColor(); } catch(_) {}
                updateTextInPreview();
            });
        }
        function spawnRisingText(cfg) {
            const target = getTextInSpawnTarget();
            const deckBLayer = document.getElementById('dj-deck-b-text-overlay-layer');
            const layer = (target === 'deck-b' && deckBLayer) ? deckBLayer : textOverlayLayer;
            if (!layer) return;
            // Apply randomizations
            try {
                if (cfg.colorRandom) {
                    cfg.color = randomHexColor();
                }
                if (cfg.fontRandom && tiFont && tiFont.options && tiFont.options.length > 0) {
                    const idx = Math.floor(Math.random() * tiFont.options.length);
                    cfg.font = tiFont.options[idx].value || cfg.font;
                }
                if (cfg.borderRandom && tiBorder) {
                    const min = Math.max(0, Number(cfg.borderMin) || Number(tiBorder.min) || 0);
                    const max = Math.max(min, Number(cfg.borderMax) || Number(tiBorder.max) || 10);
                    cfg.border = Math.floor(min + Math.random() * (max - min + 1));
                }
                if (cfg.borderColorRandom) {
                    cfg.borderColor = randomHexColor();
                }
                if (cfg.glowRandom && tiGlow) {
                    const minG = Math.max(0, Number(cfg.glowMin) || Number(tiGlow.min) || 0);
                    const maxG = Math.max(minG, Number(cfg.glowMax) || Number(tiGlow.max) || 50);
                    cfg.glow = Math.floor(minG + Math.random() * (maxG - minG + 1));
                }
                // Flash on/off random removed per new request; only speed random here
                if (cfg.speedRandom) {
                    const minS = Math.max(0, Number(cfg.speedMin) || 25);
                    const maxS = Math.max(minS, Number(cfg.speedMax) || 900);
                    cfg.speed = Math.floor(minS + Math.random() * (maxS - minS + 1));
                }
                if (cfg.flashSpeedRandom && tiFlashSpeed) {
                    const minFS = Math.max(0, Number(cfg.flashSpeedMin) || Number(tiFlashSpeed.min) || 0);
                    const maxFS = Math.max(minFS, Number(cfg.flashSpeedMax) || Number(tiFlashSpeed.max) || 5);
                    cfg.flashSpeed = (minFS + Math.random() * (maxFS - minFS));
                }
            } catch(_) {}
            const el = document.createElement('div');
            el.className = 'rising-text';
            el.textContent = cfg.text;
            applyTextStyles(el, cfg);
            try { el.style.setProperty('font-family', cfg.font, 'important'); } catch(_) {}
            // compute horizontal origin
            let xPct = Number(cfg.xPercent || 50);
            if (cfg.xRandom) {
                xPct = Math.random() * 100;
            }
            // compute size
            if (cfg.sizeRandom) {
                const min = Math.max(16, Number(tiSize?.min) || 16);
                const max = Math.max(min, Number(tiSize?.max) || 200);
                cfg.size = Math.floor(min + Math.random() * (max - min + 1));
                try { el.style.fontSize = cfg.size + 'px'; } catch(_) {}
            }
            const containerHeight = (target === 'deck-b')
                ? Math.max(1, layer.clientHeight || (layer.getBoundingClientRect && layer.getBoundingClientRect().height) || window.innerHeight)
                : window.innerHeight;
            const startY = containerHeight + 20;
            el.style.top = startY + 'px';
            el.style.left = xPct + (target === 'deck-b' ? '%' : 'vw');
            layer.appendChild(el);
            const item = { el, cfg, y: startY, start: performance.now(), last: performance.now(), target, height: containerHeight };
            activeTextOverlays.push(item);
            if (!textOverlayAnimId) requestTextOverlayFrame();
        }
        function requestTextOverlayFrame() {
            textOverlayAnimId = requestAnimationFrame(stepTextOverlays);
        }
        function stepTextOverlays(ts) {
            for (let i = activeTextOverlays.length - 1; i >= 0; i--) {
                const it = activeTextOverlays[i];
                const dt = Math.max(0, (ts - it.last) / 1000);
                it.last = ts;
                it.y -= it.cfg.speed * dt;
                it.el.style.transform = `translate(-50%, 0)`;
                it.el.style.top = it.y + 'px';
                const refH = (it.target === 'deck-b') ? Math.max(1, it.height || window.innerHeight) : window.innerHeight;
                const progress = Math.max(0, Math.min(1, ( (refH + 20) - it.y ) / refH ));
                let opacity = 1 - progress;
                if (it.cfg.flash && it.cfg.flashSpeed > 0) {
                    const phase = (ts - it.start) / 1000 * it.cfg.flashSpeed * Math.PI * 2;
                    const mod = 0.5 + 0.5 * Math.sin(phase);
                    opacity *= (0.6 + 0.4 * mod);
                }
                it.el.style.opacity = String(Math.max(0, Math.min(1, opacity)));
                if (it.y + 100 < 0 || opacity <= 0.02) {
                    try { it.el.remove(); } catch(_) {}
                    activeTextOverlays.splice(i, 1);
                }
            }
            if (activeTextOverlays.length > 0) requestTextOverlayFrame();
            else { cancelAnimationFrame(textOverlayAnimId); textOverlayAnimId = null; }
        }
        // bind preview updates
        [tiText, tiFont, tiColor, tiSize, tiSizeRand, tiX, tiXRand, tiBorder, tiBorderColor, tiGlow, tiFlash, tiFlashSpeed, tiSpeed, tiSpeedRand].forEach(el => {
            if (el) el.addEventListener('input', updateTextInPreview);
        });
        if (tiFont) tiFont.addEventListener('change', updateTextInPreview);
        // If user touches size/x sliders, turn off random check automatically
        if (tiSize) tiSize.addEventListener('input', () => { if (tiSizeRand && tiSizeRand.checked) { tiSizeRand.checked = false; updateTextInPreview(); } });
        if (tiX) tiX.addEventListener('input', () => { if (tiXRand && tiXRand.checked) { tiXRand.checked = false; updateTextInPreview(); } });
        if (tiSend) tiSend.addEventListener('click', () => { const cfg = getTextConfigFromUi(); spawnRisingText(cfg); });
        if (tiClose) tiClose.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); hideTextInPanel(); });
        updateTextInPreview();
        // Auto send controls
        function setTextAuto(on) {
            textAutoOn = !!on;
            if (textAutoTimer) { clearInterval(textAutoTimer); textAutoTimer = null; }
            if (tiAutoBtn) tiAutoBtn.classList.toggle('on', textAutoOn);
            if (textAutoOn) {
                let iv = Math.max(100, Math.min(10000, Number(tiAutoInterval?.value) || 1000));
                if (tiAutoInterval) tiAutoInterval.value = String(iv);
                textAutoTimer = setInterval(() => {
                    try {
                        const cfg = getTextConfigFromUi();
                        spawnRisingText(cfg);
                    } catch(_) {}
                }, iv);
            }
        }
        if (tiAutoBtn) tiAutoBtn.addEventListener('click', () => setTextAuto(!textAutoOn));
        if (tiAutoInterval) tiAutoInterval.addEventListener('input', () => { if (textAutoOn) setTextAuto(true); });
// 🎲 One-shot random glow colour
tiGlowColorRandBtn.addEventListener('click', () => {
    tiGlowColor.value = randomGlowColor();
});

        /** ICY/Shoutcast-style embedded metadata (StreamTitle). Requires stream + CORS to allow reading the body. */
        function parseStreamTitleFromIcyBlock(raw) {
            if (!raw) return null;
            const s = String(raw).replace(/\0+$/g, '').trim();
            let m = s.match(/StreamTitle\s*=\s*'((?:\\'|[^'])*)'/i);
            if (m) return m[1].replace(/\\'/g, "'").trim();
            m = s.match(/StreamTitle\s*=\s*"((?:\\"|[^"])*)"/i);
            if (m) return m[1].replace(/\\"/g, '"').trim();
            m = s.match(/StreamTitle=([^;]+);/i);
            if (m) return m[1].replace(/^['"]|['"]$/g, '').trim();
            return null;
        }

        async function readOneIcyStreamTitle(streamUrl) {
            const ac = new AbortController();
            const to = setTimeout(() => { try { ac.abort(); } catch (_) {} }, 10000);
            let reader = null;
            try {
                const res = await fetch(streamUrl, {
                    method: 'GET',
                    headers: { 'Icy-MetaData': '1', Accept: '*/*' },
                    mode: 'cors',
                    cache: 'no-store',
                    signal: ac.signal
                });
                if (!res.ok || !res.body) return null;
                const metaintHdr = res.headers.get('icy-metaint') || res.headers.get('Icy-Metaint');
                const metaint = parseInt(String(metaintHdr || ''), 10);
                if (!Number.isFinite(metaint) || metaint < 1) return null;

                reader = res.body.getReader();
                const chunks = [];
                let received = 0;
                const need = metaint + 1 + 512;
                while (received < need) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    if (value && value.length) {
                        chunks.push(value);
                        received += value.length;
                    }
                }
                try { await reader.cancel(); } catch (_) {}
                reader = null;

                const all = new Uint8Array(received);
                let pos = 0;
                for (const c of chunks) {
                    all.set(c, pos);
                    pos += c.length;
                }
                if (all.length <= metaint) return null;
                const metaByte = all[metaint];
                const metaLen = metaByte * 16;
                if (!metaLen) return null;
                const metaStart = metaint + 1;
                if (all.length < metaStart + metaLen) return null;
                const decoded = new TextDecoder('utf-8', { fatal: false }).decode(all.slice(metaStart, metaStart + metaLen));
                return parseStreamTitleFromIcyBlock(decoded);
            } catch (_) {
                return null;
            } finally {
                clearTimeout(to);
                try { if (reader) await reader.cancel(); } catch (_) {}
            }
        }

        function clearNowPlayingICYBanner() {
            currentNowPlayingICY = '';
            if (stationBannerNowplayingEl) stationBannerNowplayingEl.textContent = '';
            if (stationBannerMetaWrap) stationBannerMetaWrap.classList.add('display-none');
        }

        function applyNowPlayingICYTitle(title) {
            const t = title && String(title).trim();
            if (!t) {
                clearNowPlayingICYBanner();
                return;
            }
            currentNowPlayingICY = t;
            if (stationBannerNowplayingEl) stationBannerNowplayingEl.textContent = t;
            if (stationBannerMetaWrap) stationBannerMetaWrap.classList.remove('display-none');
        }

        function stopNowPlayingPoll() {
            if (nowPlayingPollTimer) {
                clearInterval(nowPlayingPollTimer);
                nowPlayingPollTimer = null;
            }
            nowPlayingPollUrl = '';
        }

        function restartNowPlayingPoll(streamUrl) {
            stopNowPlayingPoll();
            if (!streamUrl || typeof streamUrl !== 'string') return;
            if (!/^https?:\/\//i.test(streamUrl)) return;
            nowPlayingPollUrl = streamUrl;
            const tick = async () => {
                try {
                    if (!state || !state.isPlaying) return;
                    if (!audioEl || audioEl.src !== streamUrl) return;
                    if (audioEl.paused) return;
                    if (icyPollBusy) return;
                    icyPollBusy = true;
                    const ttl = await readOneIcyStreamTitle(streamUrl);
                    if (ttl && audioEl && audioEl.src === streamUrl) applyNowPlayingICYTitle(ttl);
                } catch (_) {
                } finally {
                    icyPollBusy = false;
                }
            };
            tick();
            nowPlayingPollTimer = setInterval(tick, 26000);
        }

        function isRadioVisualModeActive() {
            try {
                const vis = state.activeVisualizer;
                if (!vis || !vis.name) return false;
                if (typeof globalThis.isRadioVisualModeName === 'function') {
                    return globalThis.isRadioVisualModeName(vis.name);
                }
                return vis.name === 'Analogue radio' || vis.name === 'Digital Radio'
                    || vis.name === 'Radio' || vis.name === 'Radio Visual';
            } catch (_) {
                return false;
            }
        }

        /** Deck currently heard (crossfader ≥ 0.5 → B, else A). */
        function getCrossfaderAudibleDeckKey() {
            return getDjCrossfade01() >= 0.5 ? 'b' : 'a';
        }

        function getDeckStationDisplayName(deck) {
            const dk = deck === 'b' ? 'b' : 'a';
            try {
                if (state && state.deckSourceMode && state.deckSourceMode[dk] === 'local') {
                    const raw = (state.deckLocalDisplayName && state.deckLocalDisplayName[dk]) || '';
                    if (raw && String(raw).trim()) return String(raw).trim();
                    const el = dk === 'b' ? audioElB : audioEl;
                    const src = (el && String(el.currentSrc || el.src || '')) || '';
                    if (src && typeof deriveTitleFromUrl === 'function') return deriveTitleFromUrl(src);
                    return 'Local track';
                }
                if (!Array.isArray(stations) || !stations.length) return '—';
                if (dk === 'b') {
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

        /** Bottom HUD secondary line: audible deck station (all visual modes). */
        function updateModeSubStationLine() {
            try {
                const subEl = document.getElementById('mode-sub');
                if (subEl) {
                    subEl.textContent = getDeckStationDisplayName(getCrossfaderAudibleDeckKey());
                }
                if (isRadioVisualModeActive()) {
                    const titleEl = document.getElementById('mode-title');
                    const vis = state.activeVisualizer;
                    if (titleEl && vis && vis.name) titleEl.textContent = vis.name;
                }
                syncModeInfoHudHintForDigitalRadio();
            } catch (_) {}
        }

        function hideStationBannerPermanently() {
            if (stationBannerTimer) {
                clearTimeout(stationBannerTimer);
                stationBannerTimer = null;
            }
            if (!stationBanner) return;
            try { stationBanner.remove(); } catch (_) {}
        }

        function showStationBanner(text) {
            hideStationBannerPermanently();
            updateModeSubStationLine();
        }

        function deriveTitleFromUrl(url) {
            try {
                const u = new URL(url);
                return u.hostname.replace(/^www\./, '');
            } catch {
                return url;
            }
        }

        // Swap: left-click now opens panel; right-click picks random (guard missing quick button)
        if (radioQuickBtn) {
            radioQuickBtn.title = "Left-click: choose • Right-click: random station";
            radioQuickBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (uiLocked) return;
                toggleRadioPanel();
            });
            radioQuickBtn.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (uiLocked) return;
                pickRandomStation();
                // Keep the radio button visible for 10s after right-click shuffle
                radioQuickBtn.style.opacity = '1';
                radioQuickBtn.style.pointerEvents = 'auto';
                radioQuickHoldUntil = Date.now() + 10000;
                if (radioQuickHoldTimeout) { clearTimeout(radioQuickHoldTimeout); radioQuickHoldTimeout = null; }
                radioQuickHoldTimeout = setTimeout(() => {
                    radioQuickHoldUntil = 0;
                    // Only hide if UI is idle (layer opacity is 0)
                    if (uiLayer.style.opacity === '0') {
                        radioQuickBtn.style.opacity = '0';
                        radioQuickBtn.style.pointerEvents = 'none';
                    }
                }, 10000);
            });
        }
        document.addEventListener('click', (e) => {
            if (radioPanel && !radioPanel.classList.contains('display-none')) {
                if(!radioPanel.contains(e.target) && e.target !== radioQuickBtn) {
                    hideRadioPanel();
                }
            }
        });

        // Drag Drop
        document.body.addEventListener('dragover', e => e.preventDefault());
        document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            // Accept only for ProjectM (either version)
            const active = state.activeVisualizer;
            if(!(active instanceof MilkdropEngine) && !(active instanceof MilkdropEngineV2)) {
                return alert("Switch to ProjectM (v1 or v2) first");
            }
            const file = e.dataTransfer.files[0];
            if(!file) return;
            const lower = file.name.toLowerCase();
            if(lower.endsWith('.milk')) {
                alert("Raw .milk files are not directly supported. Please convert to butterchurn JSON first.");
                return;
            }
            const reader = new FileReader();
            reader.onload = (ev) => active.loadCustomMilk?.(ev.target.result);
            reader.readAsText(file);
        });

        // Keyboard shortcuts
        /**
         * Number-row sample shortcuts. Keys 1..9, 0 map to the WAA/WAAA/AIR/FX1..FX7 sample
         * pads (Deck A slots 5..9, then Deck B slots 5..9). Short press starts/restarts the
         * sample (so repeated taps retrigger it); holding the key past the threshold cuts the
         * sample and turns the pad off. Pad clicks are unaffected.
         */
        const SAMPLE_KEY_LONG_HOLD_MS = 550;
        const SAMPLE_KEY_BINDINGS = {
            Digit1: { deck: 'a', slot: 5 }, Numpad1: { deck: 'a', slot: 5 },
            Digit2: { deck: 'a', slot: 6 }, Numpad2: { deck: 'a', slot: 6 },
            Digit3: { deck: 'a', slot: 7 }, Numpad3: { deck: 'a', slot: 7 },
            Digit4: { deck: 'a', slot: 8 }, Numpad4: { deck: 'a', slot: 8 },
            Digit5: { deck: 'a', slot: 9 }, Numpad5: { deck: 'a', slot: 9 },
            Digit6: { deck: 'b', slot: 5 }, Numpad6: { deck: 'b', slot: 5 },
            Digit7: { deck: 'b', slot: 6 }, Numpad7: { deck: 'b', slot: 6 },
            Digit8: { deck: 'b', slot: 7 }, Numpad8: { deck: 'b', slot: 7 },
            Digit9: { deck: 'b', slot: 8 }, Numpad9: { deck: 'b', slot: 8 },
            Digit0: { deck: 'b', slot: 9 }, Numpad0: { deck: 'b', slot: 9 }
        };
        const sampleKeyHoldTimers = new Map();
        function clearSampleKeyHold(code) {
            const t = sampleKeyHoldTimers.get(code);
            if (t) {
                try { clearTimeout(t); } catch (_) {}
                sampleKeyHoldTimers.delete(code);
            }
        }
        function getDjDecksEngineIfActive() {
            const av = state && state.activeVisualizer;
            return (av && av.name === 'DJ Decks') ? av : null;
        }

        function getDigitalRadioVisualIfActive() {
            const av = state && state.activeVisualizer;
            if (!av || av.name !== 'Digital Radio') return null;
            if (typeof av.triggerAutoFadeFromShortcut === 'function') return av;
            return null;
        }

        function getActiveRadioVisualEngine() {
            const av = state && state.activeVisualizer;
            if (!av || !av.name) return null;
            const n = av.name;
            if (n === 'Digital Radio' || n === 'Analogue radio' || n === 'Radio' || n === 'Radio Visual') {
                return av;
            }
            try {
                if (typeof RadioVisualEngine !== 'undefined' && RadioVisualEngine.isRadioModeName(n)) return av;
            } catch (_) {}
            return null;
        }
        function isDigitalRadioVisualActive() {
            const av = state && state.activeVisualizer;
            return !!(av && av.name === 'Digital Radio' && av.skin === 'digital');
        }

        function cancelActiveAutoFade() {
            try {
                const dj = getDjDecksEngineIfActive();
                if (dj && typeof dj.cancelAutoFade === 'function') dj.cancelAutoFade();
            } catch (_) {}
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv.cancelAutoFade === 'function') rv.cancelAutoFade();
            } catch (_) {}
        }

        function clearCrossfadeResumeSuppress() {
            try {
                const dj = getDjDecksEngineIfActive();
                if (dj && typeof dj.clearSuppressEnsureCrossfadeDeckPlayback === 'function') {
                    dj.clearSuppressEnsureCrossfadeDeckPlayback();
                }
            } catch (_) {}
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv._clearSuppressCrossfadeResume === 'function') rv._clearSuppressCrossfadeResume();
            } catch (_) {}
        }

        function getDeckBPlaybackMedia() {
            try {
                if (typeof getDeckBRadioAudibleEl === 'function') return getDeckBRadioAudibleEl();
            } catch (_) {}
            return audioElB;
        }

        function toggleAutoMixShortcut() {
            try {
                const av = state && state.activeVisualizer;
                if (av && typeof av._toggleAutoMix === 'function') {
                    av._toggleAutoMix();
                    return;
                }
                const btn = document.getElementById('mix-automix') || document.getElementById('dj-automix');
                if (btn) btn.click();
            } catch (_) {}
        }

        function toggleDigitalAiShortcut() {
            try {
                const rv = getActiveRadioVisualEngine();
                if (rv && typeof rv.toggleDigitalAiFromShortcut === 'function') {
                    rv.toggleDigitalAiFromShortcut();
                }
            } catch (_) {}
        }

        /**
         * Space-bar long-press support.
         *
         *  - Short tap  → engine.triggerAutoFadeFromShortcut() (start fade, or reverse
         *                  direction if a fade is already in flight).
         *  - Long hold  → pauseBothDecksOrStartActive() on DJ Decks or Radio visual
         *                  (pauses both decks if anything is playing; otherwise starts
         *                  the deck currently winning the crossfader).
         *
         * spaceKeyDown / spaceLongPressTimer / spaceLongPressFired track state between
         * keydown and keyup. Long-hold only runs if Space is still down when the timer
         * fires (avoids pausing on a quick tap when keyup is delayed or missed).
         */
        const SPACE_LONG_HOLD_MS = 500;
        let spaceKeyDown = false;
        let spaceKeyDownAt = 0;
        let spaceShortcutArmed = false;
        let spaceLongPressTimer = null;
        let spaceLongPressFired = false;
        function clearSpaceLongPress() {
            if (spaceLongPressTimer) {
                try { clearTimeout(spaceLongPressTimer); } catch (_) {}
                spaceLongPressTimer = null;
            }
        }
        function resetSpaceShortcutState() {
            spaceShortcutArmed = false;
            spaceKeyDown = false;
            spaceKeyDownAt = 0;
            spaceLongPressFired = false;
            clearSpaceLongPress();
        }
        function spaceShortcutTargetsActive() {
            return !!(getDjDecksEngineIfActive() || getActiveRadioVisualEngine());
        }
        /** Stop Space from activating a focused deck/radio button (keyup click) after our shortcut runs. */
        function blurDeckUiFocusForSpace() {
            try {
                const ae = document.activeElement;
                if (!ae || ae === document.body || typeof ae.blur !== 'function') return;
                if (ae.closest && (ae.closest('#radio-visual-root') || ae.closest('#dj-visual-root'))) {
                    ae.blur();
                }
            } catch (_) {}
        }
        function clickDigitalRadioFadeButton() {
            try {
                const btn = document.querySelector(
                    '#radio-visual-root button[data-rv-digital="fade"]'
                );
                if (btn) {
                    btn.click();
                    return true;
                }
            } catch (_) {}
            return false;
        }
        function triggerSpaceShortTap() {
            try {
                const eng = getDjDecksEngineIfActive();
                const rv = getActiveRadioVisualEngine();
                if (rv && !eng) {
                    if (clickDigitalRadioFadeButton()) return;
                    if (typeof rv.triggerAutoFadeFromShortcut === 'function') {
                        rv.triggerAutoFadeFromShortcut();
                    }
                    return;
                }
                if (eng && typeof eng.triggerAutoFadeFromShortcut === 'function') {
                    eng.triggerAutoFadeFromShortcut();
                } else if (rv && typeof rv.triggerAutoFadeFromShortcut === 'function') {
                    rv.triggerAutoFadeFromShortcut();
                }
            } catch (_) {}
        }
        function triggerSpaceLongHold() {
            try {
                const eng = getDjDecksEngineIfActive();
                const rv = getActiveRadioVisualEngine();
                if (eng && typeof eng.pauseBothDecksOrStartActive === 'function') {
                    eng.pauseBothDecksOrStartActive();
                } else if (rv && typeof rv.pauseBothDecksOrStartActive === 'function') {
                    rv.pauseBothDecksOrStartActive();
                }
            } catch (_) {}
        }

        /**
         * V and B are simple play/pause toggles for Deck A and Deck B respectively.
         * Holding the key (key-repeat after the OS threshold) is suppressed by the
         * `e.repeat` guard in the keydown branch so a deliberate hold never stutters
         * play/pause back and forth — it just resolves to one toggle. Next-station is
         * Next track/station is the job of the N shortcut (queue-first on Digital Radio,
         * else `pickRandomStationForCrossfadedDeck`), not V/B.
         *
         * Each helper also handles the cold-start case: if the deck's media element
         * has no source yet, we call playRadio() / playRadioB() so the very first
         * press actually boots the deck rather than being a silent no-op.
         *
         * Deck A: use `getDeckAMediaForPlaybackState()` (same as the DJ Play button)
         * so radio warm-handoff output on `audioElRadioAAlt` is toggled — not only
         * `audioEl`, which can be the muted prep element after AUTO-FADE / retune.
         */
        function deckHasSource(el) {
            try {
                const src = el ? String(el.currentSrc || el.src || '') : '';
                return !!(src && src !== 'about:blank');
            } catch (_) { return false; }
        }
        function deckAPlayPause() {
            try {
                // Take manual control: if an AUTO-FADE animation is mid-flight,
                // its per-frame ticker will keep re-playing whatever deck has
                // crossfader gain — so V appears to do nothing for the full
                // fade. Cancelling the fade first makes the play/pause stick.
                cancelActiveAutoFade();
                const media = (typeof getDeckAMediaForPlaybackState === 'function')
                    ? getDeckAMediaForPlaybackState()
                    : audioEl;
                if (!media) return;
                const eng = getDjDecksEngineIfActive();
                if (!deckHasSource(media)) {
                    clearCrossfadeResumeSuppress();
                    if (typeof playRadio === 'function') playRadio();
                } else if (media.paused) {
                    clearCrossfadeResumeSuppress();
                    media.play().catch(() => {});
                } else {
                    clearCrossfadeResumeSuppress();
                    media.pause();
                }
            } catch (_) {}
        }
        function deckBPlayPause() {
            try {
                // Same rationale as deckAPlayPause: cancel any in-flight
                // AUTO-FADE so the user's explicit pause/play on Deck B isn't
                // overridden by the next animation tick.
                cancelActiveAutoFade();
                const mediaB = getDeckBPlaybackMedia();
                if (!mediaB) return;
                if (!deckHasSource(mediaB)) {
                    clearCrossfadeResumeSuppress();
                    if (typeof playRadioB === 'function') playRadioB();
                } else if (mediaB.paused) {
                    clearCrossfadeResumeSuppress();
                    mediaB.play().catch(() => {});
                } else {
                    clearCrossfadeResumeSuppress();
                    mediaB.pause();
                }
            } catch (_) {}
        }

        function handleGlobalKeydown(e) {
            // Allow Lock toggle (L) to work even when locked or focused in inputs
            if (e && (e.key === 'l' || e.key === 'L')) {
                e.preventDefault();
                uiLocked = !uiLocked;
                applyUiLockState();
                try { resetIdleTimer(); } catch(_) {}
                return;
            }
            try {
                if (shortcutsLocked) return;
                if (uiLocked) { e.preventDefault(); return; }
                const ae = document.activeElement;
                if (ae) {
                    if (ae.tagName === 'TEXTAREA' || ae.tagName === 'SELECT' || ae.isContentEditable || (ae.closest && ae.closest('#textin-panel'))) {
                        return;
                    }
                    if (ae.tagName === 'INPUT') {
                        const it = String(ae.type || '').toLowerCase();
                        if (it === 'range' && ae.closest && (ae.closest('#dj-visual-root') || ae.closest('#radio-visual-root'))) {
                            /* Deck VOL, crossfader, BPM inside DJ / radio visual — keep Space / shortcuts */
                        } else {
                            return;
                        }
                    }
                }
            } catch(_) {}

            const isSpaceKey = e && (e.code === 'Space' || e.key === ' ');
            if (!isSpaceKey) {
                resetSpaceShortcutState();
            }

            // Sample pad shortcuts (1..0). Ignore modifier-combined presses so browser/OS
            // shortcuts like Ctrl+1 (tab) are untouched.
            const sampleBinding = (e && e.code && SAMPLE_KEY_BINDINGS[e.code]) || null;
            if (sampleBinding && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
                const engine = getDjDecksEngineIfActive();
                if (!engine) return; // Only meaningful while DJ Decks is the active visualizer
                e.preventDefault();
                if (e.repeat) return; // Key auto-repeat must not re-trigger the sample
                engine.triggerBeatPadKey(sampleBinding.deck, sampleBinding.slot, 'start');
                clearSampleKeyHold(e.code);
                const code = e.code;
                const timer = setTimeout(() => {
                    sampleKeyHoldTimers.delete(code);
                    const eng = getDjDecksEngineIfActive();
                    if (eng) eng.triggerBeatPadKey(sampleBinding.deck, sampleBinding.slot, 'stop');
                }, SAMPLE_KEY_LONG_HOLD_MS);
                sampleKeyHoldTimers.set(code, timer);
                return;
            }

            if (e.key === 'f' || e.key === 'F') {
                toggleFullscreen();
            } else if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                var resetBtn = document.getElementById('btn-webm-reset');
                if (resetBtn) { resetBtn.click(); }
            } else if (e.key === 'c' || e.key === 'C') {
                e.preventDefault();
                if (isDigitalRadioVisualActive()) {
                    const rv = getActiveRadioVisualEngine();
                    if (rv && typeof rv.triggerCFromShortcut === 'function') {
                        try { rv.triggerCFromShortcut(); } catch (_) {}
                        return;
                    }
                }
                const av = state && state.activeVisualizer;
                if (av && av.name === 'DJ Decks' && av.deckBVizMode === 'projectm' && typeof av.nextDeckBProjectMPreset === 'function') {
                    try { av.nextDeckBProjectMPreset(); } catch (_) {}
                } else if (av && typeof av.nextPreset === 'function') {
                    try { av.nextPreset(); } catch (_) {}
                }
            } else if (e.key === 'Escape') {
                // Close any open panels first; if none open, return to start
                e.preventDefault();
                let closed = false;
                try {
                    // Options panel has priority: close it and stop further handling
                    if (!closed && typeof isOptionsOpen === 'function' && isOptionsOpen && isOptionsOpen()) {
                        try { closeOptionsPanel(); } catch(_) {}
                        closed = true;
                    }
                    // Avatar Settings (right slide) next priority
                    if (!closed && typeof isBottomMenuOpen === 'function' && isBottomMenuOpen()) {
                        try { closeBottomMenuPanel(); } catch(_) {}
                        closed = true;
                    }
                    // Mic confirm
                    if (!closed && micConfirm && micConfirm.style.display === 'flex') {
                        hideMicConfirm();
                        closed = true;
                    }
                    // Mix panel
                    if (mixPanel && !mixPanel.classList.contains('display-none') && mixPanel.classList.contains('open')) {
                        toggleMixPanel();
                        closed = true;
                    }
                    // Keyboard shortcuts sheet
                    if (!closed && keyboardShortcutsPanel && !keyboardShortcutsPanel.classList.contains('display-none') && keyboardShortcutsPanel.classList.contains('open')) {
                        closeKeyboardShortcutsPanel();
                        closed = true;
                    }
                    // Text-In panel
                    if (!closed && textInPanel && !textInPanel.classList.contains('display-none') && textInPanel.classList.contains('open')) {
                        hideTextInPanel();
                        closed = true;
                    }
                    // Top menu
                    if (!closed && typeof isTopMenuOpen === 'function' && isTopMenuOpen()) {
                        try { closeTopMenuPanel(); } catch(_) {}
                        closed = true;
                    }
                    // WebM settings
                    if (!closed && webmSettingsPanel && !webmSettingsPanel.classList.contains('display-none')) {
                        try { hideWebmSettingsPanel(); } catch(_) {}
                        closed = true;
                    }
                    // Visualizer settings panel
                    if (!closed && settingsPanel && !settingsPanel.classList.contains('display-none')) {
                        settingsPanel.classList.add('display-none');
                        settingsPanel.style.opacity = '';
                        settingsPanel.style.pointerEvents = '';
                        closed = true;
                    }
                } catch(_) {}
                if (!closed) {
                    stopAllAndShowStart();
                }
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (typeof webmOn !== 'undefined' && webmOn) {
                    adjustWebmPosition(-2, 0);
                } else {
                    nudgeCrossfade(-CROSSFADE_KEY_STEP);
                }
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                if (typeof webmOn !== 'undefined' && webmOn) {
                    adjustWebmPosition(2, 0);
                } else {
                    nudgeCrossfade(CROSSFADE_KEY_STEP);
                }
            } else if (e.key === 'v' || e.key === 'V') {
                // V → toggle play / pause on Deck A (or boot it via playRadio() if
                // no source is loaded yet). e.repeat is suppressed so a held key
                // resolves to a single toggle rather than flapping back and forth
                // at the OS auto-repeat rate.
                e.preventDefault();
                if (e.repeat) return;
                deckAPlayPause();
            } else if (e.key === 'b' || e.key === 'B') {
                // B → toggle play / pause on Deck B (or boot it via playRadioB()
                // if nothing is loaded). Same e.repeat guard as V.
                e.preventDefault();
                if (e.repeat) return;
                deckBPlayPause();
            } else if (e.key === 'o' || e.key === 'O') {
                e.preventDefault();
                if (e.repeat) return;
                toggleAutoMixShortcut();
            } else if (e.key === '/' || e.code === 'Slash') {
                e.preventDefault();
                if (e.repeat) return;
                if (isDigitalRadioVisualActive()) toggleDigitalAiShortcut();
            } else if (e.code === 'Space' || e.key === ' ') {
                e.preventDefault();
                if (e.repeat) return;
                const engine = getDjDecksEngineIfActive();
                const radioVis = getActiveRadioVisualEngine();
                if (isDigitalRadioVisualActive() && !engine) {
                    return;
                }
                if (!engine && !radioVis) {
                    try {
                        if (audioEl) {
                            if (audioEl.paused) audioEl.play().catch(() => {});
                            else audioEl.pause();
                        }
                    } catch (_) {}
                    return;
                }
                blurDeckUiFocusForSpace();
                spaceShortcutArmed = true;
                spaceKeyDown = true;
                spaceKeyDownAt = performance.now();
                clearSpaceLongPress();
                spaceLongPressFired = false;
                spaceLongPressTimer = setTimeout(() => {
                    spaceLongPressTimer = null;
                    if (!spaceShortcutArmed || !spaceKeyDown) return;
                    spaceLongPressFired = true;
                    triggerSpaceLongHold();
                }, SPACE_LONG_HOLD_MS);
            } else if (e.key === 't' || e.key === 'T') {
                // Toggle Text-In panel
                e.preventDefault();
                window.__panelGuardUntilMs = 0; // FIX: Reset guard timer for instant open
                try { toggleTextInPanel(); } catch(_) {}
            } else if (e.key === 'n' || e.key === 'N') {
                // Digital Radio: next queued local on crossfader-winning deck, else random station (like A▶ / B▶).
                e.preventDefault();
                if (isDigitalRadioVisualActive()) {
                    const rv = getActiveRadioVisualEngine();
                    if (rv && typeof rv.triggerNextFromShortcut === 'function') {
                        try { rv.triggerNextFromShortcut(); } catch (_) {}
                        return;
                    }
                }
                try { pickRandomStationForCrossfadedDeck(); } catch (_) {}
            } else if (e.key === 'p' || e.key === 'P') {
                // Toggle the top-menu Radio Stations list (formerly bound to the
                // left-click on the radio quick button only).
                e.preventDefault();
                window.__panelGuardUntilMs = 0; // FIX: Reset guard timer for instant open
                try { toggleRadioPanel(); } catch(_) {}
            } else if (e.key === 'r' || e.key === 'R') {
                // Toggle Avatar Settings (Right Slide Panel). Previously this
                // panel was opened by P; we swapped P → Radio Stations and gave
                // Avatar its own dedicated key (R) so both can be reached
                // without a detour through the top menu.
                e.preventDefault();
                window.__panelGuardUntilMs = 0;
                try { toggleBottomMenuPanel(); } catch(_) {}
            } else if (e.key === 'm' || e.key === 'M') {
                // Toggle Mix Settings panel
                e.preventDefault();
                window.__panelGuardUntilMs = 0; // FIX: Reset guard timer for instant open
                try { toggleMixPanel(); } catch(_) {}
            } else if (e.key === 'h' || e.key === 'H') {
                e.preventDefault();
                try {
                    const q = document.getElementById('dj-b-queue');
                    if (q) q.click();
                } catch (_) {}
            } else if (e.key === 'j' || e.key === 'J') {
                e.preventDefault();
                try {
                    const v = document.getElementById('dj-b-fx-low');
                    if (v) v.click();
                } catch (_) {}
            } else if (e.key === 'k' || e.key === 'K') {
                e.preventDefault();
                if (isDigitalRadioVisualActive()) {
                    const rv = getActiveRadioVisualEngine();
                    if (rv && typeof rv.toggleDigitalStagingKaraoke === 'function') {
                        try { rv.toggleDigitalStagingKaraoke(); } catch (_) {}
                        return;
                    }
                }
                try {
                    const k = document.getElementById('dj-b-sfx-w1');
                    if (k) k.click();
                } catch (_) {}
            } else if (e.key === 'g' || e.key === 'G') {
                // Toggle Options panel (repurposed)
                e.preventDefault();
                window.__panelGuardUntilMs = 0; 
                try { toggleOptionsPanel(); } catch(_) {}
            } else if (e.key === 'u' || e.key === 'U') {
                // Send Text-In immediately (shortcut swapped from I → U)
                e.preventDefault();
                try {
                    const cfg = getTextConfigFromUi();
                    spawnRisingText(cfg);
                } catch(_) {}
            } else if (e.key === 'w' || e.key === 'W') {
                e.preventDefault();
                if(!webmOn) {
                    if(webmList.length === 0) {
                        loadWebmList().finally(() => {
                            if(webmList.length > 0) showWebm();
                        });
                    } else {
                        showWebm();
                    }
                } else {
                    hideWebm();
                }
            } else if (e.key === 'q' || e.key === 'Q') {
                e.preventDefault();
                adjustWebmSpeed(-0.1);
            } else if (e.key === 'e' || e.key === 'E') {
                e.preventDefault();
                adjustWebmSpeed(+0.1);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                adjustWebmPosition(0, -2);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                adjustWebmPosition(0, 2);
            } else if (e.key === 'a' || e.key === 'A') {
                e.preventDefault();
                if (webmOn) prevWebm();
            } else if (e.key === 'd' || e.key === 'D') {
                e.preventDefault();
                if (webmOn) nextWebm();
            } else if (e.key === 'z' || e.key === 'Z') {
                e.preventDefault();
                adjustWebmOpacity(-0.05);
            } else if (e.key === 'x' || e.key === 'X') {
                e.preventDefault();
                adjustWebmOpacity(0.05);
            } else if (e.key === '=') {
                e.preventDefault();
                adjustWebmScale(2);
            } else if (e.key === '-' || e.key === '_') {
                e.preventDefault();
                adjustWebmScale(-2);
            }
            // Visual mode navigation with comma/period
            else if (e.key === ',') {
                e.preventDefault();
                loadMode(state.currentModeIdx - 1);
            } else if (e.key === '.') {
                e.preventDefault();
                loadMode(state.currentModeIdx + 1);
            }
        }
        function handleGlobalKeyup(e) {
            try {
                if (e && e.code && sampleKeyHoldTimers.has(e.code)) clearSampleKeyHold(e.code);
            } catch (_) {}
            if (!e || (e.code !== 'Space' && e.key !== ' ')) return;
            if (isDigitalRadioVisualActive() && !getDjDecksEngineIfActive()) {
                resetSpaceShortcutState();
                return;
            }
            if (!spaceShortcutArmed && !spaceKeyDownAt) return;
            try {
                if (shortcutsLocked || uiLocked) {
                    resetSpaceShortcutState();
                    return;
                }
            } catch (_) {
                resetSpaceShortcutState();
                return;
            }
            try {
                e.preventDefault();
                e.stopPropagation();
            } catch (_) {}
            const held = spaceKeyDownAt ? (performance.now() - spaceKeyDownAt) : 0;
            const doShortTap = !spaceLongPressFired && held < SPACE_LONG_HOLD_MS;
            resetSpaceShortcutState();
            if (doShortTap) triggerSpaceShortTap();
        }
        try {
            document.addEventListener('keydown', handleGlobalKeydown, false);
        } catch(e) {}
        try {
            document.addEventListener('keyup', handleGlobalKeyup, true);
        } catch(e) {}
        try {
            window.addEventListener('blur', () => {
                resetSpaceShortcutState();
            }, false);
        } catch (_) {}

        // Bootstrap station list
        loadStations();
        // Preload webm list (best-effort)
        loadWebmList();

        // --- WEBM HELPERS ---
        async function loadWebmList() {
            try {
                // Try a manifest first
                const resp = await fetch('assets/video/webms.txt', { cache: 'no-store' });
                if(resp.ok) {
                    const txt = await resp.text();
                    const files = txt.split('\n').map(s => s.trim()).filter(Boolean);
                    webmList = files.map(f => f.startsWith('assets/video/') ? f : `assets/video/${f}`);
                    webmIndex = 0;
                    return;
                }
            } catch {}
            // Try directory listing parse (best-effort)
            try {
                const dirResp = await fetch('assets/video/', { cache: 'no-store' });
                if(dirResp.ok) {
                    const html = await dirResp.text();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const anchors = Array.from(doc.querySelectorAll('a,link'));
                    const found = anchors.map(a => a.getAttribute('href') || '').filter(href => /\.webm(\?.*)?$/i.test(href));
                    if(found.length > 0) {
                        const normalized = found.map(f => f.startsWith('assets/video/') ? f : `assets/video/${f.replace(/^\//,'')}`);
                        // Deduplicate
                        webmList = Array.from(new Set(normalized));
                        webmIndex = 0;
                        return;
                    }
                }
            } catch {}
            // Fallback to known files
            const defaults = ['assets/video/red.webm', 'assets/video/grn.webm'];
            webmList = defaults;
            webmIndex = 0;
        }

        function centerAvatarInDeckB() {
            webmAnchorDeckB = true;
            webmSettings.posXvw = 50;
            webmSettings.posYvh = 50;
            try {
                if (typeof inpWebmX !== 'undefined' && inpWebmX) inpWebmX.value = '50';
                if (typeof inpWebmY !== 'undefined' && inpWebmY) inpWebmY.value = '50';
                const aX = document.getElementById('avatar-inp-x');
                const aY = document.getElementById('avatar-inp-y');
                if (aX) aX.value = '50';
                if (aY) aY.value = '50';
            } catch (_) {}
            if (webmOn) applyWebmSettings();
        }

        function showWebm(opts) {
            if(webmList.length === 0) return;
            if (opts && opts.deckBCenter) {
                centerAvatarInDeckB();
            } else {
                webmAnchorDeckB = false;
            }
            webmOn = true;
            setWebm(webmIndex);
            webmOverlayEl.classList.remove('display-none');
            // Allow interaction on overlay (for dblclick/gestures)
            try { webmOverlayEl.style.pointerEvents = 'auto'; } catch(_) {}
            webmPrevBtn.classList.remove('display-none');
            webmNextBtn.classList.remove('display-none');
            applyWebmSettings();
            // resume auto if enabled
            if (webmAutoOn) scheduleWebmAuto();
            try { updateAvatarPlayButton(); } catch(_) {}
        }

        function hideWebm() {
            webmOn = false;
            webmAnchorDeckB = false;
            try { unbindWebmDeckBLayoutWatchers(); } catch (_) {}
            try {
                webmOverlayEl.style.position = '';
                webmOverlayEl.style.left = '';
                webmOverlayEl.style.top = '';
                webmOverlayEl.style.transform = '';
            } catch (_) {}
            webmOverlayEl.classList.add('display-none');
            // Disable overlay interaction when hidden
            try { webmOverlayEl.style.pointerEvents = 'none'; } catch(_) {}
            webmPrevBtn.classList.add('display-none');
            webmNextBtn.classList.add('display-none');
            try { webmVideoEl.pause(); } catch {}
            try { webmVideoLeftEl.pause(); } catch {}
            try { webmVideoRightEl.pause(); } catch {}
            webmVideoEl.src = '';
            webmVideoLeftEl.src = '';
            webmVideoRightEl.src = '';
            // stop auto when hidden
            cancelWebmAuto();
            try { updateAvatarPlayButton(); } catch(_) {}
        }

        function isWebmOverlayVisible() {
            try {
                return !!(webmOverlayEl && !webmOverlayEl.classList.contains('display-none'));
            } catch (_) {
                return !!webmOn;
            }
        }

        function toggleWebmOverlay(opts) {
            try {
                if (uiLocked) return;
                if (isWebmOverlayVisible()) {
                    hideWebm();
                    return;
                }
                const open = () => {
                    try {
                        if (opts && opts.deckBCenter) showWebm({ deckBCenter: true });
                        else showWebm();
                    } catch (_) {}
                };
                if (!webmList.length && typeof loadWebmList === 'function') {
                    return loadWebmList().finally(open);
                }
                open();
            } catch (_) {}
        }

        function setWebm(index) {
            if(webmList.length === 0) return;
            if(index < 0) index = webmList.length - 1;
            if(index >= webmList.length) index = 0;
            webmIndex = index;
            const src = webmList[webmIndex];
            if(webmVideoEl.src.endsWith(src)) return;
            webmVideoEl.src = src;
            webmVideoEl.muted = true;
            webmVideoEl.loop = true;
            webmVideoEl.playsInline = true;
            webmVideoEl.autoplay = true;
            webmVideoEl.play().catch(()=>{ /* ignore */ });
            // mirror to duplicates
            [webmVideoLeftEl, webmVideoRightEl].forEach(v => {
                v.src = src;
                v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true;
                v.playbackRate = webmSettings.playbackRate;
                v.style.opacity = String(webmSettings.opacity);
                v.play().catch(()=>{});
            });
            // Ensure current settings are applied consistently to the main video and duplicates
            applyWebmSettings();
        }

        function nextWebm() { setWebm(webmIndex + 1); }
        function prevWebm() { setWebm(webmIndex - 1); }
        // --- WebM Auto Random ---
        function scheduleWebmAuto() {
            cancelWebmAuto();
            const delay = (30 + Math.random() * 30) * 1000; // 30-60s
            webmAutoTimer = setTimeout(() => {
                if (webmOn && webmList.length > 0) {
                    nextWebm();
                }
                scheduleWebmAuto();
            }, delay);
        }
        function cancelWebmAuto() {
            if (webmAutoTimer) { clearTimeout(webmAutoTimer); webmAutoTimer = null; }
        }
        function setWebmAuto(on) {
            webmAutoOn = !!on;
            const autoBtn = document.getElementById('btn-webm-auto');
            if (autoBtn) { autoBtn.textContent = 'Auto'; autoBtn.classList.toggle('on', webmAutoOn); }
            const autoBtn2 = document.getElementById('avatar-btn-auto');
            if (autoBtn2) { autoBtn2.textContent = 'Auto'; autoBtn2.classList.toggle('on', webmAutoOn); }
            if (webmAutoOn) scheduleWebmAuto(); else cancelWebmAuto();
        }
        
        function toggleWebmSettingsPanel() {
            if (uiLocked) return;
            if(webmSettingsPanel.classList.contains('display-none')) {
                webmSettingsPanel.classList.remove('display-none');
                webmSettingsPanel.style.display = 'block';
                webmSettingsPanel.style.opacity = '1';
                webmSettingsPanel.style.pointerEvents = 'auto';
                scheduleWebmSettingsClose();
                // sync inputs
                inpWebmScale.value = String(webmSettings.scaleVw);
                inpWebmX.value = String(webmSettings.posXvw);
                inpWebmY.value = String(webmSettings.posYvh);
                inpWebmRot.value = String(webmSettings.rotationDeg);
                inpWebmSpeed.value = String(webmSettings.playbackRate);
                inpWebmOpacity.value = String(webmSettings.opacity);
                if (inpWebmDupSpacing) inpWebmDupSpacing.value = String(Math.round((webmSettings.duplicateSpacing || WEBM_DEFAULT_DUP_SPACING) * 100));
                try { syncAllWebmDupKnobs(); } catch (_) {}
            } else {
                hideWebmSettingsPanel();
            }
        }
        function hideWebmSettingsPanel() {
            if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
            webmSettingsPanel.classList.add('display-none');
            webmSettingsPanel.style.display = 'none';
            webmSettingsPanel.style.opacity = '';
            webmSettingsPanel.style.pointerEvents = '';
        }

        // --- Top Menu Panel helpers ---
        function isTopMenuOpen() {
            const p = document.getElementById('top-menu-panel');
            return !!(p && !p.classList.contains('display-none') && p.classList.contains('open'));
        }
        function openTopMenuPanel() {
            if (uiLocked) return;
            // Prevent opening if a recent panel close just happened
            try { if (window.__panelGuardUntilMs && Date.now() < window.__panelGuardUntilMs) return; } catch(_) {}
            const p = document.getElementById('top-menu-panel');
            if (!p) return;
            // Populate fields
            try {
                const urlEl = document.getElementById('station-url');
                if (urlEl) {
                    let current = '';
                    try { current = audioEl?.src || ''; } catch(_) {}
                    if (!current && typeof currentStationIndex === 'number' && currentStationIndex >= 0 && stations[currentStationIndex]) {
                        current = stations[currentStationIndex].url || '';
                    }
                    urlEl.value = current || '';
                }
                const sel = document.getElementById('webm-select');
                if (sel) {
                    sel.innerHTML = '';
                    if (Array.isArray(webmList) && webmList.length > 0) {
                        webmList.forEach((pth, idx) => {
                            const opt = document.createElement('option');
                            opt.value = String(idx);
                            opt.textContent = (pth || '').split('/').pop() || pth || ('Item ' + (idx+1));
                            if (typeof webmIndex === 'number' && idx === webmIndex) opt.selected = true;
                            sel.appendChild(opt);
                        });
                    } else {
                        const opt = document.createElement('option');
                        opt.value = '0';
                        opt.textContent = '(load list)';
                        sel.appendChild(opt);
                    }
                }
                // Ensure volume slider is visible when opening the panel
                const vs = document.getElementById('volume-slider-container');
                if (vs) vs.style.display = 'flex';
                try { updateStationActiveHighlight(); } catch (_) {}
                try { syncTopMenuStationsLayout(); } catch (_) {}
                try {
                    if (typeof setKnobUi === 'function') {
                        setKnobUi(document.getElementById('knob-tm-shuffle-min'), 3, 120, visualSettings.shuffleMinSec);
                        setKnobUi(document.getElementById('knob-tm-shuffle-max'), 5, 180, visualSettings.shuffleMaxSec);
                        setKnobUi(document.getElementById('knob-tm-transition'), 0, 10, visualSettings.transitionSec);
                        setKnobUi(document.getElementById('knob-tm-pixelratio'), 0.5, 3, visualSettings.pixelRatio);
                    }
                } catch (_) {}
            } catch(_) {}
            p.classList.remove('display-none');
            requestAnimationFrame(() => { p.classList.add('open'); });
        }
        function closeTopMenuPanel() {
            const p = document.getElementById('top-menu-panel');
            if (!p) return;
            p.classList.remove('open');
            try { window.__panelGuardUntilMs = Date.now() + 1200; } catch(_) {}
            setTimeout(() => { p.classList.add('display-none'); }, 350);
        }
        function toggleTopMenuPanel() {
            if (uiLocked) return;
            if (isTopMenuOpen()) closeTopMenuPanel(); else openTopMenuPanel();
        }
        // Bind top menu controls
        (function bindTopMenuControls(){
            try {
                const btnClose = document.getElementById('btn-topmenu-close');
                if (btnClose) btnClose.addEventListener('click', (e)=>{ e.stopPropagation(); closeTopMenuPanel(); });
                const btnPlay = document.getElementById('topmenu-play');
                if (btnPlay) btnPlay.addEventListener('click', (e)=>{ 
                    e.stopPropagation();
                    const urlEl = document.getElementById('station-url');
                    if (urlEl && typeof playRadio === 'function') {
                        if (typeof radioInputEl !== 'undefined' && radioInputEl) radioInputEl.value = urlEl.value;
                        playRadio();
                    }
                });
                function applyTopMenuVisualSettings() {
                    try {
                        if (inpShuffleMin) inpShuffleMin.value = String(visualSettings.shuffleMinSec ?? 30);
                        if (inpShuffleMax) inpShuffleMax.value = String(visualSettings.shuffleMaxSec ?? 60);
                        if (inpTransition) inpTransition.value = String(visualSettings.transitionSec ?? 2.7);
                        if (inpPixelRatio) inpPixelRatio.value = String(visualSettings.pixelRatio ?? 1);
                        if (state.activeVisualizer && state.activeVisualizer instanceof MilkdropEngineV2) {
                            state.activeVisualizer.applySettings?.();
                        }
                    } catch (_) {}
                }
                if (typeof bindKnob === 'function') {
                    bindKnob('knob-tm-shuffle-min', 3, 120, visualSettings.shuffleMinSec, visualSettings, 'shuffleMinSec', applyTopMenuVisualSettings);
                    bindKnob('knob-tm-shuffle-max', 5, 180, visualSettings.shuffleMaxSec, visualSettings, 'shuffleMaxSec', applyTopMenuVisualSettings);
                    bindKnob('knob-tm-transition', 0, 10, visualSettings.transitionSec, visualSettings, 'transitionSec', applyTopMenuVisualSettings);
                    bindKnob('knob-tm-pixelratio', 0.5, 3, visualSettings.pixelRatio, visualSettings, 'pixelRatio', applyTopMenuVisualSettings);
                }
                const btnPrevStA = document.getElementById('topmenu-prev-station-a');
                const btnNextStA = document.getElementById('topmenu-next-station-a');
                if (btnPrevStA) btnPrevStA.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof setStation === 'function' && stations.length) {
                        setStation((currentStationIndex - 1 + stations.length) % stations.length);
                    }
                });
                if (btnNextStA) btnNextStA.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof setStation === 'function' && stations.length) {
                        setStation((currentStationIndex + 1) % stations.length);
                    }
                });
                const btnPrevStB = document.getElementById('topmenu-prev-station-b');
                const btnNextStB = document.getElementById('topmenu-next-station-b');
                if (btnPrevStB) btnPrevStB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof setStationB === 'function' && stations.length) {
                        setStationB((currentStationBIndex - 1 + stations.length) % stations.length);
                    }
                });
                if (btnNextStB) btnNextStB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof setStationB === 'function' && stations.length) {
                        setStationB((currentStationBIndex + 1) % stations.length);
                    }
                });
                const btnRandB = document.getElementById('topmenu-station-b-random');
                if (btnRandB) btnRandB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    try { pickRandomStationB(); } catch (_) {}
                });
                const btnExpandB = document.getElementById('topmenu-stations-b-expand');
                if (btnExpandB) btnExpandB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof deckBHasLoadedContent === 'function' && deckBHasLoadedContent()) return;
                    topMenuStationsBSplitManual = !topMenuStationsBSplitManual;
                    try { syncTopMenuStationsLayout(); } catch (_) {}
                });
            } catch(e) {}
        })();
        // expose for inline handlers
        window.hideWebmSettingsPanel = hideWebmSettingsPanel;
        function scheduleWebmSettingsClose() {
            if (webmSettingsTimer) clearTimeout(webmSettingsTimer);
            webmSettingsTimer = setTimeout(() => {
                webmSettingsPanel.style.opacity = '0';
                webmSettingsPanel.style.pointerEvents = 'none';
                setTimeout(() => { hideWebmSettingsPanel(); }, 1200);
            }, 30000);
        }
        if (webmCloseBtn) {
            webmCloseBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
                hideWebmSettingsPanel();
            });
            // Support pointerdown for cases where click is swallowed by other handlers
            webmCloseBtn.addEventListener('pointerdown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
                hideWebmSettingsPanel();
            });
            // Also close on right-click/contextmenu for consistency with toggle behavior
            webmCloseBtn.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
                hideWebmSettingsPanel();
            });
        }
        // Event delegation fallback in case the button is replaced dynamically
        webmSettingsPanel.addEventListener('click', (e) => {
            const t = e.target;
            if (!t) return;
            const btn = (t.id === 'btn-webm-close') ? t : (t.closest ? t.closest('#btn-webm-close') : null);
            if (btn) {
                e.preventDefault();
                e.stopPropagation();
                if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
                hideWebmSettingsPanel();
            }
        });
        webmSettingsPanel.addEventListener('pointerdown', (e) => {
            const t = e.target;
            if (!t) return;
            const btn = (t.id === 'btn-webm-close') ? t : (t.closest ? t.closest('#btn-webm-close') : null);
            if (btn) {
                e.preventDefault();
                e.stopPropagation();
                if (webmSettingsTimer) { clearTimeout(webmSettingsTimer); webmSettingsTimer = null; }
                hideWebmSettingsPanel();
            }
        });
        // Allow closing the Avatar Settings with Escape for reliability
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !webmSettingsPanel.classList.contains('display-none')) {
                hideWebmSettingsPanel();
            }
        });
        // Close when clicking outside the avatar settings panel
        document.addEventListener('click', (e) => {
            if (webmSettingsPanel.classList.contains('display-none')) return;
            const target = e.target;
            // If click is inside the panel, ignore
            if (webmSettingsPanel.contains(target)) return;
            // Otherwise, close
            hideWebmSettingsPanel();
        }, true);
        function syncAllWebmDupKnobs() {
            const n = Math.max(0, Math.min(2, Math.round(Number(webmSettings.duplicates) || 0)));
            webmSettings.duplicates = n;
            ['knob-webm-dup', 'knob-avatar-dup'].forEach((id) => {
                const el = document.getElementById(id);
                if (!el) return;
                setKnobUi(el, 0, 2, n);
                const valTooltip = el.querySelector('.knob-value');
                if (valTooltip) valTooltip.textContent = String(n);
            });
        }
        function syncWebmDupSpacingInputs() {
            const pct = Math.round((Number(webmSettings.duplicateSpacing) || WEBM_DEFAULT_DUP_SPACING) * 100);
            if (inpWebmDupSpacing) inpWebmDupSpacing.value = String(pct);
            const avatarSpacing = document.getElementById('avatar-inp-dup-spacing');
            if (avatarSpacing) avatarSpacing.value = String(pct);
        }
        function onWebmDupCountChange() {
            webmSettings.duplicates = Math.max(0, Math.min(2, Math.round(Number(webmSettings.duplicates) || 0)));
            syncAllWebmDupKnobs();
            applyWebmSettings();
            try { scheduleWebmSettingsClose(); } catch (_) {}
        }
        function bindWebmDupKnob(id) {
            bindKnob(id, 0, 2, webmSettings.duplicates, webmSettings, 'duplicates', () => {
                onWebmDupCountChange();
            });
        }
        try { bindWebmDupKnob('knob-webm-dup'); } catch (_) {}
        try { bindWebmDupKnob('knob-avatar-dup'); } catch (_) {}
        document.getElementById('btn-webm-reset').addEventListener('click', () => {
            webmSettings.scaleVw = WEBM_DEFAULT_SCALE_VW;
            webmSettings.posXvw = 50;
            webmSettings.posYvh = 50;
            webmSettings.rotationDeg = 0;
            webmSettings.playbackRate = 1.0;
            webmSettings.opacity = 0.82;
            webmSettings.duplicates = 0;
            webmSettings.duplicateSpacing = WEBM_DEFAULT_DUP_SPACING;
            inpWebmScale.value = String(WEBM_DEFAULT_SCALE_VW);
            inpWebmX.value = '50';
            inpWebmY.value = '50';
            inpWebmRot.value = '0';
            inpWebmSpeed.value = '1.0';
            inpWebmOpacity.value = '0.82';
            syncWebmDupSpacingInputs();
            syncAllWebmDupKnobs();
            applyWebmSettings();
        });
        // Live-change on input
        function updateWebmSettingsFromInputs() {
            webmSettings.scaleVw = Number(inpWebmScale.value) || WEBM_DEFAULT_SCALE_VW;
            webmSettings.posXvw = Number(inpWebmX.value) || 50;
            webmSettings.posYvh = Number(inpWebmY.value) || 50;
            webmSettings.rotationDeg = Number(inpWebmRot.value) || 0;
            webmSettings.playbackRate = Math.max(0.1, Math.min(4, Number(inpWebmSpeed.value) || 1));
            webmSettings.opacity = Math.max(0, Math.min(1, Number(inpWebmOpacity.value) || 1));
            webmSettings.duplicateSpacing = Math.max(0.15, Math.min(1, (Number(inpWebmDupSpacing?.value) || 60) / 100));
            syncWebmDupSpacingInputs();
            applyWebmSettings();
            scheduleWebmSettingsClose();
        }
        [inpWebmScale, inpWebmX, inpWebmY, inpWebmRot, inpWebmSpeed, inpWebmOpacity, inpWebmDupSpacing].forEach(el => {
            if (el) el.addEventListener('input', updateWebmSettingsFromInputs);
        });
        // Bind auto toggle
        const btnWebmAuto = document.getElementById('btn-webm-auto');
        if (btnWebmAuto) {
            btnWebmAuto.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                setWebmAuto(!webmAutoOn);
            });
        }
        function stashWebmOverlayHome() {
            try {
                if (!webmOverlayEl || webmOverlayEl.__webmHomeParent) return;
                webmOverlayEl.__webmHomeParent = webmOverlayEl.parentElement;
                webmOverlayEl.__webmHomeNext = webmOverlayEl.nextSibling;
            } catch (_) {}
        }
        function restoreWebmOverlayHome() {
            try {
                if (!webmOverlayEl || !webmOverlayEl.__webmHomeParent) return;
                if (webmOverlayEl.parentElement !== webmOverlayEl.__webmHomeParent) {
                    webmOverlayEl.__webmHomeParent.insertBefore(webmOverlayEl, webmOverlayEl.__webmHomeNext);
                }
            } catch (_) {}
        }
        function applyWebmSettings() {
            try {
                if (webmOn && webmAnchorDeckB) {
                    const mount = document.getElementById('dj-deck-b-viz-mount');
                    const fsEl = typeof getDeckBVizFullscreenEl === 'function' ? getDeckBVizFullscreenEl() : null;
                    const mountFs = !!(mount && fsEl && mount.contains(fsEl));
                    const deckCol = document.querySelector('#dj-visual-root .dj-deck-b');
                    let rect = null;
                    if (mountFs) {
                        rect = mount.getBoundingClientRect();
                    } else if (mount) {
                        const mr = mount.getBoundingClientRect();
                        if (mr.width >= 8 && mr.height >= 8) rect = mr;
                    }
                    if (!rect && deckCol) {
                        rect = deckCol.getBoundingClientRect();
                    }
                    stashWebmOverlayHome();
                    try {
                        if (webmOverlayEl) {
                            if (mountFs) {
                                if (webmOverlayEl.parentElement !== mount) mount.appendChild(webmOverlayEl);
                                webmOverlayEl.classList.add('webm-overlay--deck-b-viz-fs');
                            } else {
                                restoreWebmOverlayHome();
                                webmOverlayEl.classList.remove('webm-overlay--deck-b-viz-fs');
                            }
                        }
                    } catch (_) {}
                    if (rect && rect.width >= 8 && rect.height >= 8 && (fsEl || deckCol)) {
                        bindWebmDeckBLayoutWatchers();
                        const pxOff = ((Number(webmSettings.posXvw) || 50) - 50) / 100 * rect.width;
                        const pyOff = ((Number(webmSettings.posYvh) || 50) - 50) / 100 * rect.height;
                        const sw = Number(webmSettings.scaleVw) || WEBM_DEFAULT_SCALE_VW;
                        const wPx = Math.min(rect.width * 0.94, Math.max(72, rect.width * 0.34 * (sw / 50)));
                        if (mountFs) {
                            webmOverlayEl.style.position = 'absolute';
                            webmOverlayEl.style.left = '50%';
                            webmOverlayEl.style.top = '50%';
                            webmOverlayEl.style.transform = `translate(calc(-50% + ${pxOff}px), calc(-50% + ${pyOff}px))`;
                            webmOverlayEl.style.zIndex = '8';
                        } else {
                            const cx = rect.left + rect.width * 0.5;
                            const cy = rect.top + rect.height * 0.5;
                            const fixLeft = cx + pxOff;
                            const fixTop = cy + pyOff;
                            webmOverlayEl.style.position = 'fixed';
                            webmOverlayEl.style.left = `${fixLeft}px`;
                            webmOverlayEl.style.top = `${fixTop}px`;
                            webmOverlayEl.style.transform = 'translate(-50%, -50%)';
                            webmOverlayEl.style.zIndex = '';
                        }
                        webmVideoEl.style.width = `${wPx}px`;
                        webmVideoEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                        webmVideoEl.style.opacity = String(webmSettings.opacity);
                        try { webmVideoEl.playbackRate = webmSettings.playbackRate; } catch {}
                        const dup = webmSettings.duplicates;
                        const spacingFactor = Math.max(0.15, Math.min(1, Number(webmSettings.duplicateSpacing) || WEBM_DEFAULT_DUP_SPACING));
                        const offsetPx = wPx * spacingFactor;
                        if (dup >= 1) {
                            webmVideoLeftEl.classList.remove('display-none');
                            webmVideoLeftEl.style.width = `${wPx}px`;
                            webmVideoLeftEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                            webmVideoLeftEl.style.opacity = String(webmSettings.opacity);
                            try { webmVideoLeftEl.playbackRate = webmSettings.playbackRate; } catch {}
                        } else {
                            webmVideoLeftEl.classList.add('display-none');
                        }
                        if (dup >= 2) {
                            webmVideoRightEl.classList.remove('display-none');
                            webmVideoRightEl.style.width = `${wPx}px`;
                            webmVideoRightEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                            webmVideoRightEl.style.opacity = String(webmSettings.opacity);
                            try { webmVideoRightEl.playbackRate = webmSettings.playbackRate; } catch {}
                        } else {
                            webmVideoRightEl.classList.add('display-none');
                        }
                        webmVideoEl.style.display = 'block';
                        webmVideoLeftEl.style.display = dup >= 1 ? 'block' : 'none';
                        webmVideoRightEl.style.display = dup >= 2 ? 'block' : 'none';
                        webmVideoEl.style.margin = '0';
                        webmVideoLeftEl.style.margin = '0';
                        webmVideoRightEl.style.margin = '0';
                        webmVideoLeftEl.style.position = 'absolute';
                        webmVideoRightEl.style.position = 'absolute';
                        webmVideoEl.style.position = 'relative';
                        webmVideoLeftEl.style.left = `-${offsetPx}px`;
                        webmVideoLeftEl.style.top = '0';
                        webmVideoRightEl.style.left = `${offsetPx}px`;
                        webmVideoRightEl.style.top = '0';
                        try { if (webmVideoEl && webmVideoEl.paused) webmVideoEl.play().catch(() => {}); } catch (_) {}
                        return;
                    }
                    webmAnchorDeckB = false;
                }
                try {
                    if (webmOverlayEl) {
                        webmOverlayEl.classList.remove('webm-overlay--deck-b-viz-fs');
                        restoreWebmOverlayHome();
                    }
                } catch (_) {}
                unbindWebmDeckBLayoutWatchers();
                try {
                    webmOverlayEl.style.position = '';
                    webmOverlayEl.style.left = '';
                    webmOverlayEl.style.top = '';
                    webmOverlayEl.style.transform = '';
                } catch (_) {}
                // central (viewport vw/vh layout)
                webmVideoEl.style.width = `${webmSettings.scaleVw}vw`;
                webmOverlayEl.style.left = `${webmSettings.posXvw}vw`;
                webmOverlayEl.style.top = `${webmSettings.posYvh}vh`;
                webmOverlayEl.style.transform = `translate(-50%, -50%)`;
                webmVideoEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                webmVideoEl.style.opacity = String(webmSettings.opacity);
                try { webmVideoEl.playbackRate = webmSettings.playbackRate; } catch {}
                const dup = webmSettings.duplicates;
                const spacingFactor = Math.max(0.15, Math.min(1, Number(webmSettings.duplicateSpacing) || WEBM_DEFAULT_DUP_SPACING));
                const offsetVw = webmSettings.scaleVw * spacingFactor;
                if (dup >= 1) {
                    webmVideoLeftEl.classList.remove('display-none');
                    webmVideoLeftEl.style.width = `${webmSettings.scaleVw}vw`;
                    webmVideoLeftEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                    webmVideoLeftEl.style.opacity = String(webmSettings.opacity);
                    try { webmVideoLeftEl.playbackRate = webmSettings.playbackRate; } catch {}
                } else {
                    webmVideoLeftEl.classList.add('display-none');
                }
                if (dup >= 2) {
                    webmVideoRightEl.classList.remove('display-none');
                    webmVideoRightEl.style.width = `${webmSettings.scaleVw}vw`;
                    webmVideoRightEl.style.transform = `rotate(${webmSettings.rotationDeg}deg)`;
                    webmVideoRightEl.style.opacity = String(webmSettings.opacity);
                    try { webmVideoRightEl.playbackRate = webmSettings.playbackRate; } catch {}
                } else {
                    webmVideoRightEl.classList.add('display-none');
                }
                webmVideoEl.style.display = 'block';
                webmVideoLeftEl.style.display = dup >= 1 ? 'block' : 'none';
                webmVideoRightEl.style.display = dup >= 2 ? 'block' : 'none';
                webmVideoEl.style.margin = '0';
                webmVideoLeftEl.style.margin = '0';
                webmVideoRightEl.style.margin = '0';
                webmVideoLeftEl.style.position = 'absolute';
                webmVideoRightEl.style.position = 'absolute';
                webmVideoEl.style.position = 'relative';
                webmVideoLeftEl.style.left = `-${offsetVw}vw`;
                webmVideoLeftEl.style.top = `0`;
                webmVideoRightEl.style.left = `${offsetVw}vw`;
                webmVideoRightEl.style.top = `0`;
            } catch (_) {}
            try { saveWebmSettingsToStorage(); } catch (_) {}
        }
        // --- MODE SHUFFLE ---
        function setModeShuffle(on) {
            modeShuffleOn = on;
            if(modeShuffleTimer) { clearTimeout(modeShuffleTimer); modeShuffleTimer = null; }
            if(modeShuffleOn) {
                scheduleModeShuffle();
            }
        }
        function scheduleModeShuffle() {
            if(!modeShuffleOn) return;
            modeShuffleTimer = setTimeout(() => {
                loadMode(state.currentModeIdx + 1);
                scheduleModeShuffle();
            }, 30000);
        }


    document.addEventListener('keydown', (e) => {
        // Ignore typing in inputs / text areas
        const tag = document.activeElement?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

        // Ignore modifier keys
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        // Y key toggles TEXT-IN Auto
        if (e.key === 'y' || e.key === 'Y') {
            const autoBtn = document.getElementById('ti-auto');
            if (!autoBtn) return;

            autoBtn.click(); // reuse existing logic
            e.preventDefault();
        }
    });

try { hideStationBannerPermanently(); } catch (_) {}
exposeAppBindingsToGlobal();
