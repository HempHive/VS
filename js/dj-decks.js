/* Extracted from app.js — dj-decks. Uses globals via globalThis (see app.js exposeVsGlobals). */
        function closeDjBeatPadLoopMenu() {
            if (__djBeatPadLoopMenuCleanup) {
                try { __djBeatPadLoopMenuCleanup(); } catch (_) {}
                __djBeatPadLoopMenuCleanup = null;
            }
            const el = document.getElementById('dj-beat-pad-loop-menu');
            if (el && el.parentNode) el.parentNode.removeChild(el);
        }

        function applyDjBeatPadLoopVisual(btn, slot) {
            if (!btn || !slot) return;
            const loop = !!slot.loopMode;
            btn.classList.toggle('dj-beat-pad-mode-loop', loop);
            btn.classList.toggle('dj-beat-pad-mode-once', !loop);
            const base = ((btn.textContent || '').trim()) || slot.label || 'Sample';
            btn.title = loop
                ? (base + ' · Loop · Long-press: loop · Right-click: load sample')
                : (base + ' · Solo · Long-press: loop · Right-click: load sample');
        }

        function openDjBeatPadLoopMenu(clientX, clientY, slot, btn) {
            closeDjBeatPadLoopMenu();
            const menu = document.createElement('div');
            menu.id = 'dj-beat-pad-loop-menu';
            menu.className = 'dj-beat-pad-loop-menu';
            menu.setAttribute('role', 'menu');

            const mkItem = (label, modeLoop) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'dj-beat-pad-loop-menu-item';
                if (!!slot.loopMode === modeLoop) {
                    b.classList.add('dj-beat-pad-loop-menu-item--active');
                    b.setAttribute('aria-current', 'true');
                }
                b.textContent = label;
                b.addEventListener('click', (ev) => {
                    try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                    slot.loopMode = modeLoop;
                    applyDjBeatPadLoopVisual(btn, slot);
                    closeDjBeatPadLoopMenu();
                });
                return b;
            };
            menu.appendChild(mkItem('Loop', true));
            menu.appendChild(mkItem('Solo', false));

            document.body.appendChild(menu);
            const pad = 8;
            const mw = menu.offsetWidth;
            const mh = menu.offsetHeight;
            let x = clientX;
            let y = clientY;
            if (x + mw + pad > window.innerWidth) x = Math.max(pad, window.innerWidth - mw - pad);
            if (y + mh + pad > window.innerHeight) y = Math.max(pad, window.innerHeight - mh - pad);
            if (x < pad) x = pad;
            if (y < pad) y = pad;
            menu.style.left = x + 'px';
            menu.style.top = y + 'px';

            const onDoc = (ev) => {
                if (menu.contains(ev.target)) return;
                closeDjBeatPadLoopMenu();
            };
            const onKey = (ev) => {
                if (ev.key === 'Escape') closeDjBeatPadLoopMenu();
            };
            requestAnimationFrame(() => {
                document.addEventListener('click', onDoc, true);
                document.addEventListener('keydown', onKey, true);
            });
            __djBeatPadLoopMenuCleanup = () => {
                document.removeEventListener('click', onDoc, true);
                document.removeEventListener('keydown', onKey, true);
            };
        }

        /** Map horizontal pointer position on a range input to its value (for right-click cut preview). */
        function crossfadeValueFromPointerOnRange(rangeEl, clientX) {
            if (!rangeEl || !Number.isFinite(Number(clientX))) return 0;
            const rect = rangeEl.getBoundingClientRect();
            const w = Math.max(1, rect.width);
            const t = (Number(clientX) - rect.left) / w;
            const min = Number(rangeEl.min) || 0;
            const max = Number(rangeEl.max) || 1;
            return Math.max(min, Math.min(max, min + t * (max - min)));
        }

        /** Full-screen DJ UI: dual decks (A/B radio), crossfader, FX mirrors Mixer Settings */
        class DjDecksEngine {
            constructor() {
                this.name = 'DJ Decks';
                this.resizeHandler = this.onResize.bind(this);
                this.abortCtrl = null;
                this.animId = null;
                this.root = null;
                this.els = {};
                this.angleA = 0;
                this.angleB = 0;
                this.bufA = null;
                this.bufB = null;
                this.deckBVizMode = 'idle';
                this.deckBVizPmAnimId = null;
                this.deckBVizPmCycleTimeout = null;
                this.deckBVizPmVisualizer = null;
                this.deckBVizPmCanvas = null;
                this.deckBVizPmResize = null;
                this.deckBVizBarsAnimId = null;
                this.deckBVideoList = [];
                this.deckBVideoIndex = 0;
                this.deckBVideoEl = null;
                this.deckBVideoElA = null;
                this.deckBVideoElB = null;
                this.deckBVideoElQ = null;
                this.deckBVideoStackEl = null;
                this._deckBVideoUiAbortCtrl = null;
                this.autoFadeRafId = null;
                this.autoFadeTargetDeck = null;
                /** When true, `ensureCrossfadeDeckPlayback` must not restart decks (long Space hard-pause both). */
                this.suppressEnsureCrossfadeDeckPlayback = false;
                this.clearSuppressEnsureCrossfadeDeckPlayback = () => {
                    this.suppressEnsureCrossfadeDeckPlayback = false;
                };
                this.autoFadeHoldTimer = null;
                this.autoMixTimerId = null;
                this.autoMixPreloadTimerId = null;
                this.autoMixHoldTimer = null;
                this.autoMixSessionLimitTimerId = null;
                this.autoMixSessionRemainingIntervalId = null;
                this.autoMixNextFadeAt = 0;
                this.autoMixNextFadeIntervalId = null;
                /** Next deck AUTO-MIX will fade toward (alternates a/b each completed cycle). */
                this.autoMixNextTargetDeck = null;
                this.masterFadeOutRafId = null;
                this._panelPointerPos = null;
                this.deckBVizResizeObs = null;
                this.deckVolResizeObs = null;
                this._deckLayoutResizeRaf = null;
                this.deckBQueueVisible = false;
                this.deckBMediaPanelVisible = false;
                /** After Deck A "Audio Bars" moves Deck B from ProjectM → in-panel bars, next Deck A "Audio Bars" loads main Audio Bars. */
                this._deckAAudioBarsMainOnNextClick = false;
                /** After Deck A "Stations" moves Deck B from file QUEUE → station cycle + video queue, next Deck A "Stations" toggles the top Radio Stations menu. */
                this._deckAStationsTopMenuOnNextClick = false;
                this.beatMapSlots = null;
                this.headClockTimerId = null;
                /**
                 * DJ deck columns: stacked (A top / B bottom) vs side‑by‑side. Updated in onResize().
                 * Hysteresis keeps resize from oscillating at the breakpoint.
                 */
                this._djLandscapeLayout = false;
            }

            tickHeadDatetime() {
                try {
                    const el = this.root && this.root.querySelector('#dj-head-datetime');
                    if (!el) return;
                    const d = new Date();
                    el.textContent = d.toLocaleString(undefined, {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                    });
                } catch (_) {}
            }

            wireFooterBtnLiveFromSlider(slider, btn, opts, sig) {
                if (!slider || !btn || !sig) return;
                const restLabel = opts && opts.restLabel != null ? String(opts.restLabel) : String(btn.textContent || '').trim();
                const liveClass = (opts && opts.liveClass) || '';
                const format = (opts && opts.format) || ((raw) => String(raw));
                const restoreMs = (opts && opts.restoreMs) || 450;
                let dragging = false;
                let restoreTimer = null;
                const showLive = () => {
                    btn.textContent = format(slider.value);
                    if (liveClass) btn.classList.add(liveClass);
                };
                const restore = () => {
                    dragging = false;
                    btn.textContent = restLabel;
                    if (liveClass) btn.classList.remove(liveClass);
                };
                const scheduleRestore = () => {
                    if (restoreTimer != null) clearTimeout(restoreTimer);
                    restoreTimer = setTimeout(() => {
                        restoreTimer = null;
                        restore();
                    }, restoreMs);
                };
                const onStart = () => {
                    dragging = true;
                    if (restoreTimer != null) {
                        clearTimeout(restoreTimer);
                        restoreTimer = null;
                    }
                    showLive();
                };
                try {
                    slider.addEventListener('pointerdown', onStart, sig);
                    slider.addEventListener('input', () => { if (dragging) showLive(); }, sig);
                    slider.addEventListener('pointerup', scheduleRestore, sig);
                    slider.addEventListener('pointercancel', scheduleRestore, sig);
                    slider.addEventListener('touchend', scheduleRestore, { signal: sig.signal, passive: true });
                    slider.addEventListener('change', scheduleRestore, sig);
                } catch (_) {}
            }

            bindDeckBeatMaps(root, sig) {
                const REF_BPM = 120;
                const BEAT_MAP_DEFAULTS = [
                    { label: 'High', src: 'assets/audio/High.wav', defaultLoop: true },
                    { label: 'Drum', src: 'assets/audio/Drum.wav', defaultLoop: true },
                    { label: 'Clap', src: 'assets/audio/Clap.wav', defaultLoop: true },
                    { label: 'Perc', src: 'assets/audio/Perc.wav', defaultLoop: true },
                    { label: 'Tom', src: 'assets/audio/Tom.wav', defaultLoop: true },
                    { label: 'WAA', src: 'assets/audio/wav4.mp3', defaultLoop: false },
                    { label: 'WAAA', src: 'assets/audio/wav5.mp3', defaultLoop: false },
                    { label: 'AIR📢', src: 'assets/audio/wav6.mp3', defaultLoop: false },
                    { label: 'FX1', src: 'assets/audio/wav1.mp3', defaultLoop: false },
                    { label: 'FX2', src: 'assets/audio/wav2.mp3', defaultLoop: false },
                    { label: 'KEY', src: 'assets/audio/Key.wav', defaultLoop: false },
                    { label: 'KICK', src: 'assets/audio/Kick.wav', defaultLoop: false },
                    { label: 'BELL', src: 'assets/audio/Bell.wav', defaultLoop: false },
                    { label: 'RIM', src: 'assets/audio/Rim.wav', defaultLoop: false },
                    { label: 'HAR', src: 'assets/audio/Vocal.wav', defaultLoop: false },
                ];
                /** Deck B: independent samples/labels; shared short placeholder until user loads clips */
                const BEAT_MAP_DEFAULTS_B = [
                    { label: 'CHR', src: 'assets/audio/CHR.wav', defaultLoop: false },
                    { label: 'CYM', src: 'assets/audio/CYM.wav', defaultLoop: false },
                    { label: 'SYN', src: 'assets/audio/SYN.wav', defaultLoop: false },
                    { label: 'HARM', src: 'assets/audio/HARM.wav', defaultLoop: false },
                    { label: 'HARP', src: 'assets/audio/HARP.wav', defaultLoop: false },
                    { label: 'FX3', src: 'assets/audio/FX3.wav', defaultLoop: false },
                    { label: 'FX4', src: 'assets/audio/FX4.wav', defaultLoop: false },
                    { label: 'FX5', src: 'assets/audio/FX5.wav', defaultLoop: false },
                    { label: 'FX6', src: 'assets/audio/FX6.wav', defaultLoop: false },
                    { label: 'FX7', src: 'assets/audio/FX7.wav', defaultLoop: false },
                    { label: 'SRE', src: 'assets/audio/SRE.wav', defaultLoop: false },
                    { label: 'BME', src: 'assets/audio/BME.wav', defaultLoop: false },
                    { label: 'HRN', src: 'assets/audio/HRN.wav', defaultLoop: false },
                    { label: 'CRW', src: 'assets/audio/CRW.wav', defaultLoop: false },
                    { label: 'CHG', src: 'assets/audio/CHG.wav', defaultLoop: false },
                ];
                this.beatMapSlots = { a: [], b: [] };

                const setupDeck = (deckKey) => {
                    const prefix = deckKey === 'a' ? 'dj-a' : 'dj-b';
                    const defs = deckKey === 'b' ? BEAT_MAP_DEFAULTS_B : BEAT_MAP_DEFAULTS;
                    const slots = [];
                    for (let i = 0; i < defs.length; i++) {
                        const el = new Audio();
                        el.preload = 'auto';
                        el.src = defs[i].src;
                        try { el.muted = false; el.volume = 1; } catch (_) {}
                        slots.push({
                            audio: el,
                            sourceNode: null,
                            blobUrl: '',
                            defaultSrc: defs[i].src,
                            label: defs[i].label,
                            loopMode: defs[i].defaultLoop !== undefined ? !!defs[i].defaultLoop : true,
                            _beatEnded: null,
                        });
                    }
                    this.beatMapSlots[deckKey] = slots;

                    const bpmSlider = root.querySelector('#' + prefix + '-beat-bpm');
                    const syncBtn = root.querySelector('#' + prefix + '-beat-sync');
                    const fi = root.querySelector('#' + prefix + '-beat-file');

                    const getRate = () => {
                        const v = bpmSlider ? parseFloat(bpmSlider.value) : REF_BPM;
                        const bpm = Number.isFinite(v) ? v : REF_BPM;
                        return Math.max(0.25, Math.min(4, bpm / REF_BPM));
                    };

                    const ensureGraph = (slot) => {
                        try { initAudio(); } catch (_) {}
                        if (!state.audioCtx || !slot.audio || slot.sourceNode) return;
                        try {
                            slot.sourceNode = state.audioCtx.createMediaElementSource(slot.audio);
                            slot.sourceNode.connect(state.mixInput);
                        } catch (_) {}
                    };

                    const applyRates = () => {
                        const r = getRate();
                        slots.forEach((s) => {
                            try {
                                if (s.audio && !s.audio.paused) s.audio.playbackRate = r;
                            } catch (_) {}
                        });
                    };

                    if (bpmSlider) {
                        bpmSlider.addEventListener('input', () => {
                            applyRates();
                        }, sig);
                    }
                    if (syncBtn && bpmSlider) {
                        this.wireFooterBtnLiveFromSlider(bpmSlider, syncBtn, {
                            restLabel: 'Sync',
                            liveClass: 'dj-beatmap-sync--live',
                            format: (v) => String(Math.round(parseFloat(v) || REF_BPM))
                        }, sig);
                    }

                    if (syncBtn) {
                        syncBtn.addEventListener('click', () => {
                            try {
                                initAudio();
                                const raw = deckKey === 'a' ? bpmSmoothA : bpmSmoothB;
                                const bpm = raw && raw > 40 && raw < 300 ? Math.round(raw) : null;
                                if (bpm && bpmSlider) {
                                    bpmSlider.value = String(bpm);
                                    applyRates();
                                }
                            } catch (_) {}
                        }, sig);
                        syncBtn.title = deckKey === 'a'
                            ? 'Match tempo to Mix panel BPM A (enable BPM A first)'
                            : 'Match tempo to Mix panel BPM B (enable BPM B first)';
                    }

                    let loadSlotIdx = -1;
                    if (fi) {
                        fi.addEventListener('change', (ev) => {
                            const f = ev.target.files && ev.target.files[0];
                            try { ev.target.value = ''; } catch (_) {}
                            if (!f || loadSlotIdx < 0) return;
                            const slot = slots[loadSlotIdx];
                            try { slot.audio.pause(); } catch (_) {}
                            try {
                                if (slot.blobUrl) URL.revokeObjectURL(slot.blobUrl);
                            } catch (_) {}
                            slot.blobUrl = '';
                            try {
                                slot.blobUrl = URL.createObjectURL(f);
                                slot.audio.src = slot.blobUrl;
                            } catch (_) {}
                            const btn = root.querySelector(`button[data-beat-deck="${deckKey}"][data-beat-slot="${loadSlotIdx}"]`);
                            if (btn) {
                                const nm = (f.name || 'sample').replace(/\.[^/.]+$/, '');
                                btn.textContent = nm.length > 8 ? nm.slice(0, 7) + '…' : nm;
                                btn.title = f.name || 'Loaded sample';
                                try {
                                    const sl = slots[loadSlotIdx];
                                    if (sl && typeof applyDjBeatPadLoopVisual === 'function') applyDjBeatPadLoopVisual(btn, sl);
                                } catch (_) {}
                            }
                            loadSlotIdx = -1;
                        }, sig);
                    }

                    const pads = root.querySelectorAll(`button[data-beat-deck="${deckKey}"][data-beat-slot]`);
                    pads.forEach((btn) => {
                        const idx = parseInt(btn.getAttribute('data-beat-slot'), 10);
                        if (Number.isNaN(idx) || !slots[idx]) return;
                        try {
                            if (typeof applyDjBeatPadLoopVisual === 'function') applyDjBeatPadLoopVisual(btn, slots[idx]);
                        } catch (_) {}
                        let holdT = null;
                        let longFired = false;

                        const clearEnded = (slot) => {
                            try {
                                if (slot._beatEnded) {
                                    slot.audio.removeEventListener('ended', slot._beatEnded);
                                    slot._beatEnded = null;
                                }
                            } catch (_) {}
                        };

                        const togglePlay = () => {
                            const slot = slots[idx];
                            ensureGraph(slot);
                            const rate = getRate();
                            const loop = !!slot.loopMode;
                            try { slot.audio.loop = loop; } catch (_) {}
                            if (!slot.audio.paused) {
                                try { slot.audio.pause(); } catch (_) {}
                                btn.classList.remove('on');
                                clearEnded(slot);
                                return;
                            }
                            clearEnded(slot);
                            try { slot.audio.playbackRate = rate; } catch (_) {}
                            try { slot.audio.currentTime = 0; } catch (_) {}
                            if (!loop) {
                                const onEnd = () => {
                                    try { btn.classList.remove('on'); } catch (_) {}
                                    clearEnded(slot);
                                };
                                slot._beatEnded = onEnd;
                                try { slot.audio.addEventListener('ended', onEnd); } catch (_) {}
                            }
                            slot.audio.play().then(() => {
                                btn.classList.add('on');
                            }).catch(() => {});
                        };

                        btn.addEventListener('click', (e) => {
                            if (longFired) {
                                longFired = false;
                                return;
                            }
                            try { e.preventDefault(); } catch (_) {}
                            togglePlay();
                        }, sig);

                        btn.addEventListener('contextmenu', (e) => {
                            try {
                                e.preventDefault();
                                e.stopPropagation();
                            } catch (_) {}
                            longFired = true;
                            loadSlotIdx = idx;
                            try { if (fi) fi.click(); } catch (_) {}
                        }, sig);

                        btn.addEventListener('pointerdown', (e) => {
                            if (e.button !== 0) return;
                            longFired = false;
                            holdT = setTimeout(() => {
                                holdT = null;
                                longFired = true;
                                const slot = slots[idx];
                                if (!slot) return;
                                slot.loopMode = true;
                                try { applyDjBeatPadLoopVisual(btn, slot); } catch (_) {}
                                ensureGraph(slot);
                                clearEnded(slot);
                                try { slot.audio.loop = true; } catch (_) {}
                                try { slot.audio.playbackRate = getRate(); } catch (_) {}
                                if (slot.audio.paused) {
                                    try { slot.audio.currentTime = 0; } catch (_) {}
                                    slot.audio.play().then(() => {
                                        try { btn.classList.add('on'); } catch (_) {}
                                    }).catch(() => {});
                                }
                            }, 550);
                        }, sig);

                        const cancelHold = () => {
                            if (holdT) {
                                clearTimeout(holdT);
                                holdT = null;
                            }
                        };
                        btn.addEventListener('pointerup', cancelHold, sig);
                        btn.addEventListener('pointerleave', cancelHold, sig);
                        btn.addEventListener('pointercancel', cancelHold, sig);
                    });
                };

                setupDeck('a');
                setupDeck('b');
            }

            tearDownDeckBeatMaps() {
                try {
                    try { if (typeof closeDjBeatPadLoopMenu === 'function') closeDjBeatPadLoopMenu(); } catch (_) {}
                    if (!this.beatMapSlots) return;
                    ['a', 'b'].forEach((dk) => {
                        const arr = this.beatMapSlots[dk];
                        if (!arr) return;
                        arr.forEach((slot) => {
                            try { slot.audio.pause(); } catch (_) {}
                            try {
                                if (slot.blobUrl) URL.revokeObjectURL(slot.blobUrl);
                            } catch (_) {}
                            slot.blobUrl = '';
                        });
                    });
                    this.beatMapSlots = null;
                } catch (_) {}
            }

            /**
             * Programmatic trigger for a beat-map sample slot. Mirrors the same audio graph
             * setup, BPM-derived playback rate, loop handling, and pad "on" class behavior used
             * by the on-screen pads, so keyboard shortcuts feel identical to clicks.
             *
             * @param {('a'|'b')} deckKey
             * @param {number} slotIdx
             * @param {('start'|'stop')} action  'start' (also used for "restart") plays the slot
             *                                   from time 0; 'stop' cuts the audio and clears UI.
             * @returns {boolean} true if the slot exists and the action was applied.
             */
            triggerBeatPadKey(deckKey, slotIdx, action) {
                try {
                    if (!this.beatMapSlots) return false;
                    const arr = this.beatMapSlots[deckKey];
                    if (!arr) return false;
                    const slot = arr[slotIdx];
                    if (!slot || !slot.audio) return false;
                    const root = this.root || document;
                    const btn = root.querySelector(`button[data-beat-deck="${deckKey}"][data-beat-slot="${slotIdx}"]`);

                    const clearEnded = () => {
                        try {
                            if (slot._beatEnded) {
                                slot.audio.removeEventListener('ended', slot._beatEnded);
                                slot._beatEnded = null;
                            }
                        } catch (_) {}
                    };

                    if (action === 'stop') {
                        try { slot.audio.pause(); } catch (_) {}
                        try { slot.audio.currentTime = 0; } catch (_) {}
                        clearEnded();
                        try { if (btn) btn.classList.remove('on'); } catch (_) {}
                        return true;
                    }

                    // 'start' / 'restart': always (re)play from the beginning so repeated key
                    // presses retrigger the sample instead of toggling pause.
                    try { initAudio(); } catch (_) {}
                    if (state && state.audioCtx && !slot.sourceNode) {
                        try {
                            slot.sourceNode = state.audioCtx.createMediaElementSource(slot.audio);
                            slot.sourceNode.connect(state.mixInput);
                        } catch (_) {}
                    }

                    const REF_BPM = 120;
                    const slider = root.querySelector('#' + (deckKey === 'a' ? 'dj-a' : 'dj-b') + '-beat-bpm');
                    const v = slider ? parseFloat(slider.value) : REF_BPM;
                    const bpm = Number.isFinite(v) ? v : REF_BPM;
                    const rate = Math.max(0.25, Math.min(4, bpm / REF_BPM));
                    const loop = !!slot.loopMode;
                    try { slot.audio.loop = loop; } catch (_) {}
                    clearEnded();
                    try { slot.audio.playbackRate = rate; } catch (_) {}
                    try { slot.audio.pause(); } catch (_) {}
                    try { slot.audio.currentTime = 0; } catch (_) {}
                    if (!loop) {
                        const onEnd = () => {
                            try { if (btn) btn.classList.remove('on'); } catch (_) {}
                            clearEnded();
                        };
                        slot._beatEnded = onEnd;
                        try { slot.audio.addEventListener('ended', onEnd); } catch (_) {}
                    }
                    slot.audio.play().then(() => {
                        try { if (btn) btn.classList.add('on'); } catch (_) {}
                    }).catch(() => {});
                    return true;
                } catch (_) {
                    return false;
                }
            }

            hideDeckBQueueView() {
                try {
                    this.deckBQueueVisible = false;
                    const stage = this.root && this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.remove('dj-deck-b-queue-mode');
                    const panel = this.root && this.root.querySelector('#dj-deck-b-queue-panel');
                    if (panel) panel.classList.add('display-none');
                } catch (_) {}
                try {
                    this.syncDeckBVisualButtons();
                    this.syncFxLightsFromState();
                } catch (_) {}
            }

            hideDeckBMediaView() {
                this._deckAStationsTopMenuOnNextClick = false;
                try {
                    this.deckBMediaPanelVisible = false;
                    const stage = this.root && this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.remove('dj-deck-b-media-mode');
                    const panel = this.root && this.root.querySelector('#dj-deck-b-media-panel');
                    if (panel) panel.classList.add('display-none');
                } catch (_) {}
                try {
                    this.syncDeckBVisualButtons();
                    this.syncFxLightsFromState();
                } catch (_) {}
            }

            showDeckBMediaView() {
                if (!this.root) return;
                try { this.tearDownDeckBViz({ skipMediaStrip: true }); } catch (_) {}
                this.deckBQueueVisible = false;
                this.deckBMediaPanelVisible = true;
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    const panel = this.root.querySelector('#dj-deck-b-media-panel');
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                    if (stage) {
                        stage.classList.remove('dj-deck-b-queue-mode');
                        stage.classList.add('dj-deck-b-media-mode');
                    }
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                    if (mount) mount.innerHTML = '';
                    if (ph) ph.classList.add('display-none');
                    if (panel) panel.classList.remove('display-none');
                    this.refreshQueueUi();
                    this.syncDeckBVisualButtons();
                    this.syncFxLightsFromState();
                } catch (_) {}
            }

            toggleDeckBMediaPanel() {
                if (this.deckBMediaPanelVisible) this.hideDeckBMediaView();
                else this.showDeckBMediaView();
            }

            showDeckBQueueView() {
                if (!this.root) return;
                try { this.tearDownDeckBViz({ skipQueueStrip: true }); } catch (_) {}
                this.deckBMediaPanelVisible = false;
                this.deckBQueueVisible = true;
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    const panel = this.root.querySelector('#dj-deck-b-queue-panel');
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                    if (stage) stage.classList.add('dj-deck-b-queue-mode');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                    if (mount) mount.innerHTML = '';
                    if (ph) ph.classList.add('display-none');
                    if (panel) panel.classList.remove('display-none');
                    this.refreshQueueUi();
                    this.syncDeckBVisualButtons();
                    this.syncFxLightsFromState();
                } catch (_) {}
            }

            isDeckBVisualModeActive() {
                const m = this.deckBVizMode || 'idle';
                return m === 'bars' || m === 'projectm' || m === 'blank' || m === 'video' || m === 'karaoke' || m === 'kbop';
            }

            toggleDeckBQueuePanel() {
                if (this.deckBMediaPanelVisible) {
                    this.showDeckBQueueView();
                    return;
                }
                if (this.deckBQueueVisible) this.hideDeckBQueueView();
                else this.showDeckBQueueView();
            }

            /** Toggle Deck B file-queue strip vs media queue; from media queue, Stations opens the top radio list. */
            toggleDeckBStationMediaPanels() {
                if (this.deckBMediaPanelVisible) {
                    this.hideDeckBMediaView();
                    if (typeof toggleTopMenuPanel === 'function') toggleTopMenuPanel();
                    return;
                }
                if (this.deckBQueueVisible) {
                    this.showDeckBMediaView();
                    return;
                }
                this.showDeckBMediaView();
            }

            refreshQueueUi() {
                if (!this.root) return;
                try {
                    const ulA = this.root.querySelector('#dj-queue-list-a');
                    const ulB = this.root.querySelector('#dj-queue-list-b');
                    const fill = (ul, deckKey) => {
                        if (!ul) return;
                        ul.innerHTML = '';
                        const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
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
                                playQueuedTrackNow(deckKey, idx);
                                try { this.refreshQueueUi(); } catch (_) {}
                            });
                            const toOther = document.createElement('button');
                            toOther.type = 'button';
                            toOther.className = 'dj-queue-to-deck ' + (deckKey === 'a' ? 'dj-queue-to-deck--to-b' : 'dj-queue-to-deck--to-a');
                            toOther.textContent = deckKey === 'a' ? 'B' : 'A';
                            toOther.title = deckKey === 'a' ? 'Send to Deck B queue' : 'Send to Deck A queue';
                            toOther.setAttribute('aria-label', toOther.title);
                            toOther.addEventListener('click', (ev) => {
                                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                                moveQueuedTrackToOtherDeck(deckKey, idx);
                            });
                            const rm = document.createElement('button');
                            rm.type = 'button';
                            rm.className = 'dj-queue-remove';
                            rm.textContent = '✕';
                            rm.title = 'Remove from queue';
                            rm.addEventListener('click', (ev) => {
                                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                                removeQueuedTrack(deckKey, idx);
                                try { this.refreshQueueUi(); } catch (_) {}
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
                try { this.renderStationCycleUi(); } catch (_) {}
                try { this.renderMediaQueueUi(); } catch (_) {}
            }

            renderStationCycleUi() {
                if (!this.root) return;
                const listEl = this.root.querySelector('#dj-station-cycle-list');
                if (!listEl) return;
                listEl.innerHTML = '';
                try { syncStationCycleSelection(); } catch (_) {}
                if (!Array.isArray(stations) || stations.length === 0) {
                    const empty = document.createElement('div');
                    empty.className = 'dj-station-cycle-empty';
                    empty.textContent = 'No radio stations loaded.';
                    listEl.appendChild(empty);
                    return;
                }
                stations.forEach((st, idx) => {
                    if (!st || !st.url) return;
                    const row = document.createElement('div');
                    row.className = 'dj-station-cycle-item';
                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.checked = stationCycleEnabledByUrl.get(st.url) !== false;
                    cb.title = 'Include in Deck B random / cycle (uncheck to exclude from cycle only)';
                    cb.setAttribute(
                        'aria-label',
                        'Include ' + (st.name || ('Station ' + (idx + 1))) + ' in Deck B station cycle'
                    );
                    cb.addEventListener('click', (ev) => {
                        try { ev.stopPropagation(); } catch (_) {}
                    });
                    cb.addEventListener('change', () => {
                        stationCycleEnabledByUrl.set(st.url, !!cb.checked);
                    });
                    const txt = document.createElement('span');
                    txt.className = 'dj-station-cycle-name';
                    txt.textContent = st.name || ('Station ' + (idx + 1));
                    txt.title = 'Play this station on Deck A';
                    txt.setAttribute('role', 'button');
                    txt.tabIndex = 0;
                    txt.addEventListener('click', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        try {
                            if (typeof setStation === 'function') setStation(idx);
                        } catch (_) {}
                    });
                    txt.addEventListener('keydown', (ev) => {
                        if (ev.key !== 'Enter' && ev.key !== ' ') return;
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        try {
                            if (typeof setStation === 'function') setStation(idx);
                        } catch (_) {}
                    });
                    row.appendChild(cb);
                    row.appendChild(txt);
                    listEl.appendChild(row);
                });
            }

            renderMediaQueueUi() {
                if (!this.root) return;
                const listEl = this.root.querySelector('#dj-media-queue-list');
                const btnLoop = this.root.querySelector('#dj-media-loop');
                const btnAll = this.root.querySelector('#dj-media-all');
                const btnShuffle = this.root.querySelector('#dj-media-shuffle');
                if (btnLoop) btnLoop.classList.toggle('on', !!deckBVideoLoopEnabled);
                if (btnAll) btnAll.classList.toggle('on', !!deckBVideoPlayAllEnabled);
                if (btnShuffle) btnShuffle.classList.toggle('on', !!deckBVideoShuffleEnabled);
                if (!listEl) return;
                let curUrl = '';
                try {
                    if (this.deckBVideoEl && this.deckBVizMode === 'video' && !deckBVideoUserIdle) {
                        curUrl = sanitizeUrlForAudio(String(this.deckBVideoEl.currentSrc || this.deckBVideoEl.src || ''));
                    }
                } catch (_) {}
                const showLoopHighlight = deckBVideoSingleLoopActive() && !!curUrl;
                listEl.innerHTML = '';
                if (!Array.isArray(mediaVideoQueue) || mediaVideoQueue.length === 0) {
                    const empty = document.createElement('li');
                    empty.className = 'dj-url-item dj-station-cycle-empty';
                    empty.textContent = 'No videos in queue yet — use Local / Load above or Add files on a deck.';
                    listEl.appendChild(empty);
                    return;
                }
                mediaVideoQueue.forEach((it) => {
                    if (!it || !it.url) return;
                    const li = document.createElement('li');
                    li.className = 'dj-url-item';
                    li.setAttribute('data-id', String(it.id || ''));
                    if (showLoopHighlight && sanitizeUrlForAudio(it.url) === curUrl) {
                        li.classList.add('is-looping-current');
                    }
                    const name = document.createElement('div');
                    name.className = 'dj-url-name';
                    name.textContent = String(it.label || deriveNameFromUrl(it.url) || 'Video source');
                    const acts = document.createElement('div');
                    acts.className = 'dj-url-item-actions';
                    const mk = (act, label, title) => {
                        const b = document.createElement('button');
                        b.type = 'button';
                        b.className = 'dj-url-btn';
                        b.textContent = label;
                        b.title = title;
                        b.setAttribute('data-action', act);
                        return b;
                    };
                    acts.appendChild(mk('play', 'P', 'Play in Deck B video player'));
                    acts.appendChild(mk('queue-a', 'A', 'Send to Deck A queue'));
                    acts.appendChild(mk('queue-b', 'B', 'Send to Deck B queue'));
                    acts.appendChild(mk('delete', 'x', 'Remove from video queue'));
                    li.appendChild(name);
                    li.appendChild(acts);
                    listEl.appendChild(li);
                });
            }

            tearDownDeckBViz(opts) {
                this._deckAAudioBarsMainOnNextClick = false;
                this._deckAStationsTopMenuOnNextClick = false;
                try {
                    if (this._deckBVideoUiAbortCtrl) {
                        this._deckBVideoUiAbortCtrl.abort();
                        this._deckBVideoUiAbortCtrl = null;
                    }
                } catch (_) {}
                try {
                    if (typeof this._deckBVizMountFsCleanup === 'function') {
                        try { this._deckBVizMountFsCleanup(); } catch (_) {}
                        this._deckBVizMountFsCleanup = null;
                    }
                } catch (_) {}
                try {
                    if (typeof this._deckBVideoFullscreenCleanup === 'function') {
                        try { this._deckBVideoFullscreenCleanup(); } catch (_) {}
                        this._deckBVideoFullscreenCleanup = null;
                    }
                } catch (_) {}
                try {
                    const fe = document.fullscreenElement || document.webkitFullscreenElement;
                    const mountFs = this.root && this.root.querySelector('#dj-deck-b-viz-mount');
                    if (fe && mountFs && mountFs.contains(fe)) {
                        try {
                            if (document.exitFullscreen) document.exitFullscreen();
                            else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
                        } catch (_) {}
                    }
                } catch (_) {}
                try {
                    if (!opts || !opts.skipQueueStrip) {
                        try {
                            this.deckBQueueVisible = false;
                            const stageQ = this.root && this.root.querySelector('.dj-deck-b-stage');
                            if (stageQ) stageQ.classList.remove('dj-deck-b-queue-mode');
                            const panelQ = this.root && this.root.querySelector('#dj-deck-b-queue-panel');
                            if (panelQ) panelQ.classList.add('display-none');
                        } catch (_) {}
                    }
                } catch (_) {}
                try {
                    if (!opts || !opts.skipMediaStrip) {
                        try {
                            this.deckBMediaPanelVisible = false;
                            const stageM = this.root && this.root.querySelector('.dj-deck-b-stage');
                            if (stageM) stageM.classList.remove('dj-deck-b-media-mode');
                            const panelM = this.root && this.root.querySelector('#dj-deck-b-media-panel');
                            if (panelM) panelM.classList.add('display-none');
                        } catch (_) {}
                    }
                } catch (_) {}
                try {
                    if (this.deckBVizPmAnimId) cancelAnimationFrame(this.deckBVizPmAnimId);
                } catch (_) {}
                this.deckBVizPmAnimId = null;
                try {
                    if (this.deckBVizPmCycleTimeout) clearTimeout(this.deckBVizPmCycleTimeout);
                } catch (_) {}
                this.deckBVizPmCycleTimeout = null;
                try {
                    if (this.deckBVizBarsAnimId) cancelAnimationFrame(this.deckBVizBarsAnimId);
                } catch (_) {}
                this.deckBVizBarsAnimId = null;
                this.deckBVizPmVisualizer = null;
                this.deckBVizPmCanvas = null;
                this.deckBVizPmResize = null;
                this.nextDeckBProjectMPreset = null;
                this.deckBVideoEl = null;
                this.deckBVideoElA = null;
                this.deckBVideoElB = null;
                this.deckBVideoElQ = null;
                this.deckBVideoStackEl = null;
                this.deckBVideoList = [];
                this.deckBVideoIndex = 0;
                try {
                    const stage = this.root && this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.remove('dj-deck-b-visual-mode');
                    const layer = this.root && this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'true');
                    const mount = this.root && this.root.querySelector('#dj-deck-b-viz-mount');
                    if (mount) mount.innerHTML = '';
                    const ph = this.root && this.root.querySelector('#dj-deck-b-viz-placeholder');
                    if (ph) ph.classList.add('display-none');
                } catch (_) {}
                this.deckBVizMode = 'idle';
                try {
                    const vtp = this.root && this.root.querySelector('#dj-b-video-transport-mount');
                    if (vtp) {
                        vtp.innerHTML = '';
                        vtp.classList.add('display-none');
                        vtp.setAttribute('aria-hidden', 'true');
                        vtp.classList.remove('dj-video-fs-hide-controls');
                    }
                } catch (_) {}
                try { this.syncPlayLabels(); } catch (_) {}
                try { updateSkipPresetButtonVisibility(); } catch (_) {}
            }

            showDeckBVisualBackButton(label) {
                try {
                    if (this.els && this.els.visualBackBtn) {
                        this.els.visualBackBtn.textContent = String(label || 'Visual');
                    }
                } catch (_) {}
            }

            syncDeckBVisualButtons() {
                try {
                    const bars = this.root && this.root.querySelector('#dj-b-vis-bars');
                    const pm = this.root && this.root.querySelector('#dj-b-vis-pm2');
                    const blank = this.root && this.root.querySelector('#dj-b-vis-blank');
                    const barsA = this.root && this.root.querySelector('#dj-vis-bars');
                    const pmA = this.root && this.root.querySelector('#dj-vis-pm2');
                    const blankA = this.root && this.root.querySelector('#dj-vis-blank');
                    if (this.deckBQueueVisible || this.deckBMediaPanelVisible) {
                        if (bars) bars.classList.remove('on');
                        if (pm) pm.classList.remove('on');
                        if (blank) blank.classList.remove('on');
                        if (barsA) barsA.classList.remove('deck-b-active');
                        if (pmA) pmA.classList.remove('deck-b-active');
                        if (blankA) blankA.classList.remove('deck-b-active');
                        try { updateDjDecksShortcutVisibility(); } catch (_) {}
                        try { updateSkipPresetButtonVisibility(); } catch (_) {}
                        return;
                    }
                    const m = this.deckBVizMode || 'idle';
                    if (bars) bars.classList.toggle('on', m === 'bars');
                    if (pm) pm.classList.toggle('on', m === 'projectm');
                    if (blank) blank.classList.toggle('on', m === 'blank');
                    if (barsA) barsA.classList.toggle('deck-b-active', m === 'bars');
                    if (pmA) pmA.classList.toggle('deck-b-active', m === 'projectm');
                    if (blankA) blankA.classList.toggle('deck-b-active', m === 'blank');
                    try { updateDjDecksShortcutVisibility(); } catch (_) {}
                    try { updateSkipPresetButtonVisibility(); } catch (_) {}
                } catch (_) {}
            }

            refreshDeckBVideoSource() {
                if (this.deckBVizMode !== 'video') return;
                if (deckBVideoUserIdle) return;
                try {
                    this.deckBVideoList = getDeckBVideoPlaybackSources();
                    if (this.deckBVideoElA && this.deckBVideoElB && this.deckBVideoElQ) {
                        applyDeckBVideoCrossfadeLayers(this);
                        return;
                    }
                    if (!this.deckBVideoList.length) return;
                    if (this.deckBVideoIndex >= this.deckBVideoList.length) this.deckBVideoIndex = 0;
                    const src = this.deckBVideoList[this.deckBVideoIndex];
                    if (!src || !src.url || !this.deckBVideoEl) return;
                    applyDeckBVideoPayloadToElement(this.deckBVideoEl, src, null);
                    this.showDeckBVisualBackButton(src.label || 'Video');
                } catch (_) {}
            }

            attachDeckBVizMountFullscreen(mount) {
                try {
                    if (typeof this._deckBVizMountFsCleanup === 'function') {
                        try { this._deckBVizMountFsCleanup(); } catch (_) {}
                        this._deckBVizMountFsCleanup = null;
                    }
                } catch (_) {}
                if (!mount) return;
                try { mount.title = 'Double-click for fullscreen · Esc to exit'; } catch (_) {}
                const getFs = () => document.fullscreenElement || document.webkitFullscreenElement;
                const exitFs = () => {
                    try { exitDeckBVizFullscreen(); } catch (_) {}
                };
                const onDblClick = (e) => {
                    try {
                        if (e.target && typeof e.target.closest === 'function' && e.target.closest('#dj-deck-splitter')) return;
                    } catch (_) {}
                    try {
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                    } catch (_) {}
                    try { toggleDeckBVizMountFullscreen(); } catch (_) {}
                };
                const onKeyDown = (e) => {
                    if (e.key !== 'Escape') return;
                    if (!getDeckBVizFullscreenEl()) return;
                    try { e.preventDefault(); } catch (_) {}
                    exitFs();
                };
                mount.addEventListener('dblclick', onDblClick);
                document.addEventListener('keydown', onKeyDown, true);
                this._deckBVizMountFsCleanup = () => {
                    try { mount.removeEventListener('dblclick', onDblClick); } catch (_) {}
                    try { document.removeEventListener('keydown', onKeyDown, true); } catch (_) {}
                };
            }

            startDeckBVideoVisual(opts) {
                if (!this.root) return;
                deckBVideoUserIdle = false;
                this.tearDownDeckBViz();
                try {
                    if (this._deckBVideoUiAbortCtrl) {
                        this._deckBVideoUiAbortCtrl.abort();
                        this._deckBVideoUiAbortCtrl = null;
                    }
                } catch (_) {}
                this._deckBVideoUiAbortCtrl = new AbortController();
                const uiSig = { signal: this._deckBVideoUiAbortCtrl.signal };
                this.deckBVizMode = 'video';
                this.deckBVideoList = getDeckBVideoPlaybackSources();
                this.deckBVideoIndex = 0;
                try {
                    const pi = opts && typeof opts.mediaQueueIndex === 'number' ? opts.mediaQueueIndex : -1;
                    if (pi >= 0) {
                        this.deckBVideoIndex = pi;
                        if (this.deckBVideoList.length && this.deckBVideoIndex >= this.deckBVideoList.length) {
                            this.deckBVideoIndex = 0;
                        }
                    } else if (deckBVideoShuffleEnabled && this.deckBVideoList.length > 1) {
                        this.deckBVideoIndex = Math.floor(Math.random() * this.deckBVideoList.length);
                    }
                } catch (_) {
                    this.deckBVideoIndex = 0;
                }
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.add('dj-deck-b-visual-mode');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                    const transportMount = this.root.querySelector('#dj-b-video-transport-mount');
                    if (!mount) return;
                    mount.innerHTML = '';
                    if (ph) ph.classList.add('display-none');
                    if (transportMount) {
                        transportMount.innerHTML = '';
                        transportMount.classList.remove('dj-video-fs-hide-controls');
                    }

                    const shell = document.createElement('div');
                    shell.className = 'dj-video-shell';
                    const mkBtn = (label, title) => {
                        const b = document.createElement('button');
                        b.type = 'button';
                        b.className = 'dj-video-btn';
                        b.textContent = label;
                        b.title = title;
                        return b;
                    };
                    const loadInput = this.root.querySelector('#dj-b-video-input');
                    const btnLoadUrl = this.root.querySelector('#dj-b-video-load');
                    const btnLoadQA = this.root.querySelector('#dj-b-video-q-a');
                    const btnLoadQB = this.root.querySelector('#dj-b-video-q-b');
                    const btnLoadFile = this.root.querySelector('#dj-b-video-local');
                    const stack = document.createElement('div');
                    stack.className = 'dj-video-stack';
                    const mkLayerVid = (extraClass) => {
                        const v = document.createElement('video');
                        v.className = 'dj-video-player dj-video-player--layer' + (extraClass ? ' ' + extraClass : '');
                        v.playsInline = true;
                        v.muted = true;
                        v.loop = false;
                        return v;
                    };
                    const vA = mkLayerVid('dj-video-player--deck-a');
                    const vB = mkLayerVid('dj-video-player--deck-b');
                    const vQ = mkLayerVid('dj-video-player--queue');
                    stack.appendChild(vA);
                    stack.appendChild(vB);
                    stack.appendChild(vQ);
                    const vid = vB;
                    const ctrls = document.createElement('div');
                    ctrls.className = 'dj-video-controls';
                    const btnRestart = mkBtn('Restart', 'Restart current video');
                    const btnPlay = mkBtn('Play', 'Play/Pause current video');
                    const btnNext = mkBtn('Next', 'Next known video source');
                    const btnStop = mkBtn('Stop', 'Stop playback; disable loop/shuffle; clear video until you press Play or Load');
                    const btnFullscreen = mkBtn('Fullscreen', 'Fullscreen video (Esc to exit)');
                    const btnMediaQueue = mkBtn('Media Queue', 'Radio station cycle list and video queue (Deck B)');
                    ctrls.appendChild(btnRestart);
                    ctrls.appendChild(btnPlay);
                    ctrls.appendChild(btnNext);
                    ctrls.appendChild(btnStop);
                    ctrls.appendChild(btnFullscreen);
                    ctrls.appendChild(btnMediaQueue);
                    shell.appendChild(stack);
                    mount.appendChild(shell);
                    this.deckBVideoStackEl = stack;
                    this.deckBVideoElA = vA;
                    this.deckBVideoElB = vB;
                    this.deckBVideoElQ = vQ;
                    this.deckBVideoEl = vid;
                    if (transportMount) transportMount.appendChild(ctrls);
                    const fiVideo = document.createElement('input');
                    fiVideo.type = 'file';
                    fiVideo.accept = 'video/*';
                    fiVideo.multiple = true;
                    fiVideo.style.display = 'none';
                    fiVideo.setAttribute('aria-hidden', 'true');
                    shell.appendChild(fiVideo);

                    let fsCtrlHideTimer = null;
                    const clearFsHideTimer = () => {
                        if (fsCtrlHideTimer != null) {
                            clearTimeout(fsCtrlHideTimer);
                            fsCtrlHideTimer = null;
                        }
                    };
                    const isShellFs = () =>
                        (document.fullscreenElement === shell) ||
                        (document.webkitFullscreenElement === shell);
                    const setFsHideUi = (hide) => {
                        try {
                            if (hide) {
                                shell.classList.add('dj-video-fs-hide-controls');
                                if (transportMount) transportMount.classList.add('dj-video-fs-hide-controls');
                            } else {
                                shell.classList.remove('dj-video-fs-hide-controls');
                                if (transportMount) transportMount.classList.remove('dj-video-fs-hide-controls');
                            }
                        } catch (_) {}
                    };
                    const setCursorFsIdle = (idle) => {
                        try {
                            if (idle) document.documentElement.classList.add('dj-deck-b-video-fs-cursor-idle');
                            else document.documentElement.classList.remove('dj-deck-b-video-fs-cursor-idle');
                        } catch (_) {}
                    };
                    const scheduleFsCtrlHide = () => {
                        clearFsHideTimer();
                        setFsHideUi(false);
                        setCursorFsIdle(false);
                        if (!isShellFs()) return;
                        fsCtrlHideTimer = setTimeout(() => {
                            fsCtrlHideTimer = null;
                            if (isShellFs()) {
                                setFsHideUi(true);
                                setCursorFsIdle(true);
                            }
                        }, 3000);
                    };
                    const onFsShellActivity = () => {
                        if (!isShellFs()) return;
                        scheduleFsCtrlHide();
                    };
                    const onFsDocumentMove = () => {
                        if (!isShellFs()) return;
                        scheduleFsCtrlHide();
                    };
                    const syncFsBtn = () => {
                        const on = isShellFs();
                        btnFullscreen.textContent = on ? 'Exit FS' : 'Fullscreen';
                        btnFullscreen.title = on ? 'Exit fullscreen (Esc)' : 'Fullscreen video player';
                    };
                    const onFsChange = () => {
                        syncFsBtn();
                        if (isShellFs()) scheduleFsCtrlHide();
                        else {
                            clearFsHideTimer();
                            setFsHideUi(false);
                            setCursorFsIdle(false);
                        }
                    };
                    document.addEventListener('fullscreenchange', onFsChange);
                    document.addEventListener('webkitfullscreenchange', onFsChange);
                    document.addEventListener('mousemove', onFsDocumentMove, uiSig);
                    shell.addEventListener('mousemove', onFsShellActivity);
                    shell.addEventListener('touchstart', onFsShellActivity, { passive: true });
                    if (transportMount) {
                        transportMount.addEventListener('mousemove', onFsShellActivity);
                        transportMount.addEventListener('touchstart', onFsShellActivity, { passive: true });
                    }
                    btnFullscreen.addEventListener('click', () => {
                        try { toggleDeckBVizMountFullscreen(); } catch (_) {}
                    });
                    btnMediaQueue.addEventListener('click', () => {
                        try { this.toggleDeckBMediaPanel(); } catch (_) {}
                        try { this.syncFxLightsFromState(); } catch (_) {}
                        try { resetIdleTimer(); } catch (_) {}
                    });
                    try {
                        stack.title = 'Double-click video for fullscreen · Esc to exit';
                        [vA, vB, vQ].forEach((v) => {
                            try { v.title = stack.title; } catch (_) {}
                        });
                    } catch (_) {}
                    stack.addEventListener(
                        'dblclick',
                        (e) => {
                            try {
                                if (e.target && e.target.closest && e.target.closest('#dj-deck-splitter')) return;
                            } catch (_) {}
                            try {
                                e.preventDefault();
                                e.stopPropagation();
                                e.stopImmediatePropagation();
                            } catch (_) {}
                            try { toggleDeckBVizMountFullscreen(); } catch (_) {}
                        },
                        { capture: true, signal: uiSig.signal }
                    );
                    this._deckBVideoFullscreenCleanup = () => {
                        clearFsHideTimer();
                        try { setCursorFsIdle(false); } catch (_) {}
                        document.removeEventListener('fullscreenchange', onFsChange);
                        document.removeEventListener('webkitfullscreenchange', onFsChange);
                        try { document.removeEventListener('mousemove', onFsDocumentMove); } catch (_) {}
                        try { shell.removeEventListener('mousemove', onFsShellActivity); } catch (_) {}
                        try { shell.removeEventListener('touchstart', onFsShellActivity); } catch (_) {}
                        try {
                            if (transportMount) {
                                transportMount.removeEventListener('mousemove', onFsShellActivity);
                                transportMount.removeEventListener('touchstart', onFsShellActivity);
                                transportMount.classList.remove('dj-video-fs-hide-controls');
                            }
                        } catch (_) {}
                        shell.classList.remove('dj-video-fs-hide-controls');
                    };

                    const tryLogoIdle = () => {
                        this.showDeckBVisualBackButton('Video');
                        const logoVid = vQ;
                        try {
                            vA.style.opacity = '0';
                            vB.style.opacity = '0';
                            vQ.style.opacity = '1';
                        } catch (_) {}
                        const onLogoFail = () => {
                            try {
                                logoVid.pause();
                                logoVid.removeAttribute('src');
                                logoVid.load();
                            } catch (_) {}
                            btnPlay.textContent = 'Play';
                        };
                        logoVid.loop = true;
                        const onErr = () => {
                            try { logoVid.removeEventListener('error', onErr); } catch (_) {}
                            onLogoFail();
                        };
                        logoVid.addEventListener('error', onErr, { once: true });
                        try {
                            logoVid.src = DECK_B_IDLE_LOGO_URL;
                        } catch (_) {
                            onLogoFail();
                            return;
                        }
                        logoVid.play().then(() => {
                            btnPlay.textContent = 'Pause';
                        }).catch(() => {
                            onLogoFail();
                        });
                    };

                    const allLayerVids = () => [vA, vB, vQ];
                    const dominantLayerVid = () => this.deckBVideoEl || vB;

                    const applyDeckBVideoStopped = () => {
                        this.showDeckBVisualBackButton('Video');
                        allLayerVids().forEach((v) => {
                            try {
                                v.pause();
                                v.removeAttribute('src');
                                v.load();
                                v.style.opacity = '0';
                            } catch (_) {}
                            try { v.loop = false; } catch (_) {}
                        });
                        btnPlay.textContent = 'Play';
                    };

                    const applyCurrent = () => {
                        this.deckBVideoList = getDeckBVideoPlaybackSources();
                        if (deckBVideoUserIdle) {
                            applyDeckBVideoStopped();
                            return;
                        }
                        const plan = computeDeckBVideoCrossfadePlan(this);
                        const hasLayers = !!(plan.layerA || plan.layerB || plan.layerQ);
                        if (!hasLayers && !this.deckBVideoList.length) {
                            tryLogoIdle();
                            return;
                        }
                        if (this.deckBVideoIndex >= this.deckBVideoList.length) this.deckBVideoIndex = 0;
                        applyDeckBVideoCrossfadeLayers(this);
                        try {
                            const dom = dominantLayerVid();
                            btnPlay.textContent = dom && !dom.paused ? 'Pause' : 'Play';
                        } catch (_) {
                            btnPlay.textContent = 'Play';
                        }
                        try { this.refreshQueueUi(); } catch (_) {}
                        try { this.updateStationTitles(); } catch (_) {}
                    };

                    const syncLayerFromSource = (vd, cur) => {
                        if (!vd || !cur || !cur.syncFrom) return;
                        const op = parseFloat(vd.style.opacity || '0');
                        if (!(op > 0.02)) return;
                        if (!urlsMediaMatch(String(cur.url), String(vd.currentSrc || vd.src || ''))) return;
                        const sf = cur.syncFrom;
                        if (sf.paused || sf.ended) {
                            if (!vd.paused) vd.pause();
                            return;
                        }
                        if (vd.paused) return;
                        const drift = Math.abs(vd.currentTime - sf.currentTime);
                        if (drift > 0.35) {
                            let t = Number(sf.currentTime) || 0;
                            const md = sf.duration;
                            if (Number.isFinite(md) && md > 0) t = Math.min(Math.max(0, t), md - 0.05);
                            const vdur = vd.duration;
                            if (Number.isFinite(vdur) && vdur > 0) t = Math.min(t, vdur - 0.05);
                            try { vd.currentTime = t; } catch (_) {}
                        }
                    };

                    const syncMirrorTick = () => {
                        if (this.deckBVizMode !== 'video' || deckBVideoUserIdle) return;
                        try {
                            const plan = computeDeckBVideoCrossfadePlan(this);
                            if (plan.layerA) syncLayerFromSource(vA, plan.layerA);
                            if (plan.layerB) syncLayerFromSource(vB, plan.layerB);
                            let anyPlay = false;
                            allLayerVids().forEach((v) => {
                                const o = parseFloat(v.style.opacity || '0');
                                if (o > 0.02 && !v.paused && !v.ended) anyPlay = true;
                            });
                            btnPlay.textContent = anyPlay ? 'Pause' : 'Play';
                        } catch (_) {}
                    };
                    try {
                        if (audioEl) audioEl.addEventListener('timeupdate', syncMirrorTick, { passive: true, signal: uiSig.signal });
                    } catch (_) {}
                    try {
                        if (audioElB) audioElB.addEventListener('timeupdate', syncMirrorTick, { passive: true, signal: uiSig.signal });
                    } catch (_) {}

                    btnRestart.addEventListener('click', () => {
                        try {
                            const plan = computeDeckBVideoCrossfadePlan(this);
                            [plan.layerA, plan.layerB, plan.layerQ].forEach((layer) => {
                                if (layer && layer.syncFrom) {
                                    try { layer.syncFrom.currentTime = 0; } catch (_) {}
                                }
                            });
                            allLayerVids().forEach((v) => {
                                const o = parseFloat(v.style.opacity || '0');
                                if (o <= 0.02) return;
                                try { v.currentTime = 0; } catch (_) {}
                                v.play().catch(() => {});
                            });
                        } catch (_) {}
                        btnPlay.textContent = 'Pause';
                    });
                    btnPlay.addEventListener('click', () => {
                        let anyPlaying = false;
                        allLayerVids().forEach((v) => {
                            const o = parseFloat(v.style.opacity || '0');
                            if (o > 0.02 && !v.paused) anyPlaying = true;
                        });
                        if (anyPlaying) {
                            allLayerVids().forEach((v) => {
                                try { v.pause(); } catch (_) {}
                            });
                            btnPlay.textContent = 'Play';
                            return;
                        }
                        const dom = dominantLayerVid();
                        const noSrc = dom && !dom.getAttribute('src');
                        if (deckBVideoUserIdle && noSrc) {
                            this.deckBVideoList = getDeckBVideoPlaybackSources();
                            if (this.deckBVideoList.length || mediaVideoQueue.length) {
                                deckBVideoUserIdle = false;
                                applyCurrent();
                            } else {
                                deckBVideoUserIdle = false;
                                tryLogoIdle();
                            }
                            return;
                        }
                        applyCurrent();
                        allLayerVids().forEach((v) => {
                            const o = parseFloat(v.style.opacity || '0');
                            if (o > 0.02) v.play().catch(() => {});
                        });
                        btnPlay.textContent = 'Pause';
                    });
                    btnNext.addEventListener('click', () => {
                        deckBVideoUserIdle = false;
                        this.deckBVideoList = getDeckBVideoPlaybackSources();
                        if (!this.deckBVideoList.length) {
                            tryLogoIdle();
                            return;
                        }
                        if (deckBVideoShuffleEnabled && this.deckBVideoList.length > 1) {
                            let next = this.deckBVideoIndex;
                            while (next === this.deckBVideoIndex) next = Math.floor(Math.random() * this.deckBVideoList.length);
                            this.deckBVideoIndex = next;
                        } else {
                            this.deckBVideoIndex = (this.deckBVideoIndex + 1) % this.deckBVideoList.length;
                        }
                        applyCurrent();
                    });
                    const urlOrPendingToEntry = () => {
                        const clean = sanitizeUrlForAudio(loadInput ? loadInput.value : '');
                        if (clean) {
                            if (isLikelyRadioStreamUrl(clean)) {
                                addUserRadioStation(clean, deriveNameFromUrl(clean));
                                return { type: 'radio', url: clean };
                            }
                            if (!isVideoQueueEligibleUrl(clean)) {
                                return { type: 'audiourl', url: clean, name: deriveNameFromUrl(clean) };
                            }
                            upsertMediaVideoQueueEntry(clean, deriveNameFromUrl(clean));
                            registerDeckVideoFeed('b', clean, deriveNameFromUrl(clean), false);
                            return { type: 'url', entry: { url: clean, name: deriveNameFromUrl(clean) } };
                        }
                        return null;
                    };
                    if (btnLoadUrl) btnLoadUrl.addEventListener('click', () => {
                        const src = urlOrPendingToEntry();
                        if (!src) return;
                        if (src.type === 'radio') {
                            try { this.refreshQueueUi(); } catch (_) {}
                            return;
                        }
                        if (src.type === 'audiourl') {
                            enqueueDeckUrl('b', src.url, src.name);
                            try { this.refreshQueueUi(); } catch (_) {}
                            return;
                        }
                        deckBVideoUserIdle = false;
                        if (src.type === 'url') {
                            const mqIdx = mediaVideoQueue.findIndex((x) => x && x.url === src.entry.url);
                            this.deckBVideoIndex = mqIdx >= 0 ? mqIdx : 0;
                            applyCurrent();
                        }
                    }, uiSig);
                    btnStop.addEventListener('click', () => {
                        deckBVideoLoopEnabled = false;
                        deckBVideoShuffleEnabled = false;
                        deckBVideoUserIdle = true;
                        try { this.refreshQueueUi(); } catch (_) {}
                        applyDeckBVideoStopped();
                    });
                    if (btnLoadQA) btnLoadQA.addEventListener('click', () => {
                        const src = urlOrPendingToEntry();
                        if (!src) return;
                        if (src.type === 'radio') return;
                        if (src.type === 'audiourl') {
                            enqueueDeckUrl('a', src.url, src.name);
                            return;
                        }
                        if (src.type === 'url') enqueueDeckUrl('a', src.entry.url, src.entry.name);
                    }, uiSig);
                    if (btnLoadQB) btnLoadQB.addEventListener('click', () => {
                        const src = urlOrPendingToEntry();
                        if (!src) return;
                        if (src.type === 'radio') return;
                        if (src.type === 'audiourl') {
                            enqueueDeckUrl('b', src.url, src.name);
                            return;
                        }
                        if (src.type === 'url') enqueueDeckUrl('b', src.entry.url, src.entry.name);
                    }, uiSig);
                    if (btnLoadFile) btnLoadFile.addEventListener('click', () => { try { fiVideo.click(); } catch (_) {} }, uiSig);
                    fiVideo.addEventListener('change', (ev) => {
                        const picked = ev.target.files ? Array.from(ev.target.files) : [];
                        try { ev.target.value = ''; } catch (_) {}
                        addVideoFilesToMediaQueue(picked);
                        deckBVideoUserIdle = false;
                        if (loadInput) loadInput.value = '';
                        this.deckBVideoList = getDeckBVideoPlaybackSources();
                        this.deckBVideoIndex = 0;
                        applyCurrent();
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, uiSig);
                    const onLayerEnded = (endedEl) => {
                        if (deckBVideoUserIdle) return;
                        try {
                            const plan = computeDeckBVideoCrossfadePlan(this);
                            if (plan.layerA && plan.layerA.syncFrom && !plan.layerA.syncFrom.ended && endedEl === vA) {
                                applyCurrent();
                                return;
                            }
                            if (plan.layerB && plan.layerB.syncFrom && !plan.layerB.syncFrom.ended && endedEl === vB) {
                                applyCurrent();
                                return;
                            }
                        } catch (_) {}
                        if (!deckBVideoPlayAllEnabled) {
                            if (deckBVideoSingleLoopActive()) {
                                try { endedEl.currentTime = 0; endedEl.play().catch(() => {}); } catch (_) {}
                                return;
                            }
                            if (getDeckActiveVideoMeta('a') || getDeckActiveVideoMeta('b')) {
                                applyCurrent();
                            } else {
                                tryLogoIdle();
                            }
                            return;
                        }
                        this.deckBVideoList = getDeckBVideoPlaybackSources();
                        const qOnly = mediaVideoQueue.length && !getDeckActiveVideoMeta('a') && !getDeckActiveVideoMeta('b');
                        if (qOnly && mediaVideoQueue.length > 1) {
                            if (deckBVideoShuffleEnabled) {
                                let next = this.deckBVideoIndex;
                                while (next === this.deckBVideoIndex) next = Math.floor(Math.random() * mediaVideoQueue.length);
                                this.deckBVideoIndex = next;
                            } else {
                                this.deckBVideoIndex = (this.deckBVideoIndex + 1) % mediaVideoQueue.length;
                            }
                            applyCurrent();
                        } else if (this.deckBVideoList.length > 1) {
                            if (deckBVideoShuffleEnabled) {
                                let next = this.deckBVideoIndex;
                                while (next === this.deckBVideoIndex) next = Math.floor(Math.random() * this.deckBVideoList.length);
                                this.deckBVideoIndex = next;
                            } else {
                                this.deckBVideoIndex = (this.deckBVideoIndex + 1) % this.deckBVideoList.length;
                            }
                            applyCurrent();
                        } else if (getDeckActiveVideoMeta('a') || getDeckActiveVideoMeta('b')) {
                            applyCurrent();
                        } else {
                            tryLogoIdle();
                        }
                        try { this.refreshQueueUi(); } catch (_) {}
                    };
                    allLayerVids().forEach((v) => {
                        v.addEventListener('play', () => { btnPlay.textContent = 'Pause'; });
                        v.addEventListener('pause', () => { btnPlay.textContent = 'Play'; });
                        v.addEventListener('ended', () => onLayerEnded(v));
                    });
                    applyCurrent();
                    try { this.syncDeckBVideoTopChrome(); } catch (_) {}
                } catch (_) {}
            }

            startDeckBProjectM() {
                if (!this.root || !state || typeof butterchurn === 'undefined') return;
                try { initAudio(); } catch (_) {}
                if (!state.audioCtx) return;
                this.tearDownDeckBViz();
                this.deckBVizMode = 'projectm';
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.add('dj-deck-b-visual-mode');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                } catch (_) {}
                this.showDeckBVisualBackButton('ProjectM');
                const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                if (!mount) return;
                if (ph) ph.classList.add('display-none');

                const canvas = document.createElement('canvas');
                canvas.style.display = 'block';
                canvas.style.width = '100%';
                canvas.style.height = '100%';
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
                        if (this.deckBVizPmVisualizer && this.deckBVizPmVisualizer.setRendererSize) {
                            this.deckBVizPmVisualizer.setRendererSize(w, h);
                        }
                    } catch (_) {}
                };

                const pxRatio = Number(visualSettings.pixelRatio) || Math.min(window.devicePixelRatio || 1, 2);
                const rect0 = mount.getBoundingClientRect();
                const w0 = Math.max(64, Math.floor(rect0.width));
                const h0 = Math.max(64, Math.floor(rect0.height));
                const dpr0 = Math.min(window.devicePixelRatio || 1, 2);
                canvas.width = Math.floor(w0 * dpr0);
                canvas.height = Math.floor(h0 * dpr0);

                let viz;
                try {
                    viz = butterchurn.createVisualizer(state.audioCtx, canvas, {
                        width: w0,
                        height: h0,
                        pixelRatio: pxRatio
                    });
                } catch (_) {
                    viz = null;
                }
                if (!viz) {
                    this.tearDownDeckBViz();
                    this.syncDeckBVisualButtons();
                    return;
                }
                this.deckBVizPmVisualizer = viz;
                this.deckBVizPmCanvas = canvas;
                try {
                    // Master post-mix analyser (includes sample players on mixInput); deck A tap misses samples
                    viz.connectAudio(state.analyserNode);
                } catch (_) {}

                const loadPmPreset = (idx) => {
                    if (presetKeys.length === 0 || !this.deckBVizPmVisualizer) return;
                    const key = presetKeys[idx];
                    const transition = Number(visualSettings.transitionSec) || 2.7;
                    try {
                        this.deckBVizPmVisualizer.loadPreset(presetMap[key], transition);
                    } catch (_) {}
                };

                loadPmPreset(currentPresetIdx);

                const nextPmPreset = () => {
                    if (presetKeys.length <= 1 || !this.deckBVizPmVisualizer) return;
                    let next = currentPresetIdx;
                    let guard = 0;
                    while (next === currentPresetIdx && presetKeys.length > 1 && guard++ < 8) {
                        next = Math.floor(Math.random() * presetKeys.length);
                    }
                    currentPresetIdx = next;
                    loadPmPreset(currentPresetIdx);
                };
                this.nextDeckBProjectMPreset = () => nextPmPreset();

                const restartPmCycle = () => {
                    try {
                        if (this.deckBVizPmCycleTimeout) clearTimeout(this.deckBVizPmCycleTimeout);
                    } catch (_) {}
                    const schedule = () => {
                        const minS = Number(visualSettings.shuffleMinSec) || 12;
                        const maxS = Number(visualSettings.shuffleMaxSec) || 25;
                        const lo = Math.min(minS, maxS);
                        const hi = Math.max(minS, maxS);
                        const delay = (lo + Math.random() * (hi - lo)) * 1000;
                        this.deckBVizPmCycleTimeout = setTimeout(() => {
                            nextPmPreset();
                            schedule();
                        }, delay);
                    };
                    schedule();
                };
                restartPmCycle();

                resizePm();
                this.deckBVizPmResize = resizePm;

                const loop = () => {
                    this.deckBVizPmAnimId = requestAnimationFrame(loop);
                    try {
                        if (this.deckBVizPmVisualizer) this.deckBVizPmVisualizer.render();
                    } catch (_) {}
                };
                loop();
                this.attachDeckBVizMountFullscreen(mount);
                try { updateSkipPresetButtonVisibility(); } catch (_) {}
            }

            startDeckBAudioBars() {
                if (!this.root || !state) return;
                try { initAudio(); } catch (_) {}
                this.tearDownDeckBViz();
                this.deckBVizMode = 'bars';
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.add('dj-deck-b-visual-mode');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                } catch (_) {}
                this.showDeckBVisualBackButton('Audio Bars');
                const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                if (!mount) return;
                if (ph) ph.classList.add('display-none');

                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
                camera.position.z = 8;
                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
                const canvas = renderer.domElement;
                canvas.style.display = 'block';
                canvas.style.width = '100%';
                canvas.style.height = '100%';
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
                this.deckBVizPmResize = resizeBars;
                resizeBars();

                const loop = () => {
                    this.deckBVizBarsAnimId = requestAnimationFrame(loop);
                    try {
                        // Master analyser: beat pads / IDEAS samples route through mixInput, not deck A/B taps
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
                this.attachDeckBVizMountFullscreen(mount);
            }

            startDeckBBlankVisual() {
                if (!this.root) return;
                this.tearDownDeckBViz();
                this.deckBVizMode = 'blank';
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.add('dj-deck-b-visual-mode');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    if (mount) mount.innerHTML = '';
                    const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                    if (ph) ph.classList.add('display-none');
                } catch (_) {}
                try {
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    if (mount) this.attachDeckBVizMountFullscreen(mount);
                } catch (_) {}
            }

            startDeckBEmbed(url, mode, opts) {
                if (!this.root) return;
                let href = (url && String(url).trim()) || '';
                if (!href) href = 'about:blank';
                else if (!/^https?:\/\//i.test(href)) href = 'https://' + href.replace(/^\/+/, '');
                const iframeTitle = (opts && opts.iframeTitle) || 'Web content';
                const backLabel = (opts && opts.backLabel) || 'Web';
                this.tearDownDeckBViz();
                this.deckBVizMode = mode;
                try {
                    const stage = this.root.querySelector('.dj-deck-b-stage');
                    if (stage) stage.classList.add('dj-deck-b-visual-mode');
                    const layer = this.root.querySelector('#dj-deck-b-viz-layer');
                    if (layer) layer.setAttribute('aria-hidden', 'false');
                    const mount = this.root.querySelector('#dj-deck-b-viz-mount');
                    const ph = this.root.querySelector('#dj-deck-b-viz-placeholder');
                    if (!mount) return;
                    mount.innerHTML = '';
                    if (ph) ph.classList.add('display-none');
                    const shell = document.createElement('div');
                    shell.className = 'dj-deck-b-embed-shell';
                    const iframe = document.createElement('iframe');
                    iframe.className = 'dj-deck-b-embed-frame';
                    iframe.setAttribute('title', iframeTitle);
                    iframe.setAttribute(
                        'sandbox',
                        'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads allow-modals'
                    );
                    iframe.setAttribute('referrerpolicy', 'no-referrer-when-downgrade');
                    iframe.src = href;
                    shell.appendChild(iframe);
                    mount.appendChild(shell);
                    this.showDeckBVisualBackButton(backLabel);
                    this.attachDeckBVizMountFullscreen(mount);
                } catch (_) {}
            }

            startDeckBKaraokeEmbed(url) {
                let href = url;
                try {
                    if (typeof globalThis.normalizeKaraokeNerdsEmbedUrl === 'function') {
                        href = globalThis.normalizeKaraokeNerdsEmbedUrl(url);
                    } else if (!href) {
                        href = 'https://www.karaokenerds.com/#query';
                    }
                } catch (_) {
                    href = url || 'https://www.karaokenerds.com/#query';
                }
                this.startDeckBEmbed(
                    href,
                    'karaoke',
                    { iframeTitle: 'Karaoke Nerds', backLabel: 'Karaoke' }
                );
            }

            toggleDeckBKaraokeEmbed() {
                try {
                    if (typeof uiLocked !== 'undefined' && uiLocked) return;
                    if (this.deckBVizMode === 'karaoke') {
                        this.tearDownDeckBViz();
                        this.syncDeckBVisualButtons();
                        this.updateStationTitles();
                        this.syncFxLightsFromState();
                        try { updateDjDecksShortcutVisibility(); } catch (_) {}
                        return;
                    }
                    if (this.deckBQueueVisible) this.hideDeckBQueueView();
                    if (this.deckBMediaPanelVisible) this.hideDeckBMediaView();
                    this.startDeckBKaraokeEmbed();
                    this.syncDeckBVisualButtons();
                    this.updateStationTitles();
                    this.syncFxLightsFromState();
                    try { updateDjDecksShortcutVisibility(); } catch (_) {}
                } catch (_) {}
            }

            toggleDeckBKBopEmbed() {
                try {
                    if (typeof uiLocked !== 'undefined' && uiLocked) return;
                    if (this.deckBVizMode === 'kbop') {
                        this.tearDownDeckBViz();
                        this.syncDeckBVisualButtons();
                        this.updateStationTitles();
                        this.syncFxLightsFromState();
                        try { updateDjDecksShortcutVisibility(); } catch (_) {}
                        return;
                    }
                    if (this.deckBQueueVisible) this.hideDeckBQueueView();
                    if (this.deckBMediaPanelVisible) this.hideDeckBMediaView();
                    this.startDeckBEmbed('https://www.kbop.us/', 'kbop', { iframeTitle: 'K-BOP', backLabel: 'K-BOP' });
                    this.syncDeckBVisualButtons();
                    this.updateStationTitles();
                    this.syncFxLightsFromState();
                    try { updateDjDecksShortcutVisibility(); } catch (_) {}
                } catch (_) {}
            }

            syncDeckVolumeSliderLengths() {
                if (!this.root) return;
                try {
                    const blocks = Array.from(this.root.querySelectorAll('.dj-btn-block'));
                    const target = blocks.length
                        ? Math.max(...blocks.map((el) => Math.floor(el.getBoundingClientRect().height || 0)))
                        : 0;
                    /* When button blocks are not laid out yet (or hidden), keep a sane default so BPM still syncs. */
                    const len = target > 0 ? Math.max(200, target + 22) : 200;
                    const volA = this.root.querySelector('#dj-deck-a-vol');
                    const volB = this.root.querySelector('#dj-deck-b-vol');
                    const bpmA = this.root.querySelector('#dj-a-beat-bpm');
                    const bpmB = this.root.querySelector('#dj-b-beat-bpm');
                    /** Rotated range: pre-rotate `width` is the vertical span. Flex-centers the 14px-tall box in the wrap (height h);
                     * thumb centres travel `width - thumb` along that span. Matching crossfader geometry requires
                     * `width === h` so extremes sit at y = thumb_half and y = h - thumb_half (same as .dj-deck-vol-wrap::before). */
                    const DECK_VOL_SLIDER_LENGTH_MULT = 2;
                    /** Length from btn-block row; cap by mosaic height only (never stack/wrap — avoids ResizeObserver feedback). */
                    const volLenPx = (volEl) => {
                        let px = Math.round(len * DECK_VOL_SLIDER_LENGTH_MULT);
                        try {
                            const mosaic = volEl && volEl.closest('.dj-deck-mosaic');
                            if (mosaic) {
                                const mh = Math.floor(mosaic.getBoundingClientRect().height || 0);
                                if (mh > 80) px = Math.min(px, Math.max(120, mh - 36));
                            }
                        } catch (_) {}
                        return px;
                    };
                    const bpmTrackLenPx = (bpmEl) => {
                        const fallback = target > 0 ? Math.max(300, target + 80) : 320;
                        try {
                            const stack = bpmEl && bpmEl.closest('.dj-beatmap-tempo-stack');
                            const wrap = bpmEl && bpmEl.closest('.dj-beatmap-tempo-slider-wrap');
                            const footer = stack && stack.querySelector('.dj-beatmap-tempo-footer');
                            if (!stack || !wrap) return Math.round(fallback);
                            const sh = stack.clientHeight || 0;
                            if (sh < 64) return Math.round(fallback);
                            const fh = footer ? footer.offsetHeight || footer.clientHeight || 0 : 0;
                            let gapPx = 0;
                            try {
                                const gs = getComputedStyle(stack);
                                gapPx = parseFloat(gs.rowGap) || parseFloat(gs.gap) || 0;
                            } catch (_) {}
                            if (!Number.isFinite(gapPx)) gapPx = 0;
                            const span = Math.floor(Math.max(0, sh - fh - gapPx));
                            const wh = Math.max(
                                wrap.clientHeight || 0,
                                wrap.offsetHeight || 0,
                                Math.round(wrap.getBoundingClientRect().height || 0)
                            );
                            const use = Math.max(120, span, wh);
                            if (use > 96) return Math.round(use);
                        } catch (_) {}
                        return Math.round(fallback);
                    };
                    const applyDeckVolSliderSize = (volEl) => {
                        if (!volEl) return;
                        const px = volLenPx(volEl);
                        const key = String(px);
                        if (volEl.dataset.volSyncPx === key) return;
                        volEl.dataset.volSyncPx = key;
                        volEl.style.width = `${px}px`;
                        try {
                            const wrap = volEl.closest('.dj-deck-vol-wrap');
                            if (wrap) {
                                wrap.style.removeProperty('min-height');
                                wrap.style.removeProperty('height');
                            }
                        } catch (_) {}
                    };
                    applyDeckVolSliderSize(volA);
                    applyDeckVolSliderSize(volB);
                    if (bpmA) bpmA.style.width = `${bpmTrackLenPx(bpmA)}px`;
                    if (bpmB) bpmB.style.width = `${bpmTrackLenPx(bpmB)}px`;
                    if (bpmA || bpmB) {
                        requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                                try {
                                    if (bpmA) bpmA.style.width = `${bpmTrackLenPx(bpmA)}px`;
                                    if (bpmB) bpmB.style.width = `${bpmTrackLenPx(bpmB)}px`;
                                } catch (_) {}
                            });
                        });
                    }
                } catch (_) {}
            }

            applyLandscapeDeckSplit(ratio) {
                /*
                 * If the splitter is dragged into the last 10% of the available screen width,
                 * snap Deck B to a collapsed state (zero width, hidden) while leaving the
                 * splitter handle in place so a leftward drag — or a double-click → 50/50 —
                 * restores Deck B instantly. Below 0.15 we clamp normally so Deck A keeps a
                 * usable minimum width.
                 */
                const COLLAPSE_THRESHOLD = 0.90;
                const collapseB = Number.isFinite(ratio) && ratio >= COLLAPSE_THRESHOLD;
                const r = collapseB
                    ? COLLAPSE_THRESHOLD
                    : Math.max(0.15, Math.min(COLLAPSE_THRESHOLD, Number(ratio) || 0.5));
                this.landscapeDeckSplit = r;
                this.deckBCollapsed = collapseB;
                if (!this.root) return;
                this.root.classList.toggle('dj-deck-b-collapsed', collapseB);
                this.root.style.setProperty('--dj-landscape-split-a', String(collapseB ? 1 : r));
                this.root.style.setProperty('--dj-landscape-split-b', String(collapseB ? 0 : 1 - r));
                const spl = this.root.querySelector('#dj-deck-splitter');
                if (spl) {
                    spl.setAttribute('aria-valuenow', String(Math.round((collapseB ? 1 : r) * 100)));
                    if (collapseB) {
                        spl.setAttribute('data-deck-b-collapsed', '1');
                        spl.title = 'Deck B is collapsed — drag left or double-click to restore Deck B';
                    } else {
                        spl.removeAttribute('data-deck-b-collapsed');
                        spl.title = 'Drag sideways to give Deck A more or less room vs Deck B and the fader. Double-click for equal columns.';
                    }
                }
                try {
                    localStorage.setItem('dj.landscapeDeckSplit.v1', String(r));
                    localStorage.setItem('dj.deckBCollapsed.v1', collapseB ? '1' : '0');
                } catch (_) {}
                requestAnimationFrame(() => {
                    try {
                        this.syncDeckMosaicScale();
                        this.syncDeckVolumeSliderLengths();
                    } catch (_) {}
                });
            }

            bindLandscapeDeckSplitter() {
                if (!this.root) return;
                const splitter = this.root.querySelector('#dj-deck-splitter');
                const columns = this.root.querySelector('.dj-deck-columns');
                if (!splitter || !columns) return;
                splitter.addEventListener(
                    'pointerdown',
                    (e) => {
                        if (e.button !== 0) return;
                        e.preventDefault();
                        try {
                            e.stopPropagation();
                        } catch (_) {}
                        const dragStartX = e.clientX;
                        // When Deck B is currently collapsed, treat the drag as starting *at* the
                        // collapse threshold (0.90) rather than wherever this.landscapeDeckSplit
                        // happens to be. That way any leftward movement immediately drops below
                        // 0.90 and restores Deck B — without it the user would have to drag past
                        // the (invisible) stored ratio before Deck B reappeared.
                        const dragStartRatio = this.deckBCollapsed ? 0.90 : this.landscapeDeckSplit;
                        const trackW = columns.getBoundingClientRect().width || 1;
                        splitter.classList.add('is-dragging');
                        try {
                            splitter.setPointerCapture(e.pointerId);
                        } catch (_) {}
                        const onMove = (ev) => {
                            const dx = ev.clientX - dragStartX;
                            this.applyLandscapeDeckSplit(dragStartRatio + dx / trackW);
                        };
                        const onUp = () => {
                            splitter.classList.remove('is-dragging');
                            try {
                                splitter.releasePointerCapture(e.pointerId);
                            } catch (_) {}
                            window.removeEventListener('pointermove', onMove);
                            window.removeEventListener('pointerup', onUp);
                            window.removeEventListener('pointercancel', onUp);
                        };
                        window.addEventListener('pointermove', onMove);
                        window.addEventListener('pointerup', onUp);
                        window.addEventListener('pointercancel', onUp);
                    },
                    true
                );
                splitter.addEventListener(
                    'dblclick',
                    (e) => {
                        try {
                            e.preventDefault();
                            e.stopPropagation();
                            e.stopImmediatePropagation();
                        } catch (_) {}
                        this.applyLandscapeDeckSplit(0.5);
                    },
                    true
                );
            }

            syncDeckMosaicScale() {
                if (!this.root) return;
                try {
                    const landscape = this.root.classList.contains('dj-mode-landscape');
                    /* Portrait: one deck spans ~full width → baseline 860. Landscape: two decks side‑by‑side → each mosaic is ~half as wide; use ~half baseline so fitW is not crushed (fixes tiny right button grid). */
                    const baselineW = landscape ? 440 : 860;
                    /* Lower landscape floor so font-size/padding (driven by --dj-control-scale) keep
                     * shrinking when the splitter squeezes a deck — without this the 30/30/40 column
                     * layout would stay rigid and the right-grid button text would clip. The CSS
                     * font clamp() backstops the visual lower bound. */
                    const scaleMin = landscape ? 0.6 : 0.72;
                    const mosaics = Array.from(this.root.querySelectorAll('.dj-deck-mosaic'));
                    if (!mosaics.length) return;
                    mosaics.forEach((mosaic) => {
                        const rect = mosaic.getBoundingClientRect();
                        const w = Math.max(1, Math.floor(rect.width || 0));
                        const h = Math.max(1, Math.floor(rect.height || 0));
                        const fitW = w / baselineW;
                        const fitH = h / 235;
                        const widthBiased = Math.min(1.28, fitW);
                        const heightBiased = Math.min(1.28, fitH);
                        const scale = Math.max(scaleMin, Math.min(1.28, (widthBiased * 0.52) + (heightBiased * 0.48)));
                        const scaleStr = scale.toFixed(3);
                        const prev = mosaic.style.getPropertyValue('--dj-control-scale');
                        if (prev !== scaleStr) mosaic.style.setProperty('--dj-control-scale', scaleStr);
                    });
                } catch (_) {}
                /* After --dj-control-scale settles, fit each button's text to its own box
                 * on the next frame (layout must commit before we can measure scrollWidth
                 * vs clientWidth). rAF-coalesced so back-to-back scale changes only run
                 * one auto-fit pass. */
                try {
                    if (this._fitBtnRaf) cancelAnimationFrame(this._fitBtnRaf);
                    this._fitBtnRaf = requestAnimationFrame(() => {
                        this._fitBtnRaf = null;
                        try {
                            this.fitDeckButtonText();
                        } catch (_) {}
                        /* Beat-map row heights can shift after font auto-fit; re-measure BPM / VOL tracks on the same frame. */
                        try {
                            this.syncDeckVolumeSliderLengths();
                        } catch (_) {}
                    });
                } catch (_) {}
            }

            /**
             * Per-button text auto-fit: starts from the CSS-driven font-size and
             * shrinks in 0.5px steps until the label fits on a single line (or hits
             * the 4px floor, after which letter-spacing is nudged negative as a last
             * resort). Runs after every syncDeckMosaicScale() so the deck panel stays
             * legible when the splitter is dragged or the viewport is resized.
             */
            fitDeckButtonText() {
                if (!this.root) return;
                const selectors = [
                    '.dj-beatmap-pads .btn.dj-fx.dj-beat-pad',
                    '.dj-controls-grid > .btn.dj-fx',
                    '.dj-controls-grid > .btn.dj-sfx',
                    '.dj-controls-grid > .dj-play',
                    '.dj-controls-grid > .dj-tbtn'
                ];
                let nodes;
                try { nodes = this.root.querySelectorAll(selectors.join(', ')); }
                catch (_) { return; }
                nodes.forEach((btn) => {
                    try {
                        // Reset any prior overrides so the CSS clamp baseline is honored on
                        // each pass (otherwise a one-time shrink would compound on resize).
                        btn.style.fontSize = '';
                        btn.style.letterSpacing = '';
                        const cw = btn.clientWidth | 0;
                        if (!cw) return;
                        const fits = () => btn.scrollWidth <= cw + 1;
                        if (fits()) return;
                        const cs = window.getComputedStyle(btn);
                        let fs = parseFloat(cs.fontSize) || 12;
                        const minFs = 4;
                        let safety = 40;
                        while (!fits() && fs > minFs && safety-- > 0) {
                            fs -= 0.5;
                            btn.style.fontSize = `${fs}px`;
                        }
                        if (!fits()) {
                            // CSS already adds overflow:hidden + text-overflow:ellipsis as
                            // the visual backstop; tighten letter-spacing first to claw
                            // back a couple of pixels before falling back to ellipsis.
                            btn.style.letterSpacing = '-0.04em';
                        }
                    } catch (_) {}
                });
            }

            onResize() {
                if (!this.root) return;
                const w = window.innerWidth;
                const h = Math.max(1, window.innerHeight);
                /* Default was w > h (any non‑square landscape). Require extra width before A/B columns go left/right, and a lower threshold when shrinking back so the layout does not thrash. */
                const LANDSCAPE_ASPECT_ENTER = 1.24;
                const LANDSCAPE_ASPECT_EXIT = 1.12;
                let land = this._djLandscapeLayout;
                if (land) {
                    land = w > h * LANDSCAPE_ASPECT_EXIT;
                } else {
                    land = w > h * LANDSCAPE_ASPECT_ENTER;
                }
                this._djLandscapeLayout = land;
                this.root.classList.toggle('dj-mode-landscape', land);
                this.root.classList.toggle('dj-mode-portrait', !land);
                try {
                    if ((this.deckBVizMode === 'projectm' || this.deckBVizMode === 'bars') && typeof this.deckBVizPmResize === 'function') {
                        this.deckBVizPmResize();
                    }
                } catch (_) {}
                this.syncDeckMosaicScale();
                this.syncDeckVolumeSliderLengths();
                requestAnimationFrame(() => {
                    try {
                        this.syncDeckMosaicScale();
                        this.syncDeckVolumeSliderLengths();
                    } catch (_) {}
                });
            }

            syncDeckBVideoTopChrome() {
                try {
                    const transport = this.root && this.root.querySelector('#dj-b-video-transport-mount');
                    const fader = this.root && this.root.querySelector('.dj-deck-b .dj-fader-strip');
                    if (!fader) return;
                    const videoTransportActive = this.deckBVizMode === 'video' && transport && transport.childNodes.length > 0;
                    if (videoTransportActive) {
                        transport.classList.remove('display-none');
                        transport.setAttribute('aria-hidden', 'false');
                        fader.classList.add('display-none');
                        return;
                    }
                    if (transport) {
                        transport.classList.add('display-none');
                        transport.setAttribute('aria-hidden', 'true');
                        transport.classList.remove('dj-video-fs-hide-controls');
                    }
                    const bPlaying = !!(audioElB && audioElB.src && !audioElB.paused && !audioElB.ended);
                    const visualActiveOnDeckB = (this.deckBVizMode === 'bars' || this.deckBVizMode === 'projectm' || this.deckBVizMode === 'blank' || this.deckBVizMode === 'video' || this.deckBVizMode === 'karaoke' || this.deckBVizMode === 'kbop');
                    const hideDeckBFader = visualActiveOnDeckB && !bPlaying;
                    fader.classList.toggle('display-none', hideDeckBFader);
                } catch (_) {}
            }

            updateLocalJogProgressRings() {
                try {
                    if (this.els.jogAProgressSvg) this.els.jogAProgressSvg.style.transform = `rotate(${-this.angleA}deg)`;
                    if (this.els.jogBProgressSvg) this.els.jogBProgressSvg.style.transform = `rotate(${-this.angleB}deg)`;
                } catch (_) {}
                const jogRingLen = 2 * Math.PI * 40;
                const step = (deckKey, barEl, svgEl, media) => {
                    if (!barEl || !svgEl) return;
                    let local = false;
                    try {
                        if (deckKey === 'b' && this.deckBVizMode === 'video' && !deckBVideoUserIdle && media) {
                            local = !!(String(media.currentSrc || media.src || '').trim());
                        } else {
                            local = !!(state && state.deckSourceMode && state.deckSourceMode[deckKey === 'b' ? 'b' : 'a'] === 'local');
                        }
                    } catch (_) {}
                    let dur = NaN;
                    let t = 0;
                    try {
                        if (media) {
                            dur = Number(media.duration);
                            t = Number(media.currentTime) || 0;
                        }
                    } catch (_) {}
                    const ok = local && media && media.src && Number.isFinite(dur) && dur > 0 && dur < 1e9;
                    if (!ok) {
                        svgEl.classList.add('dj-jog-progress-svg--hidden');
                        try {
                            barEl.style.strokeDasharray = String(jogRingLen);
                            barEl.style.strokeDashoffset = String(jogRingLen);
                        } catch (_) {}
                        return;
                    }
                    const p = Math.max(0, Math.min(1, t / dur));
                    try {
                        barEl.style.strokeDasharray = String(jogRingLen);
                        barEl.style.strokeDashoffset = String(jogRingLen * (1 - p));
                    } catch (_) {}
                    svgEl.classList.remove('dj-jog-progress-svg--hidden');
                };
                try { step('a', this.els.jogAProgressBar, this.els.jogAProgressSvg, getDeckAMediaForPlaybackState()); } catch (_) {}
                try { step('b', this.els.jogBProgressBar, this.els.jogBProgressSvg, getDeckBJogSeekMedia(this)); } catch (_) {}
            }

            syncPlayLabels() {
                try {
                    const aEl = getDeckAMediaForPlaybackState();
                    const aPlaying = !!(aEl && aEl.src && !aEl.paused && !aEl.ended);
                    const bPlaying = !!(audioElB && audioElB.src && !audioElB.paused && !audioElB.ended);
                    if (this.els.playA) {
                        this.els.playA.textContent = aPlaying ? 'Pause' : 'Play';
                        this.els.playA.classList.toggle('is-playing', aPlaying);
                    }
                    if (this.els.playB) {
                        this.els.playB.textContent = bPlaying ? 'Pause' : 'Play';
                        this.els.playB.classList.toggle('is-playing', bPlaying);
                    }
                    this.syncDeckBVideoTopChrome();
                } catch(_) {}
            }

            updateStationTitles() {
                try {
                    const srcA = (audioEl && String(audioEl.currentSrc || audioEl.src || '')) || '';
                    const titleAUseLocal = !!(state && state.deckSourceMode && state.deckSourceMode.a === 'local' && audioEl && audioEl.src);
                    let na = 'Deck A';
                    let titleALocalFile = '';
                    if (titleAUseLocal) {
                        const raw = (state.deckLocalDisplayName && state.deckLocalDisplayName.a) || '';
                        titleALocalFile = (raw && String(raw).trim()) ? String(raw).trim() : (typeof deriveTitleFromUrl === 'function' ? deriveTitleFromUrl(srcA) : 'Track');
                        if (!String(titleALocalFile || '').trim()) titleALocalFile = 'Track';
                    } else {
                        if (typeof currentStationIndex === 'number' && currentStationIndex >= 0 && stations[currentStationIndex]) {
                            na = stations[currentStationIndex].name || na;
                        } else if (radioInputEl && radioInputEl.value && typeof deriveTitleFromUrl === 'function') {
                            na = deriveTitleFromUrl(radioInputEl.value);
                        } else {
                            const aAud = getDeckARadioAudibleEl();
                            if (aAud && aAud.src && typeof deriveTitleFromUrl === 'function') {
                                na = deriveTitleFromUrl(aAud.src);
                            }
                        }
                    }
                    const radioModeB = !!(state && state.deckSourceMode && state.deckSourceMode.b === 'radio');
                    const bPlaying = !!(audioElB && audioElB.src && !audioElB.paused && !audioElB.ended);
                    const srcB = (audioElB && String(audioElB.currentSrc || audioElB.src || '')) || '';
                    const isBlobB = srcB.startsWith('blob:');
                    const vidPlaying = !!(this.deckBVizMode === 'video' && !deckBVideoUserIdle && this.deckBVideoEl
                        && (this.deckBVideoEl.src || this.deckBVideoEl.currentSrc)
                        && !this.deckBVideoEl.paused && !this.deckBVideoEl.ended);

                    let nb = '—';
                    let titleBUseLocal = false;
                    let titleBLocalFile = '';
                    const showDeckBRadioHeader = !!(radioModeB && bPlaying);
                    let showDeckBBadge = showDeckBRadioHeader;

                    if (radioModeB) {
                        if (stations && stations.length) {
                            let idxB = (typeof currentStationBIndex === 'number' && !isNaN(currentStationBIndex)) ? currentStationBIndex : 0;
                            if (typeof mixStationB !== 'undefined' && mixStationB && mixStationB.value !== undefined && mixStationB.value !== '') {
                                const pv = parseInt(mixStationB.value, 10);
                                if (!isNaN(pv)) idxB = pv;
                            }
                            idxB = Math.max(0, Math.min(stations.length - 1, idxB));
                            const stB = stations[idxB];
                            if (stB) {
                                const nm = stB.name && String(stB.name).trim();
                                nb = nm || (stB.url && typeof deriveTitleFromUrl === 'function' ? deriveTitleFromUrl(stB.url) : '—');
                            }
                        }
                        if ((nb === '—' || !nb) && srcB && typeof deriveTitleFromUrl === 'function' && !isBlobB) {
                            nb = deriveTitleFromUrl(srcB);
                        }
                    } else {
                        const stage = this.root && this.root.querySelector('.dj-deck-b-stage');
                        const textBOn = !!(stage && stage.classList.contains('dj-deck-b-text-mode'));
                        const parts = [];
                        if (textBOn) parts.push('TEXT-IN');
                        const m = this.deckBVizMode || 'idle';
                        if (m === 'projectm') parts.push('PROJECTM');
                        else if (m === 'bars') parts.push('AUDIO:BAR');
                        else if (m === 'karaoke') parts.push('KARAOKE');
                        else if (m === 'kbop') parts.push('K-BOP');
                        else if (m === 'blank') parts.push('BLANK');
                        if (parts.length) {
                            nb = parts.join(' · ');
                        } else if (bPlaying && ((state.deckSourceMode && state.deckSourceMode.b === 'local') || isBlobB)) {
                            titleBUseLocal = true;
                            const raw = (state.deckLocalDisplayName && state.deckLocalDisplayName.b) || '';
                            titleBLocalFile = (raw && String(raw).trim()) ? String(raw).trim() : (srcB && typeof deriveTitleFromUrl === 'function' && !isBlobB ? deriveTitleFromUrl(srcB) : 'Track');
                            if (!String(titleBLocalFile || '').trim()) titleBLocalFile = 'Track';
                            nb = '';
                        } else {
                            nb = '—';
                        }
                    }

                    if (titleAUseLocal) {
                        try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                    }
                    if (titleBUseLocal) {
                        try { setDjDeckRadioLoadingSpinner('b', false); } catch (_) {}
                    }
                    if (this.els.titleA) {
                        if (titleAUseLocal) {
                            this.els.titleA.classList.add('dj-head-station--local');
                            /* DECK A stays on the left badge; title is the file name only */
                            this.els.titleA.innerHTML = '<span class="dj-local-filename">' + escapeHtml(titleALocalFile) + '</span>';
                        } else {
                            this.els.titleA.classList.remove('dj-head-station--local');
                            this.els.titleA.textContent = na;
                        }
                    }
                    if (this.els.headBadgeB) this.els.headBadgeB.classList.toggle('display-none', !showDeckBBadge);
                    if (this.els.titleB) {
                        if (titleBUseLocal) {
                            this.els.titleB.classList.add('dj-head-station--local');
                            this.els.titleB.classList.remove('is-deck-b-label');
                            if (showDeckBBadge) {
                                this.els.titleB.innerHTML = '<span class="dj-local-filename">' + escapeHtml(titleBLocalFile) + '</span>';
                            } else {
                                this.els.titleB.innerHTML =
                                    '<span class="dj-local-prefix">DECK B</span><span class="dj-local-filename">' +
                                    escapeHtml(titleBLocalFile) +
                                    '</span>';
                            }
                        } else {
                            this.els.titleB.classList.remove('dj-head-station--local');
                            this.els.titleB.textContent = nb;
                            this.els.titleB.classList.toggle('is-deck-b-label', !radioModeB);
                        }
                        this.els.titleB.classList.toggle('display-none', !!(radioModeB && !bPlaying));
                    }
                    try {
                        const vEl = this.root && this.root.querySelector('#dj-head-video-title');
                        if (this.els.headVideoInline) this.els.headVideoInline.classList.toggle('display-none', !vidPlaying);
                        if (vEl) {
                            let vt = '—';
                            if (vidPlaying) {
                                const list = getDeckBVideoPlaybackSources();
                                if (list.length && this.deckBVideoIndex >= 0 && this.deckBVideoIndex < list.length) {
                                    const c = list[this.deckBVideoIndex];
                                    if (c) {
                                        vt = String(c.label || (c.url && typeof deriveNameFromUrl === 'function' ? deriveNameFromUrl(c.url) : '') || '—');
                                    }
                                }
                                const idleName = String(DECK_B_IDLE_LOGO_URL || 'assets/video/logo.mp4').replace(/^.*\//, '');
                                const blankTitle = !String(vt).trim() || vt === '—';
                                if (blankTitle && this.deckBVideoEl) {
                                    try {
                                        const vu = String(this.deckBVideoEl.currentSrc || this.deckBVideoEl.src || '').trim();
                                        const pathOnly = vu.split('?')[0].toLowerCase();
                                        if (pathOnly.endsWith(idleName.toLowerCase())) vt = idleName;
                                    } catch (_) {}
                                }
                            }
                            vEl.textContent = vt;
                        }
                    } catch (_) {}
                } catch(_) {}
            }

            syncFxLightsFromState() {
                try {
                    try {
                        const karaA = this.root && this.root.querySelector('#dj-a-sfx-w1');
                        const karaB = this.root && this.root.querySelector('#dj-b-sfx-w1');
                        const ytA = this.root && this.root.querySelector('#dj-a-sfx-w2');
                        const ytB = this.root && this.root.querySelector('#dj-b-sfx-w2');
                        const karaOn = this.deckBVizMode === 'karaoke';
                        const kbopOn = this.deckBVizMode === 'kbop';
                        if (karaA) karaA.classList.toggle('on', karaOn);
                        if (karaB) karaB.classList.toggle('on', karaOn);
                        if (ytA) ytA.classList.toggle('on', kbopOn);
                        if (ytB) ytB.classList.toggle('on', kbopOn);
                    } catch (_) {}
                    if (!state || !state.fx) return;
                    const videoOn = this.deckBVizMode === 'video';
                    const kon = !!(state.fx.tk && state.fx.tk.on);
                    const avatarOn = !!(typeof webmOn !== 'undefined' && webmOn);
                    const mixOpenUi = !!(mixPanel && !mixPanel.classList.contains('display-none') && mixPanel.classList.contains('open'));
                    const stationOpenUi = (typeof isTopMenuOpen === 'function') && isTopMenuOpen();
                    if (this.els.fxLow) this.els.fxLow.classList.toggle('on', videoOn);
                    if (this.els.fxLowB) this.els.fxLowB.classList.toggle('on', videoOn);
                    const queuePanelOn = !!this.deckBQueueVisible;
                    const mediaPanelOn = !!this.deckBMediaPanelVisible;
                    const deckBPanelStrip = queuePanelOn || mediaPanelOn;
                    const setPanelBtn = (el, panelOn) => {
                        if (!el) return;
                        el.classList.toggle('on', !!panelOn);
                        el.classList.toggle('deck-b-active', !!panelOn);
                    };
                    setPanelBtn(this.els.queueA, queuePanelOn);
                    setPanelBtn(this.els.queueB, queuePanelOn);
                    if (typeof syncDjTextInDeckLights === 'function') {
                        try { syncDjTextInDeckLights(); } catch(_) {}
                    }
                    if (this.els.fxHighA) this.els.fxHighA.classList.toggle('on', avatarOn);
                    if (this.els.fxHighB) this.els.fxHighB.classList.toggle('on', avatarOn);
                    if (this.els.fxTrebleA) this.els.fxTrebleA.classList.toggle('on', mixOpenUi);
                    if (this.els.fxTrebleB) this.els.fxTrebleB.classList.toggle('on', mixOpenUi);
                    if (this.els.fxDistortA) {
                        this.els.fxDistortA.classList.toggle('on', mediaPanelOn || (stationOpenUi && !deckBPanelStrip));
                        this.els.fxDistortA.classList.toggle('deck-b-active', mediaPanelOn);
                    }
                    setPanelBtn(this.els.fxDistortB, mediaPanelOn);
                    if (this.root && typeof syncDjBeatFxKnobActiveDom === 'function') syncDjBeatFxKnobActiveDom(this.root);
                } catch (_) {}
            }

            animateFrame() {
                this.animId = requestAnimationFrame(() => this.animateFrame());
                try {
                    if (state && state.fx && state.fx.loopMusical && state.fx.loopMusical.on && typeof syncDjMusicalLoopDelayTime === 'function') {
                        syncDjMusicalLoopDelayTime();
                    }
                } catch (_) {}
                try {
                    if (state && state.analyserNodeA && state.analyserNodeB) {
                        if (!this.bufA || this.bufA.length !== state.analyserNodeA.frequencyBinCount) {
                            this.bufA = new Uint8Array(state.analyserNodeA.frequencyBinCount);
                        }
                        if (!this.bufB || this.bufB.length !== state.analyserNodeB.frequencyBinCount) {
                            this.bufB = new Uint8Array(state.analyserNodeB.frequencyBinCount);
                        }
                        state.analyserNodeA.getByteFrequencyData(this.bufA);
                        state.analyserNodeB.getByteFrequencyData(this.bufB);
                        let sumA = 0, sumB = 0;
                        const n = 24;
                        for (let i = 0; i < n; i++) { sumA += this.bufA[i]; sumB += this.bufB[i]; }
                        const driveA = (sumA / (n * 255));
                        const driveB = (sumB / (n * 255));
                        const aMed = getDeckAMediaForPlaybackState();
                        const aPlay = !!(aMed && !aMed.paused && aMed.src);
                        const bPlay = !!(audioElB && !audioElB.paused && audioElB.src);
                        this.angleA += (0.35 + driveA * 4.2) * (aPlay ? 1 : 0.08);
                        this.angleB += (0.35 + driveB * 4.2) * (bPlay ? 1 : 0.08);
                        if (this.els.jogA) this.els.jogA.style.transform = `rotate(${this.angleA}deg)`;
                        if (this.els.jogB) this.els.jogB.style.transform = `rotate(${this.angleB}deg)`;
                    }
                } catch(_) {}
                try { this.updateLocalJogProgressRings(); } catch (_) {}
                this.syncPlayLabels();
                this.updateStationTitles();
                this.syncFxLightsFromState();
            }

            init() {
                container.innerHTML = '';
                try { initAudio(); } catch(_) {}

                this.abortCtrl = new AbortController();
                const sig = { signal: this.abortCtrl.signal };

                const root = document.createElement('div');
                root.id = 'dj-visual-root';
                root.setAttribute('aria-label', 'DJ Decks');

                root.innerHTML = `
<div class="dj-layout">
  <div class="dj-deck-head-row" aria-label="Deck stations and video">
    <div class="dj-head-line dj-head-line--primary">
        <span class="dj-head-cluster dj-head-cluster--a">
        <span class="dj-deck-badge dj-deck-badge--head">DECK A</span>
        <span class="dj-head-station-name dj-head-station-name--a" id="dj-station-a-title">—</span>
        <span class="dj-station-load-spinner display-none" id="dj-station-a-load-spinner" role="status" aria-live="polite" aria-label="Station buffering"></span>
      </span>
      <span class="dj-head-datetime" id="dj-head-datetime"></span>
      <span class="dj-head-automix-timer display-none" id="dj-head-automix-timer" aria-live="polite" aria-hidden="true"></span>
      <span class="dj-head-cluster dj-head-cluster--b">
        <span class="dj-deck-badge dj-deck-badge--b dj-deck-badge--head display-none" id="dj-head-deck-b-badge">DECK B</span>
        <span class="dj-head-station-name dj-head-station-name--b dj-station-title--b display-none" id="dj-station-b-title">—</span>
        <span class="dj-station-load-spinner dj-station-load-spinner--b display-none" id="dj-station-b-load-spinner" role="status" aria-live="polite" aria-label="Deck B station buffering"></span>
        <span class="dj-head-video-inline display-none" id="dj-head-video-inline">
          <span class="dj-head-video-badge">VIDEO</span>
          <span class="dj-head-video-title" id="dj-head-video-title">—</span>
        </span>
      </span>
    </div>
  </div>
  <div class="dj-deck-columns">
  <section class="dj-deck dj-deck-a">
    <div class="dj-deck-body">
      <div class="dj-stack">
        <div class="dj-middle-row dj-deck-mosaic">
          <div class="dj-beatmap-strip" aria-label="Deck A beat map">
            <div class="dj-beatmap-main">
              <div class="dj-beatmap-fx-knobs" aria-label="Deck A beat FX">
                <div class="eq-col eq-strip" aria-label="TK filter frequency"><div class="knob-wrap" id="dj-knob-a-tk" style="--knob-color:#b8980c"><div class="knob-indicator"></div><div class="knob-value">1200</div></div><div class="eq-label">TK</div></div>
                <div class="eq-col eq-strip" aria-label="Musical loop bars"><div class="knob-wrap" id="dj-knob-a-loop" style="--knob-color:#2a8fa0"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label">Loop</div></div>
                <div class="eq-col eq-strip" aria-label="Arpeggiator rate"><div class="knob-wrap" id="dj-knob-a-arp" style="--knob-color:#8647a8"><div class="knob-indicator"></div><div class="knob-value">4</div></div><div class="eq-label">Arp</div></div>
              </div>
              <div class="dj-beatmap-pads">
                <div class="dj-beatmap-col" aria-label="Deck A beat layers">
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="0">High</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="1">Drum</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="2">Clap</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="3">Perc</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="4">Tom</button>
                </div>
                <div class="dj-beatmap-col" aria-label="Deck A key samples">
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="10">KEY</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="11">KICK</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="12">BELL</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="13">RIM</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="14">HAR</button>
                </div>
                <div class="dj-beatmap-col" aria-label="Deck A sample pads">
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="5">WAA</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="6">WAAA</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="7">AIR📢</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="8">FX1</button>
                  <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="a" data-beat-slot="9">FX2</button>
                </div>
              </div>
            </div>
            <input type="file" id="dj-a-beat-file" class="display-none" accept="audio/*" aria-hidden="true" />
            <div class="dj-beatmap-tempo-col">
              <div class="dj-beatmap-tempo-stack">
                <div class="dj-beatmap-tempo-slider-wrap">
                  <input type="range" id="dj-a-beat-bpm" class="dj-beatmap-bpm-slider" min="60" max="240" step="1" value="120" aria-label="Deck A beat tempo BPM" />
                </div>
                <div class="dj-beatmap-tempo-footer">
                  <button type="button" class="btn dj-fx dj-beatmap-sync" id="dj-a-beat-sync">Sync</button>
                </div>
              </div>
            </div>
          </div>
          <div class="dj-jog-wrap" title="Load File ⊕">
            <div class="dj-jog" id="dj-jog-a" aria-hidden="true">
              <svg class="dj-jog-progress-svg dj-jog-progress-svg--a dj-jog-progress-svg--hidden" id="dj-jog-a-progress" viewBox="0 0 100 100" aria-hidden="true">
                <g transform="translate(50 50) rotate(-90)">
                  <circle class="dj-jog-progress-track" r="40" cx="0" cy="0" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="2.4" />
                  <circle id="dj-jog-a-progress-bar" class="dj-jog-progress-bar" r="40" cx="0" cy="0" fill="none" stroke-width="2.4" stroke-linecap="round" />
                  <circle class="dj-jog-progress-hit" r="40" cx="0" cy="0" fill="none" stroke="rgba(0,0,0,0)" stroke-width="14" />
                </g>
              </svg>
              <span class="dj-jog-marker"></span>
            </div>
            <span class="dj-jog-deck-label dj-jog-deck-label--a">DECK A</span>
          </div>
          <div class="dj-deck-vol-col" aria-label="Deck A volume">
            <div class="dj-deck-vol-stack">
              <div class="dj-deck-vol-wrap">
                <input type="range" id="dj-deck-a-vol" class="dj-deck-vol" min="0" max="1" step="0.01" value="1" />
              </div>
              <button type="button" class="btn dj-fx dj-deck-vol-mute" id="dj-deck-a-vol-mute" aria-pressed="false" title="Mute Deck A (restore previous level on second press)">VOL</button>
            </div>
          </div>
          <div class="dj-controls-grid dj-btn-block" aria-label="Deck A visuals, effects, and transport">
            <div class="dj-knob-top-row" aria-label="Deck A EQ">
              <div class="eq-col eq-strip" aria-label="Deck A low"><div class="knob-wrap" id="dj-knob-a-low" style="--knob-color:#06D001"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#06D001">Lo</div></div>
              <div class="eq-col eq-strip" aria-label="Deck A mid"><div class="knob-wrap" id="dj-knob-a-mid" style="--knob-color:#9BEC00"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#9BEC00">Med</div></div>
              <div class="eq-col eq-strip" aria-label="Deck A high"><div class="knob-wrap" id="dj-knob-a-high" style="--knob-color:#F94C10"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#F94C10">High</div></div>
              <div class="eq-col eq-strip" aria-label="Deck A gain"><div class="knob-wrap" id="dj-knob-a-gain" style="--knob-color:#FF0B55"><div class="knob-indicator"></div><div class="knob-value">1</div></div><div class="eq-label" style="color:#FF0B55">Gain</div></div>
            </div>
            <button type="button" class="btn dj-fx" id="dj-vis-bars" title="Switch to Audio Bars visual">AUDIO:BAR</button>
            <button type="button" class="btn dj-fx" id="dj-vis-blank" title="Switch to no visual (Blank mode)">NO VISUAL</button>
            <button type="button" class="btn dj-fx" id="dj-vis-pm2" title="Switch visual to ProjectM v2">ProjectM</button>

            <button type="button" class="btn dj-fx" id="dj-fx-high" title="Toggle avatar / WebM overlay (centres in Deck B while a visual is playing there)">Avatar</button>
            <button type="button" class="btn dj-fx" id="dj-a-queue" title="Show local file queues (Deck A &amp; B)">QUEUE</button>
            <button type="button" class="btn dj-fx" id="dj-a-sfx-wav" title="Switch to Digital Radio visual">RADIO</button>

            <button type="button" class="btn dj-fx" id="dj-fx-treble" title="Open or close Mix Settings (bottom panel)">Mixer</button>
            <button type="button" class="btn dj-fx" id="dj-fx-low" title="Toggle Deck B video player">Video</button>
            <button type="button" class="btn dj-fx" id="dj-fx-distort" title="Open or close station list (top menu)">Stations</button>

            <button type="button" class="btn dj-sfx" id="dj-a-sfx-w2" title="Open K-BOP in Deck B (toggle)">K-BOP</button>
            <button type="button" class="btn dj-sfx" id="dj-a-sfx-w1" title="Open Karaoke Nerds in Deck B (toggle)">KARAOKE</button>
            <button type="button" class="btn dj-fx" id="dj-fx-tk" title="Open or close the TEXT-IN panel (rising text on the main screen)">TEXT-IN</button>

            <button type="button" class="dj-play" id="dj-play-a" title="Play or pause Deck A">Play</button>
            <button type="button" class="dj-tbtn" id="dj-a-next" title="Tune to another random station">Rand ⟶</button>
            <button type="button" class="dj-tbtn" id="dj-a-prev" title="Previous station (history)">⟵ Prev</button>
          </div>
        </div>
      </div>
    </div>
  </section>
  <button type="button" class="dj-deck-splitter" id="dj-deck-splitter"
    aria-label="Resize Deck A and Deck B"
    aria-orientation="vertical"
    aria-valuemin="15"
    aria-valuemax="85"
    aria-valuenow="50"
    title="Drag sideways to give Deck A more or less room vs Deck B and the fader. Double-click for equal columns.">
    <span class="dj-deck-splitter-grip" aria-hidden="true"></span>
  </button>
  <section class="dj-deck dj-deck-b">
    <div id="dj-b-video-transport-mount" class="dj-b-video-transport-mount display-none" aria-hidden="true"></div>
    <div class="dj-fader-strip">
      <div class="dj-autofade-wrap">
        <button type="button" class="dj-autofade-btn" id="dj-autofade" title="Auto-fade to the opposite deck (long-press for duration)">Auto-Fade</button>
        <div class="dj-autofade-panel" id="dj-autofade-panel" aria-hidden="true">
          <div class="dj-autofade-title">Auto-fade time</div>
          <div class="dj-autofade-row">
            <input type="range" id="dj-autofade-duration" min="2" max="15" step="1" value="5" aria-label="Auto-fade duration seconds">
            <span class="dj-autofade-readout" id="dj-autofade-readout">5s</span>
          </div>
          <div class="dj-autofade-station-block">
            <div class="dj-autofade-station-title">On Auto-Fade</div>
            <label class="dj-autofade-station-check" title="When on, Auto-Fade and Auto-Mix switch the incoming deck to a random station before each crossfade. Turning on Auto-Mix checks this. Uncheck to keep each deck's current station or local file until you use RAND or pick a station manually.">
              <input type="checkbox" id="dj-autofade-change-station" />
              <span>Change station before fading</span>
            </label>
          </div>
          <div class="dj-autofade-station-block">
            <div class="dj-autofade-station-title">Radio station change</div>
            <label class="dj-autofade-station-check">
              <input type="checkbox" id="dj-radio-station-xfade-en" />
              <span>Fade between stations (Deck A)</span>
            </label>
            <div class="dj-autofade-row">
              <input type="range" id="dj-radio-station-xfade-sec" min="0" max="15" step="0.5" value="0" aria-label="Station crossfade duration seconds" disabled>
              <span class="dj-autofade-readout" id="dj-radio-station-xfade-readout">0s</span>
            </div>
          </div>
        </div>
      </div>
      <span class="dj-fader-label-a">A</span>
      <div class="dj-crossfader-mount">
        <span class="dj-crossfader-wrap">
          <input type="range" id="dj-crossfader" min="0" max="1" step="0.01" value="0" aria-label="Crossfade from A to B" />
        </span>
      </div>
      <span class="dj-fader-label-b">B</span>
      <div class="dj-automix-wrap">
        <button type="button" class="dj-automix-btn" id="dj-automix" title="Toggle auto-mix (long-press for interval, session limit, and fade-out)">Auto-Mix</button>
        <div class="dj-automix-panel" id="dj-automix-panel" aria-hidden="true">
          <div class="dj-automix-title">Auto-mix interval</div>
          <div class="dj-automix-row">
            <input type="range" id="dj-automix-min" min="1" max="20" step="1" value="2" aria-label="Auto-mix minimum minutes">
            <span class="dj-automix-readout" id="dj-automix-min-readout">2m</span>
          </div>
          <div class="dj-automix-row">
            <input type="range" id="dj-automix-max" min="1" max="20" step="1" value="20" aria-label="Auto-mix maximum minutes">
            <span class="dj-automix-readout" id="dj-automix-max-readout">20m</span>
          </div>
          <div class="dj-automix-limit-section">
            <div class="dj-automix-title">Session limit</div>
            <label class="dj-automix-check-row">
              <input type="checkbox" id="dj-automix-limit-en" aria-label="Hold auto-mix to a session time limit">
              <span>Limit session length <span id="dj-automix-limit-remaining" class="dj-automix-limit-remaining" aria-live="polite"></span></span>
            </label>
            <div class="dj-automix-row">
              <input type="range" id="dj-automix-limit-min" min="5" max="300" step="1" value="60" aria-label="Auto-mix session time limit minutes">
              <span class="dj-automix-readout" id="dj-automix-limit-readout">1h</span>
            </div>
          </div>
          <div class="dj-automix-nextfade-section">
            <div class="dj-automix-title">Next crossfade</div>
            <div class="dj-automix-nextfade-row">
              <span>Starts in</span>
              <span class="dj-automix-nextfade-readout" id="dj-automix-next-fade" aria-live="polite"></span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <button type="button" class="dj-visual-title-btn display-none" id="dj-b-visual-back-btn">DECK B</button>
    <div class="dj-deck-body dj-deck-body-b">
      <div class="dj-deck-b-stage">
        <div id="dj-deck-b-controls" class="dj-deck-b-controls">
          <div class="dj-stack">
            <div class="dj-middle-row dj-deck-mosaic">
              <div class="dj-beatmap-strip" aria-label="Deck B beat map">
                <div class="dj-beatmap-main">
                  <div class="dj-beatmap-fx-knobs" aria-label="Deck B beat FX">
                    <div class="eq-col eq-strip" aria-label="Reverb wet mix"><div class="knob-wrap" id="dj-knob-b-reverb" style="--knob-color:#ff4040"><div class="knob-indicator"></div><div class="knob-value">50</div></div><div class="eq-label">REVERB</div></div>
                    <div class="eq-col eq-strip" aria-label="Flanger depth"><div class="knob-wrap" id="dj-knob-b-flanger" style="--knob-color:#3fff00"><div class="knob-indicator"></div><div class="knob-value">50</div></div><div class="eq-label">FLANGER</div></div>
                    <div class="eq-col eq-strip" aria-label="Rhythmic CUT: tap toggles gate; drag adjusts chop rate vs BPM"><div class="knob-wrap" id="dj-knob-b-cut" style="--knob-color:#bada55"><div class="knob-indicator"></div><div class="knob-value">50</div></div><div class="eq-label">CUT</div></div>
                  </div>
                  <div class="dj-beatmap-pads">
                    <div class="dj-beatmap-col" aria-label="Deck B CHR–HARP">
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="0">CHR</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="1">CYM</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="2">SYN</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="3">HARM</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="4">HARP</button>
                    </div>
                    <div class="dj-beatmap-col" aria-label="Deck B SRE–CHG">
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="10">SRE</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="11">BME</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="12">HRN</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="13">CRW</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="14">CHG</button>
                    </div>
                    <div class="dj-beatmap-col" aria-label="Deck B FX3–FX7">
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="5">FX3</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="6">FX4</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="7">FX5</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="8">FX6</button>
                      <button type="button" class="btn dj-fx dj-beat-pad" data-beat-deck="b" data-beat-slot="9">FX7</button>
                    </div>
                  </div>
                </div>
                <input type="file" id="dj-b-beat-file" class="display-none" accept="audio/*" aria-hidden="true" />
                <div class="dj-beatmap-tempo-col">
                  <div class="dj-beatmap-tempo-stack">
                    <div class="dj-beatmap-tempo-slider-wrap">
                      <input type="range" id="dj-b-beat-bpm" class="dj-beatmap-bpm-slider" min="60" max="240" step="1" value="120" aria-label="Deck B beat tempo BPM" />
                    </div>
                    <div class="dj-beatmap-tempo-footer">
                      <button type="button" class="btn dj-fx dj-beatmap-sync" id="dj-b-beat-sync">Sync</button>
                    </div>
                  </div>
                </div>
              </div>
              <div class="dj-jog-wrap" title="Load File ⊕">
                <div class="dj-jog" id="dj-jog-b" aria-hidden="true">
                  <svg class="dj-jog-progress-svg dj-jog-progress-svg--b dj-jog-progress-svg--hidden" id="dj-jog-b-progress" viewBox="0 0 100 100" aria-hidden="true">
                    <g transform="translate(50 50) rotate(-90)">
                      <circle class="dj-jog-progress-track" r="40" cx="0" cy="0" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="2.4" />
                      <circle id="dj-jog-b-progress-bar" class="dj-jog-progress-bar" r="40" cx="0" cy="0" fill="none" stroke-width="2.4" stroke-linecap="round" />
                      <circle class="dj-jog-progress-hit" r="40" cx="0" cy="0" fill="none" stroke="rgba(0,0,0,0)" stroke-width="14" />
                    </g>
                  </svg>
                  <span class="dj-jog-marker"></span>
                </div>
                <span class="dj-jog-deck-label dj-jog-deck-label--b">DECK B</span>
              </div>
              <div class="dj-deck-vol-col" aria-label="Deck B volume">
                <div class="dj-deck-vol-stack">
                  <div class="dj-deck-vol-wrap">
                    <input type="range" id="dj-deck-b-vol" class="dj-deck-vol" min="0" max="1" step="0.01" value="1" />
                  </div>
                  <button type="button" class="btn dj-fx dj-deck-vol-mute" id="dj-deck-b-vol-mute" aria-pressed="false" title="Mute Deck B (restore previous level on second press)">VOL</button>
                </div>
              </div>
              <div class="dj-controls-grid dj-btn-block" aria-label="Deck B visual screen &amp; transport">
                <div class="dj-knob-top-row" aria-label="Deck B EQ">
                  <div class="eq-col eq-strip" aria-label="Deck B low"><div class="knob-wrap" id="dj-knob-b-low" style="--knob-color:#9AB3F5"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#9AB3F5">Lo</div></div>
                  <div class="eq-col eq-strip" aria-label="Deck B mid"><div class="knob-wrap" id="dj-knob-b-mid" style="--knob-color:#0ff"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#0ff">Med</div></div>
                  <div class="eq-col eq-strip" aria-label="Deck B high"><div class="knob-wrap" id="dj-knob-b-high" style="--knob-color:#f0f"><div class="knob-indicator"></div><div class="knob-value">0</div></div><div class="eq-label" style="color:#f0f">High</div></div>
                  <div class="eq-col eq-strip" aria-label="Deck B gain"><div class="knob-wrap" id="dj-knob-b-gain" style="--knob-color:#5800FF"><div class="knob-indicator"></div><div class="knob-value">1</div></div><div class="eq-label" style="color:#5800FF">Gain</div></div>
                </div>
                <button type="button" class="btn dj-fx" id="dj-b-vis-bars" title="Show Deck A audio bars here (replaces controls)">AUDIO:BAR</button>
                <button type="button" class="btn dj-fx" id="dj-b-vis-blank" title="Return to deck controls">NO VISUAL</button>
                <button type="button" class="btn dj-fx" id="dj-b-vis-pm2" title="Show ProjectM driven by Deck A (replaces controls)">ProjectM</button>

                <button type="button" class="btn dj-fx" id="dj-b-fx-high" title="Toggle avatar in Deck B viz area (arrow keys still nudge position)">Avatar</button>
                <button type="button" class="btn dj-fx" id="dj-b-queue" title="Show local file queues (Deck A &amp; B)">QUEUE</button>
                <button type="button" class="btn dj-fx" id="dj-b-sfx-wav" title="Switch to Digital Radio visual">RADIO</button>

                <button type="button" class="btn dj-fx" id="dj-b-fx-treble" title="Open or close Mix Settings (bottom panel)">Mixer</button>
                <button type="button" class="btn dj-fx" id="dj-b-fx-low" title="Toggle Deck B video player">Video</button>
                <button type="button" class="btn dj-fx" id="dj-b-fx-distort" title="Show or hide station cycle list and video queue (Deck B)">Stations</button>

                <button type="button" class="btn dj-sfx" id="dj-b-sfx-w2" title="Open K-BOP in Deck B (toggle)">K-BOP</button>
                <button type="button" class="btn dj-sfx" id="dj-b-sfx-w1" title="Open Karaoke Nerds in Deck B (toggle)">KARAOKE</button>
                <button type="button" class="btn dj-fx" id="dj-b-fx-tk" title="Open or close TEXT-IN, displaying rising text inside the Deck B player">TEXT-IN</button>

                <button type="button" class="dj-play" id="dj-play-b" title="Play or pause Deck B">Play</button>
                <button type="button" class="dj-tbtn" id="dj-b-next" title="Tune deck B to another random station">Rand ⟶</button>
                <button type="button" class="dj-tbtn" id="dj-b-prev" title="Previous Station B">⟵ Prev</button>
              </div>
            </div>
          </div>
        </div>
        <div id="dj-deck-b-text-stage" class="dj-deck-b-text-stage" aria-hidden="true">
          <div id="dj-deck-b-text-overlay-layer" class="dj-deck-b-text-overlay-layer"></div>
        </div>
        <div id="dj-deck-b-viz-layer" class="dj-deck-b-viz-layer" aria-hidden="true">
          <div id="dj-deck-b-viz-mount" class="dj-deck-b-viz-mount"></div>
          <div id="dj-deck-b-viz-placeholder" class="dj-deck-b-viz-placeholder display-none">Station A · NO VISUAL to show decks</div>
          <div id="dj-deck-b-queue-panel" class="dj-deck-b-queue-panel display-none" aria-label="Local file queues">
            <div class="dj-queue-header">Local playlists · Deck A &amp; B</div>
            <div class="dj-queue-columns">
              <div class="dj-queue-col" data-deck="a">
                <div class="dj-queue-col-head">
                  <div class="dj-queue-col-title">Deck A queue</div>
                  <div class="dj-queue-col-head-btns">
                    <button type="button" class="dj-queue-add" id="dj-queue-add-a">Add files…</button>
                    <button type="button" class="dj-queue-folder" id="dj-queue-folder-a" title="Add all audio/video files from a folder">Folder…</button>
                    <button type="button" class="dj-queue-add-url" id="dj-queue-url-deck-a" title="Add a URL (radio streams go to Station cycle)">Add URL…</button>
                  </div>
                </div>
                <ul id="dj-queue-list-a" class="dj-queue-list"></ul>
              </div>
              <div class="dj-queue-col" data-deck="b">
                <div class="dj-queue-col-head">
                  <div class="dj-queue-col-title">Deck B queue</div>
                  <div class="dj-queue-col-head-btns">
                    <button type="button" class="dj-queue-add" id="dj-queue-add-b">Add files…</button>
                    <button type="button" class="dj-queue-folder" id="dj-queue-folder-b" title="Add all audio/video files from a folder">Folder…</button>
                    <button type="button" class="dj-queue-add-url" id="dj-queue-url-deck-b" title="Add a URL (radio streams go to Station cycle)">Add URL…</button>
                  </div>
                </div>
                <ul id="dj-queue-list-b" class="dj-queue-list"></ul>
              </div>
            </div>
          </div>
          <div id="dj-deck-b-media-panel" class="dj-deck-b-media-panel display-none" aria-label="Station cycle and video queue">
            <div class="dj-queue-tools">
              <div class="dj-station-cycle" aria-label="Station cycle filter">
                <div class="dj-station-cycle-radio-anchor">
                  <div class="dj-station-cycle-top">
                    <div class="dj-station-cycle-head">Radio station cycle list</div>
                    <button type="button" class="dj-video-btn dj-b-video-radio" id="dj-b-video-radio" title="Add a radio stream URL to the Station cycle list">Radio</button>
                  </div>
                  <div id="dj-radio-url-popout" class="dj-radio-popout" aria-hidden="true" role="dialog" aria-label="Radio station URL">
                    <div class="dj-radio-popout-head">
                      <span class="dj-station-cycle-head" style="margin:0;">Radio station URL</span>
                      <button type="button" class="dj-radio-popout-close" id="dj-radio-url-close" aria-label="Close">×</button>
                    </div>
                    <div class="dj-url-entry-row">
                      <input id="dj-radio-url-input" class="dj-url-input" type="text" placeholder="Paste Icecast / stream URL" autocomplete="off" />
                      <button type="button" class="dj-url-btn" id="dj-radio-url-add">Add station</button>
                    </div>
                  </div>
                </div>
                <div id="dj-station-cycle-list" class="dj-station-cycle-list"></div>
              </div>
              <div class="dj-media-queue dj-video-queue" aria-label="Video queue">
                <div class="dj-media-head">
                  <div class="dj-station-cycle-head" style="margin:0;">Video queue</div>
                  <div class="dj-media-head-actions">
                    <button type="button" class="dj-url-btn" id="dj-media-loop">Loop</button>
                    <button type="button" class="dj-url-btn" id="dj-media-all" title="Play through all queued videos; turn off to loop a single clip (use Loop)">All</button>
                    <button type="button" class="dj-url-btn" id="dj-media-shuffle">Shuffle</button>
                  </div>
                </div>
                <div class="dj-video-tools-block">
                  <div class="dj-media-deck-b-tools" aria-label="Deck B video URL and file loader">
                    <button type="button" class="dj-video-btn" id="dj-b-video-local" title="Pick local video file(s)">Local</button>
                    <button type="button" class="dj-video-btn" id="dj-b-video-folder" title="Add all videos from a folder to the queue">Folder</button>
                    <input id="dj-b-video-input" class="dj-video-loader-input" type="text" placeholder="Paste video URL" autocomplete="off" />
                    <button type="button" class="dj-video-btn" id="dj-b-video-load" title="Load URL/local into Deck B video player">Load</button>
                    <button type="button" class="dj-video-btn" id="dj-b-video-q-a" title="Load into Deck A queue">A</button>
                    <button type="button" class="dj-video-btn" id="dj-b-video-q-b" title="Load into Deck B queue">B</button>
                  </div>
                </div>
                <ul id="dj-media-queue-list" class="dj-media-list"></ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
  </div>
</div>`;

                container.appendChild(root);
                this.root = root;
                this.landscapeDeckSplit = 0.5;
                this.deckBCollapsed = false;
                try {
                    const raw = localStorage.getItem('dj.landscapeDeckSplit.v1');
                    if (raw != null) {
                        const x = parseFloat(raw);
                        // Clamp matches the COLLAPSE_THRESHOLD inside applyLandscapeDeckSplit: a
                        // stored 0.90 represents "Deck B collapsed", and the explicit collapsed
                        // flag below decides whether to enter that state on load.
                        if (!Number.isNaN(x)) this.landscapeDeckSplit = Math.max(0.15, Math.min(0.90, x));
                    }
                } catch (_) {}
                try {
                    this.deckBCollapsed = localStorage.getItem('dj.deckBCollapsed.v1') === '1';
                } catch (_) {}
                // If the user had Deck B collapsed when they last left, re-enter that state by
                // passing a value at the threshold so applyLandscapeDeckSplit re-derives the
                // collapsed flag and class.
                this.applyLandscapeDeckSplit(this.deckBCollapsed ? 0.90 : this.landscapeDeckSplit);
                this.bindLandscapeDeckSplitter();
                this.onResize();
                window.addEventListener('resize', this.resizeHandler);
                window.addEventListener('orientationchange', this.resizeHandler);

                try {
                    if (typeof ResizeObserver !== 'undefined') {
                        this.deckVolResizeObs = new ResizeObserver(() => {
                            if (this._deckLayoutResizeRaf) return;
                            this._deckLayoutResizeRaf = requestAnimationFrame(() => {
                                this._deckLayoutResizeRaf = null;
                                try {
                                    this.syncDeckMosaicScale();
                                } catch (_) {}
                            });
                        });
                        root.querySelectorAll('.dj-deck-mosaic').forEach((el) => this.deckVolResizeObs.observe(el));
                    }
                } catch (_) {}

                try {
                    const vizLayer = root.querySelector('#dj-deck-b-viz-layer');
                    if (vizLayer && typeof ResizeObserver !== 'undefined') {
                        this.deckBVizResizeObs = new ResizeObserver(() => {
                            try {
                                if ((this.deckBVizMode === 'projectm' || this.deckBVizMode === 'bars') && typeof this.deckBVizPmResize === 'function') {
                                    this.deckBVizPmResize();
                                }
                            } catch (_) {}
                        });
                        this.deckBVizResizeObs.observe(vizLayer);
                    }
                } catch (_) {}

                this.els.titleA = root.querySelector('#dj-station-a-title');
                this.els.titleB = root.querySelector('#dj-station-b-title');
                this.els.headBadgeB = root.querySelector('#dj-head-deck-b-badge');
                this.els.headVideoInline = root.querySelector('#dj-head-video-inline');
                this.els.visualBackBtn = root.querySelector('#dj-b-visual-back-btn');
                this.els.playA = root.querySelector('#dj-play-a');
                this.els.playB = root.querySelector('#dj-play-b');
                this.els.jogA = root.querySelector('#dj-jog-a');
                this.els.jogB = root.querySelector('#dj-jog-b');
                this.els.jogAProgressSvg = root.querySelector('#dj-jog-a-progress');
                this.els.jogBProgressSvg = root.querySelector('#dj-jog-b-progress');
                this.els.jogAProgressBar = root.querySelector('#dj-jog-a-progress-bar');
                this.els.jogBProgressBar = root.querySelector('#dj-jog-b-progress-bar');
                this.els.fxLow = root.querySelector('#dj-fx-low');
                this.els.fxTk = root.querySelector('#dj-fx-tk');
                this.els.queueA = root.querySelector('#dj-a-queue');
                this.els.fxHighA = root.querySelector('#dj-fx-high');
                this.els.fxTrebleA = root.querySelector('#dj-fx-treble');
                this.els.fxDistortA = root.querySelector('#dj-fx-distort');
                this.els.fxLowB = root.querySelector('#dj-b-fx-low');
                this.els.fxTkB = root.querySelector('#dj-b-fx-tk');
                this.els.queueB = root.querySelector('#dj-b-queue');
                this.els.fxHighB = root.querySelector('#dj-b-fx-high');
                this.els.fxTrebleB = root.querySelector('#dj-b-fx-treble');
                this.els.fxDistortB = root.querySelector('#dj-b-fx-distort');
                this.els.deckVolA = root.querySelector('#dj-deck-a-vol');
                this.els.deckVolB = root.querySelector('#dj-deck-b-vol');
                this.els.deckVolMuteA = root.querySelector('#dj-deck-a-vol-mute');
                this.els.deckVolMuteB = root.querySelector('#dj-deck-b-vol-mute');
                const djCross = root.querySelector('#dj-crossfader');
                const btnAutoFade = root.querySelector('#dj-autofade');
                const autoFadeWrap = root.querySelector('.dj-autofade-wrap');
                const autoFadePanel = root.querySelector('#dj-autofade-panel');
                const autoFadeDuration = root.querySelector('#dj-autofade-duration');
                const autoFadeReadout = root.querySelector('#dj-autofade-readout');
                const radioStationXfadeEn = root.querySelector('#dj-radio-station-xfade-en');
                const radioStationXfadeSec = root.querySelector('#dj-radio-station-xfade-sec');
                const radioStationXfadeReadout = root.querySelector('#dj-radio-station-xfade-readout');
                const autoFadeChangeStationEn = root.querySelector('#dj-autofade-change-station');
                const btnAutoMix = root.querySelector('#dj-automix');
                const autoMixWrap = root.querySelector('.dj-automix-wrap');
                const autoMixPanel = root.querySelector('#dj-automix-panel');
                const autoMixMin = root.querySelector('#dj-automix-min');
                const autoMixMax = root.querySelector('#dj-automix-max');
                const autoMixMinReadout = root.querySelector('#dj-automix-min-readout');
                const autoMixMaxReadout = root.querySelector('#dj-automix-max-readout');
                const autoMixLimitEn = root.querySelector('#dj-automix-limit-en');
                const autoMixLimitMinEl = root.querySelector('#dj-automix-limit-min');
                const autoMixLimitReadout = root.querySelector('#dj-automix-limit-readout');
                const autoMixLimitRemainingEl = root.querySelector('#dj-automix-limit-remaining');
                const autoMixNextFadeEl = root.querySelector('#dj-automix-next-fade');
                const autoMixHeadTimerEl = root.querySelector('#dj-head-automix-timer');
                let suppressAutoFadeClickUntil = 0;
                let autoFadeLongPressFired = false;
                const AUTOFADE_MS_KEY = 'dj.autofade.duration.ms.v1';
                const AUTOFADE_CHANGE_STATION_KEY = 'dj.autofade.changeStation.enabled.v1';
                const AUTOFADE_MIN_MS = 2000;
                const AUTOFADE_MAX_MS = 15000;
                let autoFadeDurationMs = 5000;
                // Default ON so Auto-Mix keeps doing what it used to (preload + retune before
                // each scheduled fade). Users can untick it to fade between the same stations.
                let autoFadeChangeStationEnabled = true;
                let autoMixEnabled = false;
                let suppressAutoMixClickUntil = 0;
                let autoMixLongPressFired = false;
                const AUTOMIX_ENABLED_KEY = 'dj.automix.enabled.v1';
                const AUTOMIX_MIN_KEY = 'dj.automix.min.min.v1';
                const AUTOMIX_MAX_KEY = 'dj.automix.max.min.v1';
                const AUTOMIX_LIMIT_EN_KEY = 'dj.automix.limit.enabled.v1';
                const AUTOMIX_LIMIT_MIN_KEY = 'dj.automix.limit.minutes.v1';
                let autoMixMinMin = 2;
                let autoMixMaxMin = 20;
                let autoMixLimitEnabled = false;
                let autoMixLimitMin = 60;
                let autoMixSessionStartedAt = 0;
                try {
                    const stored = Number(localStorage.getItem(AUTOFADE_MS_KEY));
                    if (Number.isFinite(stored) && stored >= AUTOFADE_MIN_MS && stored <= AUTOFADE_MAX_MS) autoFadeDurationMs = stored;
                } catch (_) {}
                try {
                    const raw = localStorage.getItem(AUTOFADE_CHANGE_STATION_KEY);
                    if (raw === '1' || raw === '0') autoFadeChangeStationEnabled = (raw === '1');
                } catch (_) {}
                try {
                    const minStored = Number(localStorage.getItem(AUTOMIX_MIN_KEY));
                    const maxStored = Number(localStorage.getItem(AUTOMIX_MAX_KEY));
                    if (Number.isFinite(minStored) && minStored >= 1 && minStored <= 20) autoMixMinMin = Math.round(minStored);
                    if (Number.isFinite(maxStored) && maxStored >= 1 && maxStored <= 20) autoMixMaxMin = Math.round(maxStored);
                    if (autoMixMinMin > autoMixMaxMin) autoMixMaxMin = autoMixMinMin;
                } catch (_) {}
                try {
                    autoMixLimitEnabled = localStorage.getItem(AUTOMIX_LIMIT_EN_KEY) === '1';
                    const limStored = Number(localStorage.getItem(AUTOMIX_LIMIT_MIN_KEY));
                    if (Number.isFinite(limStored) && limStored >= 5 && limStored <= 300) autoMixLimitMin = Math.round(limStored);
                } catch (_) {}
                // AUTO-MIX always starts off on page load; enable only via the Auto-Mix button.
                autoMixEnabled = false;
                try { state.autoMixEnabled = false; } catch (_) {}
                const syncAutoMixRuntimeState = () => {
                    try { state.autoMixEnabled = !!autoMixEnabled; } catch (_) {}
                };
                const syncAutoFadeDurationUi = () => {
                    const sec = Math.max(2, Math.min(15, Math.round((Number(autoFadeDurationMs) || 5000) / 1000)));
                    if (autoFadeDuration) autoFadeDuration.value = String(sec);
                    if (autoFadeReadout) autoFadeReadout.textContent = `${sec}s`;
                };
                const syncAutoFadeChangeStationUi = () => {
                    if (autoFadeChangeStationEn) autoFadeChangeStationEn.checked = !!autoFadeChangeStationEnabled;
                };
                const setAutoFadeChangeStationEnabled = (on, persist) => {
                    autoFadeChangeStationEnabled = !!on;
                    if (persist !== false) {
                        try { localStorage.setItem(AUTOFADE_CHANGE_STATION_KEY, autoFadeChangeStationEnabled ? '1' : '0'); } catch (_) {}
                    }
                    syncAutoFadeChangeStationUi();
                };
                const syncRadioStationCrossfadeUi = () => {
                    if (radioStationXfadeEn) radioStationXfadeEn.checked = !!radioStationCrossfadeEnabled;
                    const sec = Math.max(0, Math.min(15, Number(radioStationCrossfadeSec) || 0));
                    if (radioStationXfadeSec) {
                        radioStationXfadeSec.disabled = !radioStationCrossfadeEnabled;
                        radioStationXfadeSec.value = String(sec);
                    }
                    if (radioStationXfadeReadout) {
                        const label = Number.isInteger(sec) || Math.abs(sec - Math.round(sec)) < 1e-6
                            ? `${Math.round(sec)}`
                            : String(sec);
                        radioStationXfadeReadout.textContent = `${label}s`;
                    }
                };
                const formatLimitMinutesReadout = (m) => {
                    const n = Math.max(5, Math.min(300, Math.round(Number(m) || 60)));
                    if (n < 60) return `${n}m`;
                    const h = Math.floor(n / 60);
                    const r = n % 60;
                    if (r === 0) return h === 1 ? '1h' : `${h}h`;
                    return `${h}h ${r}m`;
                };
                const formatSessionRemainingClock = (ms) => {
                    if (!Number.isFinite(ms) || ms <= 0) return '0:00';
                    const totalSec = Math.max(0, Math.ceil(ms / 1000));
                    const h = Math.floor(totalSec / 3600);
                    const m = Math.floor((totalSec % 3600) / 60);
                    const sec = totalSec % 60;
                    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
                    return `${m}:${String(sec).padStart(2, '0')}`;
                };
                const updateAutoMixSessionRemainingUi = () => {
                    if (!autoMixLimitRemainingEl) return;
                    if (!autoMixEnabled || !autoMixLimitEnabled || !autoMixSessionStartedAt) {
                        autoMixLimitRemainingEl.textContent = '';
                        return;
                    }
                    const limitMs = Math.max(5, Math.min(300, Number(autoMixLimitMin) || 60)) * 60 * 1000;
                    const elapsed = Date.now() - autoMixSessionStartedAt;
                    const left = Math.max(0, limitMs - elapsed);
                    autoMixLimitRemainingEl.textContent = left <= 0 ? ' · 0:00 left' : ` · ${formatSessionRemainingClock(left)} left`;
                };
                const clearAutoMixNextFadeUi = () => {
                    if (autoMixNextFadeEl) autoMixNextFadeEl.textContent = '';
                    if (autoMixHeadTimerEl) {
                        autoMixHeadTimerEl.textContent = '';
                        autoMixHeadTimerEl.classList.add('display-none');
                        autoMixHeadTimerEl.setAttribute('aria-hidden', 'true');
                    }
                };
                const updateAutoMixNextFadeUi = () => {
                    if (!autoMixEnabled || !this.autoMixNextFadeAt) {
                        clearAutoMixNextFadeUi();
                        return;
                    }
                    const left = Math.max(0, this.autoMixNextFadeAt - Date.now());
                    const clock = formatSessionRemainingClock(left);
                    if (autoMixNextFadeEl) autoMixNextFadeEl.textContent = clock;
                    if (autoMixHeadTimerEl) {
                        autoMixHeadTimerEl.textContent = `· ${clock}`;
                        autoMixHeadTimerEl.classList.remove('display-none');
                        autoMixHeadTimerEl.setAttribute('aria-hidden', 'false');
                        autoMixHeadTimerEl.setAttribute('aria-label', `Next auto-mix crossfade in ${clock}`);
                    }
                };
                const clearAutoMixNextFadeTick = () => {
                    if (this.autoMixNextFadeIntervalId) {
                        try { clearInterval(this.autoMixNextFadeIntervalId); } catch (_) {}
                        this.autoMixNextFadeIntervalId = null;
                    }
                };
                const syncAutoMixNextFadeDisplay = () => {
                    clearAutoMixNextFadeTick();
                    if (!autoMixEnabled || !this.autoMixNextFadeAt) {
                        clearAutoMixNextFadeUi();
                        return;
                    }
                    updateAutoMixNextFadeUi();
                    this.autoMixNextFadeIntervalId = setInterval(() => {
                        try { updateAutoMixNextFadeUi(); } catch (_) {}
                    }, 1000);
                };
                const clearAutoMixSessionRemainingTick = () => {
                    if (this.autoMixSessionRemainingIntervalId) {
                        try { clearInterval(this.autoMixSessionRemainingIntervalId); } catch (_) {}
                        this.autoMixSessionRemainingIntervalId = null;
                    }
                    if (autoMixLimitRemainingEl) autoMixLimitRemainingEl.textContent = '';
                };
                const syncAutoMixSessionLimitDisplay = () => {
                    if (this.autoMixSessionRemainingIntervalId) {
                        try { clearInterval(this.autoMixSessionRemainingIntervalId); } catch (_) {}
                        this.autoMixSessionRemainingIntervalId = null;
                    }
                    if (!autoMixEnabled || !autoMixLimitEnabled) {
                        if (autoMixLimitRemainingEl) autoMixLimitRemainingEl.textContent = '';
                        return;
                    }
                    updateAutoMixSessionRemainingUi();
                    this.autoMixSessionRemainingIntervalId = setInterval(() => {
                        try { updateAutoMixSessionRemainingUi(); } catch (_) {}
                    }, 1000);
                };
                const syncAutoMixUi = () => {
                    autoMixMinMin = Math.max(1, Math.min(20, Math.round(Number(autoMixMinMin) || 2)));
                    autoMixMaxMin = Math.max(autoMixMinMin, Math.min(20, Math.round(Number(autoMixMaxMin) || 20)));
                    autoMixLimitMin = Math.max(5, Math.min(300, Math.round(Number(autoMixLimitMin) || 60)));
                    if (autoMixMin) autoMixMin.value = String(autoMixMinMin);
                    if (autoMixMax) autoMixMax.value = String(autoMixMaxMin);
                    if (autoMixMinReadout) autoMixMinReadout.textContent = `${autoMixMinMin}m`;
                    if (autoMixMaxReadout) autoMixMaxReadout.textContent = `${autoMixMaxMin}m`;
                    if (autoMixLimitMinEl) autoMixLimitMinEl.value = String(autoMixLimitMin);
                    if (autoMixLimitEn) autoMixLimitEn.checked = !!autoMixLimitEnabled;
                    if (autoMixLimitReadout) autoMixLimitReadout.textContent = formatLimitMinutesReadout(autoMixLimitMin);
                    if (btnAutoMix) btnAutoMix.classList.toggle('on', !!autoMixEnabled);
                    syncAutoMixRuntimeState();
                    try { updateAutoMixSessionRemainingUi(); } catch (_) {}
                    try { updateAutoMixNextFadeUi(); } catch (_) {}
                };
                const resetPopupPanelStyles = (panel) => {
                    if (!panel) return;
                    panel.classList.remove('is-fixed-popup');
                    panel.style.position = '';
                    panel.style.left = '';
                    panel.style.top = '';
                    panel.style.right = '';
                };
                const positionPopupPanelAtPointer = (panel, clientX, clientY) => {
                    if (!panel) return;
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
                };
                const closeAutoFadePanel = () => {
                    if (!autoFadePanel) return;
                    autoFadePanel.classList.remove('is-open');
                    autoFadePanel.setAttribute('aria-hidden', 'true');
                    resetPopupPanelStyles(autoFadePanel);
                    // Clear the long-press latch on every close path. Previously the
                    // flag was only reset on the next pointerdown on the AUTO-FADE
                    // button itself, so closing the panel via outside-click or Esc
                    // left `autoFadeLongPressFired === true` — a subsequent
                    // programmatic click (e.g. Space → triggerAutoFadeFromShortcut
                    // → btnAutoFade.click()) would then short-circuit at the click
                    // guard and silently do nothing. By the time we're closing, the
                    // long-press is by definition over, so clearing here is safe.
                    autoFadeLongPressFired = false;
                    suppressAutoFadeClickUntil = 0;
                };
                const closeAutoMixPanel = () => {
                    if (!autoMixPanel) return;
                    autoMixPanel.classList.remove('is-open');
                    autoMixPanel.setAttribute('aria-hidden', 'true');
                    resetPopupPanelStyles(autoMixPanel);
                    // Same latch-clear as closeAutoFadePanel — symmetric fix so a
                    // long-pressed AUTO-MIX panel closed via outside-click or Esc
                    // never leaves the click guard stuck on.
                    autoMixLongPressFired = false;
                    suppressAutoMixClickUntil = 0;
                };
                const openAutoFadePanel = () => {
                    if (!autoFadePanel) return;
                    syncAutoFadeDurationUi();
                    syncAutoFadeChangeStationUi();
                    try { syncRadioStationCrossfadeUi(); } catch (_) {}
                    autoFadePanel.classList.add('is-open');
                    autoFadePanel.setAttribute('aria-hidden', 'false');
                    const p = this._panelPointerPos;
                    positionPopupPanelAtPointer(autoFadePanel, p && p.x, p && p.y);
                };
                const openAutoMixPanel = () => {
                    if (!autoMixPanel) return;
                    syncAutoMixUi();
                    try { syncAutoMixNextFadeDisplay(); } catch (_) {}
                    autoMixPanel.classList.add('is-open');
                    autoMixPanel.setAttribute('aria-hidden', 'false');
                    const p = this._panelPointerPos;
                    positionPopupPanelAtPointer(autoMixPanel, p && p.x, p && p.y);
                };
                syncAutoFadeDurationUi();
                syncAutoFadeChangeStationUi();
                try { syncRadioStationCrossfadeUi(); } catch (_) {}
                syncAutoMixUi();

                const mirrorPairs = [
                    ['dj-knob-a-low', 'knob-a-low'],
                    ['dj-knob-a-mid', 'knob-a-mid'],
                    ['dj-knob-a-high', 'knob-a-high'],
                    ['dj-knob-a-gain', 'knob-a-gain'],
                    ['dj-knob-b-gain', 'knob-b-gain'],
                    ['dj-knob-b-high', 'knob-b-high'],
                    ['dj-knob-b-mid', 'knob-b-mid'],
                    ['dj-knob-b-low', 'knob-b-low']
                ];
                mirrorPairs.forEach(([djId, mixId]) => {
                    try {
                        const kn = root.querySelector('#' + djId);
                        if (kn && typeof wireDjKnobMirror === 'function') wireDjKnobMirror(kn, mixId, sig.signal);
                    } catch (_) {}
                });

                try {
                    if (typeof wireDjBeatFxKnobs === 'function') wireDjBeatFxKnobs(root);
                } catch (_) {}

                try {
                    if (mixCross && djCross) djCross.value = mixCross.value;
                    else if (djCross) djCross.value = '0';
                } catch (_) {}

                if (djCross) {
                    djCross.addEventListener('input', () => {
                        try {
                            if (typeof applyCrossfade === 'function') applyCrossfade(djCross.value);
                        } catch (_) {}
                        try { if (typeof clearAutoMixDeferForNonIncoming === 'function') clearAutoMixDeferForNonIncoming(); } catch (_) {}
                    }, sig);
                }

                const getSeekBounds = (media) => {
                    if (!media) return null;
                    try {
                        if (media.seekable && media.seekable.length > 0) {
                            const start = Number(media.seekable.start(0));
                            const end = Number(media.seekable.end(media.seekable.length - 1));
                            if (Number.isFinite(start) && Number.isFinite(end) && end > start) return { start, end };
                        }
                    } catch (_) {}
                    try {
                        const d = Number(media.duration);
                        if (Number.isFinite(d) && d > 0) return { start: 0, end: d };
                    } catch (_) {}
                    return null;
                };
                const attachJogSeek = (deckKey, jogEl) => {
                    if (!jogEl) return;
                    const mediaForDeck = () => {
                        if (deckKey === 'b') return getDeckBJogSeekMedia(this);
                        return audioEl;
                    };
                    const syncDeckBVideoLayersToTime = (t) => {
                        if (deckKey !== 'b' || this.deckBVizMode !== 'video') return;
                        const plan = computeDeckBVideoCrossfadePlan(this);
                        const touch = (vd, layer) => {
                            if (!vd || !layer) return;
                            const o = parseFloat(vd.style.opacity || '0');
                            if (o <= 0.02) return;
                            try { vd.currentTime = t; } catch (_) {}
                        };
                        if (plan.layerB && plan.opB > 0.02) touch(this.deckBVideoElB, plan.layerB);
                        if (plan.layerQ && plan.opQ > 0.02) touch(this.deckBVideoElQ, plan.layerQ);
                    };
                    let ptrId = null;
                    let baseTime = 0;
                    let startX = 0;
                    let baseAngle = 0;
                    const secPerPx = 0.06;
                    const degPerPx = 0.9;
                    const onMove = (ev) => {
                        if (ptrId == null || ev.pointerId !== ptrId) return;
                        const media = mediaForDeck();
                        const bounds = getSeekBounds(media);
                        if (!media || !bounds) return;
                        const dx = Number(ev.clientX) - startX;
                        const target = Math.max(bounds.start, Math.min(bounds.end, baseTime + (dx * secPerPx)));
                        try {
                            if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) {
                                releaseAutoMixDeferredLocal(deckKey, 'seek');
                            }
                        } catch (_) {}
                        try { media.currentTime = target; } catch (_) {}
                        syncDeckBVideoLayersToTime(target);
                        if (deckKey === 'b') this.angleB = baseAngle + (dx * degPerPx);
                        else this.angleA = baseAngle + (dx * degPerPx);
                        try { jogEl.style.transform = `rotate(${deckKey === 'b' ? this.angleB : this.angleA}deg)`; } catch (_) {}
                    };
                    const end = (ev) => {
                        if (ptrId == null || ev.pointerId !== ptrId) return;
                        try { jogEl.releasePointerCapture(ptrId); } catch (_) {}
                        ptrId = null;
                    };
                    jogEl.addEventListener('pointerdown', (ev) => {
                        const media = mediaForDeck();
                        const bounds = getSeekBounds(media);
                        if (!media || !media.src || !bounds) return;
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        try {
                            if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) {
                                releaseAutoMixDeferredLocal(deckKey, 'seek');
                            }
                        } catch (_) {}
                        ptrId = ev.pointerId;
                        startX = Number(ev.clientX) || 0;
                        baseTime = Number(media.currentTime) || 0;
                        baseAngle = deckKey === 'b' ? this.angleB : this.angleA;
                        try { jogEl.setPointerCapture(ptrId); } catch (_) {}
                    }, sig);
                    jogEl.addEventListener('pointermove', onMove, sig);
                    jogEl.addEventListener('pointerup', end, sig);
                    jogEl.addEventListener('pointercancel', end, sig);
                    jogEl.addEventListener('wheel', (ev) => {
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        const media = mediaForDeck();
                        const bounds = getSeekBounds(media);
                        if (!media || !media.src || !bounds) return;
                        const dy = Number(ev.deltaY) || 0;
                        if (!dy) return;
                        try {
                            if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) {
                                releaseAutoMixDeferredLocal(deckKey, 'seek');
                            }
                        } catch (_) {}
                        const next = Math.max(bounds.start, Math.min(bounds.end, (Number(media.currentTime) || 0) + (dy * 0.08)));
                        try { media.currentTime = next; } catch (_) {}
                        syncDeckBVideoLayersToTime(next);
                        if (deckKey === 'b') this.angleB += (dy * 0.45);
                        else this.angleA += (dy * 0.45);
                        try { jogEl.style.transform = `rotate(${deckKey === 'b' ? this.angleB : this.angleA}deg)`; } catch (_) {}
                    }, { passive: false, signal: this.abortCtrl.signal });
                };
                attachJogSeek('a', this.els.jogA);
                attachJogSeek('b', this.els.jogB);

                const attachJogRingPolarSeek = (deckKey, jogEl) => {
                    if (!jogEl) return;
                    const svg = deckKey === 'b' ? this.els.jogBProgressSvg : this.els.jogAProgressSvg;
                    if (!svg) return;
                    const hit = svg.querySelector('.dj-jog-progress-hit');
                    if (!hit) return;
                    const mediaForDeck = () => {
                        if (deckKey === 'b') return getDeckBJogSeekMedia(this);
                        return audioEl;
                    };
                    hit.addEventListener('pointerdown', (ev) => {
                        const media = mediaForDeck();
                        if (!media || !media.src) return;
                        let local = false;
                        try {
                            if (deckKey === 'b' && this.deckBVizMode === 'video' && !deckBVideoUserIdle) {
                                local = !!(String(media.currentSrc || media.src || '').trim());
                            } else {
                                local = !!(state && state.deckSourceMode && state.deckSourceMode[deckKey === 'b' ? 'b' : 'a'] === 'local');
                            }
                        } catch (_) {}
                        const dur = Number(media.duration);
                        if (!local || !Number.isFinite(dur) || dur <= 0) return;
                        try {
                            if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) {
                                releaseAutoMixDeferredLocal(deckKey, 'seek');
                            }
                            const rect = jogEl.getBoundingClientRect();
                            const cx = rect.left + rect.width / 2;
                            const cy = rect.top + rect.height / 2;
                            const dx = ev.clientX - cx;
                            const dy = ev.clientY - cy;
                            let a = Math.atan2(dy, dx) + Math.PI / 2;
                            if (a < 0) a += 2 * Math.PI;
                            const frac = a / (2 * Math.PI);
                            const target = Math.max(0, Math.min(dur - 0.001, frac * dur));
                            media.currentTime = target;
                            if (deckKey === 'b' && this.deckBVizMode === 'video' && !deckBVideoUserIdle) {
                                const plan = computeDeckBVideoCrossfadePlan(this);
                                const touch = (vd, layer, op) => {
                                    if (!vd || !layer || !(op > 0.02)) return;
                                    try { vd.currentTime = target; } catch (_) {}
                                };
                                touch(this.deckBVideoElB, plan.layerB, plan.opB);
                                touch(this.deckBVideoElQ, plan.layerQ, plan.opQ);
                            }
                            ev.preventDefault();
                            ev.stopPropagation();
                        } catch (_) {}
                    }, { signal: this.abortCtrl.signal, passive: false });
                };
                attachJogRingPolarSeek('a', this.els.jogA);
                attachJogRingPolarSeek('b', this.els.jogB);

                /** Keep deck audio elements playing whenever the crossfader gives them audible gain (fixes mismatches after interrupted/completed auto-fades). */
                const ensureCrossfadeDeckPlayback = () => {
                    try {
                        if (this.suppressEnsureCrossfadeDeckPlayback) return;
                        const xv = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                        const ga = 1 - xv;
                        const gb = xv;
                        const thresh = 0.03;
                        if (ga > thresh) {
                            const elA = (typeof getDeckAMediaForPlaybackState === 'function') ? getDeckAMediaForPlaybackState() : audioEl;
                            if (elA && elA.src && elA.paused) {
                                if (!(typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed('a'))) {
                                    elA.play().catch(() => {});
                                }
                            }
                        }
                        if (gb > thresh && audioElB && audioElB.src && audioElB.paused) {
                            if (!(typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed('b'))) {
                                audioElB.play().catch(() => {});
                            }
                        }
                    } catch (_) {}
                };

                /** Right-click hold: instant cut to the fader position under the cursor; on release it snaps back. */
                let cutFadeHoldActive = false;
                let cutFadeHoldRestore = 0;
                let cutFadeHoldPointerId = null;
                const wireCutFadeHoldOnDjCross = () => {
                    if (!djCross) return;
                    const restoreCutFadeHold = () => {
                        if (!cutFadeHoldActive) return;
                        cutFadeHoldActive = false;
                        cutFadeHoldPointerId = null;
                        try {
                            if (typeof applyCrossfade === 'function') applyCrossfade(cutFadeHoldRestore);
                        } catch (_) {}
                        try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                    };
                    const onCutFadePointerDown = (ev) => {
                        if (ev.button !== 2) return;
                        try { ev.preventDefault(); } catch (_) {}
                        cutFadeHoldRestore = Math.max(0, Math.min(1, Number(djCross.value) || 0));
                        const cutTo = crossfadeValueFromPointerOnRange(djCross, ev.clientX);
                        cutFadeHoldActive = true;
                        cutFadeHoldPointerId = ev.pointerId;
                        try {
                            if (typeof applyCrossfade === 'function') applyCrossfade(cutTo);
                        } catch (_) {}
                        try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                        try { djCross.setPointerCapture(ev.pointerId); } catch (_) {}
                    };
                    const onCutFadePointerEnd = (ev) => {
                        if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                        try { djCross.releasePointerCapture(ev.pointerId); } catch (_) {}
                        restoreCutFadeHold();
                    };
                    djCross.addEventListener('pointerdown', onCutFadePointerDown, sig);
                    djCross.addEventListener('pointerup', onCutFadePointerEnd, sig);
                    djCross.addEventListener('pointercancel', onCutFadePointerEnd, sig);
                    djCross.addEventListener('lostpointercapture', (ev) => {
                        if (!cutFadeHoldActive || ev.pointerId !== cutFadeHoldPointerId) return;
                        restoreCutFadeHold();
                    }, sig);
                    djCross.addEventListener('contextmenu', (e) => {
                        try { e.preventDefault(); } catch (_) {}
                    }, sig);
                };
                wireCutFadeHoldOnDjCross();

                const ensureDeckStationPlayback = (deckKey) => {
                    if (deckKey === 'b') {
                        try { state.deckSourceMode.b = 'radio'; } catch (_) {}
                        // Keep the cued Deck B station — random retune is only via RAND / retuneDeckStationForAutoMix
                        // when "Change station before fading" is on, not when ensuring playback.
                        try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                    } else {
                        try { state.deckSourceMode.a = 'radio'; } catch (_) {}
                        try {
                            if (typeof setStation === 'function') {
                                const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0) ? currentStationIndex : 0;
                                setStation(idx);
                            } else if (typeof playRadio === 'function') {
                                playRadio();
                            }
                            if (typeof uiLocked !== 'undefined' && uiLocked && typeof playRadio === 'function') {
                                try {
                                    if (audioEl && audioEl.paused) playRadio();
                                } catch (_) {}
                            }
                        } catch (_) {}
                    }
                };
                const isDeckLocalMode = (deckKey) => {
                    try {
                        return !!(state && state.deckSourceMode && state.deckSourceMode[deckKey] === 'local');
                    } catch (_) {
                        return false;
                    }
                };
                const getDeckLocalMediaEl = (deckKey) => {
                    try {
                        if (deckKey === 'b') return audioElB;
                        return (typeof getDeckAMediaForPlaybackState === 'function') ? getDeckAMediaForPlaybackState() : audioEl;
                    } catch (_) {
                        return deckKey === 'b' ? audioElB : audioEl;
                    }
                };
                const deckLocalQueueLength = (deckKey) => {
                    try {
                        const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
                        return (q && q.length) || 0;
                    } catch (_) {
                        return 0;
                    }
                };
                const isDeckLocalAudiblyReady = (deckKey) => {
                    if (!isDeckLocalMode(deckKey)) return false;
                    const el = getDeckLocalMediaEl(deckKey);
                    const src = el ? String(el.currentSrc || el.src || '') : '';
                    if (!src || src === 'about:blank') return false;
                    try {
                        if ((el.readyState | 0) >= 3) return true;
                        if (!el.paused && (Number(el.currentTime) || 0) > 0) return true;
                    } catch (_) {}
                    return false;
                };
                const shouldPreserveLocalDeckForAutoFade = (deckKey) => {
                    if (!isDeckLocalMode(deckKey)) return false;
                    if (isDeckLocalAudiblyReady(deckKey)) return true;
                    const el = getDeckLocalMediaEl(deckKey);
                    const src = el ? String(el.currentSrc || el.src || '') : '';
                    if (src && src !== 'about:blank') {
                        try { if (!el.ended) return true; } catch (_) { return true; }
                    }
                    return deckLocalQueueLength(deckKey) > 0;
                };
                const isDeckIncomingReadyForAutoFade = (deckKey) => {
                    try {
                        if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) return true;
                    } catch (_) {}
                    return isDeckLocalAudiblyReady(deckKey) || isDeckRadioAudiblyReady(deckKey);
                };
                const ensureDeckPlaybackForAutoFade = (deckKey) => {
                    try {
                        if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) return;
                    } catch (_) {}
                    if (shouldPreserveLocalDeckForAutoFade(deckKey)) {
                        const el = getDeckLocalMediaEl(deckKey);
                        const src = el ? String(el.currentSrc || el.src || '') : '';
                        if (src && src !== 'about:blank') {
                            try { if (el.paused) el.play().catch(() => {}); } catch (_) {}
                            return;
                        }
                        if (deckLocalQueueLength(deckKey) > 0) {
                            try {
                                if (deckKey === 'b') playDeckBTrackFromQueue();
                                else playDeckATrackFromQueue();
                            } catch (_) {}
                            return;
                        }
                    }
                    if (isDeckRadioAudiblyReady(deckKey)) {
                        try {
                            const el = getRadioReadinessTargetEl(deckKey);
                            if (el && el.paused) el.play().catch(() => {});
                        } catch (_) {}
                        return;
                    }
                    ensureDeckStationPlayback(deckKey);
                };
                const maybeRetuneDeckForAutoFade = (deckKey, triedIndexes) => {
                    if (!autoFadeChangeStationEnabled) return null;
                    if (shouldPreserveLocalDeckForAutoFade(deckKey)) return null;
                    try { return retuneDeckStationForAutoMix(deckKey, triedIndexes); } catch (_) {}
                    return null;
                };
                const awaitDeckLocalReady = (deckKey, timeoutMs) => {
                    return new Promise((resolve) => {
                        const el = getDeckLocalMediaEl(deckKey);
                        if (!el) { resolve('failed'); return; }
                        const isReady = () => {
                            try {
                                const src = String(el.currentSrc || el.src || '');
                                if (!src || src === 'about:blank') return false;
                                if ((el.readyState | 0) >= 3) return true;
                                if (!el.paused && (Number(el.currentTime) || 0) > 0) return true;
                            } catch (_) {}
                            return false;
                        };
                        if (isReady()) { resolve('ready'); return; }
                        const cap = Math.max(3000, Math.min(20000, Number(timeoutMs) || 12000));
                        let done = false;
                        let timer = null;
                        const cleanup = () => {
                            try { el.removeEventListener('playing', onPlaying); } catch (_) {}
                            try { el.removeEventListener('canplay', onMaybe); } catch (_) {}
                            try { el.removeEventListener('loadeddata', onMaybe); } catch (_) {}
                            try { el.removeEventListener('error', onError); } catch (_) {}
                            if (timer) { try { clearTimeout(timer); } catch (_) {} timer = null; }
                        };
                        const finish = (status) => {
                            if (done) return;
                            done = true;
                            cleanup();
                            resolve(status);
                        };
                        const onPlaying = () => finish('ready');
                        const onMaybe = () => { if (isReady()) finish('ready'); };
                        const onError = () => finish('failed');
                        try { el.addEventListener('playing', onPlaying); } catch (_) {}
                        try { el.addEventListener('canplay', onMaybe); } catch (_) {}
                        try { el.addEventListener('loadeddata', onMaybe); } catch (_) {}
                        try { el.addEventListener('error', onError); } catch (_) {}
                        timer = setTimeout(() => finish('timeout'), cap);
                    });
                };
                /**
                 * True when the deck's radio element is actually producing (or buffered to play)
                 * audio — not merely holding a URL that failed to connect.
                 */
                const isDeckRadioAudiblyReady = (deckKey) => {
                    try {
                        const mode = state && state.deckSourceMode && state.deckSourceMode[deckKey];
                        if (mode !== 'radio') return false;
                        const idx = getDeckRadioStationIndex(deckKey);
                        if (!stationIndexHasStreamUrl(idx)) return false;
                        const expected = sanitizeUrlForAudio(String(stations[idx].url || ''));
                        const elements = (deckKey === 'b')
                            ? [audioElB, audioElRadioBAlt].filter(Boolean)
                            : [audioEl, audioElRadioAAlt].filter(Boolean);
                        for (const el of elements) {
                            if (!el) continue;
                            const src = sanitizeUrlForAudio(String(el.currentSrc || el.src || ''));
                            if (!src || src === 'about:blank' || src !== expected) continue;
                            if ((el.readyState | 0) >= 3) return true;
                            if (!el.paused && (Number(el.currentTime) || 0) > 0) return true;
                        }
                    } catch (_) {}
                    return false;
                };
                /**
                 * Picks the <audio> element whose readiness *actually* reflects whether the
                 * newly-cued station is playable.
                 *
                 * Deck B is straightforward — there's only one element (`audioElB`).
                 *
                 * Deck A is tricky: `playRadio()` does a "warm crossfade" handoff when the
                 * deck is already playing. The new URL is loaded into the OFF / prep element
                 * (the one that's *not* currently audible), not into `audioEl`. If we awaited
                 * on `audioEl` in that case it would already report ready (still streaming the
                 * old station) and resolve synchronously — the AUTO-FADE would then crossfade
                 * INTO the old station instead of the new one we just retuned. That's exactly
                 * the bug the user reported: Deck B preloads correctly, Deck A does not.
                 *
                 * `radioAHandoffAbortCtrl` is non-null exactly during an active warm-crossfade
                 * handoff, so we use that as our "is a handoff in flight?" signal and pick the
                 * prep element using the same rule `playRadio()` uses internally.
                 */
                const getRadioReadinessTargetEl = (deckKey) => {
                    if (deckKey === 'b') {
                        try {
                            const idx = getDeckRadioStationIndex('b');
                            const expected = stationIndexHasStreamUrl(idx)
                                ? sanitizeUrlForAudio(String(stations[idx].url || ''))
                                : '';
                            const pickLoaded = (el) => {
                                if (!el) return null;
                                const src = sanitizeUrlForAudio(String(el.currentSrc || el.src || ''));
                                if (!src || src === 'about:blank') return null;
                                if (expected && src !== expected) return null;
                                return el;
                            };
                            if (expected) {
                                const onAlt = pickLoaded(audioElRadioBAlt);
                                if (onAlt) return onAlt;
                                const onMain = pickLoaded(audioElB);
                                if (onMain) return onMain;
                            }
                            if (
                                typeof radioBHandoffAbortCtrl !== 'undefined' && radioBHandoffAbortCtrl &&
                                audioElRadioBAlt &&
                                state && state.deckSourceMode && state.deckSourceMode.b === 'radio'
                            ) {
                                const outputFromSecondary = (typeof isDeckBRadioOutputFromAlt === 'function')
                                    ? isDeckBRadioOutputFromAlt()
                                    : false;
                                const prepEl = outputFromSecondary ? audioElB : audioElRadioBAlt;
                                const prepSrc = prepEl ? String(prepEl.currentSrc || prepEl.src || '') : '';
                                if (prepEl && prepSrc && prepSrc !== 'about:blank') return prepEl;
                            }
                        } catch (_) {}
                        return audioElB;
                    }
                    try {
                        const idx = getDeckRadioStationIndex('a');
                        const expected = stationIndexHasStreamUrl(idx)
                            ? sanitizeUrlForAudio(String(stations[idx].url || ''))
                            : '';
                        const pickLoaded = (el) => {
                            if (!el) return null;
                            const src = sanitizeUrlForAudio(String(el.currentSrc || el.src || ''));
                            if (!src || src === 'about:blank') return null;
                            if (expected && src !== expected) return null;
                            return el;
                        };
                        if (expected) {
                            const onAlt = pickLoaded(audioElRadioAAlt);
                            if (onAlt) return onAlt;
                            const onMain = pickLoaded(audioEl);
                            if (onMain) return onMain;
                        }
                        if (
                            typeof radioAHandoffAbortCtrl !== 'undefined' && radioAHandoffAbortCtrl &&
                            audioElRadioAAlt &&
                            state && state.deckSourceMode && state.deckSourceMode.a === 'radio'
                        ) {
                            const outputFromSecondary = (typeof isDeckARadioOutputFromAlt === 'function')
                                ? isDeckARadioOutputFromAlt()
                                : false;
                            const prepEl = outputFromSecondary ? audioEl : audioElRadioAAlt;
                            const prepSrc = prepEl ? String(prepEl.currentSrc || prepEl.src || '') : '';
                            if (prepEl && prepSrc && prepSrc !== 'about:blank') return prepEl;
                        }
                    } catch (_) {}
                    return audioEl;
                };

                /**
                 * Resolves once the given deck's media element is actually producing audio
                 * (or has buffered enough to start without an audible gap). Used to gate the
                 * AUTO-FADE crossfade so the "Change station before fading" path doesn't fade
                 * INTO silence while the stream is still negotiating.
                 *
                 * Resolution conditions:
                 *  - `playing` event fires (strongest signal — audio is flowing), OR
                 *  - `canplay` / `loadeddata` event fires AND readyState >= HAVE_FUTURE_DATA, OR
                 *  - already-ready check passes synchronously (just retuned to a primed deck), OR
                 *  - `timeoutMs` elapses (safety cap so a dead/slow stream cannot block the UI).
                 *
                 * Resolves with:
                 *   'ready'   — stream is audible / buffered enough to play
                 *   'failed'  — no URL, or a media error (skip to another station)
                 *   'timeout' — still connecting when the wait budget expired (do NOT skip station)
                 */
                const stationIndexHasStreamUrl = (idx) => {
                    try {
                        const st = Array.isArray(stations) && stations[idx];
                        return !!(st && st.url && String(st.url).trim());
                    } catch (_) {}
                    return false;
                };
                const getDeckRadioStationIndex = (deckKey) => {
                    try {
                        if (deckKey === 'b') {
                            return (typeof currentStationBIndex === 'number' && currentStationBIndex >= 0)
                                ? currentStationBIndex : 0;
                        }
                        return (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                            ? currentStationIndex : 0;
                    } catch (_) {}
                    return 0;
                };
                const awaitDeckRadioReady = (deckKey, timeoutMs) => {
                    return new Promise((resolve) => {
                        const stationIdx = getDeckRadioStationIndex(deckKey);
                        if (!stationIndexHasStreamUrl(stationIdx)) {
                            resolve('failed');
                            return;
                        }
                        let el = null;
                        try { el = getRadioReadinessTargetEl(deckKey); } catch (_) {}
                        if (!el) { resolve('failed'); return; }
                        const isReady = () => {
                            try {
                                if ((el.readyState | 0) >= 3) return true; // HAVE_FUTURE_DATA
                                if (!el.paused && (Number(el.currentTime) || 0) > 0) return true;
                            } catch (_) {}
                            return false;
                        };
                        if (isReady()) { resolve('ready'); return; }
                        const cap = Math.max(3000, Math.min(20000, Number(timeoutMs) || 12000));
                        let done = false;
                        let timer = null;
                        const cleanup = () => {
                            try { el.removeEventListener('playing', onPlaying); } catch (_) {}
                            try { el.removeEventListener('canplay', onMaybe); } catch (_) {}
                            try { el.removeEventListener('loadeddata', onMaybe); } catch (_) {}
                            try { el.removeEventListener('error', onError); } catch (_) {}
                            if (timer) { try { clearTimeout(timer); } catch (_) {} timer = null; }
                        };
                        const finish = (status) => {
                            if (done) return;
                            done = true;
                            cleanup();
                            resolve(status);
                        };
                        const onPlaying = () => finish('ready');
                        const onMaybe = () => { if (isReady()) finish('ready'); };
                        const onError = () => finish('failed');
                        try { el.addEventListener('playing', onPlaying); } catch (_) {}
                        try { el.addEventListener('canplay', onMaybe); } catch (_) {}
                        try { el.addEventListener('loadeddata', onMaybe); } catch (_) {}
                        try { el.addEventListener('error', onError); } catch (_) {}
                        timer = setTimeout(() => finish('timeout'), cap);
                    });
                };
                const retuneDeckStationForAutoMix = (deckKey, triedIndexes) => {
                    try {
                        if (!Array.isArray(stations) || stations.length === 0) return null;
                        const tried = (triedIndexes instanceof Set) ? triedIndexes : new Set();
                        if (deckKey === 'b') {
                            const cur = (typeof currentStationBIndex === 'number' && Number.isFinite(currentStationBIndex)) ? currentStationBIndex : 0;
                            let eligible = (typeof getCycleEligibleStationIndexes === 'function') ? getCycleEligibleStationIndexes(cur) : [];
                            if (tried.size) {
                                const filtered = eligible.filter((idx) => !tried.has(idx));
                                if (filtered.length) eligible = filtered;
                            }
                            const idx = eligible.length ? eligible[Math.floor(Math.random() * eligible.length)] : cur;
                            const next = Math.max(0, Math.min(stations.length - 1, Number(idx) || 0));
                            if (next === cur && isDeckRadioAudiblyReady('b')) {
                                try {
                                    const el = getRadioReadinessTargetEl('b');
                                    if (el && el.paused) el.play().catch(() => {});
                                } catch (_) {}
                                return next;
                            }
                            currentStationBIndex = next;
                            if (mixStationB) mixStationB.value = String(next);
                            try { if (typeof refreshMixStationB === 'function') refreshMixStationB(); } catch (_) {}
                            try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
                            return next;
                        }
                        const cur = (typeof currentStationIndex === 'number' && Number.isFinite(currentStationIndex)) ? currentStationIndex : 0;
                        let eligible = (typeof getCycleEligibleStationIndexes === 'function') ? getCycleEligibleStationIndexes(cur) : [];
                        if (tried.size) {
                            const filtered = eligible.filter((idx) => !tried.has(idx));
                            if (filtered.length) eligible = filtered;
                        }
                        const idx = eligible.length ? eligible[Math.floor(Math.random() * eligible.length)] : cur;
                        const next = Math.max(0, Math.min(stations.length - 1, Number(idx) || 0));
                        if (next === cur && isDeckRadioAudiblyReady('a')) {
                            try {
                                const el = getRadioReadinessTargetEl('a');
                                if (el && el.paused) el.play().catch(() => {});
                            } catch (_) {}
                            return next;
                        }
                        if (typeof setStation === 'function') setStation(next, { force: true });
                        else if (typeof playRadio === 'function') playRadio();
                        return next;
                    } catch (_) {}
                    return null;
                };
                /**
                 * Before AUTO-FADE / AUTO-MIX crossfades, ensure the incoming deck has a playable
                 * station. Retries other eligible stations when the first pick fails to connect.
                 */
                const prepareIncomingDeckForAutoFade = async (deckKey, timeoutMs, opts) => {
                    const wantRadioRetune = !!(opts && opts.retuneIfNeeded) && autoFadeChangeStationEnabled;
                    const retuneIfNeeded = wantRadioRetune && !shouldPreserveLocalDeckForAutoFade(deckKey);
                    const perStationMs = Math.max(8000, Math.min(20000, Number(timeoutMs) || 12000));
                    if (!retuneIfNeeded && isDeckIncomingReadyForAutoFade(deckKey)) return true;
                    if (retuneIfNeeded) {
                        try {
                            if (typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(deckKey)) return true;
                        } catch (_) {}
                    }
                    if (isDeckLocalMode(deckKey) && shouldPreserveLocalDeckForAutoFade(deckKey)) {
                        try { ensureDeckPlaybackForAutoFade(deckKey); } catch (_) {}
                        const waitResult = await awaitDeckLocalReady(deckKey, perStationMs);
                        if (waitResult === 'ready' || isDeckLocalAudiblyReady(deckKey)) return true;
                        if (waitResult === 'timeout') return isDeckLocalAudiblyReady(deckKey);
                        return isDeckLocalAudiblyReady(deckKey);
                    }
                    if (!retuneIfNeeded) {
                        try { ensureDeckPlaybackForAutoFade(deckKey); } catch (_) {}
                        if (isDeckLocalMode(deckKey)) {
                            const waitResult = await awaitDeckLocalReady(deckKey, perStationMs);
                            return waitResult === 'ready' || isDeckLocalAudiblyReady(deckKey);
                        }
                        const waitResult = await awaitDeckRadioReady(deckKey, perStationMs);
                        return waitResult === 'ready' || isDeckRadioAudiblyReady(deckKey);
                    }
                    const tried = new Set();
                    const markTried = () => {
                        try {
                            const idx = getDeckRadioStationIndex(deckKey);
                            if (typeof idx === 'number' && idx >= 0) tried.add(idx);
                        } catch (_) {}
                    };
                    markTried();
                    for (let attempt = 0; attempt < MAX_RADIO_RETRIES; attempt++) {
                        const skipFirstRetune = !!(opts && opts.skipInitialRetune) && isDeckIncomingReadyForAutoFade(deckKey);
                        if (attempt > 0 || !skipFirstRetune) {
                            try { maybeRetuneDeckForAutoFade(deckKey, tried); } catch (_) {}
                            markTried();
                        }
                        const waitResult = await awaitDeckRadioReady(deckKey, perStationMs);
                        if (waitResult === 'ready' || isDeckRadioAudiblyReady(deckKey)) return true;
                        if (waitResult === 'timeout') return isDeckRadioAudiblyReady(deckKey);
                        if (waitResult !== 'failed') return isDeckRadioAudiblyReady(deckKey);
                    }
                    return isDeckRadioAudiblyReady(deckKey);
                };
                const clearAutoFadePreloadUi = () => {
                    this.autoFadePreloadPending = false;
                    if (btnAutoFade) btnAutoFade.classList.remove('is-preloading');
                };
                const clearAutoMixTimer = () => {
                    if (this.autoMixTimerId) {
                        try { clearTimeout(this.autoMixTimerId); } catch (_) {}
                        this.autoMixTimerId = null;
                    }
                    if (this.autoMixPreloadTimerId) {
                        try { clearTimeout(this.autoMixPreloadTimerId); } catch (_) {}
                        this.autoMixPreloadTimerId = null;
                    }
                };
                const clearAutoMixSessionLimitTimeoutOnly = () => {
                    if (this.autoMixSessionLimitTimerId) {
                        try { clearTimeout(this.autoMixSessionLimitTimerId); } catch (_) {}
                        this.autoMixSessionLimitTimerId = null;
                    }
                };
                const clearAutoMixSessionLimitTimer = () => {
                    clearAutoMixSessionLimitTimeoutOnly();
                    clearAutoMixSessionRemainingTick();
                };
                const scheduleAutoMixSessionLimit = () => {
                    clearAutoMixSessionLimitTimeoutOnly();
                    if (!autoMixEnabled || !autoMixLimitEnabled) {
                        clearAutoMixSessionRemainingTick();
                        return;
                    }
                    const limitMs = Math.max(5, Math.min(300, Number(autoMixLimitMin) || 60)) * 60 * 1000;
                    if (!autoMixSessionStartedAt) autoMixSessionStartedAt = Date.now();
                    const elapsed = Date.now() - autoMixSessionStartedAt;
                    const remaining = Math.max(0, limitMs - elapsed);
                    this.autoMixSessionLimitTimerId = setTimeout(() => {
                        this.autoMixSessionLimitTimerId = null;
                        try { onAutoMixSessionLimitReached(); } catch (_) {}
                    }, remaining);
                    syncAutoMixSessionLimitDisplay();
                };
                const runMasterFadeOut = (durMs, onDone) => {
                    if (this.masterFadeOutRafId) {
                        try { cancelAnimationFrame(this.masterFadeOutRafId); } catch (_) {}
                        this.masterFadeOutRafId = null;
                    }
                    if (this.autoFadeRafId) {
                        try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                        this.autoFadeRafId = null;
                        this.autoFadeTargetDeck = null;
                        if (btnAutoFade) btnAutoFade.classList.remove('on');
                    }
                    const d = Math.max(200, Math.min(120000, Number(durMs) || 5000));
                    let startGa = 0;
                    let startGb = 0;
                    try {
                        if (state.streamAGain) startGa = Number(state.streamAGain.gain.value) || 0;
                        if (state.streamBGain) startGb = Number(state.streamBGain.gain.value) || 0;
                    } catch (_) {}
                    const startTs = performance.now();
                    const tick = (ts) => {
                        const t = Math.max(0, Math.min(1, (ts - startTs) / d));
                        const k = 1 - t;
                        try {
                            if (state.streamAGain) state.streamAGain.gain.value = startGa * k;
                            if (state.streamBGain) state.streamBGain.gain.value = startGb * k;
                        } catch (_) {}
                        if (t >= 1) {
                            this.masterFadeOutRafId = null;
                            try { if (typeof onDone === 'function') onDone(); } catch (_) {}
                            return;
                        }
                        this.masterFadeOutRafId = requestAnimationFrame(tick);
                    };
                    this.masterFadeOutRafId = requestAnimationFrame(tick);
                };
                const onAutoMixSessionLimitReached = () => {
                    if (!autoMixEnabled) return;
                    const fadeDur = Math.max(AUTOFADE_MIN_MS, Math.min(AUTOFADE_MAX_MS, Number(autoFadeDurationMs) || 5000));
                    runMasterFadeOut(fadeDur, () => {
                        autoMixEnabled = false;
                        syncAutoMixRuntimeState();
                        clearAutoMixTimer();
                        this.autoMixNextTargetDeck = null;
                        this.autoMixNextFadeAt = 0;
                        clearAutoMixNextFadeTick();
                        clearAutoMixNextFadeUi();
                        clearAutoMixSessionLimitTimer();
                        autoMixSessionStartedAt = 0;
                        try { localStorage.setItem(AUTOMIX_ENABLED_KEY, '0'); } catch (_) {}
                        syncAutoMixUi();
                        try { if (audioEl) audioEl.pause(); } catch (_) {}
                        try { if (audioElB) audioElB.pause(); } catch (_) {}
                        try {
                            const x = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                            if (typeof applyCrossfade === 'function') applyCrossfade(x);
                        } catch (_) {}
                    });
                };
                const AUTO_MIX_PRELOAD_MS = 12000;
                const scheduleNextAutoMix = () => {
                    clearAutoMixTimer();
                    this.autoMixNextFadeAt = 0;
                    clearAutoMixNextFadeTick();
                    if (!autoMixEnabled) {
                        clearAutoMixNextFadeUi();
                        try { if (typeof clearAllAutoMixDeferLocal === 'function') clearAllAutoMixDeferLocal(); } catch (_) {}
                        return;
                    }
                    try { if (typeof clearAutoMixDeferForNonIncoming === 'function') clearAutoMixDeferForNonIncoming(); } catch (_) {}
                    const minMs = Math.max(1, Math.min(20, Number(autoMixMinMin) || 2)) * 60 * 1000;
                    const maxMs = Math.max(minMs, Math.max(1, Math.min(20, Number(autoMixMaxMin) || 20)) * 60 * 1000);
                    const span = Math.max(0, maxMs - minMs);
                    const waitMs = minMs + (span > 0 ? Math.floor(Math.random() * (span + 1)) : 0);
                    this.autoMixNextFadeAt = Date.now() + waitMs;
                    syncAutoMixNextFadeDisplay();
                    // Tracks whether this cycle's preload-retune actually ran. The "Change
                    // station before fading" checkbox can flip between schedule time and fire
                    // time; this flag lets the fire callback decide whether the incoming deck
                    // still needs a station change.
                    let preloadCompleted = false;
                    const cycleTarget = (this.autoMixNextTargetDeck === 'a' || this.autoMixNextTargetDeck === 'b')
                        ? this.autoMixNextTargetDeck
                        : ((typeof getCrossfaderIncomingDeckKey === 'function') ? getCrossfaderIncomingDeckKey() : 'b');
                    this.autoMixNextTargetDeck = cycleTarget;
                    const preloadDelay = Math.max(0, waitMs - AUTO_MIX_PRELOAD_MS);
                    this.autoMixPreloadTimerId = setTimeout(() => {
                        this.autoMixPreloadTimerId = null;
                        if (!autoMixEnabled) return;
                        const preloadDeck = cycleTarget;
                        if (!autoFadeChangeStationEnabled) return;
                        if (shouldPreserveLocalDeckForAutoFade(preloadDeck)) return;
                        try { maybeRetuneDeckForAutoFade(preloadDeck); } catch (_) {}
                        preloadCompleted = true;
                    }, preloadDelay);
                    this.autoMixTimerId = setTimeout(() => {
                        this.autoMixTimerId = null;
                        if (!autoMixEnabled) return;
                        const targetDeck = cycleTarget;
                        const fireFade = () => {
                            if (!autoMixEnabled) return;
                            if (sig && sig.signal && sig.signal.aborted) return;
                            const needsIncomingReady = autoFadeChangeStationEnabled || autoMixEnabled;
                            const preserveIncomingLocal = shouldPreserveLocalDeckForAutoFade(targetDeck);
                            const incomingReadyNow = isDeckIncomingReadyForAutoFade(targetDeck);
                            let deferArmed = false;
                            try {
                                deferArmed = !!(typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(targetDeck));
                            } catch (_) {}
                            const onAutoMixCycleComplete = () => {
                                if (!autoMixEnabled) return;
                                this.autoMixNextTargetDeck = targetDeck === 'a' ? 'b' : 'a';
                                scheduleNextAutoMix();
                            };
                            const startAutoMixCrossfade = (stationPreloaded) => {
                                runAutoFade(targetDeck, {
                                    autoMixCycle: true,
                                    stationPreloaded: !!stationPreloaded,
                                    onComplete: onAutoMixCycleComplete
                                });
                            };
                            const finish = (ready) => {
                                if (!ready) {
                                    try {
                                        ensureDeckPlaybackForAutoFade(targetDeck);
                                    } catch (_) {}
                                    try {
                                        statusEl.innerText = 'Incoming deck still buffering — crossfading anyway.';
                                    } catch (_) {}
                                    startAutoMixCrossfade(false);
                                    return;
                                }
                                startAutoMixCrossfade(isDeckIncomingReadyForAutoFade(targetDeck));
                            };
                            if (!needsIncomingReady) {
                                finish(true);
                                return;
                            }
                            // With "Change station before fading" on, still retune when incoming
                            // radio is already playing — only skip prepare for cued local / deferred.
                            if (incomingReadyNow && !autoFadeChangeStationEnabled) {
                                finish(true);
                                return;
                            }
                            if (incomingReadyNow && (preserveIncomingLocal || deferArmed)) {
                                finish(true);
                                return;
                            }
                            prepareIncomingDeckForAutoFade(targetDeck, AUTO_MIX_PRELOAD_MS, {
                                retuneIfNeeded: autoFadeChangeStationEnabled,
                                skipInitialRetune: preloadCompleted
                            }).then(finish);
                        };
                        fireFade();
                    }, waitMs);
                };
                const runAutoFade = (toDeck, fadeOpts) => {
                    try { this.clearSuppressEnsureCrossfadeDeckPlayback(); } catch (_) {}
                    const targetDeck = toDeck === 'a' ? 'a' : 'b';
                    const invokeFadeComplete = () => {
                        try { if (fadeOpts && typeof fadeOpts.onComplete === 'function') fadeOpts.onComplete(); } catch (_) {}
                    };
                    try { if (typeof tryStartAutoMixDeferredLocal === 'function') tryStartAutoMixDeferredLocal(targetDeck); } catch (_) {}
                    const autoMixCycle = !!(fadeOpts && fadeOpts.autoMixCycle);
                    let incomingReady = !!(fadeOpts && fadeOpts.stationPreloaded);
                    if (incomingReady && !isDeckIncomingReadyForAutoFade(targetDeck)) incomingReady = false;
                    if (autoMixCycle || !incomingReady) {
                        ensureDeckPlaybackForAutoFade(targetDeck);
                    } else {
                        try {
                            const el = getDeckLocalMediaEl(targetDeck);
                            const deferArmed = typeof isAutoMixDeferredLocalArmed === 'function' && isAutoMixDeferredLocalArmed(targetDeck);
                            if (el && el.src && el.paused && !deferArmed) el.play().catch(() => {});
                        } catch (_) {}
                        try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                    }
                    const endVal = targetDeck === 'b' ? 1 : 0;
                    let startVal = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                    if (autoMixCycle && Math.abs(endVal - startVal) < 0.001) {
                        // Scheduled AUTO-MIX always sweeps from the opposite end so every cycle
                        // produces a visible crossfade (avoids stalling when the fader already
                        // matches the target after preload / Deck A warm handoff).
                        startVal = endVal === 1 ? 0 : 1;
                    } else if (Math.abs(endVal - startVal) < 0.001) {
                        // Already at the destination — clear any stale "in-flight" target so the
                        // Space shortcut doesn't try to "reverse" a fade that isn't running.
                        this.autoFadeTargetDeck = null;
                        try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                        invokeFadeComplete();
                        return;
                    }
                    if (this.autoFadeRafId) {
                        try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                        this.autoFadeRafId = null;
                    }
                    // Remember which deck the fade is heading to. The Space keyboard
                    // shortcut reads this to reverse direction on a second tap.
                    this.autoFadeTargetDeck = targetDeck;
                    if (btnAutoFade) btnAutoFade.classList.add('on');
                    const durMs = Math.max(AUTOFADE_MIN_MS, Math.min(AUTOFADE_MAX_MS, Number(autoFadeDurationMs) || 5000));
                    const startTs = performance.now();
                    const tick = (ts) => {
                        const t = Math.max(0, Math.min(1, (ts - startTs) / durMs));
                        const eased = 1 - Math.pow(1 - t, 3);
                        const val = startVal + ((endVal - startVal) * eased);
                        try { if (typeof applyCrossfade === 'function') applyCrossfade(val); } catch (_) {}
                        try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                        if (t >= 1) {
                            this.autoFadeRafId = null;
                            this.autoFadeTargetDeck = null;
                            if (btnAutoFade) btnAutoFade.classList.remove('on');
                            // Do not call ensureDeckStationPlayback here: setStation/playRadioB reload the
                            // stream and cause a second silence gap after the fader has already arrived.
                            try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                            invokeFadeComplete();
                            return;
                        }
                        this.autoFadeRafId = requestAnimationFrame(tick);
                    };
                    this.autoFadeRafId = requestAnimationFrame(tick);
                };
                if (btnAutoFade) {
                    btnAutoFade.addEventListener('pointerdown', (ev) => {
                        try { this._panelPointerPos = { x: ev.clientX, y: ev.clientY }; } catch (_) {}
                        autoFadeLongPressFired = false;
                        if (this.autoFadeHoldTimer) {
                            try { clearTimeout(this.autoFadeHoldTimer); } catch (_) {}
                            this.autoFadeHoldTimer = null;
                        }
                        this.autoFadeHoldTimer = setTimeout(() => {
                            this.autoFadeHoldTimer = null;
                            autoFadeLongPressFired = true;
                            suppressAutoFadeClickUntil = Date.now() + 450;
                            openAutoFadePanel();
                        }, 450);
                    }, sig);
                    const clearAutoFadeHold = () => {
                        if (this.autoFadeHoldTimer) {
                            try { clearTimeout(this.autoFadeHoldTimer); } catch (_) {}
                            this.autoFadeHoldTimer = null;
                        }
                    };
                    btnAutoFade.addEventListener('pointerup', clearAutoFadeHold, sig);
                    btnAutoFade.addEventListener('pointercancel', clearAutoFadeHold, sig);
                    btnAutoFade.addEventListener('pointerleave', clearAutoFadeHold, sig);
                    btnAutoFade.addEventListener('click', () => {
                        if (autoFadeLongPressFired || Date.now() < suppressAutoFadeClickUntil) return;
                        // Ignore double-clicks while a previous Change-Station preload is still
                        // waiting for the incoming deck's stream to be ready — otherwise we'd
                        // stack retunes / fades on top of each other.
                        if (this.autoFadePreloadPending) return;
                        // Release keyboard focus from this button (and any mirror button such as
                        // #mix-autofade that just forwarded a click here) so the browser's native
                        // button-activation behaviour can't convert a subsequent Space / Enter
                        // press into another click on this button. Without this, Space "sometimes"
                        // re-triggers AUTO-FADE instead of running the proper Space shortcut
                        // (engine.triggerAutoFadeFromShortcut) because the focused-button click
                        // races the global keydown handler.
                        try { btnAutoFade.blur(); } catch (_) {}
                        try {
                            const ae = document.activeElement;
                            if (ae && ae !== document.body && typeof ae.blur === 'function' &&
                                (ae.classList.contains('dj-autofade-btn') || ae.classList.contains('dj-automix-btn'))) {
                                ae.blur();
                            }
                        } catch (_) {}
                        closeAutoFadePanel();
                        closeAutoMixPanel();
                        try { this.clearSuppressEnsureCrossfadeDeckPlayback(); } catch (_) {}
                        if (this.autoFadeRafId && this.autoFadeTargetDeck) {
                            // Mid-fade button press = reverse direction, matching the
                            // Space shortcut. This is explicit manual steering, so do
                            // NOT run the normal Change-Station / preload-retune path:
                            // both decks are already part of the current fade and the
                            // user wants to return to the original deck, not pick a new
                            // station on the opposite side.
                            //
                            // But applyCrossfade() has a 3% edge snap, so the UI/audio can
                            // already look fully arrived while the RAF still has a few frames
                            // left. In that snapped-at-destination case, treat the click as a
                            // fresh AUTO-FADE request so "Change station before fading" still
                            // retunes the deck we are heading back to (notably Deck A after a
                            // completed-looking fade to Deck B).
                            const currentX = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                            const destinationX = this.autoFadeTargetDeck === 'b' ? 1 : 0;
                            const visuallyAtDestination = Math.abs(currentX - destinationX) <= 0.001;
                            if (!visuallyAtDestination) {
                                const opposite = this.autoFadeTargetDeck === 'a' ? 'b' : 'a';
                                runAutoFade(opposite, { stationPreloaded: true });
                                try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                                return;
                            }
                            try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                            this.autoFadeRafId = null;
                            this.autoFadeTargetDeck = null;
                            if (btnAutoFade) btnAutoFade.classList.remove('on');
                        }
                        const x = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                        const target = x < 0.5 ? 'b' : 'a';
                        // Two independent toggles, decoupled so AUTO-MIX gives us a smooth
                        // transition even when the user wants to keep their currently cued
                        // station on the incoming deck:
                        //
                        //   doRetune       = "Change station before fading" is ON
                        //                    → swap the incoming deck to a fresh random station.
                        //   doPreloadWait  = retune is happening, OR AUTO-MIX is enabled
                        //                    → wait for the incoming deck's stream to be
                        //                      audibly ready before kicking off the crossfade,
                        //                      so we never fade INTO silence / a buffering gap.
                        //
                        // When AUTO-MIX is on the user has explicitly opted into "automated
                        // mixing" — so even a manual press should benefit from the same
                        // preload-wait guarantee that AUTO-MIX's scheduled fires already use.
                        const doRetune = autoFadeChangeStationEnabled;
                        const doPreloadWait = autoFadeChangeStationEnabled || autoMixEnabled;
                        if (doPreloadWait) {
                            // Cancel any AUTO-MIX preload/fire timer that might have been
                            // about to expire on the same deck — otherwise it could re-retune
                            // mid-preload (interrupting our stream load) or run a second fade
                            // on top of ours. scheduleNextAutoMix() runs again after the manual
                            // fade completes, so the AUTO-MIX cycle simply restarts from then.
                            if (autoMixEnabled) { try { clearAutoMixTimer(); } catch (_) {} }
                            if (doRetune && !shouldPreserveLocalDeckForAutoFade(target)) {
                                try { maybeRetuneDeckForAutoFade(target); } catch (_) {}
                            } else {
                                try { ensureDeckPlaybackForAutoFade(target); } catch (_) {}
                            }
                            const myGen = (this.autoFadePreloadGen = (this.autoFadePreloadGen || 0) + 1);
                            this.autoFadePreloadPending = true;
                            if (btnAutoFade) btnAutoFade.classList.add('is-preloading');
                            // Keep mid-stream playback flowing on the outgoing deck so it
                            // doesn't pause / glitch while we wait for the incoming one.
                            try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                            prepareIncomingDeckForAutoFade(target, AUTO_MIX_PRELOAD_MS, {
                                retuneIfNeeded: doRetune,
                                skipInitialRetune: false
                            }).then((ready) => {
                                if (myGen !== this.autoFadePreloadGen) return;
                                clearAutoFadePreloadUi();
                                if (sig && sig.signal && sig.signal.aborted) return;
                                if (!ready) {
                                    try { statusEl.innerText = 'Incoming deck has no playable station — crossfade cancelled.'; } catch (_) {}
                                    try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                                    if (autoMixEnabled) scheduleNextAutoMix();
                                    return;
                                }
                                runAutoFade(target, {
                                    stationPreloaded: isDeckIncomingReadyForAutoFade(target),
                                    onComplete: autoMixEnabled ? () => {
                                        this.autoMixNextTargetDeck = target === 'a' ? 'b' : 'a';
                                        scheduleNextAutoMix();
                                    } : undefined
                                });
                                try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                            });
                        } else {
                            runAutoFade(target, {
                                stationPreloaded: isDeckIncomingReadyForAutoFade(target),
                                onComplete: autoMixEnabled ? () => {
                                    this.autoMixNextTargetDeck = target === 'a' ? 'b' : 'a';
                                    scheduleNextAutoMix();
                                } : undefined
                            });
                            try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                        }
                    }, sig);
                }

                /**
                 * Entry point for the Space keyboard shortcut (short tap).
                 *
                 *  - If an AUTO-FADE animation is already running, reverse direction
                 *    immediately: both decks are streaming during the fade, so we just
                 *    re-launch runAutoFade towards the opposite deck with
                 *    stationPreloaded:true — no Change-Station retune (the user is
                 *    steering, not picking a fresh station).
                 *  - Otherwise, simulate a click on the AUTO-FADE button so the full
                 *    Change-Station / preload-wait / Auto-Mix-reschedule pipeline runs
                 *    exactly the same way as a button click.
                 */
                /**
                 * Shared helper: are both decks currently silent?
                 *
                 * "Silent" here means the underlying <audio> element is paused (or
                 * not even instantiated). A deck that is loaded but paused still
                 * counts as silent — what matters for the Space shortcut is whether
                 * the user is hearing anything right now.
                 */
                const bothDecksSilent = () => {
                    const aPlaying = !!(audioEl && !audioEl.paused);
                    const bPlaying = !!(audioElB && !audioElB.paused);
                    return !aPlaying && !bPlaying;
                };

                /**
                 * Shared helper: start whichever deck the crossfader is currently
                 * favouring. Used by both the Space short-tap "no music yet" case
                 * and the Space long-hold "pause everything OR start the active
                 * deck" case so the two behaviours stay in lockstep.
                 *
                 *   crossfader < 0.5  → Deck A wins   → start Deck A
                 *   crossfader ≥ 0.5  → Deck B wins   → start Deck B
                 *
                 * Falls back to playRadio() / playRadioB() when the chosen deck
                 * has no media source loaded yet (cold boot from start screen).
                 */
                const startActiveDeckByCrossfader = () => {
                    try {
                        this.clearSuppressEnsureCrossfadeDeckPlayback();
                        const raw = (djCross && djCross.value) || (mixCross && mixCross.value) || 0;
                        const x = Math.max(0, Math.min(1, Number(raw) || 0));
                        if (x < 0.5) {
                            const src = audioEl ? String(audioEl.currentSrc || audioEl.src || '') : '';
                            const hasSource = !!(src && src !== 'about:blank');
                            if (hasSource && audioEl) audioEl.play().catch(() => {});
                            else if (typeof playRadio === 'function') playRadio();
                        } else {
                            const src = audioElB ? String(audioElB.currentSrc || audioElB.src || '') : '';
                            const hasSource = !!(src && src !== 'about:blank');
                            if (hasSource && audioElB) audioElB.play().catch(() => {});
                            else if (typeof playRadioB === 'function') playRadioB();
                        }
                    } catch (_) {}
                };

                this.triggerAutoFadeFromShortcut = () => {
                    try {
                        this.clearSuppressEnsureCrossfadeDeckPlayback();
                        // First Space tap when nothing is playing: boot the deck
                        // the crossfader is favouring instead of triggering an
                        // AUTO-FADE. The user explicitly asked for this so that
                        // Space can be a one-key "start the show" hotkey, and any
                        // *subsequent* tap (once audio is flowing) still kicks
                        // the AUTO-FADE off the way it always did.
                        if (bothDecksSilent()) {
                            startActiveDeckByCrossfader();
                            return;
                        }
                        if (this.autoFadeRafId && this.autoFadeTargetDeck) {
                            const currentX = Math.max(0, Math.min(1, Number((djCross && djCross.value) || (mixCross && mixCross.value) || 0)));
                            const destinationX = this.autoFadeTargetDeck === 'b' ? 1 : 0;
                            const visuallyAtDestination = Math.abs(currentX - destinationX) <= 0.001;
                            if (!visuallyAtDestination) {
                                const opposite = this.autoFadeTargetDeck === 'a' ? 'b' : 'a';
                                runAutoFade(opposite, { stationPreloaded: true });
                                try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                                return;
                            }
                            try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                            this.autoFadeRafId = null;
                            this.autoFadeTargetDeck = null;
                            if (btnAutoFade) btnAutoFade.classList.remove('on');
                        }
                        if (btnAutoFade) btnAutoFade.click();
                    } catch (_) {}
                };

                /**
                 * Entry point for the Space keyboard shortcut (long-hold).
                 *
                 *  - If anything is currently playing on Deck A or Deck B: hard-pause
                 *    both decks (and cancel any in-flight AUTO-FADE so the crossfader
                 *    stops moving). Mirrors the "pause everything" intent of a long
                 *    Space press.
                 *  - If nothing is playing on either deck: start the deck on whichever
                 *    side of the crossfader is louder (<0.5 = Deck A, >=0.5 = Deck B).
                 *    Uses playRadio() / playRadioB() when no media source is loaded
                 *    so the very first long-press boots a station from scratch.
                 */
                this.pauseBothDecksOrStartActive = () => {
                    try {
                        if (!bothDecksSilent()) {
                            this.suppressEnsureCrossfadeDeckPlayback = true;
                            if (this.autoFadeRafId) {
                                try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                                this.autoFadeRafId = null;
                                this.autoFadeTargetDeck = null;
                                if (btnAutoFade) btnAutoFade.classList.remove('on');
                            }
                            try { if (audioEl) audioEl.pause(); } catch (_) {}
                            try { if (audioElB) audioElB.pause(); } catch (_) {}
                            return;
                        }
                        startActiveDeckByCrossfader();
                    } catch (_) {}
                };

                /**
                 * Stop any in-flight AUTO-FADE animation and clear its bookkeeping.
                 * Called from the V / B keyboard shortcuts so a manual play/pause
                 * actually sticks — without this, `runAutoFade`'s per-frame
                 * `ensureCrossfadeDeckPlayback()` re-plays whichever deck the user
                 * just paused as long as that deck has audible crossfader gain,
                 * making V / B feel inert for the entire fade duration.
                 *
                 * The crossfader is intentionally left at its current value so the
                 * user keeps the mix balance they had when they took over; only
                 * the animation timer and "in-flight target" state are reset.
                 * Safe to call when no fade is active (it's a no-op).
                 */
                this.cancelAutoFade = () => {
                    if (this.autoFadeRafId) {
                        try { cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                        this.autoFadeRafId = null;
                    }
                    this.autoFadeTargetDeck = null;
                    if (btnAutoFade) btnAutoFade.classList.remove('on');
                };
                if (autoFadeDuration) {
                    autoFadeDuration.addEventListener('input', () => {
                        const sec = Math.max(2, Math.min(15, Number(autoFadeDuration.value) || 2));
                        autoFadeDurationMs = Math.round(sec * 1000);
                        if (autoFadeReadout) autoFadeReadout.textContent = `${Math.round(sec)}s`;
                    }, sig);
                    autoFadeDuration.addEventListener('change', () => {
                        try { localStorage.setItem(AUTOFADE_MS_KEY, String(Math.max(AUTOFADE_MIN_MS, Math.min(AUTOFADE_MAX_MS, Number(autoFadeDurationMs) || 5000)))); } catch (_) {}
                    }, sig);
                }
                if (autoFadeChangeStationEn) {
                    autoFadeChangeStationEn.addEventListener('change', () => {
                        setAutoFadeChangeStationEnabled(!!autoFadeChangeStationEn.checked);
                        // Toggling resets AUTO-MIX's next-crossfade timer so the upcoming scheduled
                        // fade is recomputed from now with the new station-change preference.
                        if (autoMixEnabled) {
                            try { scheduleNextAutoMix(); } catch (_) {}
                        }
                    }, sig);
                }
                if (radioStationXfadeEn) {
                    radioStationXfadeEn.addEventListener('change', () => {
                        radioStationCrossfadeEnabled = !!radioStationXfadeEn.checked;
                        try { syncRadioStationCrossfadeUi(); } catch (_) {}
                        try { saveRadioStationCrossfadeToStorage(); } catch (_) {}
                    }, sig);
                }
                if (radioStationXfadeSec) {
                    radioStationXfadeSec.addEventListener('input', () => {
                        const sec = Math.max(0, Math.min(15, Number(radioStationXfadeSec.value) || 0));
                        radioStationCrossfadeSec = sec;
                        if (radioStationXfadeReadout) {
                            const label = Number.isInteger(sec) || Math.abs(sec - Math.round(sec)) < 1e-6
                                ? `${Math.round(sec)}`
                                : String(sec);
                            radioStationXfadeReadout.textContent = `${label}s`;
                        }
                    }, sig);
                    radioStationXfadeSec.addEventListener('change', () => {
                        radioStationCrossfadeSec = Math.max(0, Math.min(15, Number(radioStationXfadeSec.value) || 0));
                        try { saveRadioStationCrossfadeToStorage(); } catch (_) {}
                    }, sig);
                }
                if (btnAutoMix) {
                    btnAutoMix.addEventListener('pointerdown', (ev) => {
                        try { this._panelPointerPos = { x: ev.clientX, y: ev.clientY }; } catch (_) {}
                        autoMixLongPressFired = false;
                        if (this.autoMixHoldTimer) {
                            try { clearTimeout(this.autoMixHoldTimer); } catch (_) {}
                            this.autoMixHoldTimer = null;
                        }
                        this.autoMixHoldTimer = setTimeout(() => {
                            this.autoMixHoldTimer = null;
                            autoMixLongPressFired = true;
                            suppressAutoMixClickUntil = Date.now() + 450;
                            openAutoMixPanel();
                        }, 450);
                    }, sig);
                    const clearAutoMixHold = () => {
                        if (this.autoMixHoldTimer) {
                            try { clearTimeout(this.autoMixHoldTimer); } catch (_) {}
                            this.autoMixHoldTimer = null;
                        }
                    };
                    btnAutoMix.addEventListener('pointerup', clearAutoMixHold, sig);
                    btnAutoMix.addEventListener('pointercancel', clearAutoMixHold, sig);
                    btnAutoMix.addEventListener('pointerleave', clearAutoMixHold, sig);
                    btnAutoMix.addEventListener('click', () => {
                        if (autoMixLongPressFired || Date.now() < suppressAutoMixClickUntil) return;
                        // Same focus-release as the AUTO-FADE click handler — Space / Enter
                        // shouldn't be able to re-toggle AUTO-MIX just because this button
                        // happens to be the most-recently-focused element after a tap.
                        try { btnAutoMix.blur(); } catch (_) {}
                        try {
                            const ae = document.activeElement;
                            if (ae && ae !== document.body && typeof ae.blur === 'function' &&
                                (ae.classList.contains('dj-autofade-btn') || ae.classList.contains('dj-automix-btn'))) {
                                ae.blur();
                            }
                        } catch (_) {}
                        closeAutoFadePanel();
                        autoMixEnabled = !autoMixEnabled;
                        syncAutoMixRuntimeState();
                        syncAutoMixUi();
                        try { localStorage.setItem(AUTOMIX_ENABLED_KEY, autoMixEnabled ? '1' : '0'); } catch (_) {}
                        if (autoMixEnabled) {
                            autoMixSessionStartedAt = Date.now();
                            // AUTO-MIX expects random incoming stations — re-enable On Auto-Fade retune.
                            setAutoFadeChangeStationEnabled(true);
                            // Do not retune/reload the audible deck — only schedule the next crossfade.
                            try { ensureCrossfadeDeckPlayback(); } catch (_) {}
                            this.autoMixNextTargetDeck = (typeof getCrossfaderIncomingDeckKey === 'function')
                                ? getCrossfaderIncomingDeckKey() : 'b';
                            scheduleNextAutoMix();
                            scheduleAutoMixSessionLimit();
                        }
                        else {
                            clearAutoMixTimer();
                            this.autoMixNextTargetDeck = null;
                            this.autoMixNextFadeAt = 0;
                            clearAutoMixNextFadeTick();
                            clearAutoMixNextFadeUi();
                            clearAutoMixSessionLimitTimer();
                            autoMixSessionStartedAt = 0;
                            try { if (typeof clearAllAutoMixDeferLocal === 'function') clearAllAutoMixDeferLocal(); } catch (_) {}
                        }
                    }, sig);
                }
                if (autoMixMin) {
                    autoMixMin.addEventListener('input', () => {
                        autoMixMinMin = Math.max(1, Math.min(20, Number(autoMixMin.value) || 2));
                        if (autoMixMaxMin < autoMixMinMin) autoMixMaxMin = autoMixMinMin;
                        syncAutoMixUi();
                    }, sig);
                    autoMixMin.addEventListener('change', () => {
                        try { localStorage.setItem(AUTOMIX_MIN_KEY, String(autoMixMinMin)); } catch (_) {}
                        try { localStorage.setItem(AUTOMIX_MAX_KEY, String(autoMixMaxMin)); } catch (_) {}
                        if (autoMixEnabled) scheduleNextAutoMix();
                    }, sig);
                }
                if (autoMixMax) {
                    autoMixMax.addEventListener('input', () => {
                        autoMixMaxMin = Math.max(1, Math.min(20, Number(autoMixMax.value) || 20));
                        if (autoMixMaxMin < autoMixMinMin) autoMixMinMin = autoMixMaxMin;
                        syncAutoMixUi();
                    }, sig);
                    autoMixMax.addEventListener('change', () => {
                        try { localStorage.setItem(AUTOMIX_MIN_KEY, String(autoMixMinMin)); } catch (_) {}
                        try { localStorage.setItem(AUTOMIX_MAX_KEY, String(autoMixMaxMin)); } catch (_) {}
                        if (autoMixEnabled) scheduleNextAutoMix();
                    }, sig);
                }
                if (autoMixLimitEn) {
                    autoMixLimitEn.addEventListener('change', () => {
                        autoMixLimitEnabled = !!autoMixLimitEn.checked;
                        try { localStorage.setItem(AUTOMIX_LIMIT_EN_KEY, autoMixLimitEnabled ? '1' : '0'); } catch (_) {}
                        if (autoMixEnabled) {
                            if (autoMixLimitEnabled) {
                                autoMixSessionStartedAt = Date.now();
                                scheduleAutoMixSessionLimit();
                            } else {
                                clearAutoMixSessionLimitTimer();
                            }
                        }
                    }, sig);
                }
                if (autoMixLimitMinEl) {
                    autoMixLimitMinEl.addEventListener('input', () => {
                        autoMixLimitMin = Math.max(5, Math.min(300, Number(autoMixLimitMinEl.value) || 60));
                        if (autoMixLimitReadout) autoMixLimitReadout.textContent = formatLimitMinutesReadout(autoMixLimitMin);
                    }, sig);
                    autoMixLimitMinEl.addEventListener('change', () => {
                        autoMixLimitMin = Math.max(5, Math.min(300, Number(autoMixLimitMinEl.value) || 60));
                        try { localStorage.setItem(AUTOMIX_LIMIT_MIN_KEY, String(autoMixLimitMin)); } catch (_) {}
                        if (autoMixEnabled && autoMixLimitEnabled) {
                            autoMixSessionStartedAt = Date.now();
                            scheduleAutoMixSessionLimit();
                        }
                    }, sig);
                }
                document.addEventListener('click', (ev) => {
                    const t = ev.target;
                    if (autoFadePanel && autoFadePanel.classList.contains('is-open')) {
                        if (!(autoFadeWrap && autoFadeWrap.contains(t))) closeAutoFadePanel();
                    }
                    if (autoMixPanel && autoMixPanel.classList.contains('is-open')) {
                        if (!(autoMixWrap && autoMixWrap.contains(t))) closeAutoMixPanel();
                    }
                }, { capture: true, signal: this.abortCtrl.signal });
                document.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Escape') {
                        closeAutoFadePanel();
                        closeAutoMixPanel();
                    }
                }, sig);

                const clampVol = (v) => Math.max(0, Math.min(1, Number(v) || 0));
                const syncDeckVolFillCss = (inputEl, cv) => {
                    try {
                        const wrap = inputEl && inputEl.closest('.dj-deck-vol-wrap');
                        if (wrap) wrap.style.setProperty('--deck-vol-pct', String(clampVol(cv)));
                    } catch (_) {}
                };
                let savedDeckVolA = 1;
                let savedDeckVolB = 1;
                const syncDeckVolMuteButtons = () => {
                    try {
                        if (this.els.deckVolMuteA && this.els.deckVolA) {
                            const m = Number(this.els.deckVolA.value) < 0.001;
                            this.els.deckVolMuteA.setAttribute('aria-pressed', m ? 'true' : 'false');
                            this.els.deckVolMuteA.classList.toggle('on', m);
                        }
                        if (this.els.deckVolMuteB && this.els.deckVolB) {
                            const m = Number(this.els.deckVolB.value) < 0.001;
                            this.els.deckVolMuteB.setAttribute('aria-pressed', m ? 'true' : 'false');
                            this.els.deckVolMuteB.classList.toggle('on', m);
                        }
                    } catch (_) {}
                };
                const setDeckVolA = (v) => {
                    const cv = clampVol(v);
                    try { if (audioEl) audioEl.volume = cv; } catch (_) {}
                    try { if (audioElRadioAAlt) audioElRadioAAlt.volume = cv; } catch (_) {}
                    try { syncDeckVolFillCss(this.els.deckVolA, cv); } catch (_) {}
                    if (cv > 0.001) savedDeckVolA = cv;
                    syncDeckVolMuteButtons();
                };
                const setDeckVolB = (v) => {
                    const cv = clampVol(v);
                    try { if (audioElB) audioElB.volume = cv; } catch (_) {}
                    try { if (audioElRadioBAlt) audioElRadioBAlt.volume = cv; } catch (_) {}
                    try { syncDeckVolFillCss(this.els.deckVolB, cv); } catch (_) {}
                    if (cv > 0.001) savedDeckVolB = cv;
                    syncDeckVolMuteButtons();
                };
                const blurDjRangeIfFocused = (el) => {
                    if (!el) return;
                    const fn = () => {
                        try {
                            if (document.activeElement === el) el.blur();
                        } catch (_) {}
                    };
                    try {
                        el.addEventListener('pointerup', fn, sig);
                        el.addEventListener('touchend', fn, { signal: sig.signal, passive: true });
                        el.addEventListener('change', fn, sig);
                    } catch (_) {}
                };
                try {
                    if (this.els.deckVolA) {
                        const initA = (audioEl && Number.isFinite(audioEl.volume)) ? audioEl.volume : 1;
                        const ia = clampVol(initA);
                        savedDeckVolA = ia > 0.001 ? ia : 1;
                        this.els.deckVolA.value = String(ia);
                        setDeckVolA(ia);
                        this.els.deckVolA.addEventListener('input', () => setDeckVolA(this.els.deckVolA.value), sig);
                        blurDjRangeIfFocused(this.els.deckVolA);
                        if (this.els.deckVolMuteA) {
                            this.wireFooterBtnLiveFromSlider(this.els.deckVolA, this.els.deckVolMuteA, {
                                restLabel: 'VOL',
                                liveClass: 'dj-deck-vol-mute--live',
                                format: (v) => String(Math.round(clampVol(v) * 100))
                            }, sig);
                        }
                    }
                    if (this.els.deckVolB) {
                        const initB = (audioElB && Number.isFinite(audioElB.volume)) ? audioElB.volume : 1;
                        const ib = clampVol(initB);
                        savedDeckVolB = ib > 0.001 ? ib : 1;
                        this.els.deckVolB.value = String(ib);
                        setDeckVolB(ib);
                        this.els.deckVolB.addEventListener('input', () => setDeckVolB(this.els.deckVolB.value), sig);
                        blurDjRangeIfFocused(this.els.deckVolB);
                        if (this.els.deckVolMuteB) {
                            this.wireFooterBtnLiveFromSlider(this.els.deckVolB, this.els.deckVolMuteB, {
                                restLabel: 'VOL',
                                liveClass: 'dj-deck-vol-mute--live',
                                format: (v) => String(Math.round(clampVol(v) * 100))
                            }, sig);
                        }
                    }
                    try {
                        blurDjRangeIfFocused(this.root && this.root.querySelector('#dj-crossfader'));
                        blurDjRangeIfFocused(this.root && this.root.querySelector('#dj-a-beat-bpm'));
                        blurDjRangeIfFocused(this.root && this.root.querySelector('#dj-b-beat-bpm'));
                    } catch (_) {}
                    if (this.els.deckVolMuteA) {
                        this.els.deckVolMuteA.addEventListener(
                            'click',
                            (e) => {
                                try {
                                    e.preventDefault();
                                } catch (_) {}
                                if (!this.els.deckVolA) return;
                                const cur = Number(this.els.deckVolA.value) || 0;
                                if (cur < 0.001) {
                                    const r = clampVol(savedDeckVolA > 0.001 ? savedDeckVolA : 1);
                                    this.els.deckVolA.value = String(r);
                                    setDeckVolA(r);
                                } else {
                                    savedDeckVolA = cur;
                                    this.els.deckVolA.value = '0';
                                    setDeckVolA(0);
                                }
                            },
                            sig
                        );
                    }
                    if (this.els.deckVolMuteB) {
                        this.els.deckVolMuteB.addEventListener(
                            'click',
                            (e) => {
                                try {
                                    e.preventDefault();
                                } catch (_) {}
                                if (!this.els.deckVolB) return;
                                const cur = Number(this.els.deckVolB.value) || 0;
                                if (cur < 0.001) {
                                    const r = clampVol(savedDeckVolB > 0.001 ? savedDeckVolB : 1);
                                    this.els.deckVolB.value = String(r);
                                    setDeckVolB(r);
                                } else {
                                    savedDeckVolB = cur;
                                    this.els.deckVolB.value = '0';
                                    setDeckVolB(0);
                                }
                            },
                            sig
                        );
                    }
                } catch (_) {}

                if (this.els.visualBackBtn) {
                    this.els.visualBackBtn.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        if (this.deckBQueueVisible) {
                            this.hideDeckBQueueView();
                            this.updateStationTitles();
                            return;
                        }
                        if (this.deckBMediaPanelVisible) {
                            this.hideDeckBMediaView();
                            this.updateStationTitles();
                            return;
                        }
                        this.tearDownDeckBViz();
                        this.syncDeckBVisualButtons();
                        this.updateStationTitles();
                    }, sig);
                }
                this.syncDeckBVisualButtons();

                const djToggleVideo = () => {
                    try {
                        if (this.deckBVizMode === 'video') {
                            this.tearDownDeckBViz();
                            this.syncDeckBVisualButtons();
                            this.syncFxLightsFromState();
                            return;
                        }
                        this.startDeckBVideoVisual();
                        this.syncDeckBVisualButtons();
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };
                const djToggleTextInDeckA = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        const stage = (typeof getDeckBStageEl === 'function') ? getDeckBStageEl() : null;
                        const deckBOn = !!(stage && stage.classList.contains('dj-deck-b-text-mode'));
                        if (deckBOn) {
                            try { if (typeof setTextAuto === 'function') setTextAuto(false); } catch (_) {}
                            try { if (typeof setDeckBTextMode === 'function') setDeckBTextMode(false); } catch (_) {}
                            try { if (textInPanel) delete textInPanel.dataset.textTarget; } catch (_) {}
                            const panelOpen = !!(textInPanel && !textInPanel.classList.contains('display-none') && textInPanel.classList.contains('open'));
                            if (panelOpen) {
                                try { hideTextInPanel({ forceCloseDeckB: true }); } catch (_) {}
                            }
                            try { if (typeof syncDjTextInDeckLights === 'function') syncDjTextInDeckLights(); } catch (_) {}
                            return;
                        }
                        if (typeof openTextInForTarget === 'function') openTextInForTarget('global');
                        if (typeof syncDjTextInDeckLights === 'function') syncDjTextInDeckLights();
                    } catch (_) {}
                };
                const djToggleTextInDeckB = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (typeof openTextInForTarget === 'function') openTextInForTarget('deck-b');
                        if (typeof syncDjTextInDeckLights === 'function') syncDjTextInDeckLights();
                    } catch (_) {}
                };
                const showDeckBAvatar = () => {
                    const open = () => {
                        try {
                            showWebm({ deckBCenter: true });
                            this.syncFxLightsFromState();
                        } catch (_) {}
                    };
                    if (!webmList.length) {
                        loadWebmList().finally(() => {
                            try {
                                if (webmList.length > 0) open();
                                else this.syncFxLightsFromState();
                            } catch (_) {}
                        });
                        return;
                    }
                    open();
                };
                const djToggleAvatarDeckB = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (!webmOn) showDeckBAvatar();
                        else hideWebm();
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };
                const djToggleAvatarDeckA = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (!webmOn) {
                            const centerInB = this.isDeckBVisualModeActive();
                            const open = () => {
                                try {
                                    if (centerInB) showWebm({ deckBCenter: true });
                                    else showWebm();
                                    this.syncFxLightsFromState();
                                } catch (_) {}
                            };
                            if (!webmList.length) {
                                loadWebmList().finally(() => {
                                    try {
                                        if (webmList.length > 0) open();
                                        else this.syncFxLightsFromState();
                                    } catch (_) {}
                                });
                                return;
                            }
                            open();
                        } else {
                            hideWebm();
                        }
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };
                const djToggleMixerUi = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (typeof toggleMixPanel === 'function') toggleMixPanel();
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };
                const djToggleStationUi = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (this.deckBQueueVisible || this.deckBMediaPanelVisible) {
                            this.toggleDeckBStationMediaPanels();
                            this.syncFxLightsFromState();
                            return;
                        }
                        if (this.isDeckBVisualModeActive()) {
                            this.showDeckBMediaView();
                            this.syncFxLightsFromState();
                            return;
                        }
                        if (this._deckAStationsTopMenuOnNextClick) {
                            this._deckAStationsTopMenuOnNextClick = false;
                        }
                        if (typeof toggleTopMenuPanel === 'function') toggleTopMenuPanel();
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };
                const djToggleDeckBStationPanel = () => {
                    try {
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        this.toggleDeckBStationMediaPanels();
                        this.syncFxLightsFromState();
                    } catch (_) {}
                };

                if (this.els.fxLow) this.els.fxLow.addEventListener('click', djToggleVideo, sig);
                if (this.els.fxLowB) this.els.fxLowB.addEventListener('click', djToggleVideo, sig);
                if (this.els.queueA) {
                    this.els.queueA.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { window.__suppressNextClick = true; } catch (_) {}
                        this.toggleDeckBQueuePanel();
                        try { resetIdleTimer(); } catch (_) {}
                    }, sig);
                }
                if (this.els.queueB) {
                    this.els.queueB.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { window.__suppressNextClick = true; } catch (_) {}
                        this.toggleDeckBQueuePanel();
                        try { resetIdleTimer(); } catch (_) {}
                    }, sig);
                }

                try {
                    const fiQA = document.createElement('input');
                    fiQA.type = 'file';
                    fiQA.accept = 'audio/*,video/*';
                    fiQA.multiple = true;
                    fiQA.style.display = 'none';
                    fiQA.setAttribute('aria-hidden', 'true');
                    root.appendChild(fiQA);
                    const fiQB = document.createElement('input');
                    fiQB.type = 'file';
                    fiQB.accept = 'audio/*,video/*';
                    fiQB.multiple = true;
                    fiQB.style.display = 'none';
                    fiQB.setAttribute('aria-hidden', 'true');
                    root.appendChild(fiQB);
                    const fiQAFolder = document.createElement('input');
                    fiQAFolder.type = 'file';
                    fiQAFolder.multiple = true;
                    fiQAFolder.style.display = 'none';
                    fiQAFolder.setAttribute('aria-hidden', 'true');
                    fiQAFolder.setAttribute('webkitdirectory', '');
                    root.appendChild(fiQAFolder);
                    const fiQBFolder = document.createElement('input');
                    fiQBFolder.type = 'file';
                    fiQBFolder.multiple = true;
                    fiQBFolder.style.display = 'none';
                    fiQBFolder.setAttribute('aria-hidden', 'true');
                    fiQBFolder.setAttribute('webkitdirectory', '');
                    root.appendChild(fiQBFolder);
                    const btnAddA = root.querySelector('#dj-queue-add-a');
                    const btnAddB = root.querySelector('#dj-queue-add-b');
                    const btnFolderA = root.querySelector('#dj-queue-folder-a');
                    const btnFolderB = root.querySelector('#dj-queue-folder-b');
                    const btnUrlDeckA = root.querySelector('#dj-queue-url-deck-a');
                    const btnUrlDeckB = root.querySelector('#dj-queue-url-deck-b');
                    if (btnAddA) btnAddA.addEventListener('click', () => { try { __djLocalPickImmediateDeck = null; fiQA.click(); } catch (_) {} }, sig);
                    if (btnAddB) btnAddB.addEventListener('click', () => { try { __djLocalPickImmediateDeck = null; fiQB.click(); } catch (_) {} }, sig);
                    if (btnFolderA) {
                        btnFolderA.addEventListener('click', () => {
                            openDeckLocalFolderPicker('a', fiQAFolder).catch(() => {});
                        }, sig);
                    }
                    if (btnFolderB) {
                        btnFolderB.addEventListener('click', () => {
                            openDeckLocalFolderPicker('b', fiQBFolder).catch(() => {});
                        }, sig);
                    }
                    if (btnUrlDeckA) btnUrlDeckA.addEventListener('click', () => { try { promptAddUrlForDeck('a'); this.refreshQueueUi(); } catch (_) {} }, sig);
                    if (btnUrlDeckB) btnUrlDeckB.addEventListener('click', () => { try { promptAddUrlForDeck('b'); this.refreshQueueUi(); } catch (_) {} }, sig);
                    const applyDeckLocalFilePick = (deckKey, files, forceNow) => {
                        if (!files || !files.length) return;
                        if (forceNow) ingestLocalFilesToDeckAndPlay(deckKey, files);
                        else addDeckLocalFilesToDeck(deckKey, files);
                        try { this.refreshQueueUi(); } catch (_) {}
                    };
                    fiQA.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        if (!files.length) return;
                        const forceNow = __djLocalPickImmediateDeck === 'a';
                        __djLocalPickImmediateDeck = null;
                        applyDeckLocalFilePick('a', files, forceNow);
                    }, sig);
                    fiQB.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        if (!files.length) return;
                        const forceNow = __djLocalPickImmediateDeck === 'b';
                        __djLocalPickImmediateDeck = null;
                        applyDeckLocalFilePick('b', files, forceNow);
                    }, sig);
                    fiQAFolder.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        applyDeckLocalFilePick('a', files, false);
                    }, sig);
                    fiQBFolder.addEventListener('change', (ev) => {
                        const files = Array.from(ev.target.files || []);
                        try { ev.target.value = ''; } catch (_) {}
                        applyDeckLocalFilePick('b', files, false);
                    }, sig);
                    const openDeckLocalFolderPicker = async (deckKey, fiFolder) => {
                        try { __djLocalPickImmediateDeck = null; } catch (_) {}
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
                        applyDeckLocalFilePick(deckKey, files, false);
                    };

                    const jogWrapA = this.els.jogA && this.els.jogA.closest('.dj-jog-wrap');
                    const jogWrapB = this.els.jogB && this.els.jogB.closest('.dj-jog-wrap');
                    const wireJogWrapLocalLoad = (wrap, deckKey, fiDeck) => {
                        if (!wrap) return;
                        wrap.addEventListener('dblclick', (e) => {
                            try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                            __djLocalPickImmediateDeck = deckKey;
                            scheduleClearDjLocalPickImmediate(deckKey);
                            try { fiDeck.click(); } catch (_) {}
                        }, sig);
                        wrap.addEventListener('dragover', (e) => {
                            try {
                                const dt = e.dataTransfer;
                                if (!dt || !Array.from(dt.types || []).includes('Files')) return;
                                e.preventDefault();
                                dt.dropEffect = 'copy';
                            } catch (_) {}
                        }, sig);
                        wrap.addEventListener('drop', (e) => {
                            try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                            const dt = e.dataTransfer;
                            if (!dt) return;
                            const applyDrop = (files) => {
                                if (!files || !files.length) return;
                                addDeckLocalFilesToDeck(deckKey, files);
                                try { this.refreshQueueUi(); } catch (_) {}
                            };
                            if (typeof collectMediaFilesFromDataTransfer === 'function') {
                                collectMediaFilesFromDataTransfer(dt).then(applyDrop).catch(() => {
                                    if (dt.files && dt.files.length) applyDrop(Array.from(dt.files));
                                });
                            } else if (dt.files && dt.files.length) {
                                applyDrop(Array.from(dt.files));
                            }
                        }, sig);
                    };
                    wireJogWrapLocalLoad(jogWrapA, 'a', fiQA);
                    wireJogWrapLocalLoad(jogWrapB, 'b', fiQB);

                    const sharedQueueUrlInput = root.querySelector('#dj-b-video-input');
                    const btnVideoLoad = root.querySelector('#dj-b-video-load');
                    const btnVideoLocal = root.querySelector('#dj-b-video-local');
                    const btnVideoQueueA = root.querySelector('#dj-b-video-q-a');
                    const btnVideoQueueB = root.querySelector('#dj-b-video-q-b');
                    const btnRadioToggle = root.querySelector('#dj-b-video-radio');
                    const radioPopout = root.querySelector('#dj-radio-url-popout');
                    const radioUrlInput = root.querySelector('#dj-radio-url-input');
                    const btnRadioUrlAdd = root.querySelector('#dj-radio-url-add');
                    const btnRadioUrlClose = root.querySelector('#dj-radio-url-close');
                    const mediaList = root.querySelector('#dj-media-queue-list');
                    const btnMediaLoop = root.querySelector('#dj-media-loop');
                    const btnMediaAll = root.querySelector('#dj-media-all');
                    const btnMediaShuffle = root.querySelector('#dj-media-shuffle');
                    const btnVideoFolder = root.querySelector('#dj-b-video-folder');
                    const fiQueueVideo = document.createElement('input');
                    fiQueueVideo.type = 'file';
                    fiQueueVideo.accept = 'video/*';
                    fiQueueVideo.multiple = true;
                    fiQueueVideo.style.display = 'none';
                    fiQueueVideo.setAttribute('aria-hidden', 'true');
                    root.appendChild(fiQueueVideo);
                    const fiQueueVideoFolder = document.createElement('input');
                    fiQueueVideoFolder.type = 'file';
                    fiQueueVideoFolder.multiple = true;
                    fiQueueVideoFolder.style.display = 'none';
                    fiQueueVideoFolder.setAttribute('aria-hidden', 'true');
                    fiQueueVideoFolder.setAttribute('webkitdirectory', '');
                    root.appendChild(fiQueueVideoFolder);
                    const getQueueInputSource = () => {
                        const clean = sanitizeUrlForAudio(sharedQueueUrlInput ? sharedQueueUrlInput.value : '');
                        if (!clean) return null;
                        if (isLikelyRadioStreamUrl(clean)) {
                            addUserRadioStation(clean, deriveNameFromUrl(clean));
                            if (sharedQueueUrlInput) sharedQueueUrlInput.value = '';
                            return { type: 'radio' };
                        }
                        if (!isVideoQueueEligibleUrl(clean)) {
                            return { type: 'audiourl', url: clean, name: deriveNameFromUrl(clean) };
                        }
                        upsertMediaVideoQueueEntry(clean, deriveNameFromUrl(clean));
                        registerDeckVideoFeed('b', clean, deriveNameFromUrl(clean), false);
                        return { type: 'url', entry: { url: clean, name: deriveNameFromUrl(clean) } };
                    };
                    const closeRadioPopout = () => {
                        if (!radioPopout) return;
                        radioPopout.classList.remove('is-open');
                        radioPopout.setAttribute('aria-hidden', 'true');
                    };
                    const openRadioPopout = () => {
                        if (!radioPopout) return;
                        radioPopout.classList.add('is-open');
                        radioPopout.setAttribute('aria-hidden', 'false');
                        try { if (radioUrlInput) radioUrlInput.focus(); } catch (_) {}
                    };
                    if (btnRadioToggle) btnRadioToggle.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        if (!radioPopout) return;
                        if (radioPopout.classList.contains('is-open')) closeRadioPopout();
                        else openRadioPopout();
                    }, sig);
                    if (btnRadioUrlClose) btnRadioUrlClose.addEventListener('click', () => { closeRadioPopout(); }, sig);
                    if (btnVideoLocal) btnVideoLocal.addEventListener('click', () => {
                        try { fiQueueVideo.click(); } catch (_) {}
                    }, sig);
                    if (btnVideoFolder) btnVideoFolder.addEventListener('click', () => {
                        try { fiQueueVideoFolder.click(); } catch (_) {}
                    }, sig);
                    fiQueueVideo.addEventListener('change', (ev) => {
                        const picked = ev.target.files ? Array.from(ev.target.files) : [];
                        try { ev.target.value = ''; } catch (_) {}
                        addVideoFilesToMediaQueue(picked);
                        if (sharedQueueUrlInput) sharedQueueUrlInput.value = '';
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    fiQueueVideoFolder.addEventListener('change', (ev) => {
                        const picked = ev.target.files ? Array.from(ev.target.files) : [];
                        try { ev.target.value = ''; } catch (_) {}
                        addVideoFilesToMediaQueue(picked);
                        if (sharedQueueUrlInput) sharedQueueUrlInput.value = '';
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnVideoLoad) btnVideoLoad.addEventListener('click', () => {
                        const src = getQueueInputSource();
                        if (!src) return;
                        if (src.type === 'radio') {
                            try { this.refreshQueueUi(); } catch (_) {}
                            return;
                        }
                        if (src.type === 'audiourl') {
                            enqueueDeckUrl('b', src.url, src.name);
                            try { this.refreshQueueUi(); } catch (_) {}
                            return;
                        }
                        if (src.type === 'url') {
                            const mqIdx = mediaVideoQueue.findIndex((x) => x && x.url === src.entry.url);
                            this.startDeckBVideoVisual({ mediaQueueIndex: mqIdx >= 0 ? mqIdx : 0 });
                        }
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnVideoQueueA) btnVideoQueueA.addEventListener('click', () => {
                        const src = getQueueInputSource();
                        if (!src || src.type === 'radio') return;
                        if (src.type === 'audiourl') enqueueDeckUrl('a', src.url, src.name);
                        else if (src.type === 'url') enqueueDeckUrl('a', src.entry.url, src.entry.name);
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnVideoQueueB) btnVideoQueueB.addEventListener('click', () => {
                        const src = getQueueInputSource();
                        if (!src || src.type === 'radio') return;
                        if (src.type === 'audiourl') enqueueDeckUrl('b', src.url, src.name);
                        else if (src.type === 'url') enqueueDeckUrl('b', src.entry.url, src.entry.name);
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnRadioUrlAdd) btnRadioUrlAdd.addEventListener('click', () => {
                        const clean = sanitizeUrlForAudio(radioUrlInput ? radioUrlInput.value : '');
                        if (!clean) return;
                        addUserRadioStation(clean, deriveNameFromUrl(clean));
                        if (radioUrlInput) radioUrlInput.value = '';
                        closeRadioPopout();
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (radioUrlInput) radioUrlInput.addEventListener('keydown', (ev) => {
                        if (ev.key !== 'Enter') return;
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        try { if (btnRadioUrlAdd) btnRadioUrlAdd.click(); } catch (_) {}
                    }, sig);
                    document.addEventListener('click', (e) => {
                        try {
                            if (!radioPopout || !radioPopout.classList.contains('is-open')) return;
                            const t = ev.target;
                            if (radioPopout.contains(t)) return;
                            if (btnRadioToggle && btnRadioToggle.contains(t)) return;
                            closeRadioPopout();
                        } catch (_) {}
                    }, { capture: true, signal: this.abortCtrl.signal });
                    if (sharedQueueUrlInput && btnVideoLoad) sharedQueueUrlInput.addEventListener('keydown', (ev) => {
                        if (ev.key !== 'Enter') return;
                        try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                        try { btnVideoLoad.click(); } catch (_) {}
                    }, sig);
                    if (mediaList) mediaList.addEventListener('click', (ev) => {
                        const btn = ev.target && ev.target.closest ? ev.target.closest('button[data-action]') : null;
                        if (!btn) return;
                        const row = btn.closest('li[data-id]');
                        if (!row) return;
                        const id = row.getAttribute('data-id');
                        const it = mediaVideoQueue.find((x) => x && String(x.id) === String(id));
                        if (!it || !it.url) return;
                        const action = btn.getAttribute('data-action');
                        if (action === 'play') {
                            const mqIdx = mediaVideoQueue.findIndex((m) => m && m.url === it.url);
                            this.startDeckBVideoVisual({ mediaQueueIndex: mqIdx >= 0 ? mqIdx : 0 });
                        } else if (action === 'queue-a') enqueueDeckUrl('a', it.url, it.label);
                        else if (action === 'queue-b') enqueueDeckUrl('b', it.url, it.label);
                        else if (action === 'delete') removeMediaVideoQueueItem(id);
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnMediaLoop) btnMediaLoop.addEventListener('click', () => {
                        deckBVideoLoopEnabled = !deckBVideoLoopEnabled;
                        try {
                            const loopOn = !!deckBVideoSingleLoopActive();
                            [this.deckBVideoElA, this.deckBVideoElB, this.deckBVideoElQ, this.deckBVideoEl].forEach((v) => {
                                if (v && !deckBVideoUserIdle) v.loop = loopOn;
                            });
                        } catch (_) {}
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnMediaAll) btnMediaAll.addEventListener('click', () => {
                        deckBVideoPlayAllEnabled = !deckBVideoPlayAllEnabled;
                        try {
                            const loopOn = !!deckBVideoSingleLoopActive();
                            [this.deckBVideoElA, this.deckBVideoElB, this.deckBVideoElQ, this.deckBVideoEl].forEach((v) => {
                                if (v && !deckBVideoUserIdle) v.loop = loopOn;
                            });
                        } catch (_) {}
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    if (btnMediaShuffle) btnMediaShuffle.addEventListener('click', () => {
                        deckBVideoShuffleEnabled = !deckBVideoShuffleEnabled;
                        try { this.refreshQueueUi(); } catch (_) {}
                    }, sig);
                    const stationCycleListEl = root.querySelector('#dj-station-cycle-list');
                    if (stationCycleListEl) stationCycleListEl.addEventListener('wheel', (ev) => {
                        try { ev.stopPropagation(); } catch (_) {}
                    }, { passive: true, signal: this.abortCtrl.signal });
                } catch (_) {}

                if (this.els.fxHighA) this.els.fxHighA.addEventListener('click', djToggleAvatarDeckA, sig);
                if (this.els.fxHighB) this.els.fxHighB.addEventListener('click', djToggleAvatarDeckB, sig);
                if (this.els.fxTrebleA) this.els.fxTrebleA.addEventListener('click', djToggleMixerUi, sig);
                if (this.els.fxTrebleB) this.els.fxTrebleB.addEventListener('click', djToggleMixerUi, sig);
                if (this.els.fxDistortA) this.els.fxDistortA.addEventListener('click', djToggleStationUi, sig);
                if (this.els.fxDistortB) this.els.fxDistortB.addEventListener('click', djToggleDeckBStationPanel, sig);

                const TK_HOLD_MS = 450;
                const TK_F_MIN = 200;
                const TK_F_MAX = 8000;
                const TK_Q_MIN = 0.5;
                const TK_Q_MAX = 18;
                const tkHoldPanel = document.createElement('div');
                tkHoldPanel.className = 'dj-tk-hold-panel display-none';
                tkHoldPanel.setAttribute('aria-hidden', 'true');
                tkHoldPanel.innerHTML = '<div class="dj-tk-hold-head">TK filter · keep holding · drag to set freq / Q</div><canvas class="dj-tk-hold-graph" width="220" height="72"></canvas><div class="dj-tk-hold-pad" id="dj-tk-hold-pad"><span class="dj-tk-hold-dot" id="dj-tk-hold-dot"></span></div><div class="dj-tk-hold-readout"><span class="dj-tk-f">1200</span> Hz · Q <span class="dj-tk-q">4.0</span></div>';
                root.appendChild(tkHoldPanel);
                const tkGraphEl = tkHoldPanel.querySelector('.dj-tk-hold-graph');
                const tkPadEl = tkHoldPanel.querySelector('#dj-tk-hold-pad');
                const tkDotEl = tkHoldPanel.querySelector('#dj-tk-hold-dot');
                const tkFRead = tkHoldPanel.querySelector('.dj-tk-f');
                const tkQRead = tkHoldPanel.querySelector('.dj-tk-q');

                let tkHoldTimer = null;
                let tkHoldUiOpen = false;
                let tkPressPointerId = null;
                let tkAnchorX = 0;
                let tkAnchorY = 0;
                let tkDocMove = null;
                let tkDocUp = null;
                let tkPreOpenMove = null;
                let tkSkipNextTkBtnUp = false;

                const removeTkPreOpenMove = () => {
                    if (!tkPreOpenMove) return;
                    try {
                        document.removeEventListener('pointermove', tkPreOpenMove, true);
                    } catch (_) {}
                    tkPreOpenMove = null;
                };

                const positionTkHoldPanelAtCursor = () => {
                    try {
                        const rr = root.getBoundingClientRect();
                        const pw = tkHoldPanel.offsetWidth || 268;
                        const ph = tkHoldPanel.offsetHeight || 240;
                        const margin = 10;
                        let left = tkAnchorX - rr.left - pw / 2;
                        let top = tkAnchorY - rr.top - 28;
                        left = Math.max(margin, Math.min(rr.width - pw - margin, left));
                        top = Math.max(margin, Math.min(rr.height - ph - margin, top));
                        tkHoldPanel.style.left = Math.round(left) + 'px';
                        tkHoldPanel.style.top = Math.round(top) + 'px';
                    } catch (_) {}
                };

                const tkFreqToNorm = (f) => {
                    const lf = Math.max(TK_F_MIN, Math.min(TK_F_MAX, f));
                    return Math.log(lf / TK_F_MIN) / Math.log(TK_F_MAX / TK_F_MIN);
                };
                const tkNormToFreq = (nx) => TK_F_MIN * Math.pow(TK_F_MAX / TK_F_MIN, Math.max(0, Math.min(1, nx)));
                const tkQToNorm = (q) => {
                    const lq = Math.max(TK_Q_MIN, Math.min(TK_Q_MAX, q));
                    return 1 - (lq - TK_Q_MIN) / (TK_Q_MAX - TK_Q_MIN);
                };
                const tkNormToQ = (ny) => TK_Q_MIN + (1 - Math.max(0, Math.min(1, ny))) * (TK_Q_MAX - TK_Q_MIN);

                const closeTkHoldUi = () => {
                    const wasOpen = tkHoldUiOpen;
                    try {
                        if (tkDocMove) {
                            document.removeEventListener('pointermove', tkDocMove, true);
                            tkDocMove = null;
                        }
                        if (tkDocUp) {
                            document.removeEventListener('pointerup', tkDocUp, true);
                            tkDocUp = null;
                        }
                    } catch (_) {}
                    tkHoldUiOpen = false;
                    tkPressPointerId = null;
                    tkHoldPanel.classList.add('display-none');
                    tkHoldPanel.setAttribute('aria-hidden', 'true');
                    /* Only needed when button still receives pointerup after close (same-target release). */
                    if (wasOpen) tkSkipNextTkBtnUp = true;
                };

                const drawTkHoldGraph = () => {
                    try {
                        const ctx = tkGraphEl.getContext('2d');
                        const w = tkGraphEl.width;
                        const h = tkGraphEl.height;
                        ctx.clearRect(0, 0, w, h);
                        ctx.fillStyle = 'rgba(0,0,0,0.45)';
                        ctx.fillRect(0, 0, w, h);
                        if (!state || !state.audioCtx || !state.fx || !state.fx.tk || !state.fx.tk.node) return;
                        const nyquist = state.audioCtx.sampleRate / 2;
                        const n = 160;
                        const freqs = new Float32Array(n);
                        for (let i = 0; i < n; i++) freqs[i] = (i / (n - 1)) * nyquist;
                        const mag = new Float32Array(n);
                        const phase = new Float32Array(n);
                        state.fx.tk.node.getFrequencyResponse(freqs, mag, phase);
                        ctx.strokeStyle = 'rgba(212,175,55,0.95)';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        for (let i = 0; i < n; i++) {
                            const db = 20 * Math.log10(mag[i] + 1e-15);
                            const x = (i / (n - 1)) * w;
                            const normDb = Math.max(0, Math.min(1, (db + 48) / 48));
                            const y = h - 4 - normDb * (h - 8);
                            if (i === 0) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                        }
                        ctx.stroke();
                    } catch (_) {}
                };

                const syncTkHoldUiFromNode = () => {
                    try {
                        if (!state || !state.fx || !state.fx.tk || !state.fx.tk.node) return;
                        const f = state.fx.tk.node.frequency.value;
                        const q = state.fx.tk.node.Q.value;
                        const nx = tkFreqToNorm(f);
                        const ny = tkQToNorm(q);
                        tkDotEl.style.left = (nx * 100) + '%';
                        tkDotEl.style.top = (ny * 100) + '%';
                        tkFRead.textContent = String(Math.round(f));
                        tkQRead.textContent = q.toFixed(2);
                    } catch (_) {}
                };

                const applyTkHoldNorm = (nx, ny) => {
                    try {
                        initAudio();
                        if (!state || !state.fx || !state.fx.tk || !state.fx.tk.node) return;
                        state.fx.tk.node.frequency.value = tkNormToFreq(nx);
                        state.fx.tk.node.Q.value = tkNormToQ(ny);
                        syncTkHoldUiFromNode();
                        drawTkHoldGraph();
                    } catch (_) {}
                };

                const tkPadFromClient = (ev) => {
                    const rect = tkPadEl.getBoundingClientRect();
                    let nx = (ev.clientX - rect.left) / rect.width;
                    let ny = (ev.clientY - rect.top) / rect.height;
                    applyTkHoldNorm(nx, ny);
                };

                const openTkHoldUi = () => {
                    removeTkPreOpenMove();
                    tkHoldUiOpen = true;
                    tkHoldPanel.classList.remove('display-none');
                    tkHoldPanel.setAttribute('aria-hidden', 'false');
                    syncTkHoldUiFromNode();
                    drawTkHoldGraph();
                    requestAnimationFrame(() => {
                        positionTkHoldPanelAtCursor();
                        requestAnimationFrame(() => positionTkHoldPanelAtCursor());
                    });
                    // Same pointer that held the TK button: drag anywhere to hit-test the pad; release closes.
                    tkDocMove = (e) => {
                        if (!tkHoldUiOpen || tkPressPointerId == null || e.pointerId !== tkPressPointerId) return;
                        tkPadFromClient(e);
                    };
                    tkDocUp = (e) => {
                        if (tkPressPointerId == null || e.pointerId !== tkPressPointerId) return;
                        closeTkHoldUi();
                    };
                    document.addEventListener('pointermove', tkDocMove, true);
                    document.addEventListener('pointerup', tkDocUp, true);
                };

                const bindTkHoldForButton = (btn) => {
                    if (!btn) return;
                    btn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); }, sig);
                    btn.addEventListener('pointerdown', (e) => {
                        if (e.button !== 0) return;
                        tkSkipNextTkBtnUp = false;
                        tkPressPointerId = e.pointerId;
                        tkAnchorX = e.clientX;
                        tkAnchorY = e.clientY;
                        tkHoldUiOpen = false;
                        removeTkPreOpenMove();
                        tkPreOpenMove = (ev) => {
                            if (ev.pointerId !== tkPressPointerId) return;
                            tkAnchorX = ev.clientX;
                            tkAnchorY = ev.clientY;
                        };
                        document.addEventListener('pointermove', tkPreOpenMove, true);
                        tkHoldTimer = setTimeout(() => {
                            tkHoldTimer = null;
                            openTkHoldUi();
                        }, TK_HOLD_MS);
                    }, sig);
                    btn.addEventListener('pointerup', (e) => {
                        if (e.button !== 0) return;
                        if (tkSkipNextTkBtnUp) {
                            tkSkipNextTkBtnUp = false;
                            tkPressPointerId = null;
                            removeTkPreOpenMove();
                            return;
                        }
                        if (tkHoldTimer) {
                            clearTimeout(tkHoldTimer);
                            tkHoldTimer = null;
                            tkPressPointerId = null;
                            removeTkPreOpenMove();
                            if (!tkHoldUiOpen) djToggleTk();
                            return;
                        }
                        // Long-press: release is handled by document capture (tkDocUp) so we can drag first.
                        tkPressPointerId = null;
                    }, sig);
                    btn.addEventListener('pointercancel', () => {
                        if (tkHoldTimer) {
                            clearTimeout(tkHoldTimer);
                            tkHoldTimer = null;
                        }
                        tkPressPointerId = null;
                        removeTkPreOpenMove();
                        if (tkHoldUiOpen) closeTkHoldUi();
                    }, sig);
                };
                if (this.els.fxTk) this.els.fxTk.addEventListener('click', (e) => {
                    try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                    djToggleTextInDeckA();
                }, sig);
                if (this.els.fxTkB) this.els.fxTkB.addEventListener('click', (e) => {
                    try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                    djToggleTextInDeckB();
                }, sig);

                const djPrevA = root.querySelector('#dj-a-prev');
                const djNextA = root.querySelector('#dj-a-next');
                if (djPrevA) {
                    djPrevA.addEventListener('click', () => {
                        try {
                            if (typeof uiLocked !== 'undefined' && uiLocked) return;
                            if (typeof goPreviousStation === 'function') goPreviousStation();
                        } catch (_) {}
                    }, sig);
                }
                if (djNextA) {
                    djNextA.addEventListener('click', () => {
                        try {
                            if (typeof uiLocked !== 'undefined' && uiLocked) return;
                            if (state.deckSourceMode && state.deckSourceMode.a === 'local') {
                                playDeckATrackFromQueue();
                                return;
                            }
                            if (typeof pickRandomStation === 'function') pickRandomStation();
                        } catch (_) {}
                    }, sig);
                }

                const djAdvanceB = (delta) => {
                    try {
                        initAudio();
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        const n = stations.length;
                        if (!n) return;
                        let i = (typeof currentStationBIndex === 'number') ? currentStationBIndex : 0;
                        i = (((i + delta) % n) + n) % n;
                        currentStationBIndex = i;
                        if (mixStationB) mixStationB.value = String(i);
                        if (typeof playRadioB === 'function') playRadioB();
                    } catch (_) {}
                };
                const djRandomB = () => {
                    try {
                        initAudio();
                        if (typeof uiLocked !== 'undefined' && uiLocked) return;
                        if (state.deckSourceMode && state.deckSourceMode.b === 'local') {
                            playDeckBTrackFromQueue();
                            return;
                        }
                        if (!stations || stations.length === 0) return;
                        const cur = (typeof currentStationBIndex === 'number') ? currentStationBIndex : 0;
                        const eligible = getCycleEligibleStationIndexes(cur);
                        if (!eligible.length) return;
                        const idx = eligible[Math.floor(Math.random() * eligible.length)];
                        currentStationBIndex = idx;
                        if (mixStationB) mixStationB.value = String(idx);
                        if (typeof playRadioB === 'function') playRadioB();
                    } catch (_) {}
                };
                const djPrevB = root.querySelector('#dj-b-prev');
                const djNextB = root.querySelector('#dj-b-next');
                if (djPrevB) djPrevB.addEventListener('click', () => djAdvanceB(-1), sig);
                if (djNextB) djNextB.addEventListener('click', djRandomB, sig);

                try {
                    const karaA = root.querySelector('#dj-a-sfx-w1');
                    const karaB = root.querySelector('#dj-b-sfx-w1');
                    const onKaraokeClick = (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { window.__suppressNextClick = true; } catch (_) {}
                        this.toggleDeckBKaraokeEmbed();
                    };
                    if (karaA) karaA.addEventListener('click', onKaraokeClick, sig);
                    if (karaB) karaB.addEventListener('click', onKaraokeClick, sig);
                } catch (_) {}

                try {
                    const kbopBtnA = root.querySelector('#dj-a-sfx-w2');
                    const kbopBtnB = root.querySelector('#dj-b-sfx-w2');
                    const onKBopClick = (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { window.__suppressNextClick = true; } catch (_) {}
                        this.toggleDeckBKBopEmbed();
                    };
                    if (kbopBtnA) kbopBtnA.addEventListener('click', onKBopClick, sig);
                    if (kbopBtnB) kbopBtnB.addEventListener('click', onKBopClick, sig);
                } catch (_) {}

                try { this.bindDeckBeatMaps(root, sig); } catch (_) {}

                const switchVisualByName = (name) => {
                    try {
                        if (!Array.isArray(modes) || typeof loadMode !== 'function') return;
                        const idx = modes.findIndex((m) => m && m.name === name);
                        if (idx >= 0) loadMode(idx);
                    } catch (_) {}
                };
                const bindVisualSwitch = (id, name) => {
                    const el = root.querySelector('#' + id);
                    if (!el) return;
                    el.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch(_) {}
                        // Prevent global canvas click handler from treating this as a random-station click
                        try { window.__suppressNextClick = true; } catch(_) {}
                        try {
                            if (window.__randomClickTimer) {
                                clearTimeout(window.__randomClickTimer);
                                window.__randomClickTimer = null;
                            }
                        } catch(_) {}
                        // If matching Deck B visual is active, first click should clear Deck B visual mode.
                        const deckBModeFromDeckAName = (
                            name === 'Audio Bars' ? 'bars' :
                            (name === 'ProjectM v2' ? 'projectm' :
                            (name === 'Blank' ? 'blank' : null))
                        );
                        if (this.deckBVizMode === 'blank' && (name === 'Audio Bars' || name === 'ProjectM v2')) {
                            if (this.deckBQueueVisible) {
                                try { this.hideDeckBQueueView(); } catch (_) {}
                            }
                            if (this.deckBMediaPanelVisible) {
                                try { this.hideDeckBMediaView(); } catch (_) {}
                            }
                            this._deckAAudioBarsMainOnNextClick = (name === 'Audio Bars');
                            if (name === 'Audio Bars') this.startDeckBAudioBars();
                            else this.startDeckBProjectM();
                            this.syncDeckBVisualButtons();
                            try { this.updateStationTitles(); } catch (_) {}
                            return;
                        }
                        if (name === 'Blank') {
                            if (this.deckBQueueVisible) {
                                try { this.hideDeckBQueueView(); } catch (_) {}
                            }
                            if (this.deckBMediaPanelVisible) {
                                try { this.hideDeckBMediaView(); } catch (_) {}
                            }
                            if (this.isDeckBVisualModeActive()) {
                                if (this.deckBVizMode === 'blank') {
                                    this.tearDownDeckBViz();
                                    this.syncDeckBVisualButtons();
                                    try { this.updateStationTitles(); } catch (_) {}
                                    return;
                                }
                                this.startDeckBBlankVisual();
                                this.syncDeckBVisualButtons();
                                try { this.updateStationTitles(); } catch (_) {}
                                return;
                            }
                        }
                        if (name === 'Audio Bars' && this.deckBVizMode === 'projectm') {
                            this.startDeckBAudioBars();
                            this._deckAAudioBarsMainOnNextClick = true;
                            this.syncDeckBVisualButtons();
                            try { this.updateStationTitles(); } catch (_) {}
                            return;
                        }
                        if (name === 'ProjectM v2' && this.deckBVizMode === 'bars') {
                            this._deckAAudioBarsMainOnNextClick = false;
                            this.startDeckBProjectM();
                            this.syncDeckBVisualButtons();
                            try { this.updateStationTitles(); } catch (_) {}
                            return;
                        }
                        if (deckBModeFromDeckAName && this.deckBVizMode === deckBModeFromDeckAName) {
                            if (name === 'Audio Bars' && this._deckAAudioBarsMainOnNextClick) {
                                this.tearDownDeckBViz();
                                this.syncDeckBVisualButtons();
                                try { this.updateStationTitles(); } catch (_) {}
                                switchVisualByName(name);
                                return;
                            }
                            this.tearDownDeckBViz();
                            this.syncDeckBVisualButtons();
                            this.updateStationTitles();
                            return;
                        }
                        this._deckAAudioBarsMainOnNextClick = false;
                        switchVisualByName(name);
                    }, sig);
                };
                bindVisualSwitch('dj-vis-blank', 'Blank');
                bindVisualSwitch('dj-vis-bars', 'Audio Bars');
                bindVisualSwitch('dj-vis-pm2', 'ProjectM v2');
                bindVisualSwitch('dj-a-sfx-wav', 'Digital Radio');
                bindVisualSwitch('dj-b-sfx-wav', 'Digital Radio');

                const bindDeckBPreviewMode = (id, mode) => {
                    const el = root.querySelector('#' + id);
                    if (!el) return;
                    el.addEventListener('click', (e) => {
                        try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                        try { window.__suppressNextClick = true; } catch (_) {}
                        try {
                            if (window.__randomClickTimer) {
                                clearTimeout(window.__randomClickTimer);
                                window.__randomClickTimer = null;
                            }
                        } catch (_) {}
                        if (this.deckBQueueVisible) this.hideDeckBQueueView();
                        if (this.deckBMediaPanelVisible) this.hideDeckBMediaView();
                        // Toggle off if this visual is already active
                        if (this.deckBVizMode === mode) {
                            this.tearDownDeckBViz();
                            this.syncDeckBVisualButtons();
                            return;
                        }
                        if (mode === 'blank') this.startDeckBBlankVisual();
                        else if (mode === 'bars') this.startDeckBAudioBars();
                        else if (mode === 'projectm') this.startDeckBProjectM();
                        this.syncDeckBVisualButtons();
                    }, sig);
                };
                bindDeckBPreviewMode('dj-b-vis-blank', 'blank');
                bindDeckBPreviewMode('dj-b-vis-bars', 'bars');
                bindDeckBPreviewMode('dj-b-vis-pm2', 'projectm');

                if (this.els.playA) {
                    this.els.playA.addEventListener('click', async () => {
                        try {
                            initAudio();
                            const aMedia = getDeckAMediaForPlaybackState();
                            if (aMedia && aMedia.src && !aMedia.paused) {
                                aMedia.pause();
                            } else if (aMedia && aMedia.src && aMedia.paused) {
                                try { if (typeof releaseAutoMixDeferredLocal === 'function') releaseAutoMixDeferredLocal('a', 'play'); } catch (_) {}
                                await aMedia.play().catch(() => {
                                    if (deckFileQueues.a.length > 0) playDeckATrackFromQueue({ forceImmediate: true });
                                    else if (typeof playRadio === 'function') playRadio();
                                });
                            } else if (deckFileQueues.a.length > 0) {
                                playDeckATrackFromQueue({ forceImmediate: true });
                            } else if (typeof playRadio === 'function') {
                                playRadio();
                            }
                        } catch(_) {}
                        this.syncPlayLabels();
                    }, sig);
                }
                if (this.els.playB) {
                    this.els.playB.addEventListener('click', async () => {
                        try {
                            initAudio();
                            if (audioElB && audioElB.src && !audioElB.paused) {
                                audioElB.pause();
                            } else if (audioElB && audioElB.src && audioElB.paused) {
                                try { if (typeof releaseAutoMixDeferredLocal === 'function') releaseAutoMixDeferredLocal('b', 'play'); } catch (_) {}
                                await audioElB.play().catch(() => {
                                    if (deckFileQueues.b.length > 0) playDeckBTrackFromQueue({ forceImmediate: true });
                                    else if (typeof playRadioB === 'function') playRadioB();
                                });
                            } else if (deckFileQueues.b.length > 0) {
                                playDeckBTrackFromQueue({ forceImmediate: true });
                            } else if (typeof playRadioB === 'function') {
                                playRadioB();
                            }
                        } catch(_) {}
                        this.syncPlayLabels();
                    }, sig);
                }

                try {
                    ['play', 'pause', 'ended', 'stalled'].forEach((ev) => {
                        if (audioEl) {
                            audioEl.addEventListener(ev, () => {
                                try { this.syncPlayLabels(); } catch (_) {}
                                try { this.refreshDeckBVideoSource(); } catch (_) {}
                                try { this.updateStationTitles(); } catch (_) {}
                            }, sig);
                        }
                        if (audioElRadioAAlt) {
                            audioElRadioAAlt.addEventListener(ev, () => {
                                try { this.syncPlayLabels(); } catch (_) {}
                                try { this.refreshDeckBVideoSource(); } catch (_) {}
                                try { this.updateStationTitles(); } catch (_) {}
                            }, sig);
                        }
                        if (audioElB) {
                            audioElB.addEventListener(ev, () => {
                                try { this.syncPlayLabels(); } catch (_) {}
                                try { this.refreshDeckBVideoSource(); } catch (_) {}
                                try { this.updateStationTitles(); } catch (_) {}
                            }, sig);
                        }
                    });
                } catch(_) {}

                this.syncFxLightsFromState();
                this.syncPlayLabels();
                this.updateStationTitles();
                if (typeof applyCrossfade === 'function' && djCross) applyCrossfade(djCross.value);

                try { globalThis.updateModeSubStationLine?.(); } catch (_) {}

                this.syncDeckBVisualButtons();
                this.syncDeckMosaicScale();
                this.syncDeckVolumeSliderLengths();

                try {
                    window.__refreshDjQueueUi = () => { try { this.refreshQueueUi(); } catch (_) {} };
                } catch (_) {}

                try {
                    if (typeof wireMixPanelAutoButtonsToDjDeck === 'function') wireMixPanelAutoButtonsToDjDeck();
                } catch (_) {}

                this.animateFrame();
                try {
                    this.tickHeadDatetime();
                    if (this.headClockTimerId) clearInterval(this.headClockTimerId);
                    this.headClockTimerId = setInterval(() => {
                        try { this.tickHeadDatetime(); } catch (_) {}
                    }, 1000);
                } catch (_) {}
            }

            destroy() {
                try { if (this.headClockTimerId) clearInterval(this.headClockTimerId); } catch (_) {}
                this.headClockTimerId = null;
                try { cancelAnimationFrame(this.animId); } catch(_) {}
                this.animId = null;
                try { if (this.autoFadeRafId) cancelAnimationFrame(this.autoFadeRafId); } catch (_) {}
                this.autoFadeRafId = null;
                this.autoFadeTargetDeck = null;
                try { if (this.autoFadeHoldTimer) clearTimeout(this.autoFadeHoldTimer); } catch (_) {}
                this.autoFadeHoldTimer = null;
                try { if (this.autoMixTimerId) clearTimeout(this.autoMixTimerId); } catch (_) {}
                this.autoMixTimerId = null;
                try { if (this.autoMixPreloadTimerId) clearTimeout(this.autoMixPreloadTimerId); } catch (_) {}
                this.autoMixPreloadTimerId = null;
                this.autoMixNextTargetDeck = null;
                try { if (this.autoMixSessionLimitTimerId) clearTimeout(this.autoMixSessionLimitTimerId); } catch (_) {}
                this.autoMixSessionLimitTimerId = null;
                try { if (this.masterFadeOutRafId) cancelAnimationFrame(this.masterFadeOutRafId); } catch (_) {}
                this.masterFadeOutRafId = null;
                try { if (this.autoMixHoldTimer) clearTimeout(this.autoMixHoldTimer); } catch (_) {}
                this.autoMixHoldTimer = null;
                try { if (this.autoMixNextFadeIntervalId) clearInterval(this.autoMixNextFadeIntervalId); } catch (_) {}
                this.autoMixNextFadeIntervalId = null;
                try {
                    window.removeEventListener('resize', this.resizeHandler);
                    window.removeEventListener('orientationchange', this.resizeHandler);
                } catch(_) {}
                try {
                    if (this.deckBVizResizeObs) {
                        this.deckBVizResizeObs.disconnect();
                        this.deckBVizResizeObs = null;
                    }
                    if (this.deckVolResizeObs) {
                        this.deckVolResizeObs.disconnect();
                        this.deckVolResizeObs = null;
                    }
                } catch (_) {}
                try { this.tearDownDeckBeatMaps(); } catch (_) {}
                try { this.tearDownDeckBViz(); } catch (_) {}
                try {
                    const KNOB_IDS = ['knob-a-low', 'knob-a-mid', 'knob-a-high', 'knob-a-gain', 'knob-b-gain', 'knob-b-high', 'knob-b-mid', 'knob-b-low'];
                    if (window.djKnobMirrors) {
                        KNOB_IDS.forEach((k) => { try { delete window.djKnobMirrors[k]; } catch (_) {} });
                    }
                } catch (_) {}
                try { this.abortCtrl && this.abortCtrl.abort(); } catch(_) {}
                this.abortCtrl = null;
                try {
                    if (this._deckBVideoUiAbortCtrl) {
                        this._deckBVideoUiAbortCtrl.abort();
                        this._deckBVideoUiAbortCtrl = null;
                    }
                } catch (_) {}
                try { window.__refreshDjQueueUi = null; } catch (_) {}
                this.root = null;
                this.els = {};
                container.innerHTML = '';
            }
        }
