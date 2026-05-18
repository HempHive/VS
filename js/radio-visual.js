/* Extracted from app.js — radio-visual. Uses globals via globalThis (see app.js exposeVsGlobals). */
        class RadioVisualEngine {
            constructor() {
                this.name = 'Radio Visual';
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
            }

            _stopClick(ev) {
                try { ev.preventDefault(); ev.stopPropagation(); } catch (_) {}
                try { window.__suppressNextClick = true; } catch (_) {}
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

            _needlePercent() {
                if (!Array.isArray(stations) || stations.length <= 1) return 50;
                const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                    ? currentStationIndex : 0;
                return (idx / Math.max(1, stations.length - 1)) * 100;
            }

            _updateStationUi() {
                let name = '—';
                try {
                    const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0)
                        ? currentStationIndex : -1;
                    if (idx >= 0 && stations[idx]) name = stations[idx].name || name;
                    else if (stationBannerNameEl && stationBannerNameEl.textContent) {
                        name = stationBannerNameEl.textContent;
                    }
                } catch (_) {}
                if (this.els.stationAnalog) this.els.stationAnalog.textContent = name;
                if (this.els.stationDigital) this.els.stationDigital.textContent = name;
                const pct = this._needlePercent();
                if (this.els.needle) this.els.needle.style.left = `${pct}%`;
                if (this.els.tunerGlow) this.els.tunerGlow.style.left = `${pct}%`;
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

            _wirePointerKnob(knobEl, onValue) {
                if (!knobEl) return;
                let active = false;
                let startY = 0;
                let startVal = 0;
                const onDown = (ev) => {
                    this._stopClick(ev);
                    active = true;
                    startY = ev.clientY;
                    startVal = onValue.get();
                    try { knobEl.setPointerCapture(ev.pointerId); } catch (_) {}
                };
                const onMove = (ev) => {
                    if (!active) return;
                    onValue.set(Math.max(0, Math.min(1, startVal + (startY - ev.clientY) * 0.004)));
                };
                const onUp = (ev) => {
                    if (!active) return;
                    active = false;
                    try { knobEl.releasePointerCapture(ev.pointerId); } catch (_) {}
                };
                knobEl.addEventListener('pointerdown', onDown, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointermove', onMove, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointerup', onUp, { signal: this.abortCtrl.signal });
                knobEl.addEventListener('pointercancel', onUp, { signal: this.abortCtrl.signal });
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
                root.setAttribute('aria-label', 'Radio Visual');
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
                const stA = document.createElement('h1');
                stA.className = 'radio-visual-station-title';
                stA.id = 'radio-visual-station-a';
                stA.textContent = '—';
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
                const needle = document.createElement('div');
                needle.className = 'radio-visual-tuner-needle';
                needle.id = 'radio-visual-needle';
                const vuWrap = document.createElement('div');
                vuWrap.className = 'radio-visual-vu-wrap';
                const vuCanvas = document.createElement('canvas');
                vuCanvas.className = 'radio-visual-vu-canvas';
                vuCanvas.id = 'radio-visual-vu';
                vuWrap.appendChild(vuCanvas);
                tunerRail.appendChild(ticks);
                tunerRail.appendChild(glow);
                tunerRail.appendChild(needle);
                tunerShell.appendChild(tunerRail);
                tunerShell.appendChild(vuWrap);
                const knobs = document.createElement('div');
                knobs.className = 'radio-visual-knobs-row';
                const volKnob = document.createElement('div');
                volKnob.className = 'radio-visual-knob';
                volKnob.id = 'radio-visual-vol-knob';
                volKnob.setAttribute('role', 'slider');
                volKnob.setAttribute('aria-label', 'Volume');
                const tuneKnob = document.createElement('div');
                tuneKnob.className = 'radio-visual-knob';
                tuneKnob.id = 'radio-visual-tune-knob';
                tuneKnob.setAttribute('role', 'slider');
                tuneKnob.setAttribute('aria-label', 'Station tuning');
                const mkKnob = (label, knob) => {
                    const b = document.createElement('div');
                    b.className = 'radio-visual-knob-block';
                    const l = document.createElement('div');
                    l.className = 'radio-visual-knob-label';
                    l.textContent = label;
                    b.appendChild(l);
                    b.appendChild(knob);
                    return b;
                };
                knobs.appendChild(mkKnob('Volume', volKnob));
                knobs.appendChild(mkKnob('Tuning', tuneKnob));
                const analogBtns = document.createElement('div');
                analogBtns.className = 'radio-visual-btn-grid';
                analogBtns.id = 'radio-visual-analog-btns';
                stageA.appendChild(onAir);
                stageA.appendChild(stA);
                stageA.appendChild(tunerShell);
                stageA.appendChild(knobs);
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
                [['prev', 'Prev'], ['next', 'Next'], ['rand', 'Rand']].forEach(([act, lab]) => {
                    const b = document.createElement('button');
                    b.type = 'button';
                    b.className = 'radio-visual-btn';
                    b.dataset.rvAction = act;
                    b.textContent = lab;
                    stRow.appendChild(b);
                });
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
                    stationAnalog: stA,
                    stationDigital: stD,
                    ticks,
                    needle,
                    tunerGlow: glow,
                    tunerRail,
                    vuCanvas,
                    digitalEqCanvas: eqCanvas,
                    volKnob,
                    tuneKnob,
                    volDigital: volDig,
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
                this._updateStationUi();
                this._tickClock();

                btnA.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('analogue'); }, sig);
                btnD.addEventListener('click', (ev) => { this._stopClick(ev); this._setSkin('digital'); }, sig);
                volDig.addEventListener('input', () => this._applyVolume(volDig.value), sig);
                stRow.querySelectorAll('[data-rv-action]').forEach((b) => {
                    b.addEventListener('click', (ev) => {
                        this._stopClick(ev);
                        const a = b.dataset.rvAction;
                        if (a === 'prev') this._stationPrev();
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
                this._wirePointerKnob(tuneKnob, {
                    get: () => this._needlePercent() / 100,
                    set: (t) => {
                        if (!Array.isArray(stations) || !stations.length) return;
                        const idx = Math.round(Math.max(0, Math.min(1, t)) * (stations.length - 1));
                        if (typeof setStation === 'function') setStation(idx);
                    }
                });

                root.addEventListener('click', (ev) => ev.stopPropagation(), sig);
                root.addEventListener('pointerdown', (ev) => ev.stopPropagation(), sig);

                window.addEventListener('resize', this.resizeHandler, sig);
                this.onResize();
                this.clockTimerId = setInterval(() => { try { this._tickClock(); } catch (_) {} }, 1000);
                this.animateFrame();
                document.getElementById('mode-sub').innerText = 'Radio Station Visual';
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
