/* Extracted from app.js — audio-engine. Uses globals via globalThis (see app.js exposeVsGlobals). */
        // --- AUDIO ENGINE ---
        function makeReverbImpulseBuffer(audioCtx, durationSec, decay) {
            const d = Math.max(0.5, Math.min(4.5, Number(durationSec) || 2.2));
            const dc = Math.max(1.2, Math.min(5, Number(decay) || 2.45));
            const rate = audioCtx.sampleRate;
            const len = Math.floor(rate * d);
            const buf = audioCtx.createBuffer(2, len, rate);
            for (let c = 0; c < buf.numberOfChannels; c++) {
                const ch = buf.getChannelData(c);
                for (let i = 0; i < len; i++) {
                    ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, dc);
                }
            }
            // Peak-normalize so ConvolverNode wet path is clearly audible in parallel with dry
            let peak = 0;
            for (let c = 0; c < buf.numberOfChannels; c++) {
                const ch = buf.getChannelData(c);
                for (let i = 0; i < len; i++) {
                    const a = Math.abs(ch[i]);
                    if (a > peak) peak = a;
                }
            }
            if (peak > 1e-8) {
                const scale = 0.95 / peak;
                for (let c = 0; c < buf.numberOfChannels; c++) {
                    const ch = buf.getChannelData(c);
                    for (let i = 0; i < len; i++) ch[i] *= scale;
                }
            }
            return buf;
        }
        function initAudio() {
            if(!state.audioCtx) {
                state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                state.analyserNode = state.audioCtx.createAnalyser();
                // Per-stream analysers for scope/BPM
                state.analyserNodeA = state.audioCtx.createAnalyser();
                state.analyserNodeB = state.audioCtx.createAnalyser();
                [state.analyserNodeA, state.analyserNodeB].forEach(an => {
                    try {
                        an.fftSize = 1024;
                        an.smoothingTimeConstant = 0.8;
                    } catch(_) {}
                });
                state.analyserNode.fftSize = 2048; 
				state.gainNode = state.audioCtx.createGain();
				state.gainNode.gain.value = 1.0;

                // --- MIXER CHANNELS ---
                state.streamAGain = state.audioCtx.createGain();
                state.streamBGain = state.audioCtx.createGain();
                state.streamAGain.gain.value = 1.0; 
                state.streamBGain.gain.value = 0.0; 
                state.mixInput = state.audioCtx.createGain();

                // --- CHANNEL A EQ & TRIM ---
                state.trimA = state.audioCtx.createGain(); state.trimA.gain.value = eqState.a.gain;
                state.eqA = {
                    high: state.audioCtx.createBiquadFilter(),
                    mid: state.audioCtx.createBiquadFilter(),
                    low: state.audioCtx.createBiquadFilter()
                };
                // Configure A Filters
                state.eqA.high.type = 'highshelf'; state.eqA.high.frequency.value = 3000; state.eqA.high.gain.value = eqState.a.high;
                state.eqA.mid.type = 'peaking'; state.eqA.mid.frequency.value = 1000; state.eqA.mid.gain.value = eqState.a.mid;
                state.eqA.low.type = 'lowshelf'; state.eqA.low.frequency.value = 200; state.eqA.low.gain.value = eqState.a.low;
                
                // Chain A: High -> Mid -> Low -> Trim -> CrossfaderA
                state.eqA.high.connect(state.eqA.mid);
                state.eqA.mid.connect(state.eqA.low);
                state.eqA.low.connect(state.trimA);
                state.trimA.connect(state.streamAGain);
                // Deck A radio: two <audio> sources summed into EQ highshelf (crossfade = no dead air while next station buffers)
                state.gainRadioPrimaryPath = state.audioCtx.createGain();
                state.gainRadioPrimaryPath.gain.value = 1.0;
                state.gainRadioSecondaryPath = state.audioCtx.createGain();
                state.gainRadioSecondaryPath.gain.value = 0.0;
                // Deck B radio: dual <audio> summed into EQ highshelf (gapless station changes)
                state.gainRadioBPrimaryPath = state.audioCtx.createGain();
                state.gainRadioBPrimaryPath.gain.value = 1.0;
                state.gainRadioBSecondaryPath = state.audioCtx.createGain();
                state.gainRadioBSecondaryPath.gain.value = 0.0;
                // Tap A into analyser
                try { state.trimA.connect(state.analyserNodeA); } catch(_) {}
                // Beat worklet for A (lazy: only when enabled)
                // no-op here; created on demand by BPM A toggle

                // --- CHANNEL B EQ & TRIM ---
                state.trimB = state.audioCtx.createGain(); state.trimB.gain.value = eqState.b.gain;
                state.eqB = {
                    high: state.audioCtx.createBiquadFilter(),
                    mid: state.audioCtx.createBiquadFilter(),
                    low: state.audioCtx.createBiquadFilter()
                };
                // Configure B Filters
                state.eqB.high.type = 'highshelf'; state.eqB.high.frequency.value = 3000; state.eqB.high.gain.value = eqState.b.high;
                state.eqB.mid.type = 'peaking'; state.eqB.mid.frequency.value = 1000; state.eqB.mid.gain.value = eqState.b.mid;
                state.eqB.low.type = 'lowshelf'; state.eqB.low.frequency.value = 200; state.eqB.low.gain.value = eqState.b.low;

                // Chain B: High -> Mid -> Low -> Trim -> CrossfaderB
                state.eqB.high.connect(state.eqB.mid);
                state.eqB.mid.connect(state.eqB.low);
                state.eqB.low.connect(state.trimB);
                state.trimB.connect(state.streamBGain);
                // Tap B into analyser
                try { state.trimB.connect(state.analyserNodeB); } catch(_) {}
                // Beat worklet for B (lazy: only when enabled)
                // no-op here; created on demand by BPM B toggle

                // Connect to Main Mix
                state.streamAGain.connect(state.mixInput);
                state.streamBGain.connect(state.mixInput);
                
                // --- EFFECTS & MASTER ---
                state.fx = {
                    low: { node: state.audioCtx.createBiquadFilter(), on: false },
                    high: { node: state.audioCtx.createBiquadFilter(), on: false },
                    bass: { node: state.audioCtx.createBiquadFilter(), on: false },
                    treble: { node: state.audioCtx.createBiquadFilter(), on: false },
                    arp: { filter: state.audioCtx.createBiquadFilter(), lfo: state.audioCtx.createOscillator(), lfoGain: state.audioCtx.createGain(), on: false },
                    tk: { node: state.audioCtx.createBiquadFilter(), on: false },
                    echoDelay: { node: state.audioCtx.createDelay(5.0), on: false },
                    echoFeedback: { node: state.audioCtx.createGain(), on: false },
                    loopMusical: { delay: state.audioCtx.createDelay(15.0), feedback: state.audioCtx.createGain(), on: false },
                    distort: { node: state.audioCtx.createWaveShaper(), on: false }
                };
                // (Standard FX configuration...)
                state.fx.low.node.type = 'lowpass'; state.fx.low.node.frequency.value = 1800;
                state.fx.high.node.type = 'highpass'; state.fx.high.node.frequency.value = 200;
                state.fx.bass.node.type = 'peaking'; state.fx.bass.node.frequency.value = 100; state.fx.bass.node.Q.value = 1.0; state.fx.bass.node.gain.value = 6;
                state.fx.treble.node.type = 'peaking'; state.fx.treble.node.frequency.value = 4000; state.fx.treble.node.Q.value = 0.7; state.fx.treble.node.gain.value = 6;
                state.fx.arp.filter.type = 'bandpass'; state.fx.arp.filter.frequency.value = 800; state.fx.arp.filter.Q.value = 2.0;
                state.fx.arp.lfo.type = 'square'; state.fx.arp.lfo.frequency.value = 4.0; state.fx.arp.lfoGain.gain.value = 0; 
                state.fx.arp.lfo.connect(state.fx.arp.lfoGain);
                try { state.fx.arp.lfoGain.connect(state.fx.arp.filter.frequency); } catch(_) {}
                try { state.fx.arp.lfo.start(); } catch(_) {}
                state.fx.tk.node.type = 'bandpass'; state.fx.tk.node.frequency.value = 1200; state.fx.tk.node.Q.value = 4.0;
                state.fx.echoDelay.node.delayTime.value = 0.28; state.fx.echoFeedback.node.gain.value = 0.35;
                try {
                    state.fx.loopMusical.feedback.gain.value = 0.38;
                    state.fx.loopMusical.delay.delayTime.value = 0.001;
                    state.fx.loopMusical.delay.connect(state.fx.loopMusical.feedback);
                    state.fx.loopMusical.feedback.connect(state.fx.loopMusical.delay);
                } catch (_) {}
                state.djBeatFx = { tkHz: 1200, loopBars: 0, arpHz: 4, loopArm: false, reverbPct: 50, flangerPct: 50, noVocalPct: 100, cutRatePct: 50 };
                const curveLen = 44100; const curve = new Float32Array(curveLen); const amount = 120;
                for (let i=0;i<curveLen;i++){ const x = i/curveLen*2-1; curve[i] = ((3+amount)*x*20*Math.PI/180)/(Math.PI+amount*Math.abs(x)); }
                state.fx.distort.node.curve = curve; state.fx.distort.node.oversample = '4x';

                state.fx.noVocal = {
                    on: false,
                    f1: state.audioCtx.createBiquadFilter(),
                    f2: state.audioCtx.createBiquadFilter(),
                    f3: state.audioCtx.createBiquadFilter()
                };
                state.fx.noVocal.f1.type = 'peaking';
                state.fx.noVocal.f1.frequency.value = 450;
                state.fx.noVocal.f1.Q.value = 0.55;
                state.fx.noVocal.f1.gain.value = -14;
                state.fx.noVocal.f2.type = 'peaking';
                state.fx.noVocal.f2.frequency.value = 1400;
                state.fx.noVocal.f2.Q.value = 0.48;
                state.fx.noVocal.f2.gain.value = -17;
                state.fx.noVocal.f3.type = 'peaking';
                state.fx.noVocal.f3.frequency.value = 3200;
                state.fx.noVocal.f3.Q.value = 0.52;
                state.fx.noVocal.f3.gain.value = -12;
                state.fx.flanger = {
                    on: false,
                    delay: state.audioCtx.createDelay(0.08),
                    feedbackGain: state.audioCtx.createGain(),
                    dryTap: state.audioCtx.createGain(),
                    wetTap: state.audioCtx.createGain(),
                    outMix: state.audioCtx.createGain(),
                    lfo: state.audioCtx.createOscillator(),
                    lfoDepth: state.audioCtx.createGain()
                };
                state.fx.flanger.dryTap.gain.value = 0.65;
                state.fx.flanger.wetTap.gain.value = 0.65;
                state.fx.flanger.outMix.gain.value = 1;
                state.fx.flanger.delay.delayTime.value = 0.004;
                state.fx.flanger.feedbackGain.gain.value = 0;
                state.fx.flanger.lfo.type = 'sine';
                state.fx.flanger.lfo.frequency.value = 0.42;
                state.fx.flanger.lfoDepth.gain.value = 0;
                try {
                    state.fx.flanger.lfo.connect(state.fx.flanger.lfoDepth);
                    state.fx.flanger.lfoDepth.connect(state.fx.flanger.delay.delayTime);
                } catch (_) {}
                try { state.fx.flanger.lfo.start(); } catch (_) {}
                state.fx.reverb = {
                    on: false,
                    dryGain: state.audioCtx.createGain(),
                    wetGain: state.audioCtx.createGain(),
                    convolver: state.audioCtx.createConvolver(),
                    sum: state.audioCtx.createGain()
                };
                state.fx.reverb.dryGain.gain.value = 1;
                state.fx.reverb.wetGain.gain.value = 0;
                try {
                    state.fx.reverb.convolver.buffer = makeReverbImpulseBuffer(state.audioCtx);
                } catch (_) {}

                // Rhythmic CUT: square-wave amplitude gate (16th-note rate follows BPM when available)
                state.fx.rhythmCut = {
                    on: false,
                    gateGain: state.audioCtx.createGain(),
                    lfo: state.audioCtx.createOscillator(),
                    lfoScale: state.audioCtx.createGain(),
                    offset: state.audioCtx.createConstantSource(),
                    _running: false
                };
                state.fx.rhythmCut.gateGain.gain.value = 1;
                state.fx.rhythmCut.lfo.type = 'square';
                state.fx.rhythmCut.lfo.frequency.value = 8;
                state.fx.rhythmCut.lfoScale.gain.value = 0.5;
                state.fx.rhythmCut.offset.offset.value = 0.5;
                try {
                    state.fx.rhythmCut.lfo.connect(state.fx.rhythmCut.lfoScale);
                } catch (_) {}

                // Master Output
                try { state.mixInput.disconnect(); } catch(_) {}
                state.mixInput.connect(state.analyserNode);
                try { state.analyserNode.disconnect(); } catch(_) {}
                state.analyserNode.connect(state.gainNode);
                try { state.gainNode.disconnect(); } catch(_) {}
                state.gainNode.connect(state.audioCtx.destination);

				if (volumeSlider) {
					volumeSlider.value = String(0.5);
					volumeSlider.addEventListener('input', (e) => setVolume(volumeSlider.value));
					try { setVolume(0.5); } catch(e) {}
				}
                try { applyCrossfade(0); } catch(_) {}
                
                // --- BIND KNOBS ---
                // Bind A
                bindKnob('knob-a-gain', 0, 2, eqState.a.gain, eqState.a, 'gain', v => { if(state.trimA) state.trimA.gain.value = v; });
                bindKnob('knob-a-high', -20, 20, eqState.a.high, eqState.a, 'high', v => { if(state.eqA.high) state.eqA.high.gain.value = v; });
                bindKnob('knob-a-mid', -20, 20, eqState.a.mid, eqState.a, 'mid', v => { if(state.eqA.mid) state.eqA.mid.gain.value = v; });
                bindKnob('knob-a-low', -20, 20, eqState.a.low, eqState.a, 'low', v => { if(state.eqA.low) state.eqA.low.gain.value = v; });
                // Bind B
                bindKnob('knob-b-gain', 0, 2, eqState.b.gain, eqState.b, 'gain', v => { if(state.trimB) state.trimB.gain.value = v; });
                bindKnob('knob-b-high', -20, 20, eqState.b.high, eqState.b, 'high', v => { if(state.eqB.high) state.eqB.high.gain.value = v; });
                bindKnob('knob-b-mid', -20, 20, eqState.b.mid, eqState.b, 'mid', v => { if(state.eqB.mid) state.eqB.mid.gain.value = v; });
                bindKnob('knob-b-low', -20, 20, eqState.b.low, eqState.b, 'low', v => { if(state.eqB.low) state.eqB.low.gain.value = v; });
                try { rebuildEffectsChain(); } catch (_) {}
            }
            if(state.audioCtx.state === 'suspended') state.audioCtx.resume();
        }
        function applyDeckBNoVocalPeakingGains() {
            if (!state.fx || !state.fx.noVocal) return;
            const pct = (state.djBeatFx && typeof state.djBeatFx.noVocalPct === 'number') ? state.djBeatFx.noVocalPct : 100;
            const m = Math.max(0.15, Math.min(1.15, pct / 100));
            try {
                state.fx.noVocal.f1.gain.value = -14 * m;
                state.fx.noVocal.f2.gain.value = -17 * m;
                state.fx.noVocal.f3.gain.value = -12 * m;
            } catch (_) {}
        }
        function applyRhythmCutGateRate() {
            if (!state.fx || !state.fx.rhythmCut || !state.fx.rhythmCut.lfo) return;
            let bpm = 120;
            try {
                if (typeof bpmSmoothB === 'number' && bpmSmoothB > 40 && bpmSmoothB < 300) bpm = bpmSmoothB;
                else if (typeof bpmSmoothA === 'number' && bpmSmoothA > 40 && bpmSmoothA < 300) bpm = bpmSmoothA;
            } catch (_) {}
            let mult = 1;
            try {
                const pct = (state.djBeatFx && typeof state.djBeatFx.cutRatePct === 'number') ? state.djBeatFx.cutRatePct : 50;
                mult = 0.25 + (pct / 100) * 1.75;
            } catch (_) {}
            const hz = (bpm / 60) * 4 * mult;
            try {
                state.fx.rhythmCut.lfo.frequency.value = Math.max(0.5, Math.min(32, hz));
            } catch (_) {}
        }

        function setRhythmCutModulationActive(active) {
            const rc = state.fx && state.fx.rhythmCut;
            if (!rc || !rc.gateGain || !rc.lfoScale || !rc.offset) return;
            try { rc.lfoScale.disconnect(rc.gateGain.gain); } catch (_) {}
            try { rc.offset.disconnect(rc.gateGain.gain); } catch (_) {}
            if (active) {
                try { rc.gateGain.gain.value = 0; } catch (_) {}
                try { rc.lfoScale.connect(rc.gateGain.gain); } catch (_) {}
                try { rc.offset.connect(rc.gateGain.gain); } catch (_) {}
            } else {
                try { rc.gateGain.gain.value = 1; } catch (_) {}
            }
        }

        /** Butterchurn taps `state.analyserNode`; `rebuildEffectsChain` calls analyser.disconnect() and drops that tap. */
        function reconnectDjDecksDeckBProjectMIfActive() {
            try {
                const av = state && state.activeVisualizer;
                if (!av || av.name !== 'DJ Decks' || av.deckBVizMode !== 'projectm') return;
                const viz = av.deckBVizPmVisualizer;
                if (!viz || typeof viz.connectAudio !== 'function' || !state.analyserNode) return;
                viz.connectAudio(state.analyserNode);
            } catch (_) {}
        }

        function rebuildEffectsChain() {
            if (!state.audioCtx) return;
            // Disconnect everything from mixInput forward
            try { state.mixInput.disconnect(); } catch(_) {}
            try {
                if (state.fx && state.fx.rhythmCut && state.fx.rhythmCut.gateGain) {
                    try { state.fx.rhythmCut.gateGain.disconnect(); } catch (_) {}
                }
                if (state.fx && state.fx.noVocal) {
                    try { state.fx.noVocal.f1.disconnect(); } catch (_) {}
                    try { state.fx.noVocal.f2.disconnect(); } catch (_) {}
                    try { state.fx.noVocal.f3.disconnect(); } catch (_) {}
                }
                if (state.fx && state.fx.flanger) {
                    try { state.fx.flanger.delay.disconnect(); } catch (_) {}
                    try { state.fx.flanger.feedbackGain.disconnect(); } catch (_) {}
                    if (state.fx.flanger.dryTap) try { state.fx.flanger.dryTap.disconnect(); } catch (_) {}
                    if (state.fx.flanger.wetTap) try { state.fx.flanger.wetTap.disconnect(); } catch (_) {}
                    if (state.fx.flanger.outMix) try { state.fx.flanger.outMix.disconnect(); } catch (_) {}
                }
                if (state.fx && state.fx.reverb) {
                    try { state.fx.reverb.dryGain.disconnect(); } catch (_) {}
                    try { state.fx.reverb.wetGain.disconnect(); } catch (_) {}
                    try { state.fx.reverb.convolver.disconnect(); } catch (_) {}
                    try { state.fx.reverb.sum.disconnect(); } catch (_) {}
                }
            } catch (_) {}
            const enabled = [];
            if (state.fx.low.on) enabled.push(state.fx.low.node);
            if (state.fx.high.on) enabled.push(state.fx.high.node);
            if (state.fx.bass.on) enabled.push(state.fx.bass.node);
            if (state.fx.treble.on) enabled.push(state.fx.treble.node);
            if (state.fx.arp.on) enabled.push(state.fx.arp.filter);
            if (state.fx.tk.on) enabled.push(state.fx.tk.node);
            if (state.fx.echoDelay.on) {
                // echo network: mix -> delay -> feedback -> delay
                try { state.fx.echoDelay.node.disconnect(); } catch(_) {}
                try { state.fx.echoFeedback.node.disconnect(); } catch(_) {}
                state.fx.echoDelay.node.connect(state.fx.echoFeedback.node);
                state.fx.echoFeedback.node.connect(state.fx.echoDelay.node);
                enabled.push(state.fx.echoDelay.node);
            }
            if (state.fx.loopMusical && state.fx.loopMusical.on) {
                try { state.fx.loopMusical.delay.disconnect(); } catch (_) {}
                try { state.fx.loopMusical.feedback.disconnect(); } catch (_) {}
                state.fx.loopMusical.delay.connect(state.fx.loopMusical.feedback);
                state.fx.loopMusical.feedback.connect(state.fx.loopMusical.delay);
                enabled.push(state.fx.loopMusical.delay);
            }
            if (state.fx.distort.on) enabled.push(state.fx.distort.node);
            // Chain: mixInput -> ...enabled... -> analyser
            let last = state.mixInput;
            for (const node of enabled) {
                try { last.disconnect(); } catch(_) {}
                last.connect(node);
                last = node;
            }
            if (state.fx.noVocal && state.fx.noVocal.on) {
                try { last.disconnect(); } catch (_) {}
                last.connect(state.fx.noVocal.f1);
                state.fx.noVocal.f1.connect(state.fx.noVocal.f2);
                state.fx.noVocal.f2.connect(state.fx.noVocal.f3);
                last = state.fx.noVocal.f3;
                try { applyDeckBNoVocalPeakingGains(); } catch (_) {}
            }
            if (state.fx.flanger && state.fx.flanger.on && state.fx.flanger.dryTap && state.fx.flanger.wetTap && state.fx.flanger.outMix) {
                try { last.disconnect(); } catch (_) {}
                const fp = (state.djBeatFx && typeof state.djBeatFx.flangerPct === 'number') ? state.djBeatFx.flangerPct : 50;
                const ft = Math.max(0, Math.min(1, fp / 100));
                state.fx.flanger.feedbackGain.gain.value = 0.12 + 0.48 * ft;
                state.fx.flanger.lfoDepth.gain.value = 0.002 + 0.012 * ft;
                state.fx.flanger.dryTap.gain.value = 0.52 + 0.38 * (1 - ft * 0.35);
                state.fx.flanger.wetTap.gain.value = 0.32 + 0.58 * ft;
                try { state.fx.flanger.delay.disconnect(); } catch (_) {}
                try { state.fx.flanger.feedbackGain.disconnect(); } catch (_) {}
                try { state.fx.flanger.dryTap.disconnect(); } catch (_) {}
                try { state.fx.flanger.wetTap.disconnect(); } catch (_) {}
                try { state.fx.flanger.outMix.disconnect(); } catch (_) {}
                last.connect(state.fx.flanger.delay);
                last.connect(state.fx.flanger.dryTap);
                state.fx.flanger.dryTap.connect(state.fx.flanger.outMix);
                state.fx.flanger.delay.connect(state.fx.flanger.feedbackGain);
                state.fx.flanger.feedbackGain.connect(state.fx.flanger.delay);
                state.fx.flanger.delay.connect(state.fx.flanger.wetTap);
                state.fx.flanger.wetTap.connect(state.fx.flanger.outMix);
                last = state.fx.flanger.outMix;
            } else if (state.fx.flanger) {
                state.fx.flanger.feedbackGain.gain.value = 0;
                state.fx.flanger.lfoDepth.gain.value = 0;
            }
            if (state.fx.reverb && state.fx.reverb.on) {
                try { last.disconnect(); } catch (_) {}
                const rp = (state.djBeatFx && typeof state.djBeatFx.reverbPct === 'number') ? state.djBeatFx.reverbPct : 50;
                const rt = Math.max(0, Math.min(1, rp / 100));
                state.fx.reverb.dryGain.gain.value = 1 - 0.44 * rt;
                state.fx.reverb.wetGain.gain.value = 0.76 * rt;
                try { state.fx.reverb.dryGain.disconnect(); } catch (_) {}
                try { state.fx.reverb.wetGain.disconnect(); } catch (_) {}
                try { state.fx.reverb.convolver.disconnect(); } catch (_) {}
                last.connect(state.fx.reverb.dryGain);
                last.connect(state.fx.reverb.convolver);
                state.fx.reverb.convolver.connect(state.fx.reverb.wetGain);
                state.fx.reverb.dryGain.connect(state.fx.reverb.sum);
                state.fx.reverb.wetGain.connect(state.fx.reverb.sum);
                last = state.fx.reverb.sum;
            } else if (state.fx.reverb) {
                state.fx.reverb.dryGain.gain.value = 1;
                state.fx.reverb.wetGain.gain.value = 0;
            }
            setRhythmCutModulationActive(false);
            try { last.disconnect(); } catch(_) {}
            if (state.fx && state.fx.rhythmCut && state.fx.rhythmCut.on) {
                const rc = state.fx.rhythmCut;
                try {
                    if (!rc._running) {
                        rc.offset.start();
                        rc.lfo.start();
                        rc._running = true;
                    }
                } catch (_) {}
                applyRhythmCutGateRate();
                try { last.connect(rc.gateGain); } catch (_) {}
                try { rc.gateGain.connect(state.analyserNode); } catch (_) {}
                setRhythmCutModulationActive(true);
            } else {
                try { last.connect(state.analyserNode); } catch (_) {}
            }
            // Ensure analyser -> master gain -> dest remains
            try { state.analyserNode.disconnect(); } catch(_) {}
            state.analyserNode.connect(state.gainNode);
            try { state.gainNode.disconnect(); } catch(_) {}
            state.gainNode.connect(state.audioCtx.destination);
            // Reconnect visualizer if needed
            try {
                if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                    state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                }
            } catch(_) {}
            try { reconnectDjDecksDeckBProjectMIfActive(); } catch (_) {}
        }

        function syncDjMusicalLoopDelayTime() {
            if (!state.audioCtx || !state.fx || !state.fx.loopMusical || !state.djBeatFx) return;
            const bars = Math.round(Math.max(0, Math.min(8, state.djBeatFx.loopBars)));
            const bpm = (typeof bpmSmoothA === 'number' && bpmSmoothA > 40 && bpmSmoothA < 300)
                ? bpmSmoothA : ((typeof bpmSmoothB === 'number' && bpmSmoothB > 40 && bpmSmoothB < 300) ? bpmSmoothB : 120);
            const beatSec = 60 / bpm;
            const barSec = beatSec * 4;
            const dt = bars <= 0 ? 0.001 : Math.min(14.9, Math.max(0.03, bars * barSec));
            try {
                state.fx.loopMusical.delay.delayTime.value = dt;
            } catch (_) {}
        }

        function applyDjBeatFxTk(val) {
            if (!state.djBeatFx) return;
            state.djBeatFx.tkHz = val;
            try {
                if (state.fx && state.fx.tk && state.fx.tk.node) state.fx.tk.node.frequency.value = val;
            } catch (_) {}
        }

        function applyDjBeatFxLoop(val) {
            if (!state.djBeatFx || !state.fx || !state.fx.loopMusical) return;
            const bars = Math.round(Math.max(0, Math.min(8, val)));
            state.djBeatFx.loopBars = bars;
            if (bars <= 0) state.djBeatFx.loopArm = false;
            state.fx.loopMusical.on = !!(state.djBeatFx.loopArm && bars > 0);
            syncDjMusicalLoopDelayTime();
            rebuildEffectsChain();
        }

        function applyDjBeatFxArp(val) {
            if (!state.djBeatFx || !state.fx || !state.fx.arp) return;
            state.djBeatFx.arpHz = val;
            const hz = Math.max(0.25, Math.min(32, val < 0.35 ? 4 : val));
            try {
                state.fx.arp.lfo.frequency.value = hz;
                state.fx.arp.lfoGain.gain.value = state.fx.arp.on ? 600 : 0;
            } catch (_) {}
            rebuildEffectsChain();
            try {
                const fxArp = document.getElementById('fx-arp');
                if (fxArp) fxArp.classList.toggle('on', !!state.fx.arp.on);
            } catch (_) {}
        }

        function syncDjBeatFxKnobActiveDom(root) {
            try {
                if (!root) root = document.getElementById('dj-visual-root');
                if (!root || !state.fx) return;
                const tkOn = !!(state.fx.tk && state.fx.tk.on);
                const loopOn = !!(state.fx.loopMusical && state.fx.loopMusical.on);
                const arpOn = !!(state.fx.arp && state.fx.arp.on);
                const revOn = !!(state.fx.reverb && state.fx.reverb.on);
                const flOn = !!(state.fx.flanger && state.fx.flanger.on);
                const nvOn = !!(state.fx.noVocal && state.fx.noVocal.on);
                const rcOn = !!(state.fx.rhythmCut && state.fx.rhythmCut.on);
                ['a', 'b'].forEach((d) => {
                    const tk = root.querySelector('#dj-knob-' + d + '-tk');
                    const loop = root.querySelector('#dj-knob-' + d + '-loop');
                    const arp = root.querySelector('#dj-knob-' + d + '-arp');
                    if (tk) tk.classList.toggle('dj-beatfx-on', tkOn);
                    if (loop) loop.classList.toggle('dj-beatfx-on', loopOn);
                    if (arp) arp.classList.toggle('dj-beatfx-on', arpOn);
                });
                const kbRev = root.querySelector('#dj-knob-b-reverb');
                const kbFl = root.querySelector('#dj-knob-b-flanger');
                const kbCut = root.querySelector('#dj-knob-b-cut');
                if (kbRev) kbRev.classList.toggle('dj-beatfx-on', revOn);
                if (kbFl) kbFl.classList.toggle('dj-beatfx-on', flOn);
                if (kbCut) kbCut.classList.toggle('dj-beatfx-on', rcOn);
                try {
                    const fxNvBtn = document.getElementById('fx-novocal');
                    if (fxNvBtn) fxNvBtn.classList.toggle('on', nvOn);
                } catch (_) {}
            } catch (_) {}
        }

        function toggleRhythmCutFx() {
            try { initAudio(); } catch (_) {}
            if (!state.fx || !state.fx.rhythmCut) return;
            state.fx.rhythmCut.on = !state.fx.rhythmCut.on;
            rebuildEffectsChain();
            try {
                const b = document.getElementById('fx-cut');
                if (b) b.classList.toggle('on', !!state.fx.rhythmCut.on);
            } catch (_) {}
            try { syncDjBeatFxKnobActiveDom(); } catch (_) {}
        }

        function refreshDjBeatLoopKnobVisuals(root) {
            if (!state.djBeatFx) return;
            if (!root) root = document.getElementById('dj-visual-root');
            if (!root) return;
            const v = state.djBeatFx.loopBars;
            ['a', 'b'].forEach((d) => {
                const el = root.querySelector('#dj-knob-' + d + '-loop');
                if (!el) return;
                setKnobUi(el, 0, 8, v);
                try {
                    const vv = el.querySelector('.knob-value');
                    if (vv) vv.textContent = String(Math.round(v));
                } catch (_) {}
            });
        }

        function refreshDjBeatArpKnobVisuals(root) {
            if (!state.djBeatFx) return;
            if (!root) root = document.getElementById('dj-visual-root');
            if (!root) return;
            const v = state.djBeatFx.arpHz;
            ['a', 'b'].forEach((d) => {
                const el = root.querySelector('#dj-knob-' + d + '-arp');
                if (!el) return;
                setKnobUi(el, 0, 16, v);
            });
        }

        function toggleDjBeatFxTkFromKnob() {
            try { initAudio(); } catch (_) {}
            if (!state.fx || !state.fx.tk) return;
            state.fx.tk.on = !state.fx.tk.on;
            rebuildEffectsChain();
            try {
                const fxTk = document.getElementById('fx-tk');
                if (fxTk) fxTk.classList.toggle('on', state.fx.tk.on);
            } catch (_) {}
            syncDjBeatFxKnobActiveDom();
        }

        function toggleDjBeatFxLoopFromKnob() {
            try { initAudio(); } catch (_) {}
            if (!state.djBeatFx || !state.fx || !state.fx.loopMusical) return;
            state.djBeatFx.loopArm = !state.djBeatFx.loopArm;
            if (state.djBeatFx.loopArm && state.djBeatFx.loopBars <= 0) {
                state.djBeatFx.loopBars = 1;
            }
            applyDjBeatFxLoop(state.djBeatFx.loopBars);
            refreshDjBeatLoopKnobVisuals();
            syncDjBeatFxKnobActiveDom();
        }

        function toggleDjBeatFxArpFromKnob() {
            try { initAudio(); } catch (_) {}
            if (!state.fx || !state.fx.arp || !state.djBeatFx) return;
            state.fx.arp.on = !state.fx.arp.on;
            state.fx.arp.lfoGain.gain.value = state.fx.arp.on ? 600 : 0;
            if (state.fx.arp.on && state.djBeatFx.arpHz < 0.35) {
                state.djBeatFx.arpHz = 4;
            }
            applyDjBeatFxArp(state.djBeatFx.arpHz);
            refreshDjBeatArpKnobVisuals();
            syncDjBeatFxKnobActiveDom();
        }

        function toggleDeckBMixerFxFromKnob(kind) {
            try { initAudio(); } catch (_) {}
            const keyMap = { reverb: 'reverb', flanger: 'flanger', noVocal: 'noVocal' };
            const fxKey = keyMap[kind];
            if (!fxKey || !state.fx || !state.fx[fxKey]) return;
            state.fx[fxKey].on = !state.fx[fxKey].on;
            rebuildEffectsChain();
            try {
                if (fxKey === 'noVocal') {
                    const nvBtn = document.getElementById('fx-novocal');
                    if (nvBtn) nvBtn.classList.toggle('on', state.fx.noVocal.on);
                } else {
                    const btn = document.getElementById('fx-' + fxKey);
                    if (btn) btn.classList.toggle('on', state.fx[fxKey].on);
                }
            } catch (_) {}
            syncDjBeatFxKnobActiveDom();
        }

        function applyDjBeatFxDeckBReverbPct(val) {
            if (!state.djBeatFx) return;
            state.djBeatFx.reverbPct = Math.round(Math.max(0, Math.min(100, val)));
            rebuildEffectsChain();
        }
        function applyDjBeatFxDeckBFlangerPct(val) {
            if (!state.djBeatFx) return;
            state.djBeatFx.flangerPct = Math.round(Math.max(0, Math.min(100, val)));
            rebuildEffectsChain();
        }
        function applyDjBeatFxDeckBNoVocalPct(val) {
            if (!state.djBeatFx) return;
            state.djBeatFx.noVocalPct = Math.round(Math.max(0, Math.min(100, val)));
            if (state.fx && state.fx.noVocal && state.fx.noVocal.on) {
                try { applyDeckBNoVocalPeakingGains(); } catch (_) {}
            }
        }

        function applyDjBeatFxDeckBCutRatePct(val) {
            if (!state.djBeatFx) return;
            state.djBeatFx.cutRatePct = Math.round(Math.max(0, Math.min(100, val)));
            try { applyRhythmCutGateRate(); } catch (_) {}
        }

        function bindDeckBMixerFxKnob(root, suffix, min, max, key, applyFn, resetVal, loopIntegerUi, toggleKind) {
            const el = root.querySelector('#dj-knob-b-' + suffix);
            if (!el) return;
            if (!state.djBeatFx) state.djBeatFx = { tkHz: 1200, loopBars: 0, arpHz: 4, loopArm: false, reverbPct: 50, flangerPct: 50, noVocalPct: 100, cutRatePct: 50 };
            const TH = 6;
            let downY = 0;
            let startValAtDown = 0;
            let maxAbsDy = 0;

            const updateVisuals = (val) => {
                state.djBeatFx[key] = val;
                setKnobUi(el, min, max, val);
                if (loopIntegerUi) {
                    const t = String(Math.round(val));
                    try {
                        const vv = el.querySelector('.knob-value');
                        if (vv) vv.textContent = t;
                    } catch (_) {}
                }
            };

            updateVisuals(typeof state.djBeatFx[key] === 'number' ? state.djBeatFx[key] : resetVal);

            const clientYOf = (e) => {
                if (e.clientY != null) return e.clientY;
                if (e.touches && e.touches[0]) return e.touches[0].clientY;
                if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientY;
                return 0;
            };

            const onMove = (e) => {
                const y = clientYOf(e);
                maxAbsDy = Math.max(maxAbsDy, Math.abs(y - downY));
                if (maxAbsDy <= TH) return;
                const dy = downY - y;
                const range = max - min;
                const delta = (dy / 200) * range;
                let newVal = Math.max(min, Math.min(max, startValAtDown + delta));
                updateVisuals(newVal);
                applyFn(newVal);
            };

            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onMove);
                window.removeEventListener('mouseup', onUp);
                window.removeEventListener('touchend', onUp);
                window.removeEventListener('touchcancel', onUp);
                if (maxAbsDy <= TH && toggleKind) {
                    if (toggleKind === 'reverb') toggleDeckBMixerFxFromKnob('reverb');
                    else if (toggleKind === 'flanger') toggleDeckBMixerFxFromKnob('flanger');
                    else if (toggleKind === 'noVocal') toggleDeckBMixerFxFromKnob('noVocal');
                    else if (toggleKind === 'rhythmCut') toggleRhythmCutFx();
                }
                maxAbsDy = 0;
            };

            const onDown = (e) => {
                e.preventDefault();
                downY = clientYOf(e);
                startValAtDown = state.djBeatFx[key];
                maxAbsDy = 0;
                window.addEventListener('mousemove', onMove);
                window.addEventListener('touchmove', onMove, { passive: false });
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchend', onUp);
                window.addEventListener('touchcancel', onUp);
            };

            el.addEventListener('mousedown', onDown);
            el.addEventListener('touchstart', onDown, { passive: false });
            el.addEventListener('dblclick', (e) => {
                try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                updateVisuals(resetVal);
                applyFn(resetVal);
                syncDjBeatFxKnobActiveDom(root);
            });
        }

        function bindDjBeatFxKnobPair(root, suffix, min, max, key, applyFn, resetVal, loopIntegerUi, toggleKind) {
            const elA = root.querySelector('#dj-knob-a-' + suffix);
            const elB = root.querySelector('#dj-knob-b-' + suffix);
            if (!elA && !elB) return;
            if (!state.djBeatFx) state.djBeatFx = { tkHz: 1200, loopBars: 0, arpHz: 4, loopArm: false, reverbPct: 50, flangerPct: 50, noVocalPct: 100, cutRatePct: 50 };
            const TH = 6;
            let downY = 0;
            let startValAtDown = 0;
            let maxAbsDy = 0;

            const updateVisuals = (val) => {
                state.djBeatFx[key] = val;
                if (elA) setKnobUi(elA, min, max, val);
                if (elB) setKnobUi(elB, min, max, val);
                if (loopIntegerUi) {
                    const t = String(Math.round(val));
                    try {
                        if (elA) {
                            const va = elA.querySelector('.knob-value');
                            if (va) va.textContent = t;
                        }
                        if (elB) {
                            const vb = elB.querySelector('.knob-value');
                            if (vb) vb.textContent = t;
                        }
                    } catch (_) {}
                }
            };

            updateVisuals(typeof state.djBeatFx[key] === 'number' ? state.djBeatFx[key] : resetVal);

            const clientYOf = (e) => {
                if (e.clientY != null) return e.clientY;
                if (e.touches && e.touches[0]) return e.touches[0].clientY;
                if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientY;
                return 0;
            };

            const onMove = (e) => {
                const y = clientYOf(e);
                maxAbsDy = Math.max(maxAbsDy, Math.abs(y - downY));
                if (maxAbsDy <= TH) return;
                const dy = downY - y;
                const range = max - min;
                const delta = (dy / 200) * range;
                let newVal = Math.max(min, Math.min(max, startValAtDown + delta));
                updateVisuals(newVal);
                applyFn(newVal);
            };

            const onUp = () => {
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('touchmove', onMove);
                window.removeEventListener('mouseup', onUp);
                window.removeEventListener('touchend', onUp);
                window.removeEventListener('touchcancel', onUp);
                if (maxAbsDy <= TH && toggleKind) {
                    if (toggleKind === 'tk') toggleDjBeatFxTkFromKnob();
                    else if (toggleKind === 'loop') toggleDjBeatFxLoopFromKnob();
                    else if (toggleKind === 'arp') toggleDjBeatFxArpFromKnob();
                }
                maxAbsDy = 0;
            };

            const onDown = (e) => {
                e.preventDefault();
                downY = clientYOf(e);
                startValAtDown = state.djBeatFx[key];
                maxAbsDy = 0;
                window.addEventListener('mousemove', onMove);
                window.addEventListener('touchmove', onMove, { passive: false });
                window.addEventListener('mouseup', onUp);
                window.addEventListener('touchend', onUp);
                window.addEventListener('touchcancel', onUp);
            };

            const bindEl = (el) => {
                el.addEventListener('mousedown', onDown);
                el.addEventListener('touchstart', onDown, { passive: false });
                el.addEventListener('dblclick', (e) => {
                    try { e.preventDefault(); e.stopPropagation(); } catch (_) {}
                    updateVisuals(resetVal);
                    applyFn(resetVal);
                    syncDjBeatFxKnobActiveDom(root);
                });
            };
            if (elA) bindEl(elA);
            if (elB) bindEl(elB);
        }

        function wireDjBeatFxKnobs(root) {
            try { initAudio(); } catch (_) {}
            if (!state.djBeatFx) state.djBeatFx = { tkHz: 1200, loopBars: 0, arpHz: 4, loopArm: false, reverbPct: 50, flangerPct: 50, noVocalPct: 100, cutRatePct: 50 };
            if (typeof state.djBeatFx.loopArm !== 'boolean') state.djBeatFx.loopArm = false;
            if (typeof state.djBeatFx.reverbPct !== 'number') state.djBeatFx.reverbPct = 50;
            if (typeof state.djBeatFx.flangerPct !== 'number') state.djBeatFx.flangerPct = 50;
            if (typeof state.djBeatFx.noVocalPct !== 'number') state.djBeatFx.noVocalPct = 100;
            if (typeof state.djBeatFx.cutRatePct !== 'number') state.djBeatFx.cutRatePct = 50;
            bindDjBeatFxKnobPair(root, 'tk', 200, 8000, 'tkHz', applyDjBeatFxTk, 1200, false, 'tk');
            bindDjBeatFxKnobPair(root, 'loop', 0, 8, 'loopBars', applyDjBeatFxLoop, 0, true, 'loop');
            bindDjBeatFxKnobPair(root, 'arp', 0, 16, 'arpHz', applyDjBeatFxArp, 4, false, 'arp');
            bindDeckBMixerFxKnob(root, 'reverb', 0, 100, 'reverbPct', applyDjBeatFxDeckBReverbPct, 50, true, 'reverb');
            bindDeckBMixerFxKnob(root, 'flanger', 0, 100, 'flangerPct', applyDjBeatFxDeckBFlangerPct, 50, true, 'flanger');
            bindDeckBMixerFxKnob(root, 'cut', 0, 100, 'cutRatePct', applyDjBeatFxDeckBCutRatePct, 50, true, 'rhythmCut');
            try {
                applyDjBeatFxTk(state.djBeatFx.tkHz);
                applyDjBeatFxLoop(state.djBeatFx.loopBars);
                applyDjBeatFxArp(state.djBeatFx.arpHz);
                rebuildEffectsChain();
                syncDjBeatFxKnobActiveDom(root);
            } catch (_) {}
        }

        // Enable/disable start controls (radio/mic/file/url)
        function setStartControlsEnabled(enabled) {
            const btnRadio = document.getElementById('btn-radio');
            const btnMic = document.getElementById('btn-mic');
            const btnFile = document.getElementById('btn-file');
            const urlInput = document.getElementById('radio-url') || document.getElementById('station-url');
            [btnRadio, btnMic, btnFile, urlInput].forEach(el => {
                if(!el) return;
                if(enabled) {
                    el.removeAttribute('disabled');
                    el.setAttribute('tabindex', '0');
                } else {
                    el.setAttribute('disabled', 'true');
                    el.setAttribute('tabindex', '-1');
                }
            });
        }
		/** Pulsing coloured edge glow on #overlay — only after tap.gif is ready (see revealStartScreenEdgeFx). */
		function applyOverlayGlowFx() {
			const overlay = document.getElementById('overlay');
			if (!overlay) return;
			const glowHex = randomHexColor();
			overlay.style.setProperty('--glowColor', glowHex);
			overlay.classList.add('glow-on');
			overlayGlowCycleCount = 0;
			if (overlayGlowColorTimer) { clearInterval(overlayGlowColorTimer); overlayGlowColorTimer = null; }
			overlayGlowColorTimer = setInterval(() => {
				if (overlay.classList.contains('glow-on')) {
					overlay.style.setProperty('--glowColor', randomHexColor());
				}
			}, overlayGlowDurationMs * 2);
		}
		function applyOverlayLogoFx() {
			const logo = document.getElementById('logo-omni');
			if (logo) {
				const omniHex = randomHexColor();
				logo.style.setProperty('--omniColor', omniHex);
				// Hide the entire "OMNI>" group until pat.gif has decoded. The
				// reveal happens on the parent element so all letters, including
				// the pulsing ">" caret and the clipped pat.gif texture, share
				// the same 3s fade rather than each letter racing its own opacity.
				logo.classList.remove('pat-revealed');
				// retrigger animation
				logo.classList.remove('logo-animate');
				// force reflow
				void logo.offsetWidth;
				logo.classList.add('logo-animate');
				// ensure persistent glow pulse is active
				logo.classList.add('logo-glow');
			}
		}
		function applyOverlayGlowAndLogo() {
			try { applyOverlayLogoFx(); } catch (_) {}
			try { applyOverlayGlowFx(); } catch (_) {}
		}
		function showOverlay() {
			const overlay = document.getElementById('overlay');
			overlay.classList.remove('hidden');
			overlay.style.display = 'flex';
			try {
				if (typeof resetStartScreenReveal === 'function') resetStartScreenReveal();
			} catch (_) {}
			try { applyOverlayLogoFx(); } catch (_) {}
            // Restore tiled background, logo, and tap.gif border once all start GIFs decode
            try { revealStartScreenAfterAssets(); } catch (_) {
				try { fadeInPtaStartBg(); } catch (_) {}
			}
			// Hide quick radio button on start page
			if (typeof radioQuickBtn !== 'undefined' && radioQuickBtn) {
				radioQuickBtn.style.display = 'none';
			}
			// Hide volume slider on start overlay
			(() => { const vs = document.getElementById('volume-slider-container'); if (vs) vs.style.display = 'none'; })();
			// tap.gif frame + pulsing edge glow — both after tap.gif loads (see revealStartScreenEdgeFx)
			// Type the current status line on start screen, then shortcuts
			const shortcuts = [
					'F Fullscreen  •  C Next Visual  •  ,/. Visual',
					'V Play/Pause A  •  B Play/Pause B  •  N Next Station  •  L Lock',
					'T Text-In Panel  •  Y Text-In Auto  •  U Send Text',
					'P Radio Stations  •  R Avatar Settings  •  W Toggle Avatar  •  Arrows Move Avatar',
					'H QUEUE  •  J VIDEO  •  K KARAOKE (Deck B)',
					'+/− Size  •  Q/E Speed  •  Z/X Opacity  •  Space Auto-Fade  •  Esc Back'
				].join('\n');
			if (statusEl && statusEl.innerText && statusEl.innerText.trim().length > 0) {
				typeStatus(statusEl.innerText.trim(), () => {
					typeStatusTo('shortcuts-status', shortcuts, 30);
					try { layoutOverlayElements(); } catch(e) {}
					try { scheduleStartTextLoop(); } catch(e) {}
				});
			} else {
				typeStatusTo('shortcuts-status', shortcuts, 30);
				try { layoutOverlayElements(); } catch(e) {}
				try { scheduleStartTextLoop(); } catch(e) {}
			}
            setStartControlsEnabled(true);
        }
        function hideOverlay() {
            const overlay = document.getElementById('overlay');
            overlay.classList.add('hidden');
            overlay.style.display = 'none';
			overlay.classList.remove('glow-on');
			try {
				if (typeof resetStartScreenReveal === 'function') resetStartScreenReveal();
			} catch (_) {}
            try { cancelStartTextLoop(); } catch(e) {}
			if (overlayGlowColorTimer) { try { clearInterval(overlayGlowColorTimer); } catch(e) {} overlayGlowColorTimer = null; }
            setStartControlsEnabled(false);
        }
		// Apply initial glow/shimmer on first startup if overlay is visible
		(() => {
			const overlay = document.getElementById('overlay');
			if (overlay && !overlay.classList.contains('hidden') && overlay.style.display !== 'none') {

				try {
					const logo = document.getElementById('logo-omni');
					if (logo) logo.classList.remove('pat-revealed');
				} catch(e) {}

				try { applyOverlayLogoFx(); } catch(e) {}

				try { revealStartScreenAfterAssets(); } catch(e) {
					try { fadeInPtaStartBg(); } catch(_) {}
				}

				// Ensure radio quick button is hidden on the start screen
				try {
					const rq = document.getElementById('radio-quick');
					if (rq) rq.style.display = 'none';
				} catch(e) {}

                // Type initial status then shortcuts (if text present)
				const s = document.getElementById('loading-status');
			const shortcuts = [
				'F Fullscreen  •  C Next Visual  •  ,/. Visual',
				'V Play/Pause A  •  B Play/Pause B  •  N Next Station  •  L Lock',
				'T Text-In Panel  •  Y Text-In Auto  •  U Send Text',
				'P Radio Stations  •  R Avatar Settings  •  W Toggle Avatar  •  Arrows Move Avatar',
				'H QUEUE  •  J VIDEO  •  K KARAOKE (Deck B)',
				'+/− Size  •  Q/E Speed  •  Z/X Opacity  •  Space Auto-Fade  •  Esc Back'
			].join('\n');
				if (s && s.innerText && s.innerText.trim().length > 0) {
					typeStatus(s.innerText.trim(), () => { typeStatusTo('shortcuts-status', shortcuts, 30); try { layoutOverlayElements(); } catch(e) {} try { scheduleStartTextLoop(); } catch(e) {} });
				} else {
					typeStatusTo('shortcuts-status', shortcuts, 30); try { layoutOverlayElements(); } catch(e) {} try { scheduleStartTextLoop(); } catch(e) {}
				}
			}
			// pta.gif is preloaded in <head>; fade-in runs after decode (see fadeInPtaStartBg)
		})();

        /** Revoke blob: URL when replacing deck media element src */
        function revokeBlobSrc(mediaEl) {
            try {
                const u = (mediaEl && (mediaEl.currentSrc || mediaEl.src)) || '';
                if (typeof u === 'string' && u.startsWith('blob:')) URL.revokeObjectURL(u);
            } catch (_) {}
        }

        function isFirefoxBrowser() {
            try {
                return typeof InstallTrigger !== 'undefined' || /firefox/i.test(navigator.userAgent);
            } catch (_) {
                return false;
            }
        }

        /**
         * Chrome: createMediaElementSource disconnects speaker output.
         * Firefox: element + Web Audio both play unless we tap captureStream first, then mute speakers.
         * (Mute before captureStream silences the stream — do not do that.)
         */
        function createMediaSourceFromElement(ctx, media) {
            if (!ctx || !media) return null;
            if (isFirefoxBrowser()) {
                const captureFn = typeof media.captureStream === 'function'
                    ? media.captureStream.bind(media)
                    : (typeof media.mozCaptureStream === 'function' ? media.mozCaptureStream.bind(media) : null);
                if (captureFn && (media.readyState | 0) >= 2) {
                    try {
                        const stream = captureFn();
                        const tracks = stream && stream.getAudioTracks ? stream.getAudioTracks() : [];
                        if (stream && tracks.length > 0) {
                            const src = ctx.createMediaStreamSource(stream);
                            try { media.muted = true; } catch (_) {}
                            return src;
                        }
                    } catch (e) {
                        console.warn('Firefox captureStream routing failed:', e);
                    }
                }
            }
            try {
                media.muted = false;
                return ctx.createMediaElementSource(media);
            } catch (e) {
                console.warn('createMediaElementSource failed:', e);
                return null;
            }
        }

        /** Drop legacy MediaStreamSource nodes from an earlier Firefox experiment. */
        function clearStaleDeckMediaSource(stateKey) {
            try {
                const node = state[stateKey];
                if (!node) return;
                if (node.mediaElement) return;
                try { node.disconnect(); } catch (_) {}
                state[stateKey] = null;
            } catch (_) {}
        }

        /** Route Deck A/B HTMLAudioElement into EQ chain (same path as streaming radio). */
        function connectDeckMediaToEq(deck) {
            if (!state.audioCtx) return;
            const isA = deck === 'a';
            const media = isA ? audioEl : audioElB;
            const srcKey = isA ? 'radioElementSource' : 'radioElementSourceB';
            const eqHigh = isA ? state.eqA && state.eqA.high : state.eqB && state.eqB.high;
            const gainFb = isA ? state.streamAGain : state.streamBGain;
            if (!media) return;
            if (isFirefoxBrowser() && media.paused) {
                const onPlaying = () => {
                    try { media.removeEventListener('playing', onPlaying); } catch (_) {}
                    connectDeckMediaToEq(deck);
                };
                try { media.addEventListener('playing', onPlaying, { once: true }); } catch (_) {}
                return;
            }
            clearStaleDeckMediaSource(srcKey);
            if (
                isFirefoxBrowser() &&
                state[srcKey] &&
                state[srcKey].mediaElement &&
                (typeof media.captureStream === 'function' || typeof media.mozCaptureStream === 'function')
            ) {
                try { state[srcKey].disconnect(); } catch (_) {}
                state[srcKey] = null;
            }
            if (!state[srcKey]) {
                state[srcKey] = createMediaSourceFromElement(state.audioCtx, media);
                if (!state[srcKey]) return;
            }
            const srcNode = state[srcKey];
            if (isA) state.sourceNode = srcNode;
            else state.sourceNodeB = srcNode;
            try { srcNode.disconnect(); } catch (_) {}
            try {
                if (isA && state.gainRadioPrimaryPath && eqHigh) {
                    try { state.gainRadioPrimaryPath.disconnect(); } catch (_) {}
                    srcNode.connect(state.gainRadioPrimaryPath);
                    state.gainRadioPrimaryPath.connect(eqHigh);
                } else if (!isA && state.gainRadioBPrimaryPath && eqHigh) {
                    try { state.gainRadioBPrimaryPath.disconnect(); } catch (_) {}
                    srcNode.connect(state.gainRadioBPrimaryPath);
                    state.gainRadioBPrimaryPath.connect(eqHigh);
                } else if (eqHigh) srcNode.connect(eqHigh);
                else if (gainFb) srcNode.connect(gainFb);
            } catch (_) {}
            rebuildEffectsChain();
            try { applyCrossfade(mixCross ? mixCross.value : 0); } catch (_) {}
        }

        /** Folder picks often omit MIME — infer from extension. */
        function inferLocalMediaKind(file) {
            if (!file) return null;
            const mime = (file.type || '').toLowerCase();
            if (mime.startsWith('audio')) return 'audio';
            if (mime.startsWith('video')) return 'video';
            const n = (file.name || '').toLowerCase();
            if (/\.(mp3|aac|m4a|m4b|wav|flac|ogg|opus|oga|weba|aiff|aif|wma)(\?|$)/i.test(n)) return 'audio';
            if (/\.(mp4|webm|mkv|mov|m4v|ogv|avi|mpeg|mpg|wmv|3gp|flv)(\?|$)/i.test(n)) return 'video';
            return null;
        }
        function sortDeckLocalFileList(files) {
            const list = Array.from(files || []).filter(Boolean);
            list.sort((a, b) => {
                const pa = String((a && (a.webkitRelativePath || a.name)) || '').toLowerCase();
                const pb = String((b && (b.webkitRelativePath || b.name)) || '').toLowerCase();
                return pa.localeCompare(pb, undefined, { numeric: true, sensitivity: 'base' });
            });
            return list;
        }
        async function readDirectoryEntryFiles(dirEntry) {
            const out = [];
            if (!dirEntry || !dirEntry.isDirectory) return out;
            const reader = dirEntry.createReader();
            const entries = [];
            let batch;
            do {
                batch = await new Promise((resolve, reject) => {
                    try { reader.readEntries(resolve, reject); } catch (e) { reject(e); }
                });
                if (batch && batch.length) entries.push(...batch);
            } while (batch && batch.length > 0);
            for (let i = 0; i < entries.length; i++) {
                const ent = entries[i];
                try {
                    if (ent.isFile) {
                        const file = await new Promise((resolve, reject) => {
                            try { ent.file(resolve, reject); } catch (e) { reject(e); }
                        });
                        if (file) out.push(file);
                    } else if (ent.isDirectory) {
                        const nested = await readDirectoryEntryFiles(ent);
                        if (nested.length) out.push(...nested);
                    }
                } catch (_) {}
            }
            return out;
        }
        /** Expand dropped folders (and plain files) into a flat File list. */
        async function collectMediaFilesFromDataTransfer(dt) {
            if (!dt) return [];
            const out = [];
            const items = dt.items ? Array.from(dt.items) : [];
            if (items.length) {
                for (let i = 0; i < items.length; i++) {
                    const item = items[i];
                    if (!item || item.kind !== 'file') continue;
                    let entry = null;
                    try {
                        if (typeof item.webkitGetAsEntry === 'function') entry = item.webkitGetAsEntry();
                        else if (typeof item.getAsEntry === 'function') entry = item.getAsEntry();
                    } catch (_) {}
                    if (entry) {
                        try {
                            if (entry.isFile) {
                                const file = await new Promise((resolve, reject) => {
                                    try { entry.file(resolve, reject); } catch (e) { reject(e); }
                                });
                                if (file) out.push(file);
                            } else if (entry.isDirectory) {
                                const nested = await readDirectoryEntryFiles(entry);
                                if (nested.length) out.push(...nested);
                            }
                        } catch (_) {}
                    } else {
                        try {
                            const f = item.getAsFile();
                            if (f) out.push(f);
                        } catch (_) {}
                    }
                }
            }
            if (!out.length && dt.files && dt.files.length) {
                return sortDeckLocalFileList(dt.files);
            }
            return sortDeckLocalFileList(out);
        }
        /**
         * Pick a folder via File System Access API (no bulk “upload” prompt).
         * Returns File[] on success, [] if cancelled, or null if caller should use <input webkitdirectory>.
         */
        async function pickDeckLocalFolderFiles() {
            try {
                if (typeof window.showDirectoryPicker !== 'function') return null;
                const dir = await window.showDirectoryPicker({ mode: 'read' });
                const files = [];
                async function walk(handle) {
                    for await (const entry of handle.values()) {
                        try {
                            if (entry.kind === 'file') {
                                const f = await entry.getFile();
                                if (f) files.push(f);
                            } else if (entry.kind === 'directory') {
                                await walk(entry);
                            }
                        } catch (_) {}
                    }
                }
                await walk(dir);
                return sortDeckLocalFileList(files);
            } catch (e) {
                if (e && e.name === 'AbortError') return [];
                return null;
            }
        }
        function getDeckLocalPlaybackMedia(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            try {
                if (!state || !state.deckSourceMode || state.deckSourceMode[dk] !== 'local') return null;
                if (dk === 'a' && typeof getDeckAMediaForPlaybackState === 'function') {
                    return getDeckAMediaForPlaybackState();
                }
            } catch (_) {}
            return dk === 'b' ? audioElB : audioEl;
        }

        function deckLocalPlaybackInProgress(deckKey) {
            const media = getDeckLocalPlaybackMedia(deckKey);
            try {
                return !!(media && media.src && media.src !== 'about:blank' && !media.paused && !media.ended);
            } catch (_) {
                return false;
            }
        }

        function shouldAutoplayDeckLocalQueue(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            try {
                if (deckLocalPlaybackInProgress(dk)) return false;
                if (otherDeckIsAudiblyPlaying(dk) && !deckWinsCrossfade(dk)) return false;
            } catch (_) {}
            return true;
        }
        function addDeckLocalFilesToDeck(deckKey, files, opts) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const sorted = sortDeckLocalFileList(files);
            if (!sorted.length) return;
            initAudio();
            const forceImmediate = !!(opts && opts.forceImmediate);
            if (forceImmediate) {
                const items = buildDeckLocalItemsFromFiles(dk, sorted);
                if (!items.length) return;
                prependDeckLocalItems(dk, items);
                const playOpts = { forceImmediate: true };
                if (opts && opts.preserveCrossfade) playOpts.preserveCrossfade = true;
                if (dk === 'b') playDeckBTrackFromQueue(playOpts);
                else playDeckATrackFromQueue(playOpts);
                return;
            }
            enqueueDeckLocalFiles(dk, sorted);
            if (shouldAutoplayDeckLocalQueue(dk)) {
                if (dk === 'b') playDeckBTrackFromQueue(opts);
                else playDeckATrackFromQueue(opts);
            }
        }
        function isVideoFileForMediaQueue(file) {
            const k = inferLocalMediaKind(file);
            return k === 'video';
        }
        /** Push picked video files into VIDEO QUEUE (newest-first display order — iterate newest last). */
        function addVideoFilesToMediaQueue(fileList) {
            const arr = Array.from(fileList || []).filter((f) => f && isVideoFileForMediaQueue(f));
            for (let i = arr.length - 1; i >= 0; i--) {
                const f = arr[i];
                let blobUrl = '';
                try { blobUrl = URL.createObjectURL(f); } catch (_) {}
                if (!blobUrl) continue;
                registerDeckVideoFeed('b', blobUrl, f.name || 'Local video', true);
            }
        }
        function enqueueDeckLocalFiles(deckKey, files) {
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            const deferVideoFeed = deckLocalPlaybackInProgress(deckKey);
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const kind = inferLocalMediaKind(file);
                if (!file || !kind) continue;
                const isVideo = kind === 'video';
                const url = URL.createObjectURL(file);
                q.push({ name: file.name || (isVideo ? 'Video track' : 'Audio track'), url, isVideo });
                if (isVideo && !deferVideoFeed) {
                    registerDeckVideoFeed(deckKey, url, file.name || 'Video track', true);
                }
            }
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        /** Next Deck A/B file-picker result from jog double-click prepends and starts playback immediately. */
        let __djLocalPickImmediateDeck = null;
        function scheduleClearDjLocalPickImmediate(deckKey) {
            const onFocus = () => {
                window.removeEventListener('focus', onFocus);
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        if (__djLocalPickImmediateDeck === deckKey) __djLocalPickImmediateDeck = null;
                    });
                });
            };
            try { window.addEventListener('focus', onFocus, { capture: false, once: true }); } catch (_) {}
        }
        function buildDeckLocalItemsFromFiles(deckKey, files) {
            const items = [];
            const list = Array.isArray(files) ? files : Array.from(files || []);
            for (let i = 0; i < list.length; i++) {
                const file = list[i];
                const kind = inferLocalMediaKind(file);
                if (!file || !kind) continue;
                const isVideo = kind === 'video';
                let url = '';
                try { url = URL.createObjectURL(file); } catch (_) { continue; }
                if (!url) continue;
                items.push({
                    name: file.name || (isVideo ? 'Video track' : 'Audio track'),
                    url,
                    isVideo
                });
            }
            return items;
        }
        function prependDeckLocalItems(deckKey, items) {
            if (!items || !items.length) return;
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            for (let i = items.length - 1; i >= 0; i--) q.unshift(items[i]);
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }
        function ingestLocalFilesToDeckAndPlay(deckKey, files, opts) {
            addDeckLocalFilesToDeck(deckKey, files, { forceImmediate: true, ...(opts || {}) });
        }

        /** Read crossfader position (Digital Radio dash, DJ, or Mix panel). */
        function readCrossfadePosition() {
            try {
                const dig = document.getElementById('radio-visual-cross-digital');
                const mc = document.getElementById('mix-crossfader');
                const dj = document.getElementById('dj-crossfader');
                const raw = (dig && dig.value) || (mc && mc.value) || (dj && dj.value) || 0;
                return Math.max(0, Math.min(1, Number(raw) || 0));
            } catch (_) {
                return 0;
            }
        }

        /** True when the crossfader position gives this deck audible gain (A: x < 0.5, B: x >= 0.5). */
        function deckWinsCrossfade(deckKey) {
            const x = readCrossfadePosition();
            return deckKey === 'b' ? x >= 0.5 : x < 0.5;
        }

        function otherDeckIsAudiblyPlaying(deckKey) {
            const other = deckKey === 'b' ? 'a' : 'b';
            const otherMedia = other === 'b' ? audioElB : audioEl;
            try {
                return !!(otherMedia && otherMedia.src && otherMedia.src !== 'about:blank'
                    && !otherMedia.paused && !otherMedia.ended);
            } catch (_) {
                return false;
            }
        }

        /** When starting local playback, ensure the target deck is not fully muted by the crossfader. */
        function ensureLocalDeckCrossfadeAudible(deckKey, opts) {
            if (opts && opts.preserveCrossfade) return;
            const force = !!(opts && opts.forceImmediate);
            if (!force && !deckWinsCrossfade(deckKey)) return;
            const isB = deckKey === 'b';
            const x = readCrossfadePosition();
            const MIN = 0.5;
            let target = x;
            if (isB && x < MIN) target = MIN;
            else if (!isB && x > 1 - MIN) target = 1 - MIN;
            if (target === x) return;
            try {
                if (typeof applyCrossfade === 'function') applyCrossfade(target);
            } catch (_) {}
            try {
                const dig = document.getElementById('radio-visual-cross-digital');
                if (dig) dig.value = String(target);
            } catch (_) {}
            try {
                const av = state && state.activeVisualizer;
                if (av && typeof av._setCrossfadeX === 'function') av._setCrossfadeX(target);
            } catch (_) {}
        }

        function prepareDeckBLocalPlayback() {
            try { abortRadioBHandoff(); } catch (_) {}
            try { resetRadioBDualStreamHandoff(); } catch (_) {}
        }
        function enqueueDeckUrl(deckKey, url, label) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean) return false;
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            q.push({ name: String(label || deriveNameFromUrl(clean) || 'URL source'), url: clean });
            const isVid = isLikelyVideoUrl(clean) || isVideoQueueEligibleUrl(clean);
            if (isVid && !deckLocalPlaybackInProgress(deckKey)) {
                registerDeckVideoFeed(deckKey, clean, label);
            }
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            return true;
        }
        function promptAddUrlForDeck(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            const raw = prompt(dk === 'b' ? 'Paste URL for Deck B (music/video). Radio streams go to Station cycle.' : 'Paste URL for Deck A (music/video). Radio streams go to Station cycle.', '');
            const clean = sanitizeUrlForAudio(raw);
            if (!clean) return;
            if (isLikelyRadioStreamUrl(clean)) {
                addUserRadioStation(clean, deriveNameFromUrl(clean));
                return;
            }
            enqueueDeckUrl(dk, clean, deriveNameFromUrl(clean));
        }
        function playDeckUrlNow(deckKey, url, label) {
            const clean = sanitizeUrlForAudio(url);
            if (!clean) return false;
            const media = deckKey === 'b' ? audioElB : audioEl;
            if (!media) return false;
            initAudio();
            try {
                if (deckKey === 'b') {
                    prepareDeckBLocalPlayback();
                    state.deckSourceMode.b = 'local';
                    try { state.deckLocalDisplayName.b = label ? String(label) : ''; } catch (_) {}
                } else {
                    state.deckSourceMode.a = 'local';
                    try { state.deckLocalDisplayName.a = label ? String(label) : ''; } catch (_) {}
                    resetRadioADualStreamHandoff();
                }
                revokeBlobSrc(media);
                try { media.pause(); } catch (_) {}
                media.crossOrigin = 'anonymous';
                media.src = clean;
                registerDeckVideoFeed(deckKey, clean, label);
                const deferForAutoMix = shouldDeferLocalPlayForAutoMix(deckKey);
                if (deferForAutoMix) {
                    try { media.currentTime = 0; } catch (_) {}
                    try { media.pause(); } catch (_) {}
                    try { connectDeckMediaToEq(deckKey === 'b' ? 'b' : 'a'); } catch (_) {}
                    markAutoMixDeferredLocal(deckKey === 'b' ? 'b' : 'a', true);
                    try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                    return true;
                }
                ensureLocalDeckCrossfadeAudible(deckKey === 'b' ? 'b' : 'a', opts);
                media.play().then(() => {
                    connectDeckMediaToEq(deckKey === 'b' ? 'b' : 'a');
                    try {
                        if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                            state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                        }
                    } catch (_) {}
                    if (!state.isPlaying) startGame();
                    if (deckKey === 'b') {
                        try { if (typeof updateMixBStatus === 'function') updateMixBStatus(); } catch (_) {}
                    } else {
                        try { showStationBanner(String(label || deriveNameFromUrl(clean) || 'URL source')); } catch (_) {}
                    }
                }).catch(() => {});
                return true;
            } catch (_) { return false; }
        }

        function removeQueuedTrack(deckKey, index) {
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            if (index < 0 || index >= q.length) return;
            const item = q.splice(index, 1)[0];
            try {
                if (item && item.url && String(item.url).startsWith('blob:')) URL.revokeObjectURL(item.url);
            } catch (_) {}
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function clearDeckFileQueue(deckKey) {
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            while (q.length) {
                const item = q.pop();
                try {
                    if (item && item.url && String(item.url).startsWith('blob:')) URL.revokeObjectURL(item.url);
                } catch (_) {}
            }
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function moveQueuedTrackToOtherDeck(deckKey, index) {
            const src = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            const dstKey = deckKey === 'b' ? 'a' : 'b';
            const dst = dstKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            if (index < 0 || index >= src.length) return;
            const item = src.splice(index, 1)[0];
            if (!item || !item.url) return;
            dst.push({
                name: item.name || 'Track',
                url: item.url,
                isVideo: !!item.isVideo
            });
            if (item.isVideo) {
                try { registerDeckVideoFeed(dstKey, item.url, item.name, true); } catch (_) {}
            }
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function playQueuedTrackNow(deckKey, index) {
            initAudio();
            const q = deckKey === 'b' ? deckFileQueues.b : deckFileQueues.a;
            if (index < 0 || index >= q.length) return;
            const item = q.splice(index, 1)[0];
            if (!item || !item.url) return;
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            try {
                if (deckKey === 'b') {
                    if (!item.isVideo) {
                        try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('b'); } catch (_) {}
                    }
                    state.deckSourceMode.b = 'local';
                    try { state.deckLocalDisplayName.b = item.name ? String(item.name) : ''; } catch (_) {}
                    revokeBlobSrc(audioElB);
                    try { audioElB.pause(); } catch (_) {}
                    audioElB.src = item.url;
                    if (item.isVideo) registerDeckVideoFeed('b', item.url, item.name, true);
                    audioElB.play().then(() => {
                        connectDeckMediaToEq('b');
                        try {
                            if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                                state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                            }
                        } catch (_) {}
                        if (!state.isPlaying) startGame();
                        try { if (typeof updateMixBStatus === 'function') updateMixBStatus(); } catch (_) {}
                    }).catch(() => {});
                } else {
                    if (!item.isVideo) {
                        try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('a'); } catch (_) {}
                    }
                    state.deckSourceMode.a = 'local';
                    try { state.deckLocalDisplayName.a = item.name ? String(item.name) : ''; } catch (_) {}
                    resetRadioADualStreamHandoff();
                    revokeBlobSrc(audioEl);
                    try { audioEl.pause(); } catch (_) {}
                    audioEl.src = item.url;
                    if (item.isVideo) registerDeckVideoFeed('a', item.url, item.name, true);
                    audioEl.play().then(() => {
                        connectDeckMediaToEq('a');
                        try {
                            if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                                state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                            }
                        } catch (_) {}
                        if (!state.isPlaying) startGame();
                    }).catch(() => {});
                }
            } catch (_) {}
        }

        function resetDeckFileQueuesAndRevoke() {
            try { if (typeof releaseDeckVideoFeed === 'function') { releaseDeckVideoFeed('a'); releaseDeckVideoFeed('b'); } } catch (_) {}
            try { resetRadioADualStreamHandoff(); } catch (_) {}
            try { resetRadioBDualStreamHandoff(); } catch (_) {}
            revokeBlobSrc(audioEl);
            revokeBlobSrc(audioElB);
            ['a', 'b'].forEach((k) => {
                const q = deckFileQueues[k];
                while (q.length) {
                    const item = q.pop();
                    try {
                        if (item && item.url && item.url.startsWith('blob:')) URL.revokeObjectURL(item.url);
                    } catch (_) {}
                }
            });
            try {
                state.deckSourceMode.a = 'radio';
                state.deckSourceMode.b = 'radio';
                state.deckLocalDisplayName.a = '';
                state.deckLocalDisplayName.b = '';
            } catch (_) {}
        }

        function resumeRandomRadioForDeck(deckKey) {
            const dk = deckKey === 'b' ? 'b' : 'a';
            initAudio();
            try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed(dk); } catch (_) {}
            if (dk === 'b') {
                try { revokeBlobSrc(audioElB); } catch (_) {}
                state.deckSourceMode.b = 'radio';
                try { state.deckLocalDisplayName.b = ''; } catch (_) {}
                try {
                    audioElB.removeAttribute('src');
                    audioElB.load();
                } catch (_) {}
                try {
                    if (typeof pickRandomStationB === 'function') pickRandomStationB();
                    else if (typeof playRadioB === 'function') playRadioB();
                } catch (_) {}
                try { if (typeof refreshActiveDeckVideoDisplays === 'function') refreshActiveDeckVideoDisplays(); } catch (_) {}
                return;
            }
            try { abortRadioAHandoff(); } catch (_) {}
            try { resetRadioADualStreamHandoff(); } catch (_) {}
            try { revokeBlobSrc(audioEl); } catch (_) {}
            state.deckSourceMode.a = 'radio';
            try { state.deckLocalDisplayName.a = ''; } catch (_) {}
            try {
                audioEl.removeAttribute('src');
                audioEl.load();
            } catch (_) {}
            try {
                if (typeof pickRandomStation === 'function') pickRandomStation();
                else if (typeof playRadio === 'function') playRadio();
            } catch (_) {}
            try { if (typeof refreshActiveDeckVideoDisplays === 'function') refreshActiveDeckVideoDisplays(); } catch (_) {}
        }

        function playDeckATrackFromQueue(opts) {
            initAudio();
            try { stopNowPlayingPoll(); } catch (_) {}
            const q = deckFileQueues.a;
            const deferForAutoMix = shouldDeferLocalPlayForAutoMix('a', opts);
            if (!q.length) {
                resumeRandomRadioForDeck('a');
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                return;
            }

            const item = q[0];
            if (item && !item.isVideo) {
                try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('a'); } catch (_) {}
            }
            const audibleEl = getDeckAMediaForPlaybackState();
            const wasLocal = state.deckSourceMode.a === 'local';
            const audibleSrc = audibleEl ? sanitizeUrlForAudio(String(audibleEl.currentSrc || audibleEl.src || '')) : '';
            const nextClean = item.url ? sanitizeUrlForAudio(String(item.url)) : '';
            const audiblePlaying = !!(audibleEl && audibleSrc && !audibleEl.paused && !audibleEl.ended);
            const warmLocal = !!(
                wasLocal &&
                !item.isVideo &&
                state.gainRadioPrimaryPath &&
                state.gainRadioSecondaryPath &&
                audioElRadioAAlt &&
                audiblePlaying &&
                nextClean &&
                audibleSrc &&
                nextClean !== audibleSrc
            );

            const runColdDeckALocal = (trackItem, autoplay) => {
                abortRadioAHandoff();
                resetRadioADualStreamHandoff();
                revokeBlobSrc(audioEl);
                state.deckSourceMode.a = 'local';
                try { state.deckLocalDisplayName.a = trackItem.name ? String(trackItem.name) : ''; } catch (_) {}
                try { audioEl.crossOrigin = 'anonymous'; } catch (_) {}
                audioEl.src = trackItem.url;
                if (trackItem.isVideo) registerDeckVideoFeed('a', trackItem.url, trackItem.name, true);
                if (autoplay === false) {
                    try { audioEl.currentTime = 0; } catch (_) {}
                    try { audioEl.pause(); } catch (_) {}
                    try { connectDeckMediaToEq('a'); } catch (_) {}
                    markAutoMixDeferredLocal('a', true);
                    try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                    return;
                }
                audioEl.play().then(() => {
                    connectDeckMediaToEq('a');
                    try {
                        if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                            state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                        }
                    } catch (e) { /* ignore */ }
                    if (!state.isPlaying) startGame();
                    try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                }).catch((e) => {
                    console.warn('Deck A local playback failed:', e);
                    state.deckSourceMode.a = 'radio';
                    try { state.deckLocalDisplayName.a = ''; } catch (_) {}
                    try { playRadio(); } catch (_) {}
                    try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                });
            };

            if (deferForAutoMix) {
                const item = q[0];
                q.shift();
                runColdDeckALocal(item, false);
                return;
            }

            if (!warmLocal) {
                q.shift();
                runColdDeckALocal(item);
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                return;
            }

            ensureRadioAAltWired();
            if (!state.radioAltAMediaWired || !state.radioElementSourceAAlt) {
                q.shift();
                runColdDeckALocal(item);
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                return;
            }

            q.shift();
            abortRadioAHandoff();
            state.deckSourceMode.a = 'local';
            try { state.deckLocalDisplayName.a = item.name ? String(item.name) : ''; } catch (_) {}
            try { showStationBanner(item.name || 'Local track'); } catch (_) {}

            radioAHandoffAbortCtrl = new AbortController();
            const signal = radioAHandoffAbortCtrl.signal;

            const outputFromSecondary = isDeckARadioOutputFromAlt();
            const liveEl = outputFromSecondary ? audioElRadioAAlt : audioEl;
            const prepEl = outputFromSecondary ? audioEl : audioElRadioAAlt;
            const liveGain = outputFromSecondary ? state.gainRadioSecondaryPath : state.gainRadioPrimaryPath;
            const prepGain = outputFromSecondary ? state.gainRadioPrimaryPath : state.gainRadioSecondaryPath;

            const afterLocalHandoff = () => {
                try {
                    if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                        state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                    }
                } catch (e) { /* ignore */ }
                if (!state.isPlaying) startGame();
                try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            };

            const onLocalPrepFail = (e) => {
                console.warn('Deck A local gapless handoff failed:', e);
                try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                try { deckFileQueues.a.unshift(item); } catch (_) {}
                runColdDeckALocal(item);
            };

            try { setDjDeckRadioLoadingSpinner('a', true); } catch (_) {}
            beginDeckAPingPongMediaHandoff(
                prepEl, liveEl, liveGain, prepGain,
                item.url, signal, 30000, afterLocalHandoff, onLocalPrepFail
            );
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function playDeckBTrackFromQueue(opts) {
            initAudio();
            prepareDeckBLocalPlayback();
            revokeBlobSrc(audioElB);
            const q = deckFileQueues.b;
            const deferForAutoMix = shouldDeferLocalPlayForAutoMix('b', opts);
            if (!q.length) {
                resumeRandomRadioForDeck('b');
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                return;
            }
            const item = q.shift();
            if (item && !item.isVideo) {
                try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('b'); } catch (_) {}
            }
            state.deckSourceMode.b = 'local';
            try { state.deckLocalDisplayName.b = item.name ? String(item.name) : ''; } catch (_) {}
            try { audioElB.crossOrigin = 'anonymous'; } catch (_) {}
            audioElB.src = item.url;
            if (item.isVideo) registerDeckVideoFeed('b', item.url, item.name, true);
            if (deferForAutoMix) {
                try { audioElB.currentTime = 0; } catch (_) {}
                try { audioElB.pause(); } catch (_) {}
                try { connectDeckMediaToEq('b'); } catch (_) {}
                markAutoMixDeferredLocal('b', true);
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
                return;
            }
            ensureLocalDeckCrossfadeAudible('b', opts);
            audioElB.play().then(() => {
                connectDeckMediaToEq('b');
                try {
                    if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                        state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                    }
                } catch (_) {}
                if (!state.isPlaying) startGame();
                try { if (typeof updateMixBStatus === 'function') updateMixBStatus(); } catch (_) {}
                try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
            }).catch((e) => {
                console.warn('Deck B local playback failed:', e);
                state.deckSourceMode.b = 'radio';
                try { state.deckLocalDisplayName.b = ''; } catch (_) {}
                try { if (typeof playRadioB === 'function') playRadioB(); } catch (_) {}
            });
            try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function onDeckAEndedForQueue(ev) {
            try {
                if (state.deckSourceMode.a !== 'local') return;
                const target = ev && ev.target;
                const audible = (typeof getDeckAMediaForPlaybackState === 'function')
                    ? getDeckAMediaForPlaybackState()
                    : audioEl;
                if (target && audible && target !== audible) return;
                playDeckATrackFromQueue();
            } catch (_) {}
        }

        function onDeckBEndedForQueue(ev) {
            try {
                if (state.deckSourceMode.b !== 'local') return;
                const target = ev && ev.target;
                if (target && target !== audioElB) return;
                playDeckBTrackFromQueue();
            } catch (_) {}
        }

        function abortRadioAHandoff() {
            try {
                if (radioAHandoffAbortCtrl) radioAHandoffAbortCtrl.abort();
            } catch (_) {}
            radioAHandoffAbortCtrl = null;
        }

        function resetRadioADualStreamHandoff() {
            abortRadioAHandoff();
            try {
                if (state.gainRadioPrimaryPath) state.gainRadioPrimaryPath.gain.value = 1;
                if (state.gainRadioSecondaryPath) state.gainRadioSecondaryPath.gain.value = 0;
            } catch (_) {}
            try { revokeBlobSrc(audioElRadioAAlt); } catch (_) {}
            try {
                if (audioElRadioAAlt) {
                    audioElRadioAAlt.pause();
                    audioElRadioAAlt.removeAttribute('src');
                    audioElRadioAAlt.load();
                }
            } catch (_) {}
            try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
        }

        /** Deck A/B header: show indeterminate ring while stream connects or crossfade-buffering */
        function setDjDeckRadioLoadingSpinner(deck, on) {
            const id = deck === 'b' ? 'dj-station-b-load-spinner' : 'dj-station-a-load-spinner';
            try {
                const el = document.getElementById(id);
                if (!el) return;
                el.classList.toggle('display-none', !on);
                el.setAttribute('aria-busy', on ? 'true' : 'false');
                el.setAttribute('aria-hidden', on ? 'false' : 'true');
            } catch (_) {}
        }

        function isDeckARadioOutputFromAlt() {
            try {
                if (!audioElRadioAAlt || !state.gainRadioPrimaryPath || !state.gainRadioSecondaryPath) return false;
                const gS = Number(state.gainRadioSecondaryPath.gain.value);
                const gP = Number(state.gainRadioPrimaryPath.gain.value);
                return gS >= gP && gS > 0.01;
            } catch (_) {
                return false;
            }
        }

        function getDeckARadioAudibleEl() {
            if (!audioEl) return audioEl;
            try {
                if (state && state.deckSourceMode && state.deckSourceMode.a === 'radio' && isDeckARadioOutputFromAlt()) {
                    return audioElRadioAAlt;
                }
            } catch (_) {}
            return audioEl;
        }

        function getDeckAMediaForPlaybackState() {
            try {
                if (state && state.deckSourceMode && state.deckSourceMode.a === 'radio') return getDeckARadioAudibleEl();
                if (state && state.deckSourceMode && state.deckSourceMode.a === 'local' && audioElRadioAAlt && isDeckARadioOutputFromAlt()) {
                    return audioElRadioAAlt;
                }
            } catch (_) {}
            return audioEl;
        }

        function ensureRadioAAltWired() {
            if (!state.audioCtx || !state.eqA || !state.eqA.high || !state.gainRadioSecondaryPath || !audioElRadioAAlt) return;
            if (state.radioAltAMediaWired) return;
            clearStaleDeckMediaSource('radioElementSourceAAlt');
            try {
                state.radioElementSourceAAlt = createMediaSourceFromElement(state.audioCtx, audioElRadioAAlt);
                if (!state.radioElementSourceAAlt) return;
                state.radioElementSourceAAlt.connect(state.gainRadioSecondaryPath);
                state.gainRadioSecondaryPath.connect(state.eqA.high);
                state.radioAltAMediaWired = true;
            } catch (e) {
                console.warn('Deck A radio alt element wiring failed:', e);
            }
        }

        function abortRadioBHandoff() {
            try {
                if (radioBHandoffAbortCtrl) radioBHandoffAbortCtrl.abort();
            } catch (_) {}
            radioBHandoffAbortCtrl = null;
        }

        function resetRadioBDualStreamHandoff() {
            abortRadioBHandoff();
            try {
                if (state.gainRadioBPrimaryPath) state.gainRadioBPrimaryPath.gain.value = 1;
                if (state.gainRadioBSecondaryPath) state.gainRadioBSecondaryPath.gain.value = 0;
            } catch (_) {}
            try { revokeBlobSrc(audioElRadioBAlt); } catch (_) {}
            try {
                if (audioElRadioBAlt) {
                    audioElRadioBAlt.pause();
                    audioElRadioBAlt.removeAttribute('src');
                    audioElRadioBAlt.load();
                }
            } catch (_) {}
            try { setDjDeckRadioLoadingSpinner('b', false); } catch (_) {}
        }

        function isDeckBRadioOutputFromAlt() {
            try {
                if (!audioElRadioBAlt || !state.gainRadioBPrimaryPath || !state.gainRadioBSecondaryPath) return false;
                const gS = Number(state.gainRadioBSecondaryPath.gain.value);
                const gP = Number(state.gainRadioBPrimaryPath.gain.value);
                return gS >= gP && gS > 0.01;
            } catch (_) {
                return false;
            }
        }

        function getDeckBRadioAudibleEl() {
            if (!audioElB) return audioElB;
            try {
                if (state && state.deckSourceMode && state.deckSourceMode.b === 'radio' && isDeckBRadioOutputFromAlt()) {
                    return audioElRadioBAlt;
                }
            } catch (_) {}
            return audioElB;
        }

        function ensureRadioBAltWired() {
            if (!state.audioCtx || !state.eqB || !state.eqB.high || !state.gainRadioBSecondaryPath || !audioElRadioBAlt) return;
            if (state.radioAltBMediaWired) return;
            clearStaleDeckMediaSource('radioElementSourceBAlt');
            try {
                state.radioElementSourceBAlt = createMediaSourceFromElement(state.audioCtx, audioElRadioBAlt);
                if (!state.radioElementSourceBAlt) return;
                state.radioElementSourceBAlt.connect(state.gainRadioBSecondaryPath);
                state.gainRadioBSecondaryPath.connect(state.eqB.high);
                state.radioAltBMediaWired = true;
            } catch (e) {
                console.warn('Deck B radio alt element wiring failed:', e);
            }
        }

        function waitForRadioStreamAudible(mediaEl, opts) {
            const maxMs = (opts && opts.timeoutMs) || 14000;
            const signal = opts && opts.signal;
            const t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            return new Promise((resolve, reject) => {
                let done = false;
                const cleanup = () => {
                    try { clearInterval(iv); } catch (_) {}
                    try { mediaEl.removeEventListener('playing', onPlaying); } catch (_) {}
                    try { mediaEl.removeEventListener('timeupdate', onTick); } catch (_) {}
                };
                const finish = () => {
                    if (done) return;
                    done = true;
                    cleanup();
                    resolve();
                };
                const fail = (err) => {
                    if (done) return;
                    done = true;
                    cleanup();
                    reject(err || new Error('radio buffer wait'));
                };
                const elapsed = () => ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) - t0;
                const check = () => {
                    if (signal && signal.aborted) return fail(new DOMException('aborted', 'AbortError'));
                    if (elapsed() > maxMs) return fail(new Error('timeout'));
                    try {
                        if (!mediaEl || mediaEl.paused || mediaEl.ended) return;
                        const rd = mediaEl.readyState;
                        if (rd >= 4) return finish();
                        if (rd >= 3 && mediaEl.buffered && mediaEl.buffered.length) {
                            const end = mediaEl.buffered.end(mediaEl.buffered.length - 1);
                            if (end - mediaEl.currentTime > 0.12) return finish();
                        }
                    } catch (_) {}
                };
                const onTick = () => check();
                const onPlaying = () => {
                    try { check(); } catch (_) {}
                };
                mediaEl.addEventListener('playing', onPlaying);
                mediaEl.addEventListener('timeupdate', onTick);
                const iv = setInterval(check, 100);
                check();
            });
        }

        /**
         * Deck A: load newUrl on prepEl (gain 0), wait for buffer, ramp liveGain down / prepGain up, then tear down live element.
         * Used for gapless radio station changes and gapless local queue advances (same dual <audio> + gain paths).
         */
        function beginDeckAPingPongMediaHandoff(prepEl, liveEl, liveGain, prepGain, newUrl, signal, bufferTimeoutMs, onComplete, onPrepPlayFail) {
            try {
                revokeBlobSrc(prepEl);
                prepEl.crossOrigin = 'anonymous';
                prepEl.src = newUrl;
                prepGain.gain.value = 0;
            } catch (e) {
                if (typeof onPrepPlayFail === 'function') onPrepPlayFail(e);
                return;
            }
            prepEl.play().then(async () => {
                try {
                    await waitForRadioStreamAudible(prepEl, { signal, timeoutMs: bufferTimeoutMs || 14000 });
                } catch (err) {
                    if (signal && (signal.aborted || (err && err.name === 'AbortError'))) {
                        try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                        return;
                    }
                    console.warn('Deck A handoff buffer wait failed, switching immediately:', err);
                }
                if (signal && signal.aborted) {
                    try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                    return;
                }
                try { await state.audioCtx.resume(); } catch (_) {}
                const t = state.audioCtx.currentTime;
                const rampSec = (typeof getDeckARadioCrossfadeRampSec === 'function')
                    ? getDeckARadioCrossfadeRampSec()
                    : 0.06;
                const handoffCleanupMs = Math.ceil(Math.max(rampSec, 0.06) * 1000) + 100;
                try {
                    liveGain.gain.cancelScheduledValues(t);
                    prepGain.gain.cancelScheduledValues(t);
                    liveGain.gain.setValueAtTime(Math.max(0, Math.min(1, liveGain.gain.value)), t);
                    prepGain.gain.setValueAtTime(0, t);
                    liveGain.gain.linearRampToValueAtTime(0, t + rampSec);
                    prepGain.gain.linearRampToValueAtTime(1, t + rampSec);
                } catch (_) {
                    try { liveGain.gain.value = 0; prepGain.gain.value = 1; } catch (_) {}
                }
                setTimeout(() => {
                    if (signal && signal.aborted) {
                        try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                        return;
                    }
                    try { liveEl.pause(); } catch (_) {}
                    try { liveEl.removeAttribute('src'); liveEl.load(); } catch (_) {}
                    try { connectDeckMediaToEq('a'); } catch (_) {}
                    if (typeof onComplete === 'function') onComplete();
                }, handoffCleanupMs);
            }).catch((e) => {
                if (typeof onPrepPlayFail === 'function') onPrepPlayFail(e);
            });
        }

        // --- FIXED PLAYRADIO FUNCTION ---
        // --- FIXED PLAYRADIO FUNCTION (Routes through EQ; warm station changes pre-buffer on alt element) ---
        function playRadio() {
            initAudio();
            try { if (typeof releaseDeckVideoFeed === 'function') releaseDeckVideoFeed('a'); } catch (_) {}
            state.deckSourceMode.a = 'radio';
            try { state.deckLocalDisplayName.a = ''; } catch (_) {}
            let url = '';
            try {
                url = (radioInputEl && radioInputEl.value) ? radioInputEl.value : '';
                if (!url && Array.isArray(stations) && stations.length > 0) {
                    const idx = (typeof currentStationIndex === 'number' && currentStationIndex >= 0) ? currentStationIndex : 0;
                    url = stations[idx]?.url || '';
                }
            } catch (_) {}
            if (!url) return;
            abortRadioAHandoff();
            const urlClean = sanitizeUrlForAudio(String(url));

            const audibleEl = getDeckARadioAudibleEl();
            const audibleSrc = audibleEl ? sanitizeUrlForAudio(String(audibleEl.currentSrc || audibleEl.src || '')) : '';
            const audiblePlaying = !!(audibleEl && audibleSrc && !audibleEl.paused && !audibleEl.ended);
            const warmCrossfade = !!(
                state.gainRadioPrimaryPath &&
                state.gainRadioSecondaryPath &&
                audiblePlaying &&
                urlClean &&
                audibleSrc &&
                urlClean !== audibleSrc
            );

            if (!warmCrossfade) {
                resetRadioADualStreamHandoff();
            }

            if (!warmCrossfade) {
                revokeBlobSrc(audioEl);
            }
            try { statusEl.innerText = ''; } catch (_) {}
            try { setDjDeckRadioLoadingSpinner('a', true); } catch (_) {}
            try { stopNowPlayingPoll(); } catch (_) {}

            (function updateInitialStationTitle() {
                let nameToShow = null;
                if (currentStationIndex >= 0 && stations[currentStationIndex] && stations[currentStationIndex].url === url) {
                    nameToShow = stations[currentStationIndex].name;
                }
                showStationBanner(nameToShow || deriveTitleFromUrl(url));
            })();

            const afterConnect = () => {
                try {
                    if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                        state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                    }
                } catch (e) { /* ignore */ }
                if (!state.isPlaying) startGame();
                radioRetryAttempts = 0;
                try { restartNowPlayingPoll(url); } catch (_) {}
                try { statusEl.innerText = ''; } catch (_) {}
                try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
            };

            const onPlayFail = (e) => {
                console.error('Stream playback failed:', e);
                try { setDjDeckRadioLoadingSpinner('a', false); } catch (_) {}
                radioRetryAttempts = (radioRetryAttempts || 0) + 1;
                if (radioRetryAttempts <= MAX_RADIO_RETRIES && stations.length > 0) {
                    statusEl.innerText = 'Stream failed. Trying another station...';
                    if (currentStationIndex < 0 || currentStationIndex >= stations.length) {
                        currentStationIndex = Math.floor(Math.random() * stations.length);
                        setStation(currentStationIndex);
                    } else {
                        pickRandomStation();
                    }
                } else {
                    statusEl.innerText = 'No playable station found. Check console for CORS/Mixed Content errors.';
                }
            };

            if (!warmCrossfade) {
                audioEl.crossOrigin = 'anonymous';
                audioEl.src = url;
                audioEl.play().then(() => {
                    try { if (state.audioCtx && state.audioCtx.state === 'suspended') state.audioCtx.resume(); } catch (_) {}
                    connectDeckMediaToEq('a');
                    afterConnect();
                }).catch(onPlayFail);
                return;
            }

            ensureRadioAAltWired();
            if (!state.radioAltAMediaWired || !state.radioElementSourceAAlt) {
                resetRadioADualStreamHandoff();
                revokeBlobSrc(audioEl);
                audioEl.crossOrigin = 'anonymous';
                audioEl.src = url;
                audioEl.play().then(() => {
                    connectDeckMediaToEq('a');
                    afterConnect();
                }).catch(onPlayFail);
                return;
            }

            const outputFromSecondary = isDeckARadioOutputFromAlt();
            const liveEl = outputFromSecondary ? audioElRadioAAlt : audioEl;
            const prepEl = outputFromSecondary ? audioEl : audioElRadioAAlt;
            const liveGain = outputFromSecondary ? state.gainRadioSecondaryPath : state.gainRadioPrimaryPath;
            const prepGain = outputFromSecondary ? state.gainRadioPrimaryPath : state.gainRadioSecondaryPath;

            radioAHandoffAbortCtrl = new AbortController();
            const signal = radioAHandoffAbortCtrl.signal;

            beginDeckAPingPongMediaHandoff(
                prepEl, liveEl, liveGain, prepGain,
                url, signal, 14000, afterConnect, onPlayFail
            );
        }

        async function useMic() {
            initAudio();
            resetRadioADualStreamHandoff();
            statusEl.innerText = "Requesting Mic Access...";
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                if(state.sourceNode) state.sourceNode.disconnect();
                state.sourceNode = state.audioCtx.createMediaStreamSource(stream);
                state.mediaStream = stream;
                try { state.sourceNode.disconnect(); } catch(_) {}
                state.sourceNode.connect(state.streamAGain);
                rebuildEffectsChain();
                // Ensure active visualizer (e.g., ProjectM) is connected to the current analyser
                try {
                    if (state.activeVisualizer && state.activeVisualizer.visualizer && typeof state.activeVisualizer.visualizer.connectAudio === 'function') {
                        state.activeVisualizer.visualizer.connectAudio(state.analyserNode);
                    }
                } catch(e) { /* ignore */ }
                startGame();
            } catch(e) { statusEl.innerText = "Error: " + e.message; }
        }

        function useFile(e) {
            initAudio();
            const files = Array.from(e.target.files || []);
            try { e.target.value = ''; } catch (_) {}
            if (!files.length) return;
            enqueueDeckLocalFiles('a', files);
            statusEl.innerText = `Queued ${files.length} track(s) on Deck A`;
            const playingLocal = state.deckSourceMode.a === 'local';
            const mediaGoing = audioEl && audioEl.src && !audioEl.paused && !audioEl.ended;
            const startNow = !playingLocal || !mediaGoing;
            if (startNow) playDeckATrackFromQueue();
            else try { if (typeof window.__refreshDjQueueUi === 'function') window.__refreshDjQueueUi(); } catch (_) {}
        }

        function startGame() {
            state.isPlaying = true;
            const overlay = document.getElementById('overlay');
            // Digital transition: animate overlay out, then start visuals
            overlay.classList.add('digital-out');
            document.getElementById('ui-layer').classList.remove('hidden');
            setTimeout(() => {
                hideOverlay();
                overlay.classList.remove('digital-out');
                // Ensure background gif is not shown behind canvas once playing
                try { hidePtaStartBg(); } catch(e) {}
                // Start with last visual if saved, else Digital Radio
                try {
                    let idxToLoad = -1;
                    try {
                        const savedName = localStorage.getItem('lastModeName');
                        if (savedName) {
                            const savedIdx = modes.findIndex(m => m && m.name === savedName);
                            if (savedIdx >= 0) idxToLoad = savedIdx;
                        } else {
                            const savedIdxStr = localStorage.getItem('lastModeIndex');
                            const savedIdxNum = savedIdxStr != null ? parseInt(savedIdxStr, 10) : -1;
                            if (!isNaN(savedIdxNum) && savedIdxNum >= 0 && savedIdxNum < modes.length) {
                                idxToLoad = savedIdxNum;
                            }
                        }
                    } catch(_) {}
                    if (idxToLoad < 0) {
                        idxToLoad = Math.max(0, modes.findIndex(m => m && m.name === 'Digital Radio'));
                    }
                    loadMode(idxToLoad);
                } catch(e) {
                    loadMode(0);
                }
                try { cancelStartTextLoop(); } catch(e) {}
                // Show quick radio button once the app has started
                if (typeof radioQuickBtn !== 'undefined' && radioQuickBtn) {
                    radioQuickBtn.style.display = '';
                }
                // Build top bar: show quick radio (volume slider lives in top menu)
                try {
                    if (topBar && radioQuickBtn) {
                        topBar.innerHTML = '';
                        // Ensure the top bar is actually shown (remove class that forces display:none)
                        topBar.classList.remove('display-none');
                        topBar.appendChild(radioQuickBtn);
                        topBar.style.display = 'flex';
                        topBar.style.opacity = '1';
                        topBar.style.pointerEvents = 'auto';
                    }
                } catch(e) {}
            resetIdleTimer();
            }, 520);
        }

function stopAllAndShowStart() {
    // Stop active visualizer
    if (state.activeVisualizer && typeof state.activeVisualizer.destroy === 'function') {
        try { state.activeVisualizer.destroy(); } catch(e) {}
    }
    state.activeVisualizer = null;
    try { resetDeckFileQueuesAndRevoke(); } catch (_) {}
    // Stop audio sources
    try {
        if (state.sourceNode) {
            try { state.sourceNode.disconnect(); } catch(e) {}
            try { if (typeof state.sourceNode.stop === 'function') state.sourceNode.stop(0); } catch(e) {}
            state.sourceNode = null;
        }
        if (state.mediaStream) {
            try { state.mediaStream.getTracks().forEach(t => t.stop()); } catch(e) {}
            state.mediaStream = null;
        }
        if (audioEl) {
            try { audioEl.pause(); } catch(e) {}
            try { audioEl.currentTime = 0; } catch(e) {}
            try { audioEl.src = ''; audioEl.load(); } catch(e) {}
        }
    } catch (err) { /* ignore */ }
    // Hide player UI and other overlays, show start overlay
    document.getElementById('ui-layer').classList.add('hidden');
    if (typeof hideRadioPanel === 'function') { try { hideRadioPanel(); } catch(e) {} }
    try {
        if (typeof hideStationBannerPermanently === 'function') hideStationBannerPermanently();
    } catch (_) {}
    try { stopNowPlayingPoll(); } catch (e) {}
    try { clearNowPlayingICYBanner(); } catch (e) {}
    if (typeof hideWebm === 'function' && typeof webmOn !== 'undefined' && webmOn) {
        try { hideWebm(); } catch(e) {}
    }
	state.isPlaying = false;
	if (typeof statusEl !== 'undefined' && statusEl) {
		try { typeStatus("Not streaming — choose a source to begin"); } catch(e) {}
	}
    showOverlay();
}
        try { globalThis.applyOverlayGlowFx = applyOverlayGlowFx; } catch (_) {}
        try { globalThis.applyOverlayLogoFx = applyOverlayLogoFx; } catch (_) {}
