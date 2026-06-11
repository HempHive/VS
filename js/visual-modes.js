/* Extracted from app.js — visual-modes (loaded in 3 ordered parts from app.js). */
// --- ENGINE: BUTTERCHURN ---
        class MilkdropEngine {
            constructor() {
                this.name = "ProjectM / Milkdrop";
                this.presets = butterchurnPresets.getPresets();
                this.presetKeys = Object.keys(this.presets);
                this.currentPresetIdx = Math.floor(Math.random() * this.presetKeys.length);
                this.resizeHandler = this.onResize.bind(this);
                this.canvas = null; // Track canvas reference
            }

            init() {
                container.innerHTML = '';
                this.canvas = document.createElement('canvas');
                
                // FIX: Explicitly set canvas attributes to window size
                this.canvas.width = window.innerWidth;
                this.canvas.height = window.innerHeight;
                
                container.appendChild(this.canvas);

                this.visualizer = butterchurn.createVisualizer(state.audioCtx, this.canvas, {
                    width: window.innerWidth, 
                    height: window.innerHeight,
                    pixelRatio: window.devicePixelRatio || 1
                });
                this.visualizer.connectAudio(state.analyserNode);
                // Randomize preset on each init/selection
                this.currentPresetIdx = Math.floor(Math.random() * this.presetKeys.length);
                this.loadPreset(this.currentPresetIdx);

                window.addEventListener('resize', this.resizeHandler);
                
                this.animate();
                this.restartCycle();
            }

            loadPreset(idx) {
                const key = this.presetKeys[idx];
                this.visualizer.loadPreset(this.presets[key], 2.7); 
                if (state.activeVisualizer === this) {
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                }
				// Change bottom HUD color on each preset load
				try { setBottomTextRandomColor(); } catch(e) {}
            }

            nextPreset() {
                // Pick a random preset different from current
                let next = this.currentPresetIdx;
                if(this.presetKeys.length > 1) {
                    while(next === this.currentPresetIdx) {
                        next = Math.floor(Math.random() * this.presetKeys.length);
                    }
                }
                this.currentPresetIdx = next;
                this.loadPreset(this.currentPresetIdx);
                this.restartCycle();
            }

            restartCycle() {
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                const schedule = () => {
                    const minS = Number(visualSettings.shuffleMinSec) || 30;
                    const maxS = Number(visualSettings.shuffleMaxSec) || 60;
                    const lo = Math.min(minS, maxS);
                    const hi = Math.max(minS, maxS);
                    const delay = (lo + Math.random() * (hi - lo)) * 1000;
                    this.cycleTimeout = setTimeout(() => {
                        this.nextPreset();
                        schedule();
                    }, delay);
                };
                schedule();
            }
            
            loadCustomMilk(jsonTxt) {
                try {
                    const preset = JSON.parse(jsonTxt);
                    this.visualizer.loadPreset(preset, 1.0);
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                    if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
					try { setBottomTextRandomColor(); } catch(e) {}
                } catch(e) { alert("Invalid JSON"); }
            }

            onResize() { 
                if(this.visualizer && this.canvas) {
                    const w = window.innerWidth;
                    const h = window.innerHeight;
                    // FIX: Update attributes AND renderer
                    this.canvas.width = w;
                    this.canvas.height = h;
                    this.visualizer.setRendererSize(w, h);
                }
            }

            animate() {
                this.animId = requestAnimationFrame(this.animate.bind(this));
                this.visualizer.render();
            }

            destroy() {
                cancelAnimationFrame(this.animId);
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                window.removeEventListener('resize', this.resizeHandler);
                container.innerHTML = '';
            }
        }

        // --- ENGINE: BUTTERCHURN v2 (Enhanced) ---
        class MilkdropEngineV2 {
            constructor() {
                this.name = "ProjectM v2";
                this.resizeHandler = this.onResize.bind(this);
                this.canvas = null;
                this.transitionSec = visualSettings.transitionSec;
                // Use global preset map directly (largest available pack)
                this.presets = butterchurnPresets.getPresets?.() || {};
                this.presetKeys = Object.keys(this.presets);
                this.currentPresetIdx = Math.floor(Math.random() * Math.max(1, this.presetKeys.length));
            }

            init() {
                container.innerHTML = '';
                this.canvas = document.createElement('canvas');
                this.canvas.width = window.innerWidth;
                this.canvas.height = window.innerHeight;
                container.appendChild(this.canvas);

                const pxRatio = Number(visualSettings.pixelRatio) || (window.devicePixelRatio || 1);
                this.visualizer = butterchurn.createVisualizer(state.audioCtx, this.canvas, {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    pixelRatio: pxRatio
                });
                this.visualizer.connectAudio(state.analyserNode);
                // Randomize preset on each init/selection
                this.currentPresetIdx = Math.floor(Math.random() * Math.max(1, this.presetKeys.length));
                this.loadPreset(this.currentPresetIdx);

                window.addEventListener('resize', this.resizeHandler);
                this.animate();
                this.restartCycle();
            }

            loadPreset(idx) {
                if(this.presetKeys.length === 0) return;
                const key = this.presetKeys[idx];
                const transition = Number(visualSettings.transitionSec) || this.transitionSec || 2.7;
                this.visualizer.loadPreset(this.presets[key], transition);
                if (state.activeVisualizer === this) {
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                }
				// Change bottom HUD color on each preset load
				try { setBottomTextRandomColor(); } catch(e) {}
            }

            loadCustomMilk(jsonTxt) {
                try {
                    const preset = JSON.parse(jsonTxt);
                    this.visualizer.loadPreset(preset, Number(visualSettings.transitionSec) || 1.0);
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                    if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
					try { setBottomTextRandomColor(); } catch(e) {}
                } catch(e) { alert("Invalid JSON"); }
            }

            nextPreset() {
                if(this.presetKeys.length === 0) return;
                let next = this.currentPresetIdx;
                if(this.presetKeys.length > 1) {
                    while(next === this.currentPresetIdx) {
                        next = Math.floor(Math.random() * this.presetKeys.length);
                    }
                }
                this.currentPresetIdx = next;
                this.loadPreset(this.currentPresetIdx);
                this.restartCycle();
            }

            restartCycle() {
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                const schedule = () => {
                    const minS = Number(visualSettings.shuffleMinSec) || 12;
                    const maxS = Number(visualSettings.shuffleMaxSec) || 25;
                    const lo = Math.min(minS, maxS);
                    const hi = Math.max(minS, maxS);
                    const delay = (lo + Math.random() * (hi - lo)) * 1000;
                    this.cycleTimeout = setTimeout(() => {
                        this.nextPreset();
                        schedule();
                    }, delay);
                };
                schedule();
            }

            applySettings() {
                // Restart scheduling with new shuffle window
                this.restartCycle();
                // Transition seconds will be used on next loadPreset call automatically
            }

            onResize() {
                if(this.visualizer && this.canvas) {
                    const w = window.innerWidth;
                    const h = window.innerHeight;
                    this.canvas.width = w;
                    this.canvas.height = h;
                    this.visualizer.setRendererSize(w, h);
                }
            }

            animate() {
                this.animId = requestAnimationFrame(this.animate.bind(this));
                this.visualizer.render();
            }

            destroy() {
                cancelAnimationFrame(this.animId);
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                window.removeEventListener('resize', this.resizeHandler);
                container.innerHTML = '';
            }
        }

        // --- ENGINE: ProjectM v3 with Bars Overlay ---
        class MilkdropEngineV3 {
            constructor() {
                this.name = "ProjectM v3 (Bars Overlay)";
                this.resizeHandler = this.onResize.bind(this);
                this.pmCanvas = null;
                this.overlayRenderer = null;
                this.overlayScene = null;
                this.overlayCamera = null;
                this.bars = [];
                this.dataArray = null;
                this.transitionSec = visualSettings.transitionSec;
                this.presets = butterchurnPresets.getPresets?.() || {};
                this.presetKeys = Object.keys(this.presets);
                this.currentPresetIdx = Math.floor(Math.random() * Math.max(1, this.presetKeys.length));
            }

            init() {
                container.innerHTML = '';
                // ProjectM canvas (background)
                this.pmCanvas = document.createElement('canvas');
                this.pmCanvas.width = window.innerWidth;
                this.pmCanvas.height = window.innerHeight;
                container.appendChild(this.pmCanvas);

                const pxRatio = Number(visualSettings.pixelRatio) || (window.devicePixelRatio || 1);
                this.visualizer = butterchurn.createVisualizer(state.audioCtx, this.pmCanvas, {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    pixelRatio: pxRatio
                });
                this.visualizer.connectAudio(state.analyserNode);
                // Randomize preset on each init/selection
                this.currentPresetIdx = Math.floor(Math.random() * Math.max(1, this.presetKeys.length));
                this.loadPreset(this.currentPresetIdx);

                // Overlay bars (transparent WebGL)
                this.overlayScene = new THREE.Scene();
                this.overlayCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                this.overlayCamera.position.z = 8;
                this.overlayRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                this.overlayRenderer.setSize(window.innerWidth, window.innerHeight);
                this.overlayRenderer.setPixelRatio(window.devicePixelRatio || 1);
                this.overlayRenderer.domElement.style.position = 'absolute';
                this.overlayRenderer.domElement.style.top = '0';
                this.overlayRenderer.domElement.style.left = '0';
                this.overlayRenderer.domElement.style.pointerEvents = 'none';
                container.appendChild(this.overlayRenderer.domElement);

                // Build bars similar to sceneBars
                const group = new THREE.Group();
                this.bars = [];
                const numBars = 64;
                const spacing = 0.25;
                for (let i = 0; i < numBars; i++) {
                    const geo = new THREE.BoxGeometry(0.18, 0.5, 0.18);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x00ffff, emissive: 0x001f1f, metalness: 0.1, roughness: 0.6 });
                    const mesh = new THREE.Mesh(geo, mat);
                    mesh.position.x = (i - numBars / 2) * spacing;
                    group.add(mesh);
                    this.bars.push(mesh);
                }
                const ambient = new THREE.AmbientLight(0xffffff, 0.4);
                const point = new THREE.PointLight(0x00ffff, 1.2, 50);
                point.position.set(0, 5, 5);
                this.overlayScene.add(ambient, point, group);

                this.dataArray = new Uint8Array(state.analyserNode.frequencyBinCount);

                window.addEventListener('resize', this.resizeHandler);
                this.animate = this.animate.bind(this);
                this.animate();
                this.restartCycle();
            }

            loadPreset(idx) {
                if (this.presetKeys.length === 0) return;
                const key = this.presetKeys[idx];
                const transition = Number(visualSettings.transitionSec) || this.transitionSec || 2.7;
                this.visualizer.loadPreset(this.presets[key], transition);
                if (state.activeVisualizer === this) {
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                }
                try { setBottomTextRandomColor(); } catch (e) {}
            }

            nextPreset() {
                if (this.presetKeys.length === 0) return;
                let next = this.currentPresetIdx;
                if (this.presetKeys.length > 1) {
                    while (next === this.currentPresetIdx) {
                        next = Math.floor(Math.random() * this.presetKeys.length);
                    }
                }
                this.currentPresetIdx = next;
                this.loadPreset(this.currentPresetIdx);
                this.restartCycle();
            }

            restartCycle() {
                if (this.cycleTimeout) clearTimeout(this.cycleTimeout);
                const schedule = () => {
                    const minS = Number(visualSettings.shuffleMinSec) || 12;
                    const maxS = Number(visualSettings.shuffleMaxSec) || 25;
                    const lo = Math.min(minS, maxS);
                    const hi = Math.max(minS, maxS);
                    const delay = (lo + Math.random() * (hi - lo)) * 1000;
                    this.cycleTimeout = setTimeout(() => {
                        this.nextPreset();
                        schedule();
                    }, delay);
                };
                schedule();
            }

            onResize() {
                // PM
                if (this.visualizer && this.pmCanvas) {
                    const w = window.innerWidth;
                    const h = window.innerHeight;
                    this.pmCanvas.width = w;
                    this.pmCanvas.height = h;
                    this.visualizer.setRendererSize(w, h);
                }
                // Overlay
                if (this.overlayRenderer && this.overlayCamera) {
                    this.overlayCamera.aspect = window.innerWidth / window.innerHeight;
                    this.overlayCamera.updateProjectionMatrix();
                    this.overlayRenderer.setSize(window.innerWidth, window.innerHeight);
                }
            }

            animate() {
                this.animId = requestAnimationFrame(this.animate);
                // Render ProjectM (background)
                try {
                    if (this.visualizer && typeof this.visualizer.render === 'function') {
                        this.visualizer.render();
                    }
                } catch(e) { /* ignore */ }
                // Update bars overlay
                if (state && state.analyserNode && this.dataArray) {
                    state.analyserNode.getByteFrequencyData(this.dataArray);
                    const step = Math.floor(this.dataArray.length / Math.max(1, this.bars.length));
                    const t = performance.now() * 0.0008;
                    for (let i = 0; i < this.bars.length; i++) {
                        const v = this.dataArray[i * step] / 255;
                        const h = 0.2 + v * 3.0;
                        this.bars[i].scale.y = h;
                        this.bars[i].position.y = h * 0.25;
                        const hue = (i / this.bars.length + t) % 1;
                        this.bars[i].material.color.setHSL(hue, 1, 0.5);
                        this.bars[i].material.emissive.setHSL(hue, 1, 0.2);
                    }
                }
                if (this.overlayRenderer && this.overlayScene && this.overlayCamera) {
                    this.overlayRenderer.render(this.overlayScene, this.overlayCamera);
                }
            }

            destroy() {
                cancelAnimationFrame(this.animId);
                if (this.cycleTimeout) clearTimeout(this.cycleTimeout);
                window.removeEventListener('resize', this.resizeHandler);
                
                // NEW: Clean up the overlay scene memory
                if (this.overlayScene) {
                    this.overlayScene.traverse((object) => {
                        if (object.geometry) object.geometry.dispose();
                        if (object.material) {
                            if (Array.isArray(object.material)) {
                                object.material.forEach(m => m.dispose());
                            } else {
                                object.material.dispose();
                            }
                        }
                    });
                }

                if (this.overlayRenderer) {
                    try { this.overlayRenderer.dispose(); } catch(e) {}
                }
                container.innerHTML = '';
            }
        }
        // --- ENGINE: THREE.JS ---
        class ThreeEngine {
            constructor(name, sceneFn) {
                this.name = name;
                this.sceneFn = sceneFn;
                this.resizeHandler = this.onResize.bind(this);
				// Adaptive quality settings (auto-tune for stronger/weaker machines)
				this.quality = {
					mode: 'auto',
					minPixelRatio: 1,
					maxPixelRatio: 2.5,
					adjustIntervalMs: 1000,
					lastAdjustTs: 0,
					frameTimesMs: []
				};
            }

            init() {
                container.innerHTML = '';
                this.scene = new THREE.Scene();
                this.camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
                this.camera.position.z = 4;
                
                this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
				// Prefer physically-correct lighting and good tone mapping on capable devices
				if (this.renderer) {
					// Guard for different three.js versions
					if ('outputColorSpace' in this.renderer) {
						this.renderer.outputColorSpace = THREE.SRGBColorSpace;
					} else if ('outputEncoding' in this.renderer) {
						this.renderer.outputEncoding = THREE.sRGBEncoding;
					}
					if ('toneMapping' in this.renderer && THREE.ACESFilmicToneMapping) {
						this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
					}
					if ('physicallyCorrectLights' in this.renderer) {
						this.renderer.physicallyCorrectLights = true;
					}
				}
				// Initial pixel ratio tuned for device; will auto-adjust at runtime
				const initialPR = Math.min(window.devicePixelRatio || 1, QUALITY.pixelRatioCap);
				this.renderer.setPixelRatio(initialPR);
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                container.appendChild(this.renderer.domElement);

                this.composer = new EffectComposer(this.renderer);
                this.composer.addPass(new RenderPass(this.scene, this.camera));
                const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
                bloom.threshold = 0; bloom.strength = 1.2; bloom.radius = 0.5;
                this.composer.addPass(bloom);

                this.dataArray = new Uint8Array(state.analyserNode.frequencyBinCount);
                this.updateFn = this.sceneFn(this.scene, this.camera, this.composer);
                
                window.addEventListener('resize', this.resizeHandler);
				this._lastFrameTs = performance.now();
				this.quality.lastAdjustTs = this._lastFrameTs;
                this.animate();
            }

            onResize() {
                if(!this.camera || !this.renderer) return;
                this.camera.aspect = window.innerWidth / window.innerHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(window.innerWidth, window.innerHeight);
                this.composer.setSize(window.innerWidth, window.innerHeight);
            }

            animate() {
                this.animId = requestAnimationFrame(this.animate.bind(this));
				// Track frame time for adaptive quality
				const now = performance.now();
				const dt = now - (this._lastFrameTs || now);
				this._lastFrameTs = now;
				if (isFinite(dt) && dt > 0 && dt < 250) {
					const q = this.quality;
					q.frameTimesMs.push(dt);
					if (q.frameTimesMs.length > 90) q.frameTimesMs.shift();
					// Adjust roughly once per second
					if (q.mode === 'auto' && (now - q.lastAdjustTs) >= q.adjustIntervalMs && q.frameTimesMs.length >= 30) {
						const avg = q.frameTimesMs.reduce((a, b) => a + b, 0) / q.frameTimesMs.length;
						const currentPR = this.renderer.getPixelRatio ? this.renderer.getPixelRatio() : 1;
						let targetPR = currentPR;
						// Targets ~60fps. If consistently faster, try increase detail; if slower, reduce.
						if (avg < 14 && currentPR < q.maxPixelRatio) targetPR = Math.min(q.maxPixelRatio, currentPR + 0.25);
						if (avg > 20 && currentPR > q.minPixelRatio) targetPR = Math.max(q.minPixelRatio, currentPR - 0.25);
						if (Math.abs(targetPR - currentPR) >= 0.1) {
							const w = window.innerWidth, h = window.innerHeight;
							this.renderer.setPixelRatio(targetPR);
							this.renderer.setSize(w, h, false);
							this.composer.setSize(w, h);
						}
						q.lastAdjustTs = now;
					}
				}
                state.analyserNode.getByteFrequencyData(this.dataArray);
                if(this.updateFn) this.updateFn(this.dataArray, performance.now());
                this.composer.render();
            }

            destroy() {
                cancelAnimationFrame(this.animId);
                window.removeEventListener('resize', this.resizeHandler);
                
                // FORCE MEMORY CLEANUP
                this.scene.traverse((object) => {
                    if (!object.isMesh) return;
                    
                    if (object.geometry) object.geometry.dispose();
                    
                    if (object.material) {
                        if (Array.isArray(object.material)) {
                            object.material.forEach(m => m.dispose());
                        } else {
                            object.material.dispose();
                        }
                    }
                });

                this.renderer.dispose();
                container.innerHTML = '';
            }
        }

        function attachLogoSpinInput(domEl, spin, opts = {}) {
            if (!domEl || !spin) return () => {};
            const WHEEL_GAIN = 0.0045;
            const DRAG_GAIN = 0.014;
            const CLICK_MOVE_THRESH = 8;
            let pointerDown = false;
            let pointerMoved = false;
            let lastX = 0;
            let lastY = 0;
            let downX = 0;
            let downY = 0;

            const activeAxis = () => (typeof opts.getSpinAxis === 'function' ? opts.getSpinAxis() : null);

            const applySpinDelta = (dx, dy, gain) => {
                if (spin.homeActive) return;
                if (dx || dy) {
                    try { spin.touchInteraction?.(); } catch (_) {}
                }
                const axis = activeAxis();
                if (axis === 'y') {
                    if (dx) spin.velY += dx * gain;
                    else if (dy) spin.velY += dy * gain;
                    return;
                }
                if (axis === 'x') {
                    if (dy) spin.velX += dy * gain;
                    else if (dx) spin.velX += dx * gain;
                    return;
                }
                spin.velY += dx * gain;
                spin.velX += dy * gain;
            };

            const onWheel = (e) => {
                e.preventDefault();
                e.stopPropagation();
                applySpinDelta(e.deltaX || 0, e.deltaY || 0, WHEEL_GAIN);
            };
            const onPointerDown = (e) => {
                if (e.button !== 0) return;
                pointerDown = true;
                pointerMoved = false;
                downX = lastX = e.clientX;
                downY = lastY = e.clientY;
                if (spin.homeActive) spin.homeActive = false;
                try { spin.touchInteraction?.(); } catch (_) {}
                try { domEl.setPointerCapture(e.pointerId); } catch (_) {}
            };
            const onPointerMove = (e) => {
                if (!pointerDown) return;
                const dx = e.clientX - lastX;
                const dy = e.clientY - lastY;
                if (Math.abs(e.clientX - downX) > CLICK_MOVE_THRESH || Math.abs(e.clientY - downY) > CLICK_MOVE_THRESH) {
                    pointerMoved = true;
                }
                lastX = e.clientX;
                lastY = e.clientY;
                applySpinDelta(dx, dy, DRAG_GAIN);
            };
            const onPointerUp = () => {
                if (!pointerDown) return;
                pointerDown = false;
                if (!pointerMoved && typeof opts.onClick === 'function') opts.onClick();
            };

            domEl.addEventListener('wheel', onWheel, { passive: false, capture: true });
            domEl.addEventListener('pointerdown', onPointerDown);
            domEl.addEventListener('pointermove', onPointerMove);
            domEl.addEventListener('pointerup', onPointerUp);
            domEl.addEventListener('pointercancel', onPointerUp);
            return () => {
                domEl.removeEventListener('wheel', onWheel, { capture: true });
                domEl.removeEventListener('pointerdown', onPointerDown);
                domEl.removeEventListener('pointermove', onPointerMove);
                domEl.removeEventListener('pointerup', onPointerUp);
                domEl.removeEventListener('pointercancel', onPointerUp);
            };
        }

        class LogoEngine extends ThreeEngine {
            init() {
                super.init();
                try {
                    const bloomPass = this.composer && this.composer.passes
                        ? this.composer.passes.find((p) => p && p.strength != null && p.threshold != null && p.radius != null)
                        : null;
                    if (bloomPass) bloomPass.strength = 0.35;
                } catch (_) {}
                const spin = this.scene && this.scene.userData && this.scene.userData.logoSpin;
                if (spin && this.renderer) {
                    const inputOpts = {
                        onClick: () => {
                            if (spin.fxAxis && typeof spin.toggleSpinAxis === 'function') spin.toggleSpinAxis();
                            else if (typeof spin.homeToRest === 'function') spin.homeToRest();
                        },
                    };
                    if (spin.fxAxis) inputOpts.getSpinAxis = () => spin.spinAxis;
                    this._detachLogoInput = attachLogoSpinInput(this.renderer.domElement, spin, inputOpts);
                }
            }

            destroy() {
                if (this._detachLogoInput) {
                    this._detachLogoInput();
                    this._detachLogoInput = null;
                }
                const spin = this.scene && this.scene.userData && this.scene.userData.logoSpin;
                if (spin) {
                    if (typeof spin.dispose === 'function') spin.dispose();
                }
                super.destroy();
            }
        }

        let __djBeatPadLoopMenuCleanup = null;

// --- SCENES ---
        const sceneSphere = (scene) => {
            // Use SphereGeometry for clean UVs and per-vertex coloring
            const geo = new THREE.SphereGeometry(0.9, QUALITY.sphereSegs, QUALITY.sphereSegs);
            const base = geo.attributes.position.array.slice();
            // Allocate vertex colors
            const colorAttr = new THREE.Float32BufferAttribute(geo.attributes.position.count * 3, 3);
            geo.setAttribute('color', colorAttr);
            const mat = new THREE.MeshBasicMaterial({ vertexColors: true });
            const mesh = new THREE.Mesh(geo, mat);
            scene.add(mesh);
            return (data, time) => {
                const bass = data[8] / 255;
                const mid = data[32] / 255;
                const tre = data[64] / 255;
                const t = time * 0.001;
                // Audio-driven subtle vertex displacement
                const arr = geo.attributes.position.array;
                for (let i = 0; i < arr.length; i += 3) {
                    const bx = base[i], by = base[i + 1], bz = base[i + 2];
                    const n = Math.sin(bx * 1.1 + t * 1.8) * 0.015 + Math.cos(by * 1.3 - t * 1.6) * 0.015;
                    const a = (i / 3) % data.length;
                    const v = data[a] / 255;
                    const push = 0.04 + bass * 0.22 + mid * 0.12 + v * 0.06;
                    const s = 1.0 + n + push * 0.02;
                    arr[i] = bx * s; arr[i + 1] = by * s; arr[i + 2] = bz * s;
                }
                geo.attributes.position.needsUpdate = true;
                // Radiating multi-colour effect based on spherical coords + time + audio
                const cols = colorAttr.array;
                for (let i = 0, vi = 0; i < arr.length; i += 3, vi += 3) {
                    const x = arr[i], y = arr[i + 1], z = arr[i + 2];
                    const ang = Math.atan2(y, x);              // [-PI, PI]
                    const r = Math.sqrt(x * x + y * y + z * z);
                    const hue = ((ang / (Math.PI * 2)) + 0.5 + t * 0.25 + bass * 0.15) % 1;
                    const sat = 0.9;
                    const lig = 0.45 + 0.25 * (mid + tre * 0.5);
                    const c = new THREE.Color().setHSL(hue, sat, lig);
                    cols[vi] = c.r; cols[vi + 1] = c.g; cols[vi + 2] = c.b;
                }
                colorAttr.needsUpdate = true;
                // Rotation and scale breathing
                mesh.rotation.y += 0.004 + mid * 0.02;
                mesh.rotation.x += 0.002 + tre * 0.01;
                mesh.scale.setScalar(1 + bass * 0.3 + mid * 0.12);
            };
        };

        const sceneTunnel = (scene, camera) => {
            const group = new THREE.Group();
            const count = 40;
            const meshes = [];
            for(let i=0; i<count; i++){
                const geo = new THREE.TorusGeometry(1+(i*0.5), 0.05, 8, 50);
                const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 });
                const mesh = new THREE.Mesh(geo, mat);
                group.add(mesh);
                meshes.push(mesh);
            }
            scene.add(group);
            return (data, time) => {
                const step = Math.floor(data.length/count);
                for(let i=0; i<count; i++){
                    const val = data[i*step]/255;
                    meshes[i].scale.setScalar(1 + val);
                    meshes[i].material.color.setHSL(val + (time*0.0002), 1, 0.5);
                    meshes[i].rotation.z += 0.01 * (i%2?1:-1);
                }
                camera.position.z = 25 + Math.sin(time*0.001)*5;
                camera.rotation.z = time * 0.0005;
            };
        };

        const sceneBars = (scene, camera) => {
            const group = new THREE.Group();
            const bars = [];
            const numBars = 64;
            for(let i = 0; i < numBars; i++) {
                const geo = new THREE.BoxGeometry(0.18, 0.5, 0.18);
                const mat = new THREE.MeshStandardMaterial({ color: 0x00ffff, emissive: 0x001f1f, metalness: 0.1, roughness: 0.6 });
                const mesh = new THREE.Mesh(geo, mat);
				mesh.position.x = 0; // will be positioned in update based on current viewport width
                group.add(mesh);
                bars.push(mesh);
            }
            const ambient = new THREE.AmbientLight(0xffffff, 0.4);
            const point = new THREE.PointLight(0x00ffff, 1.2, 50);
            point.position.set(0, 5, 5);
            scene.add(ambient, point, group);
            camera.position.z = 8;
            return (data, time) => {
                const step = Math.floor(data.length / numBars);
				const t = time * 0.0008;
				// Compute current frustum width at z = 0 to make bars span visible width
				const fovRad = THREE.MathUtils.degToRad(camera.fov);
				const frustumHeight = 2 * Math.tan(fovRad / 2) * camera.position.z;
				const frustumWidth = frustumHeight * camera.aspect;
				const spacing = frustumWidth / numBars;
				const baseWidth = 0.18;
				const targetWidth = spacing * 0.7;
				const xScale = Math.max(0.1, targetWidth / baseWidth);
                for(let i = 0; i < numBars; i++) {
                    const v = data[i * step] / 255;
                    const h = 0.2 + v * 3.0;
					// Position bars evenly across the visible width
					bars[i].position.x = (i - (numBars - 1) / 2) * spacing;
					// Width to fit lane, height driven by audio
					bars[i].scale.x = xScale;
                    bars[i].scale.y = h;
                    bars[i].position.y = h * 0.25;
                    const hue = (i / numBars + t) % 1;
                    bars[i].material.color.setHSL(hue, 1, 0.5);
                    bars[i].material.emissive.setHSL(hue, 1, 0.2);
                }
				// No side-to-side motion
				group.rotation.y = 0;
            };
        };

        // Vertical Mirror Audio Bars (fills viewport height and resizes)
        const sceneBarsVerticalMirror = (scene, camera) => {
            const group = new THREE.Group();
            // Rotate 180 degrees
            group.rotation.x = Math.PI;
            const bars = [];
            const numBars = 64;
            const baseWidth = 0.2;
            const baseHeight = 0.2;
            for (let i = 0; i < numBars; i++) {
                const geo = new THREE.BoxGeometry(baseWidth, baseHeight, 0.18);
                const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x0a0a0a, metalness: 0.15, roughness: 0.55 });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.set(0, 0, 0); // y will be set in update
                group.add(mesh);
                bars.push(mesh);
            }
            const ambient = new THREE.AmbientLight(0xffffff, 0.45);
            const point = new THREE.PointLight(0x88ccff, 1.1, 50);
            point.position.set(0, 0, 6);
            scene.add(ambient, point, group);
            camera.position.z = 8;
            return (data, time) => {
                const t = time * 0.0006;
                // Compute visible height for current camera for consistent vertical fit
                const visibleHeight = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)) * Math.abs(camera.position.z);
                const spacing = visibleHeight / (numBars + 2); // a small top/bottom margin
                const targetBarHeight = spacing * 0.6;
                const heightScale = targetBarHeight / baseHeight;
                const step = Math.max(1, Math.floor(data.length / numBars));
                for (let i = 0; i < numBars; i++) {
                    const mesh = bars[i];
                    // Position bars vertically, centered around y=0
                    mesh.position.y = (i - (numBars - 1) / 2) * spacing;
                    // Fixed bar height to align cleanly in the stack
                    mesh.scale.y = heightScale;
                    // Width responds to audio; centered so it extends both left and right
                    const v = data[i * step] / 255;
                    const width = baseWidth * (1 + v * 8.0); // responsive extent to both sides
                    mesh.scale.x = width / baseWidth;
                    // Color/emissive shift over time and index
                    const hue = (i / numBars + t) % 1;
                    mesh.material.color.setHSL(hue, 1, 0.55);
                    mesh.material.emissive.setHSL(hue, 1, 0.18);
                }
            };
        };

        // Logo — full-viewport fidget card with trackpad / drag spin
        const LOGO_FIDGET_IMAGE = 'assets/fidget/cdl.png';
        const LOGO_IDLE_SETTLE_MS = 5000;

        function nearestFlatAngle(angle) {
            return Math.round(angle / Math.PI) * Math.PI;
        }

        function getNearestFlatTargets(spin) {
            if (spin.fxAxis && spin.spinAxis === 'y') {
                return { x: 0, y: nearestFlatAngle(spin.rotY) };
            }
            if (spin.fxAxis && spin.spinAxis === 'x') {
                return { x: nearestFlatAngle(spin.rotX), y: 0 };
            }
            return {
                x: nearestFlatAngle(spin.rotX),
                y: nearestFlatAngle(spin.rotY),
            };
        }

        function createLogoFidgetSpinState({ fxAxis = false } = {}) {
            const spin = {
                fxAxis,
                spinAxis: 'y',
                card: null,
                cardMesh: null,
                velX: 0,
                velY: 0,
                rotX: 0,
                rotY: 0,
                imgAspect: 1,
                textures: [],
                materials: [],
                lastInteractionAt: performance.now(),
                homeActive: false,
                homeFromX: 0,
                homeFromY: 0,
                homeToX: 0,
                homeToY: 0,
                homeStart: 0,
                homeDuration: 4000,
                touchInteraction() {
                    this.lastInteractionAt = performance.now();
                },
                beginHomeTo(targetX, targetY) {
                    this.velX = 0;
                    this.velY = 0;
                    this.homeActive = true;
                    this.homeFromX = this.rotX;
                    this.homeFromY = this.rotY;
                    this.homeToX = targetX;
                    this.homeToY = targetY;
                    this.homeStart = performance.now();
                    const span = Math.max(
                        Math.abs(this.homeToX - this.homeFromX),
                        Math.abs(this.homeToY - this.homeFromY)
                    );
                    const turns = span / (Math.PI * 2);
                    this.homeDuration = (2 + Math.min(2, turns * 2)) * 1000;
                },
                homeToRest() {
                    if (this.fxAxis) return;
                    this.beginHomeTo(0, 0);
                    this.homeDuration = (3 + Math.min(2, (Math.max(Math.abs(this.rotX), Math.abs(this.rotY)) / (Math.PI * 2)) * 2)) * 1000;
                },
                homeToNearestFlat() {
                    const target = getNearestFlatTargets(this);
                    this.beginHomeTo(target.x, target.y);
                },
                toggleSpinAxis() {
                    if (!this.fxAxis) return;
                    this.spinAxis = this.spinAxis === 'y' ? 'x' : 'y';
                    this.velX = 0;
                    this.velY = 0;
                    if (this.spinAxis === 'y') this.rotX = 0;
                    else this.rotY = 0;
                    if (typeof this.applyFxBackFace === 'function') this.applyFxBackFace();
                },
                applyFxBackFace() {
                    updateLogoCardBackFace(this);
                },
            };
            return spin;
        }

        function pickLogoBackMaterial(spin) {
            if (!spin.backMatY || !spin.frontMat) return null;
            const twoPi = Math.PI * 2;
            const norm = (a) => {
                let v = a % twoPi;
                if (v < 0) v += twoPi;
                return v;
            };
            const rx = norm(spin.rotX);
            const ry = norm(spin.rotY);
            const xFlipped = rx > Math.PI / 2 && rx < (3 * Math.PI) / 2;
            const yFlipped = ry > Math.PI / 2 && ry < (3 * Math.PI) / 2;
            if (yFlipped && !xFlipped) return spin.frontMat;
            if (xFlipped && !yFlipped) return spin.backMatY;
            if (yFlipped && xFlipped) {
                const xDev = Math.abs(Math.min(rx, twoPi - rx) - Math.PI / 2);
                const yDev = Math.abs(Math.min(ry, twoPi - ry) - Math.PI / 2);
                return yDev >= xDev ? spin.frontMat : spin.backMatY;
            }
            return spin.backMatY;
        }

        function applyLogoFxPlaneLayout(spin) {
            if (!spin.fxAxis || !spin.fxFrontMesh || !spin.fxBackMesh) return;
            spin.fxFrontMesh.rotation.set(0, 0, 0);
            spin.fxFrontMesh.position.z = 0.001;
            if (spin.frontMat) spin.fxFrontMesh.material = spin.frontMat;
            spin.fxBackMesh.position.z = -0.001;
            if (spin.spinAxis === 'y') {
                spin.fxBackMesh.rotation.set(0, Math.PI, 0);
            } else {
                spin.fxBackMesh.rotation.set(Math.PI, 0, 0);
            }
            if (spin.backMatY) spin.fxBackMesh.material = spin.backMatY;
        }

        function updateLogoCardBackFace(spin) {
            if (!spin || !spin.cardMesh) return;
            if (spin.fxAxis && spin.fxBackMesh) {
                applyLogoFxPlaneLayout(spin);
                return;
            }
            if (!spin.frontMat || !spin.backMatY) return;
            const mats = spin.cardMesh.material;
            if (!Array.isArray(mats)) return;
            const next = pickLogoBackMaterial(spin);
            if (next) {
                mats[4] = spin.frontMat;
                mats[5] = next;
            }
        }

        function assignLogoFidgetMaterials(spin, tex, placeholderMats) {
            if ('colorSpace' in tex) tex.colorSpace = THREE.SRGBColorSpace;
            else if ('encoding' in tex) tex.encoding = THREE.sRGBEncoding;
            tex.anisotropy = 8;
            const img = tex.image;
            if (img && img.width && img.height) spin.imgAspect = img.width / img.height;

            const frontMat = new THREE.MeshStandardMaterial({
                map: tex,
                metalness: 0.04,
                roughness: 0.42,
            });
            const backTexRotY = tex.clone();
            backTexRotY.center.set(0.5, 0.5);
            backTexRotY.rotation = Math.PI;
            const backMatY = new THREE.MeshStandardMaterial({
                map: backTexRotY,
                metalness: 0.04,
                roughness: 0.42,
            });
            spin.frontMat = frontMat;
            spin.backMatY = backMatY;
            spin.backMatX = frontMat;
            spin.textures.push(tex, backTexRotY);
            spin.materials.push(frontMat, backMatY, ...(placeholderMats || []));

            if (spin.fxAxis && spin.fxFrontMesh && spin.fxBackMesh) {
                applyLogoFxPlaneLayout(spin);
            } else if (spin.cardMesh && spin.cardMesh.isMesh) {
                const edgeMat = placeholderMats && placeholderMats[0];
                spin.cardMesh.material = [edgeMat, edgeMat, edgeMat, edgeMat, frontMat, backMatY];
            }
        }

        function mountLogoFidgetCard(scene, camera, spin) {
            scene.background = new THREE.Color(0x000000);
            camera.position.z = 5;

            const cardGroup = new THREE.Group();
            scene.add(cardGroup);
            spin.card = cardGroup;

            const placeholderFront = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, metalness: 0.05, roughness: 0.9 });
            const placeholderBack = new THREE.MeshStandardMaterial({ color: 0x121212, metalness: 0.05, roughness: 0.9 });

            if (spin.fxAxis) {
                const frontMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), placeholderFront);
                const backMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), placeholderBack);
                cardGroup.add(frontMesh, backMesh);
                spin.cardMesh = cardGroup;
                spin.fxFrontMesh = frontMesh;
                spin.fxBackMesh = backMesh;
                applyLogoFxPlaneLayout(spin);
            } else {
                const edgeMat = new THREE.MeshStandardMaterial({ color: 0x141414, metalness: 0.15, roughness: 0.85 });
                const cardDepth = 0.035;
                const cardMesh = new THREE.Mesh(
                    new THREE.BoxGeometry(1, 1, cardDepth),
                    [edgeMat, edgeMat, edgeMat, edgeMat, placeholderFront, placeholderBack]
                );
                cardGroup.add(cardMesh);
                spin.cardMesh = cardMesh;
                spin.edgeMat = edgeMat;
            }

            const amb = new THREE.AmbientLight(0xffffff, 0.9);
            const key = new THREE.DirectionalLight(0xffffff, 0.55);
            key.position.set(1.5, 2.5, 6);
            scene.add(amb, key);

            const fitCardToView = () => {
                const aspect = window.innerWidth / Math.max(1, window.innerHeight);
                const vFov = (camera.fov * Math.PI) / 180;
                const dist = Math.abs(camera.position.z);
                const viewH = 2 * Math.tan(vFov / 2) * dist;
                const viewW = viewH * aspect;
                const imgAspect = Math.max(0.05, spin.imgAspect || 1);
                let w;
                let h;
                if (viewW / viewH > imgAspect) {
                    h = viewH;
                    w = h * imgAspect;
                } else {
                    w = viewW;
                    h = w / imgAspect;
                }
                w *= 0.98;
                h *= 0.98;
                spin.cardMesh.scale.set(w, h, 1);
            };

            const onResize = () => fitCardToView();
            window.addEventListener('resize', onResize);
            const prevDispose = spin.dispose;
            spin.dispose = () => {
                window.removeEventListener('resize', onResize);
                spin.textures.forEach((t) => { try { t.dispose(); } catch (_) {} });
                spin.materials.forEach((m) => { try { m.dispose(); } catch (_) {} });
                spin.textures = [];
                spin.materials = [];
                if (typeof prevDispose === 'function') prevDispose();
            };

            const loader = new THREE.TextureLoader();
            const placeholderMats = spin.fxAxis
                ? [placeholderFront, placeholderBack]
                : [spin.edgeMat, placeholderFront, placeholderBack];
            loader.load(
                LOGO_FIDGET_IMAGE,
                (tex) => {
                    assignLogoFidgetMaterials(spin, tex, placeholderMats);
                    fitCardToView();
                },
                undefined,
                () => { fitCardToView(); }
            );
            fitCardToView();

            return cardGroup;
        }

        function stepLogoFidgetSpin(spin, cardGroup, dt) {
            const step = dt / 16.67;
            if (spin.homeActive) {
                const t = (performance.now() - spin.homeStart) / Math.max(1, spin.homeDuration);
                if (t >= 1) {
                    spin.rotX = spin.homeToX;
                    spin.rotY = spin.homeToY;
                    spin.homeActive = false;
                    spin.touchInteraction();
                } else {
                    const ease = 1 - Math.pow(1 - t, 3);
                    spin.rotX = spin.homeFromX + (spin.homeToX - spin.homeFromX) * ease;
                    spin.rotY = spin.homeFromY + (spin.homeToY - spin.homeFromY) * ease;
                }
            } else {
                const friction = Math.pow(0.9, step);
                spin.rotX += spin.velX * step;
                spin.rotY += spin.velY * step;
                spin.velX *= friction;
                spin.velY *= friction;
                if (Math.abs(spin.velX) < 1e-5) spin.velX = 0;
                if (Math.abs(spin.velY) < 1e-5) spin.velY = 0;
                const moving = Math.abs(spin.velX) > 1e-4 || Math.abs(spin.velY) > 1e-4;
                if (moving) {
                    spin.touchInteraction();
                } else {
                    const target = getNearestFlatTargets(spin);
                    const settled = Math.abs(spin.rotX - target.x) < 0.02
                        && Math.abs(spin.rotY - target.y) < 0.02;
                    const idleMs = performance.now() - (spin.lastInteractionAt || 0);
                    if (!settled && idleMs >= LOGO_IDLE_SETTLE_MS) {
                        spin.homeToNearestFlat();
                    }
                }
            }
            if (spin.fxAxis) {
                if (spin.spinAxis === 'y') {
                    spin.rotX = 0;
                    spin.velX = 0;
                } else {
                    spin.rotY = 0;
                    spin.velY = 0;
                }
            }
            cardGroup.rotation.x = spin.rotX;
            cardGroup.rotation.y = spin.rotY;
            updateLogoCardBackFace(spin);
        }

        const sceneLogo = (scene, camera) => {
            const spin = createLogoFidgetSpinState({ fxAxis: false });
            scene.userData.logoSpin = spin;
            const cardGroup = mountLogoFidgetCard(scene, camera, spin);
            let lastT = performance.now();
            return () => {
                const now = performance.now();
                const dt = Math.min(48, Math.max(1, now - lastT));
                lastT = now;
                stepLogoFidgetSpin(spin, cardGroup, dt);
            };
        };

        const sceneLogoFx = (scene, camera) => {
            const spin = createLogoFidgetSpinState({ fxAxis: true });
            scene.userData.logoSpin = spin;
            const cardGroup = mountLogoFidgetCard(scene, camera, spin);
            let lastT = performance.now();
            return () => {
                const now = performance.now();
                const dt = Math.min(48, Math.max(1, now - lastT));
                lastT = now;
                stepLogoFidgetSpin(spin, cardGroup, dt);
            };
        };

        // Circular Audio Bars
        const sceneBarsCircle = (scene, camera) => {
            const group = new THREE.Group();
            const bars = [];
            const num = 72;
            const radius = 3.0;
            for (let i = 0; i < num; i++) {
                const geo = new THREE.BoxGeometry(0.15, 0.4, 0.15);
                const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x101010, metalness: 0.2, roughness: 0.5 });
                const m = new THREE.Mesh(geo, mat);
                const a = (i / num) * Math.PI * 2;
                m.position.set(Math.cos(a) * radius, Math.sin(a) * radius, 0);
                m.lookAt(0, 0, 0);
                group.add(m); bars.push(m);
            }
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const pl = new THREE.PointLight(0x88bbff, 1.2, 40); pl.position.set(0, 0, 6); scene.add(pl, group);
            camera.position.z = 8;
            return (data, time) => {
                const step = Math.floor(data.length / bars.length);
                const t = time * 0.0005;
                bars.forEach((b, i) => {
                    const v = data[i * step] / 255;
                    const h = 0.2 + v * 2.8;
                    b.scale.y = h;
                    const hue = (i / bars.length + t) % 1;
                    b.material.color.setHSL(hue, 1, 0.55);
                    b.material.emissive.setHSL(hue, 1, 0.2);
                });
                group.rotation.z += 0.002;
            };
        };

        // 3D Audio Bars (radial tiers)
        const sceneBars3D = (scene, camera) => {
            const group = new THREE.Group();
            const tiers = 4;
            const perTier = 48;
            const bars = [];
            for (let t = 0; t < tiers; t++) {
                const r = 1.5 + t * 0.9;
                for (let i = 0; i < perTier; i++) {
                    const a = (i / perTier) * Math.PI * 2;
                    const geo = new THREE.BoxGeometry(0.12 + t*0.02, 0.5, 0.12 + t*0.02);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.3, roughness: 0.5, emissive: 0x000000 });
                    const m = new THREE.Mesh(geo, mat);
                    m.position.set(Math.cos(a) * r, Math.sin(a) * r, -t * 0.6);
                    m.lookAt(0, 0, -t * 0.6 - 1);
                    group.add(m); bars.push(m);
                }
            }
            const amb = new THREE.AmbientLight(0xffffff, 0.35);
            const pl = new THREE.PointLight(0xff88cc, 1.4, 60); pl.position.set(0, 0, 8);
            scene.add(amb, pl, group);
            camera.position.z = 10;
            return (data, time) => {
                const step = Math.floor(data.length / bars.length);
                const t = time * 0.0006;
                bars.forEach((b, i) => {
                    const v = data[(i * step) % data.length] / 255;
                    const h = 0.2 + v * 3.2;
                    b.scale.y = h;
                    b.position.z = -Math.floor(i / perTier) * 0.6 - v * 0.8;
                    const hue = (i / bars.length + t) % 1;
                    b.material.color.setHSL(hue, 1, 0.55);
                    b.material.emissive.setHSL(hue, 1, 0.15);
                });
                group.rotation.z = Math.sin(t * 0.8) * 0.2;
                group.rotation.x = Math.cos(t * 0.6) * 0.15;
            };
        };

        // Blank (no visualizer, solid black background)
        const sceneBlank = (scene /*, camera, composer */) => {
            try {
                scene.background = new THREE.Color(0x000000);
            } catch(_) {}
            return () => { /* no-op */ };
        };

        // Bars Vortex (wormhole made of bars)
        const sceneBarsVortex = (scene, camera) => {
            const group = new THREE.Group();
            const rings = [];
            const bars = [];
            const ringCount = 24;      // depth of the tunnel
            const barsPerRing = 56;    // bars around the ring
            const startRadius = 3.0;
            const endRadius = 0.6;     // taper to a small radius to form a wormhole
            const depthStep = 0.4;     // spacing between rings along -Z
            const twistPerRing = 0.18; // spiral twist down the tunnel
            for (let r = 0; r < ringCount; r++) {
                const radius = startRadius + (endRadius - startRadius) * (r / (ringCount - 1));
                const ring = [];
                for (let i = 0; i < barsPerRing; i++) {
                    const angle = (i / barsPerRing) * Math.PI * 2 + r * twistPerRing;
                    const geo = new THREE.BoxGeometry(0.08, 0.5, 0.08);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x000000, metalness: 0.35, roughness: 0.55 });
                    const m = new THREE.Mesh(geo, mat);
                    m.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, -r * depthStep);
                    m.lookAt(0, 0, -r * depthStep - 1);
                    group.add(m);
                    bars.push(m);
                    ring.push(m);
                }
                rings.push({ radius, meshes: ring });
            }
            // Lighting
            const amb = new THREE.AmbientLight(0xffffff, 0.3);
            const pl1 = new THREE.PointLight(0x66ccff, 1.3, 80); pl1.position.set(0, 0, 8);
            const pl2 = new THREE.PointLight(0xff66aa, 1.1, 80); pl2.position.set(0, 0, -ringCount * depthStep * 0.5);
            scene.add(amb, pl1, pl2, group);
            camera.position.z = 8.5;
            // Animation
            let scrollZ = 0;
            // Smoothing buffers
            const smoothedVals = new Array(ringCount * barsPerRing).fill(0);
            let smoothedBass = 0;
            const smooth = (prev, next, alpha) => prev + (next - prev) * alpha; // alpha ~ 0.15
            return (data, time) => {
                const t = time * 0.0006;
                const step = Math.max(1, Math.floor(data.length / bars.length));
                // Smooth bass to avoid jerkiness in forward motion
            const bassRaw = (data[8] || 0) / 255;
            smoothedBass = smooth(smoothedBass, bassRaw, 0.08);
                // Create a forward motion through the tunnel (smoothed)
            scrollZ += 0.025 + smoothedBass * 0.06;
                const baseZShift = scrollZ % depthStep;
                bars.forEach((b, i) => {
                    const vRaw = (data[(i * step) % data.length] || 0) / 255;
                    const v = smoothedVals[i] = smooth(smoothedVals[i], vRaw, 0.08);
                    const h = 0.22 + v * 2.0; // reduced range for steadier bars
                    b.scale.y = h;
                    // slight breathing of radius per ring by mids
                    const ringIdx = Math.floor(i / barsPerRing);
                    const mid = smooth(0, (data[24] || 0) / 255, 0.08);
                    const ring = rings[ringIdx];
                    const frac = ringIdx / (ringCount - 1);
                    const breathe = 0.04 * Math.sin(t * 1.8 + ringIdx * 0.35) * (0.5 + mid * 0.5);
                    const angle = (i % barsPerRing) / barsPerRing * Math.PI * 2 + ringIdx * twistPerRing + t * 0.8;
                    const cr = Math.max(0.1, ring.radius * (1.0 + breathe));
                    b.position.x = Math.cos(angle) * cr;
                    b.position.y = Math.sin(angle) * cr;
                    // continuous flow along z to simulate vortex pull
                    const z = -ringIdx * depthStep + baseZShift - v * 0.35; // less aggressive z wobble
                    b.position.z = z;
                    b.lookAt(0, 0, z - 1);
                    // color cycle along tunnel
                    const hue = (frac + t * 0.06 + v * 0.12) % 1;
                    b.material.color.setHSL(hue, 1, 0.55);
                    b.material.emissive.setHSL(hue, 1, 0.15);
                });
                // gentle whole-tunnel rotation for a wormhole effect
                group.rotation.z = Math.sin(t * 0.55) * 0.22;
                group.rotation.x = Math.cos(t * 0.45) * 0.14;
            };
        };
        // High-energy neon sphere variants
        const sceneElectroSphere = (scene, camera) => {
            const geo = new THREE.IcosahedronGeometry(1.6, 64);
            const mat = new THREE.MeshStandardMaterial({ color: 0x66ccff, metalness: 0.8, roughness: 0.2, emissive: 0x000000 });
            const mesh = new THREE.Mesh(geo, mat);
            scene.add(mesh);
            const base = geo.attributes.position.array.slice();
            const noise = new SimplexNoise();
            const light = new THREE.PointLight(0x66ccff, 2.2, 40);
            light.position.set(2, 2, 4);
            scene.add(new THREE.AmbientLight(0xffffff, 0.25), light);
            camera.position.z = 5;
            return (data, time) => {
                const t = time*0.0007;
                const arr = geo.attributes.position.array;
                const bass = data[8]/255;
                for(let i=0;i<arr.length;i+=3){
                    const x = base[i], y = base[i+1], z = base[i+2];
                    const n = noise.noise3d(x*0.8 + t*0.9, y*0.8 - t*0.7, z*0.8 + t*0.6);
                    const scale = 1.0 + n*(0.12 + bass*0.8);
                    arr[i] = x*scale; arr[i+1] = y*scale; arr[i+2] = z*scale;
                }
                geo.attributes.position.needsUpdate = true;
                geo.computeVertexNormals();
                mesh.rotation.y += 0.01 + bass*0.06;
                mesh.rotation.x += 0.004;
                const hue = (t*0.3 + bass*0.4) % 1;
                mat.color.setHSL(hue, 1, 0.55);
                light.intensity = 1.8 + bass*3.2;
            };
        };

        const scenePhotonShell = (scene, camera) => {
            // Points on a sphere shell
            const num = 4000;
            const positions = new Float32Array(num*3);
            for(let i=0;i<num;i++){
                const u = Math.random();
                const v = Math.random();
                const theta = 2*Math.PI*u;
                const phi = Math.acos(2*v-1);
                const r = 1.8;
                positions[i*3+0] = r*Math.sin(phi)*Math.cos(theta);
                positions[i*3+1] = r*Math.sin(phi)*Math.sin(theta);
                positions[i*3+2] = r*Math.cos(phi);
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.04, transparent: true, opacity: 0.9 });
            const points = new THREE.Points(geo, mat);
            scene.add(points);
            // Orbiting rings
            const rings = [];
            for(let i=0;i<3;i++){
                const ring = new THREE.TorusGeometry(2.0 + i*0.15, 0.03, 12, 200);
                const m = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 });
                const tor = new THREE.Mesh(ring, m);
                scene.add(tor);
                rings.push(tor);
            }
            camera.position.z = 6;
            return (data, time) => {
                const t = time*0.001;
                const bass = data[10]/255;
                points.rotation.y = t*0.4 + bass*0.6;
                points.rotation.x = Math.sin(t*0.4)*0.2;
                const hue = (t*0.2 + bass*0.4) % 1;
                mat.color.setHSL(hue, 1, 0.6);
                mat.size = 0.03 + bass*0.15;
                rings.forEach((r, i) => {
                    r.rotation.x = t*(i+1);
                    r.rotation.y = t*(i+1)*0.6;
                    r.material.color.setHSL((hue + i*0.1)%1, 1, 0.6);
                    r.material.opacity = 0.4 + bass*0.5;
                });
            };
        };

        const scenePulseOrb = (scene, camera) => {
            const layers = [];
            const layerCount = 5;
            for(let i=0;i<layerCount;i++){
                const radius = 1.2 + i*0.22;
                const geo = new THREE.SphereGeometry(radius, QUALITY.orbSegs, QUALITY.orbSegs);
                const mat = new THREE.MeshStandardMaterial({
                    color: 0x55ccff,
                    metalness: 0.4,
                    roughness: 0.6,
                    transparent: true,
                    opacity: 0.18 + i*0.08,
                    emissive: new THREE.Color(0x113355),
                    emissiveIntensity: 0.25 + i*0.05
                });
                const mesh = new THREE.Mesh(geo, mat);
                layers.push(mesh);
                scene.add(mesh);
            }
            // Dimmer, colored lighting to reduce central white clipping
            const ambient = new THREE.AmbientLight(0x223344, 0.18);
            const light = new THREE.PointLight(0x66ccff, 0.7, 60);
            light.position.set(1.5, 2.0, 6.0);
            scene.add(ambient, light);
            camera.position.z = 7.5;
            return (data, time) => {
                const t = time*0.001;
                const bass = data[6]/255;
                const mid = data[24]/255;
                layers.forEach((m, i) => {
                    const hue = (0.55 + 0.08*i + t*0.06) % 1;
                    m.material.color.setHSL(hue, 0.85, 0.55 - i*0.04);
                    const s = 1.0 + bass*(0.35 + i*0.1) + Math.sin(t*1.6 + i)*0.03;
                    m.scale.setScalar(s);
                    m.rotation.y += 0.002 + mid*0.015 + i*0.0015;
                    m.rotation.x += 0.0015 + i*0.001;
                    m.material.opacity = 0.18 + i*0.08 + bass*0.08;
                });
                light.intensity = 0.6 + bass*1.2;
                light.position.x = 1.5 + Math.sin(t*0.7)*1.0;
                light.position.y = 2.0 + Math.cos(t*0.5)*0.6;
            };
        };

        // New scenes
        const sceneStarfield = (scene, camera) => {
            const stars = [];
            const geo = new THREE.SphereGeometry(0.02, 6, 6);
            for(let i=0;i<1200;i++){
                const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: Math.random()*0.8+0.2 });
                const m = new THREE.Mesh(geo, mat);
                m.position.set((Math.random()-0.5)*80, (Math.random()-0.5)*80, (Math.random()-0.5)*80);
                scene.add(m); stars.push(m);
            }
            camera.position.z = 6;
            return (data, time) => {
                const bass = data[8]/255;
                const t = time*0.0002;
                stars.forEach((s, i) => {
                    s.position.z += 0.02 + bass*0.3;
                    if(s.position.z > 40) s.position.z = -40;
                    const hue = (i/1200 + t) % 1;
                    s.material.color.setHSL(hue, 0.7, 0.6);
                });
                camera.rotation.z = Math.sin(t*2)*0.1;
            };
        };

        const sceneWaveGrid = (scene, camera) => {
            const gridSize = 64;
            const spacing = 0.2;
            const group = new THREE.Group();
            const planes = [];
            for(let x=0;x<gridSize;x++){
                for(let y=0;y<gridSize;y++){
                    const geo = new THREE.PlaneGeometry(0.18, 0.18);
                    const mat = new THREE.MeshBasicMaterial({ color: 0x66ccff, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
                    const p = new THREE.Mesh(geo, mat);
                    p.position.set((x - gridSize/2)*spacing, (y - gridSize/2)*spacing, 0);
                    group.add(p);
                    planes.push(p);
                }
            }
            scene.add(group);
            camera.position.z = 8;
            return (data, time) => {
                const step = Math.floor(data.length / gridSize);
                const t = time * 0.0015;
                planes.forEach((p, idx) => {
                    const i = idx % gridSize;
                    const j = Math.floor(idx / gridSize);
                    const v = data[(i*step) % data.length]/255;
                    p.position.z = Math.sin(i*0.2 + j*0.25 + t*4) * (0.1 + v*0.8);
                    const hue = (v + (i+j)/ (gridSize*2) + t*0.2)%1;
                    p.material.color.setHSL(hue, 1, 0.5);
                    p.material.opacity = 0.35 + v*0.65;
                });
                group.rotation.x = Math.sin(t*0.2)*0.4;
                group.rotation.y = Math.cos(t*0.25)*0.4;
            };
        };

        const sceneParticles = (scene, camera) => {
            const num = 1500;
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(num*3);
            for(let i=0;i<num;i++){
                positions[i*3+0] = (Math.random()-0.5)*40;
                positions[i*3+1] = (Math.random()-0.5)*40;
                positions[i*3+2] = (Math.random()-0.5)*40;
            }
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.08, transparent: true, opacity: 0.8 });
            const points = new THREE.Points(geo, mat);
            scene.add(points);
            camera.position.z = 10;
            return (data, time) => {
                const bass = data[4]/255;
                points.rotation.y += 0.001 + bass*0.01;
                points.rotation.x += 0.0006;
                const hue = (time*0.0001)%1;
                points.material.color.setHSL(hue, 1, 0.6);
                points.material.size = 0.05 + bass*0.25;
            };
        };

        // --- Additional Tunnel Variants ---
        const sceneNeonTunnel = (scene, camera) => {
            const rings = [];
            const count = 36;
            for(let i=0;i<count;i++){
                const geo = new THREE.TorusGeometry(2.5, 0.08, 16, 120);
                const mat = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.6 });
                const m = new THREE.Mesh(geo, mat);
                m.position.z = -i*1.4;
                rings.push(m);
                scene.add(m);
            }
            camera.position.z = 4;
            return (data, time) => {
                const bass = data[8]/255;
                rings.forEach((r, i) => {
                    r.position.z += 0.12 + bass*0.6;
                    if(r.position.z > 3.5) r.position.z -= count*1.4;
                    const hue = ((i/count) + time*0.0002) % 1;
                    r.material.color.setHSL(hue, 1, 0.6);
                    r.rotation.x = time*0.0006 + i*0.03;
                });
                camera.rotation.z = Math.sin(time*0.0006)*0.15;
            };
        };

        const sceneTwistTunnel = (scene, camera) => {
            const group = new THREE.Group();
            const segs = [];
            for(let i=0;i<90;i++){
                const geo = new THREE.CylinderGeometry(2.2, 2.2, 0.4, 60, 1, true);
                const mat = new THREE.MeshBasicMaterial({ color: 0xff66ff, wireframe: true, transparent: true, opacity: 0.5 });
                const m = new THREE.Mesh(geo, mat);
                m.position.z = -i*0.9;
                m.rotation.z = i*0.1;
                group.add(m);
                segs.push(m);
            }
            scene.add(group);
            camera.position.z = 3.5;
            return (data, time) => {
                const mid = data[24]/255;
                segs.forEach((s, i) => {
                    s.position.z += 0.08 + mid*0.5;
                    if(s.position.z > 2.5) s.position.z -= 90*0.9;
                    s.rotation.z += 0.003 + mid*0.03;
                    const hue = ((i/90) + time*0.00015) % 1;
                    s.material.color.setHSL(hue, 1, 0.6);
                    s.material.opacity = 0.25 + mid*0.75;
                });
                group.rotation.y = Math.sin(time*0.0005)*0.3;
            };
        };

        const sceneParticleTunnel = (scene, camera) => {
            const num = 3000;
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(num*3);
            for(let i=0;i<num;i++){
                const r = 2 + Math.random()*2.5;
                const a = Math.random()*Math.PI*2;
                const z = -Math.random()*60;
                positions[i*3+0] = Math.cos(a)*r;
                positions[i*3+1] = Math.sin(a)*r;
                positions[i*3+2] = z;
            }
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.05, transparent: true, opacity: 0.9 });
            const points = new THREE.Points(geo, mat);
            scene.add(points);
            camera.position.z = 3;
            return (data, time) => {
                const bass = data[6]/255;
                const arr = geo.attributes.position.array;
                for(let i=0;i<arr.length;i+=3){
                    arr[i+2] += 0.22 + bass*1.6;
                    if(arr[i+2] > 2.5) arr[i+2] -= 62.5;
                }
                geo.attributes.position.needsUpdate = true;
                const hue = (time*0.0002)%1;
                mat.color.setHSL(hue, 1, 0.7);
                points.rotation.z += 0.001 + bass*0.02;
            };
        };

        const sceneGalaxy = (scene, camera) => {
            const count = QUALITY.galaxyCount;
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(count*3);
            const colors = new Float32Array(count*3);
            for(let i=0;i<count;i++){
                const r = Math.random() * 20;
                const angle = Math.random()*Math.PI*2;
                const arm = ((i%2) * 2 - 1) * (r*0.1);
                const x = Math.cos(angle + arm) * r;
                const y = (Math.random()-0.5) * 2;
                const z = Math.sin(angle + arm) * r;
                positions[i*3+0] = x; positions[i*3+1] = y; positions[i*3+2] = z;
                const c = new THREE.Color().setHSL((r/20)*0.6 + 0.2, 1, 0.6);
                colors[i*3+0] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
            }
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            const mat = new THREE.PointsMaterial({ size: 0.06, vertexColors: true, transparent: true, opacity: 0.9 });
            const points = new THREE.Points(geo, mat);
            scene.add(points);
            camera.position.z = 16;
            return (data, time) => {
                const bass = data[10]/255;
                points.rotation.z = time*0.0001 + bass*0.2;
                mat.size = 0.04 + bass*0.18;
            };
        };

        const sceneKaleidoRings = (scene, camera) => {
            const group = new THREE.Group();
            const rings = [];
            for(let i=0;i<14;i++){
                const geo = new THREE.TorusGeometry(0.8 + i*0.35, 0.04, 16, 120);
                const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
                const torus = new THREE.Mesh(geo, mat);
                group.add(torus);
                rings.push(torus);
            }
            scene.add(group);
            camera.position.z = 8;
            return (data, time) => {
                const step = Math.floor(data.length / rings.length);
                const t = time*0.0006;
                rings.forEach((r, i) => {
                    const v = data[i*step]/255;
                    r.rotation.x = t*(i%3+1);
                    r.rotation.y = -t*(i%2+1);
                    r.material.color.setHSL(((i/rings.length)+t)%1, 1, 0.55);
                    r.material.opacity = 0.3 + v*0.7;
                    r.scale.setScalar(1 + v*0.4);
                });
                group.rotation.z = Math.sin(t*2)*0.3;
            };
        };

        const sceneTerrain = (scene, camera) => {
            const size = 80;
            const segments = 120;
            const geo = new THREE.PlaneGeometry(size, size, segments, segments);
            const mat = new THREE.MeshStandardMaterial({ color: 0x4444ff, wireframe: false, metalness: 0.2, roughness: 0.9, side: THREE.DoubleSide });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.rotation.x = -Math.PI/2.5;
            scene.add(mesh);
            const light = new THREE.PointLight(0x88ccff, 2, 120);
            light.position.set(0, 20, 20);
            scene.add(light, new THREE.AmbientLight(0xffffff, 0.35));
            const noise = new SimplexNoise();
            camera.position.set(0, 10, 22);
            return (data, time) => {
                const arr = geo.attributes.position.array;
                const t = time*0.0003;
                const bass = data[20]/255;
                for(let i=0;i<arr.length;i+=3){
                    const x = arr[i+0];
                    const y = arr[i+1];
                    const n = noise.noise3d(x*0.05, y*0.05, t)*2.0;
                    arr[i+2] = n + bass*3.0;
                }
                geo.attributes.position.needsUpdate = true;
                geo.computeVertexNormals();
                mesh.material.color.setHSL((t*0.2 + bass*0.3)%1, 0.8, 0.5);
                light.intensity = 1.5 + bass*2.5;
            };
        };

        const sceneHexGrid = (scene, camera) => {
            const group = new THREE.Group();
            const hexes = [];
            const radius = 0.25;
            const rows = 22, cols = 22;
            for(let r=0;r<rows;r++){
                for(let c=0;c<cols;c++){
                    const x = (c + (r%2?0.5:0)) * radius*1.8 - cols*radius*0.9;
                    const y = r * radius*1.6 - rows*radius*0.8;
                    const geo = new THREE.CylinderGeometry(radius, radius, 0.2, 6, 1);
                    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x000000, metalness: 0.1, roughness: 0.7 });
                    const m = new THREE.Mesh(geo, mat);
                    m.position.set(x, y, 0);
                    m.rotation.x = Math.PI/2;
                    group.add(m);
                    hexes.push(m);
                }
            }
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const light = new THREE.PointLight(0xff66ff, 1.6, 60);
            light.position.set(0, 0, 10);
            scene.add(light, group);
            camera.position.z = 10;
            return (data, time) => {
                const step = Math.floor(data.length / hexes.length) || 1;
                const t = time*0.001;
                hexes.forEach((h, i) => {
                    const v = data[(i*step)%data.length]/255;
                    h.scale.z = 0.4 + v*2.8;
                    const hue = (v*0.6 + ((i%cols)/cols) + t*0.1)%1;
                    h.material.color.setHSL(hue, 1, 0.5);
                    h.material.emissive.setHSL(hue, 1, 0.2);
                });
                group.rotation.z = Math.sin(t*0.5)*0.2;
            };
        };

        const sceneRibbons = (scene, camera) => {
            const group = new THREE.Group();
            const ribbons = [];
            const createRibbon = (color) => {
                const points = [];
                for(let i=0;i<80;i++){
                    points.push(new THREE.Vector3(i*0.08, Math.sin(i*0.2)*0.4, 0));
                }
                const curve = new THREE.CatmullRomCurve3(points);
                const tube = new THREE.TubeGeometry(curve, 200, 0.03, 8, false);
                const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.2, emissive: 0x000000 });
                const mesh = new THREE.Mesh(tube, mat);
                return mesh;
            };
            for(let i=0;i<6;i++){
                const hue = i/6;
                const col = new THREE.Color().setHSL(hue, 1, 0.6);
                const m = createRibbon(col);
                m.position.y = (i-3)*0.4;
                m.position.x = -3;
                group.add(m);
                ribbons.push(m);
            }
            scene.add(new THREE.AmbientLight(0xffffff, 0.6), group);
            const light = new THREE.PointLight(0x66ffff, 2, 40);
            light.position.set(0, 2, 8);
            scene.add(light);
            camera.position.set(0, 2, 10);
            return (data, time) => {
                const bass = data[12]/255;
                const t = time*0.0012;
                ribbons.forEach((m, i) => {
                    m.rotation.y = t*(i%3+1);
                    m.rotation.x = Math.sin(t + i)*0.3;
                    m.position.x = -3 + Math.sin(t*0.7 + i)*2.5 + bass*2.0;
                });
                group.rotation.z = Math.sin(t*0.6)*0.2;
                light.intensity = 1.2 + bass*2.0;
            };
        };

        // --- Infinity Pattern Tunnels ---
        const sceneInfinityTunnel = (scene, camera, composer) => {
            const rings = [];
            const count = 64;
            const baseRadius = 2.2;
            for(let i=0;i<count;i++){
                const geo = new THREE.TorusGeometry(baseRadius, 0.06, 12, 120);
                const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 });
                const m = new THREE.Mesh(geo, mat);
                m.position.z = -i*1.0;
                rings.push(m);
                scene.add(m);
            }
            const after = new AfterimagePass(0.88);
            composer.addPass(after);
            camera.position.z = 4;
            return (data, time) => {
                const t = time * 0.001;
                const bass = data[10]/255;
                rings.forEach((r, i) => {
                    // Figure-eight cross-section via Lissajous pattern
                    const phase = (i*0.22) + t*1.2;
                    r.position.x = Math.sin(phase) * (baseRadius * 0.6);
                    r.position.y = Math.sin(phase*2.0) * (baseRadius * 0.35);
                    r.position.z += 0.18 + bass*0.8;
                    if(r.position.z > 2.5) r.position.z -= count*1.0;
                    const hue = ((i/count) + t*0.05) % 1;
                    r.material.color.setHSL(hue, 1, 0.6);
                    r.material.opacity = 0.35 + bass*0.6;
                    r.rotation.x = phase*0.2;
                    r.rotation.y = -phase*0.15;
                });
                camera.rotation.z = Math.sin(t*0.8)*0.2;
            };
        };

        const sceneInfinityMirror = (scene, camera, composer) => {
            const frames = [];
            const count = 40;
            for(let i=0;i<count;i++){
                const geo = new THREE.RingGeometry(0.8, 3.0, 6, 1);
                const mat = new THREE.MeshBasicMaterial({ color: 0x88ccff, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
                const ring = new THREE.Mesh(geo, mat);
                ring.position.z = -i*1.4;
                frames.push(ring);
                scene.add(ring);
            }
            const after = new AfterimagePass(0.9);
            composer.addPass(after);
            camera.position.z = 3.8;
            return (data, time) => {
                const t = time*0.001;
                const mid = data[24]/255;
                frames.forEach((f, i) => {
                    f.position.z += 0.14 + mid*0.7;
                    if(f.position.z > 2.5) f.position.z -= count*1.4;
                    const s = 0.9 + Math.sin(i*0.2 + t*2.0)*0.08 + mid*0.2;
                    f.scale.set(s, s, s);
                    f.rotation.z = t*0.7 + i*0.05;
                    const hue = ((i/count) + t*0.1) % 1;
                    f.material.color.setHSL(hue, 1, 0.65);
                    f.material.opacity = 0.3 + mid*0.7;
                });
            };
        };
        // --- Kaleidoscope Shader ---
        const KaleidoShader = {
            uniforms: {
                tDiffuse: { value: null },
                sides: { value: 6.0 },
                angle: { value: 0.0 },
                rotation: { value: 0.0 },
                time: { value: 0.0 },
                gain: { value: 0.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float sides;
                uniform float angle;
                uniform float rotation;
                uniform float time;
                uniform float gain;
                varying vec2 vUv;
                
                float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
                float noise(vec2 p){
                    vec2 i = floor(p);
                    vec2 f = fract(p);
                    float a = hash(i);
                    float b = hash(i + vec2(1.0, 0.0));
                    float c = hash(i + vec2(0.0, 1.0));
                    float d = hash(i + vec2(1.0, 1.0));
                    vec2 u = f*f*(3.0-2.0*f);
                    return mix(a, b, u.x) + (c - a)*u.y*(1.0 - u.x) + (d - b)*u.x*u.y;
                }
                float fbm(vec2 p){
                    float v = 0.0;
                    float a = 0.5;
                    for(int i=0;i<5;i++){
                        v += a * noise(p);
                        p = p * 2.0 + vec2(23.1, 17.7);
                        a *= 0.5;
                    }
                    return v;
                }
                vec3 palette(float t){
                    return 0.5 + 0.5*cos(6.2831853*(vec3(0.0,0.33,0.67) + t));
                }
                vec4 fractalColor(vec2 uv){
                    // Centered uv for fractal evolution
                    vec2 q = uv - 0.5;
                    float r = length(q);
                    float ang = atan(q.y, q.x);
                    vec2 p = vec2(cos(ang), sin(ang)) * (r * 2.2);
                    float f = fbm(p * 1.8 + vec2(time*0.15, -time*0.11 + gain*0.3));
                    vec3 col = palette(f + r*0.25 + gain*0.2);
                    // Radial darkening to avoid bright center
                    float vign = smoothstep(0.95, 0.3, r);
                    return vec4(col * vign, 1.0);
                }
                void main() {
                    vec2 uv = vUv - 0.5;
                    float r = length(uv);
                    float a = atan(uv.y, uv.x) + rotation;
                    float sector = 6.28318530718 / max(1.0, sides);
                    a = mod(a, sector);
                    a = abs(a - sector * 0.5);
                    a += angle;
                    vec2 pos = vec2(cos(a), sin(a)) * r;
                    vec2 sampleUv = pos + 0.5;
                    bool inBounds = all(greaterThanEqual(sampleUv, vec2(0.0))) && all(lessThanEqual(sampleUv, vec2(1.0)));
                    vec4 src = texture2D(tDiffuse, clamp(sampleUv, 0.0, 1.0));
                    vec4 fractCol = fractalColor(sampleUv);
                    // Blend: use fractal when out-of-bounds, otherwise softly mix by radial factor to eliminate white cores
                    float mixAmt = smoothstep(0.0, 0.35, r) * 0.7 + gain*0.2;
                    vec4 col = mix(src, fractCol, mixAmt);
                    if(!inBounds){
                        col = fractCol;
                    }
                    gl_FragColor = col;
                }
            `
        };

        const RadialZoomShader = {
            uniforms: {
                tDiffuse: { value: null },
                center: { value: new THREE.Vector2(0.5, 0.5) },
                strength: { value: 0.2 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform vec2 center;
                uniform float strength;
                varying vec2 vUv;
                void main() {
                    vec4 color = vec4(0.0);
                    float total = 0.0;
                    vec2 toCenter = center - vUv;
                    float offset = strength * 0.02;
                    for (float t = 0.0; t <= 1.0; t += 0.1) {
                        vec2 sampleUv = vUv + toCenter * t * offset;
                        color += texture2D(tDiffuse, sampleUv);
                        total += 1.0;
                    }
                    gl_FragColor = color / total;
                }
            `
        };

        // Helper: build a simple colorful source for kaleidoscope
        function buildKaleidoSource(scene) {
            const group = new THREE.Group();
            const num = 24;
            for(let i=0;i<num;i++){
                const hue = i/num;
                const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(hue, 1, 0.6) });
                const geo = new THREE.TorusKnotGeometry(0.5 + (i%6)*0.05, 0.12, 90, 18, 2 + (i%3), 3 + (i%4));
                const m = new THREE.Mesh(geo, mat);
                m.position.set(Math.cos(i)*2.2, Math.sin(i*1.3)*1.2, -1 - (i%5)*0.2);
                m.rotation.x = i*0.2;
                m.rotation.y = i*0.15;
                group.add(m);
            }
            scene.add(group);
            return group;
        }

        const sceneKaleido = (sides) => (scene, camera, composer) => {
            const group = buildKaleidoSource(scene);
            const pass = new ShaderPass(KaleidoShader);
            pass.uniforms.sides.value = sides;
            composer.addPass(pass);
            camera.position.z = 4;
            return (data, time) => {
                const bass = data[12]/255;
                group.rotation.y = time*0.0006 + bass*0.2;
                group.rotation.x = Math.sin(time*0.0004)*0.3;
                pass.uniforms.rotation.value = time*0.0002 + bass*0.6;
                pass.uniforms.angle.value = Math.sin(time*0.0005)*0.5;
                pass.uniforms.time.value = time*0.001;
                pass.uniforms.gain.value = bass;
            };
        };

        const sceneKaleidoSpiral = (scene, camera, composer) => {
            const group = buildKaleidoSource(scene);
            const pass = new ShaderPass(KaleidoShader);
            composer.addPass(pass);
            camera.position.z = 5;
            return (data, time) => {
                const bass = data[6]/255;
                pass.uniforms.sides.value = 5.0 + Math.floor((Math.sin(time*0.0007)+1.0)*3.5); // 5..12
                pass.uniforms.rotation.value = time*0.00035 + bass*0.8;
                pass.uniforms.angle.value = Math.cos(time*0.0006)*0.8;
                group.rotation.y += 0.002 + bass*0.02;
                group.rotation.z = Math.sin(time*0.0003)*0.4;
                pass.uniforms.time.value = time*0.001;
                pass.uniforms.gain.value = (data[12]+data[24])/(2.0*255.0);
            };
        };

        const sceneKaleidoLayered = (scene, camera, composer) => {
            const group = buildKaleidoSource(scene);
            const k1 = new ShaderPass(KaleidoShader);
            const k2 = new ShaderPass(KaleidoShader);
            k1.uniforms.sides.value = 7.0;
            k2.uniforms.sides.value = 11.0;
            composer.addPass(k1);
            composer.addPass(k2);
            camera.position.z = 5;
            // Smoothed, time-integrated rotations to avoid jitter
            let rot1 = 0, rot2 = 0, rotY = 0, rotX = 0;
            let speed1 = 0.00022, speed2 = -0.00026, speedY = 0.00028, speedX = 0.00018;
            let gSmooth = 0;
            let lastTime = 0;
            return (data, time) => {
                const gRaw = (data[6] + data[10] + data[12] + data[24]) / (4.0 * 255.0);
                gSmooth = gSmooth + (gRaw - gSmooth) * 0.08; // low-pass filter
                const dt = lastTime ? (time - lastTime) : 16.0; // ms
                lastTime = time;
                // Gently modulate speeds with smoothed gain
                const target1 = 0.00022 + gSmooth * 0.00040;
                const target2 = -0.00026 - gSmooth * 0.00035;
                const targetY = 0.00028 + gSmooth * 0.00025;
                const targetX = 0.00018 + gSmooth * 0.00010;
                speed1 += (target1 - speed1) * 0.05;
                speed2 += (target2 - speed2) * 0.05;
                speedY += (targetY - speedY) * 0.05;
                speedX += (targetX - speedX) * 0.05;
                // Integrate rotations (smooth)
                rot1 += speed1 * dt;
                rot2 += speed2 * dt;
                rotY += speedY * dt;
                rotX += speedX * dt;
                group.rotation.x = rotX;
                group.rotation.y = rotY;
                k1.uniforms.rotation.value = rot1;
                k2.uniforms.rotation.value = rot2;
                // Slow, continuous angle sweep
                k1.uniforms.angle.value = Math.sin(time * 0.00025) * 0.55;
                k2.uniforms.angle.value = Math.cos(time * 0.00023) * 0.55;
                // Pass time/gain to shaders
                const tSec = time * 0.001;
                k1.uniforms.time.value = tSec;
                k2.uniforms.time.value = tSec;
                k1.uniforms.gain.value = gSmooth;
                k2.uniforms.gain.value = gSmooth * 0.85;
            };
        };

        const sceneKaleidoZoom = (scene, camera, composer) => {
            const group = buildKaleidoSource(scene);
            const kaleido = new ShaderPass(KaleidoShader);
            const zoom = new ShaderPass(RadialZoomShader);
            composer.addPass(kaleido);
            composer.addPass(zoom);
            camera.position.z = 4.5;
            return (data, time) => {
                const bass = data[14]/255;
                group.rotation.y = time*0.0005 + bass*0.3;
                kaleido.uniforms.sides.value = 10.0;
                kaleido.uniforms.rotation.value = time*0.00025;
                zoom.uniforms.center.value.set(0.5, 0.5);
                zoom.uniforms.strength.value = 0.1 + bass*0.6;
                kaleido.uniforms.time.value = time*0.001;
                kaleido.uniforms.gain.value = (data[10]+data[14]+data[18])/(3.0*255.0);
            };
        };

        // --- Self-Replicating / Fractal Scenes ---
        const sceneMengerSponge = (scene, camera) => {
            const depth = 2; // keep counts reasonable
            const positions = [];
            function addLevel(center, size, d){
                if(d === 0){
                    positions.push({ center, size });
                    return;
                }
                const step = size / 3;
                for(let x=0;x<3;x++){
                    for(let y=0;y<3;y++){
                        for(let z=0;z<3;z++){
                            // Skip the center cross sections
                            const mid = (x===1) + (y===1) + (z===1);
                            if(mid >= 2) continue;
                            const nx = center.x + (x-1)*step;
                            const ny = center.y + (y-1)*step;
                            const nz = center.z + (z-1)*step;
                            addLevel(new THREE.Vector3(nx, ny, nz), step, d-1);
                        }
                    }
                }
            }
            addLevel(new THREE.Vector3(0,0,0), 6, depth);
            const count = positions.length;
            const geo = new THREE.BoxGeometry(1,1,1);
            const mat = new THREE.MeshStandardMaterial({ color: 0x66ccff, metalness: 0.3, roughness: 0.6, emissive: 0x000000 });
            const mesh = new THREE.InstancedMesh(geo, mat, count);
            const dummy = new THREE.Object3D();
            positions.forEach((p, i) => {
                dummy.position.copy(p.center);
                dummy.scale.setScalar(p.size * 0.95);
                dummy.updateMatrix();
                mesh.setMatrixAt(i, dummy.matrix);
            });
            mesh.instanceMatrix.needsUpdate = true;
            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const light = new THREE.PointLight(0xffffff, 2, 100);
            light.position.set(6, 8, 10);
            scene.add(light, mesh);
            camera.position.set(0, 0, 14);
            return (data, time) => {
                const g = (data[6]+data[12]+data[24])/(3*255);
                mesh.rotation.x = time*0.0003 + g*0.4;
                mesh.rotation.y = time*0.0004 + g*0.5;
                mat.emissiveIntensity = 0.2 + g*2.0;
                const hue = (time*0.00008 + g*0.3) % 1;
                mat.color.setHSL(hue, 0.7, 0.6);
            };
        };

        const sceneSierpinskiTetra = (scene, camera) => {
            // Base tetra geometry
            const r = 1;
            const v = [
                new THREE.Vector3(1, 1, 1),
                new THREE.Vector3(-1, -1, 1),
                new THREE.Vector3(-1, 1, -1),
                new THREE.Vector3(1, -1, -1)
            ];
            const base = new THREE.TetrahedronGeometry(r, 0);
            const mat = new THREE.MeshStandardMaterial({ color: 0xff88aa, metalness: 0.2, roughness: 0.7 });
            const depth = 4;
            const transforms = [];
            function add(level, center, scale){
                if(level === 0){
                    transforms.push({ center, scale });
                    return;
                }
                const s = scale * 0.5;
                for(let i=0;i<4;i++){
                    add(level-1, new THREE.Vector3(
                        center.x + v[i].x * s,
                        center.y + v[i].y * s,
                        center.z + v[i].z * s
                    ), s);
                }
            }
            add(depth, new THREE.Vector3(0,0,0), 3.0);
            const inst = new THREE.InstancedMesh(base, mat, transforms.length);
            const dummy = new THREE.Object3D();
            transforms.forEach((t, i) => {
                dummy.position.copy(t.center);
                dummy.scale.setScalar(t.scale);
                dummy.updateMatrix();
                inst.setMatrixAt(i, dummy.matrix);
            });
            inst.instanceMatrix.needsUpdate = true;
            scene.add(new THREE.AmbientLight(0xffffff, 0.4));
            const light = new THREE.PointLight(0xff66ff, 2.4, 100);
            light.position.set(6, 6, 12);
            scene.add(light, inst);
            camera.position.set(0, 0, 16);
            return (data, time) => {
                const bass = data[8]/255;
                inst.rotation.y = time*0.00035 + bass*0.6;
                inst.rotation.x = time*0.0002;
                const hue = (time*0.00012) % 1;
                mat.color.setHSL(hue, 0.8, 0.6);
                light.intensity = 1.4 + bass*3.0;
            };
        };

        const sceneGameOfLife = (scene, camera) => {
            const W = 128, H = 128;
            const cells = new Uint8Array(W*H);
            const nextCells = new Uint8Array(W*H);
            // random init
            for(let i=0;i<W*H;i++){ cells[i] = Math.random() < 0.15 ? 1 : 0; }
            const data = new Uint8Array(W*H*3);
            const tex = new THREE.DataTexture(data, W, H, THREE.RGBFormat);
            tex.needsUpdate = true;
            const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.95 });
            const plane = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), mat);
            scene.add(plane);
            camera.position.z = 10;
            function step() {
                for(let y=0;y<H;y++){
                    for(let x=0;x<W;x++){
                        const idx = y*W + x;
                        let n = 0;
                        for(let dy=-1;dy<=1;dy++){
                            for(let dx=-1;dx<=1;dx++){
                                if(dx===0 && dy===0) continue;
                                const nx = (x+dx+W)%W;
                                const ny = (y+dy+H)%H;
                                n += cells[ny*W + nx];
                            }
                        }
                        const c = cells[idx];
                        nextCells[idx] = (c===1 && (n===2 || n===3)) || (c===0 && n===3) ? 1 : 0;
                    }
                }
                cells.set(nextCells);
            }
            let acc = 0;
            return (audio, time) => {
                const g = (audio[6]+audio[12]+audio[24])/(3*255);
                // step a few times per second + boost with audio
                acc += 0.016 + g*0.05;
                while(acc > 0.1){ step(); acc -= 0.1; }
                // write texture
                const t = time*0.0002;
                for(let y=0;y<H;y++){
                    for(let x=0;x<W;x++){
                        const idx = y*W + x;
                        const alive = cells[idx];
                        const i3 = idx*3;
                        const hue = (t + x/W*0.3 + y/H*0.2) % 1;
                        const c = alive ? hue : 0.0;
                        // hsv to rgb approx via palette
                        const r = 0.5 + 0.5*Math.cos(6.2831*(c + 0.0));
                        const gch = 0.5 + 0.5*Math.cos(6.2831*(c + 0.33));
                        const b = 0.5 + 0.5*Math.cos(6.2831*(c + 0.67));
                        data[i3+0] = Math.floor(r*255);
                        data[i3+1] = Math.floor(gch*255);
                        data[i3+2] = Math.floor(b*255);
                    }
                }
                tex.needsUpdate = true;
                plane.rotation.z = Math.sin(time*0.0003)*0.2;
                plane.scale.setScalar(1.0 + g*0.3);
            };
        };

        // Mandelbrot (shader)
        const sceneMandelbrot = (scene, camera) => {
            const uni = {
                time: { value: 0.0 },
                center: { value: new THREE.Vector2(-0.5, 0.0) },
                zoom: { value: 1.0 },
                gain: { value: 0.0 },
                aspect: { value: window.innerWidth / window.innerHeight }
            };
            const mat = new THREE.ShaderMaterial({
                uniforms: uni,
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
                    }
                `,
                fragmentShader: `
                    precision highp float;
                    varying vec2 vUv;
                    uniform vec2 center;
                    uniform float zoom;
                    uniform float time;
                    uniform float gain;
                    uniform float aspect;
                    vec3 palette(float t){
                        return 0.5 + 0.5*cos(6.28318*(vec3(0.0,0.33,0.67) + t));
                    }
                    void main(){
                        vec2 uv = (vUv - 0.5);
                        uv.x *= aspect;
                        vec2 c = center + uv / zoom;
                        vec2 z = vec2(0.0);
                        float it = 0.0;
                        const int MAX_IT = 200;
                        float maxIter = 60.0 + gain*120.0;
                        for(int i=0; i<MAX_IT; i++){
                            if(it > maxIter) break;
                            float x = (z.x*z.x - z.y*z.y) + c.x;
                            float y = (2.0*z.x*z.y) + c.y;
                            z = vec2(x,y);
                            if(dot(z,z) > 4.0) break;
                            it += 1.0;
                        }
                        float t = it / maxIter;
                        float sm = t + 1.0 - log2(log(length(z)))/log(2.0);
                        sm = clamp(sm, 0.0, 1.0);
                        vec3 col = palette(sm + time*0.02);
                        gl_FragColor = vec4(col * pow(t, 0.5), 1.0);
                    }
                `
            });
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
            scene.add(mesh);
            camera.position.z = 1;
            return (audio, time) => {
                const g = (audio[6]+audio[12]+audio[24])/(3.0*255.0);
                uni.time.value = time*0.001;
                uni.gain.value = g;
                const baseZoom = 0.85 + Math.sin(time*0.0001)*0.15;
                uni.zoom.value = 1.5 + baseZoom * (1.0 + g*4.0);
                uni.center.value.set(-0.5 + Math.sin(time*0.00008)*0.15, Math.cos(time*0.00006)*0.1);
                uni.aspect.value = window.innerWidth / window.innerHeight;
            };
        };

        // Julia Set (shader)
        const sceneJulia = (scene, camera) => {
            const uni = {
                time: { value: 0.0 },
                c: { value: new THREE.Vector2(-0.70176, -0.3842) },
                zoom: { value: 1.0 },
                gain: { value: 0.0 },
                aspect: { value: window.innerWidth / window.innerHeight }
            };
            const mat = new THREE.ShaderMaterial({
                uniforms: uni,
                vertexShader: `
                    varying vec2 vUv;
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
                    }
                `,
                fragmentShader: `
                    precision highp float;
                    varying vec2 vUv;
                    uniform vec2 c;
                    uniform float zoom;
                    uniform float time;
                    uniform float gain;
                    uniform float aspect;
                    vec3 palette(float t){
                        return 0.5 + 0.5*cos(6.28318*(vec3(0.0,0.33,0.67) + t));
                    }
                    void main(){
                        vec2 uv = (vUv - 0.5);
                        uv.x *= aspect;
                        vec2 z = uv / zoom * 2.0;
                        float it = 0.0;
                        const int MAX_IT = 200;
                        float maxIter = 60.0 + gain*120.0;
                        for(int i=0; i<MAX_IT; i++){
                            if(it > maxIter) break;
                            z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
                            if(dot(z,z) > 16.0) break;
                            it += 1.0;
                        }
                        float t = it / maxIter;
                        float sm = t + 1.0 - log2(log(length(z)))/log(2.0);
                        sm = clamp(sm, 0.0, 1.0);
                        vec3 col = palette(sm + time*0.02);
                        gl_FragColor = vec4(col * pow(t, 0.55), 1.0);
                    }
                `
            });
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
            scene.add(mesh);
            camera.position.z = 1;
            return (audio, time) => {
                const g = (audio[8]+audio[16]+audio[24])/(3.0*255.0);
                uni.time.value = time*0.001;
                uni.gain.value = g;
                uni.zoom.value = 1.4 + 0.6*Math.sin(time*0.0002) + g*2.2;
                const ang = time*0.00015;
                uni.c.value.set(Math.cos(ang)*0.7, Math.sin(ang*1.3)*0.5);
                uni.aspect.value = window.innerWidth / window.innerHeight;
            };
        };

        // Barnsley Fern (points)
        const sceneBarnsleyFern = (scene, camera) => {
            const num = QUALITY.fernCount;
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(num*3);
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.PointsMaterial({ color: 0x66ff99, size: 0.03, transparent: true, opacity: 0.9 });
            const points = new THREE.Points(geo, mat);
            scene.add(points);
            camera.position.z = 8;
            let x = 0, y = 0;
            function step() {
                const r = Math.random();
                let nx, ny;
                if (r < 0.01) { nx = 0.0; ny = 0.16*y; }
                else if (r < 0.86) { nx = 0.85*x + 0.04*y; ny = -0.04*x + 0.85*y + 1.6; }
                else if (r < 0.93) { nx = 0.2*x - 0.26*y; ny = 0.23*x + 0.22*y + 1.6; }
                else { nx = -0.15*x + 0.28*y; ny = 0.26*x + 0.24*y + 0.44; }
                x = nx; y = ny;
            }
            return (audio, time) => {
                const g = (audio[6]+audio[12]+audio[24])/(3.0*255.0);
                for (let i=0;i<num;i++) {
                    step();
                    positions[i*3+0] = x*1.6 - 2.0;
                    positions[i*3+1] = y*1.6 - 3.0;
                    positions[i*3+2] = 0;
                }
                geo.attributes.position.needsUpdate = true;
                const hue = (time*0.0001 + g*0.5) % 1;
                mat.color.setHSL(hue, 0.8, 0.6);
                points.rotation.z = Math.sin(time*0.0003)*0.1;
                points.scale.setScalar(1.0 + g*0.2);
            };
        };

        // Sailing Ship (stylized low-poly with audio-reactive sea)
        /* Removed Sailing Ship visualizer per request */
        /* const sceneSailingShip = (scene, camera) => {
            const group = new THREE.Group();
            // Sea plane (dynamic wave via vertex displacement in JS)
            const seaSize = 40;
            const seaSeg = 128;
            const seaGeo = new THREE.PlaneGeometry(seaSize, seaSize, seaSeg, seaSeg);
            const seaMat = new THREE.MeshStandardMaterial({ color: 0x115577, metalness: 0.0, roughness: 0.9, transparent: true, opacity: 0.95, side: THREE.DoubleSide });
            const sea = new THREE.Mesh(seaGeo, seaMat);
            sea.rotation.x = -Math.PI/2;
            sea.position.y = -1.0;
            group.add(sea);
            // Ship: hull
            const hullMat = new THREE.MeshStandardMaterial({ color: 0x8b5a2b, metalness: 0.1, roughness: 0.8, emissive: 0x000000 });
            const hull = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.6, 1.0), hullMat);
            hull.position.y = -0.4;
            group.add(hull);
            // Prow and stern wedges
            const prow = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.0, 4), hullMat);
            prow.rotation.z = Math.PI/2;
            prow.position.set(1.9, -0.4, 0);
            group.add(prow);
            const stern = new THREE.Mesh(new THREE.ConeGeometry(0.5, 0.8, 4), hullMat);
            stern.rotation.z = -Math.PI/2;
            stern.position.set(-1.9, -0.4, 0);
            group.add(stern);
            // Mast
            const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 3.2, 12), new THREE.MeshStandardMaterial({ color: 0xdddddd, metalness: 0.2, roughness: 0.6 }));
            mast.position.y = 1.0;
            group.add(mast);
            // Sails
            const sailMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.0, roughness: 0.9, side: THREE.DoubleSide, emissive: 0x101010, emissiveIntensity: 0.08 });
            const sail1 = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 2.0), sailMat);
            sail1.position.set(0.6, 1.1, 0);
            sail1.rotation.y = Math.PI/10;
            group.add(sail1);
            const sail2 = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.6), sailMat);
            sail2.position.set(-0.3, 0.9, 0);
            sail2.rotation.y = Math.PI/12;
            group.add(sail2);
            // Flag
            const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.6, 0.25), new THREE.MeshStandardMaterial({ color: 0xff3366, side: THREE.DoubleSide, metalness: 0, roughness: 1 }));
            flag.position.set(0, 2.0, 0);
            flag.rotation.y = Math.PI/8;
            group.add(flag);
            // Lights
            const amb = new THREE.AmbientLight(0xffffff, 0.45);
            const sun = new THREE.DirectionalLight(0xffe0b0, 0.9);
            sun.position.set(4, 6, 3);
            scene.add(amb, sun, group);
            camera.position.set(0, 1.6, 7.5);
            camera.lookAt(0, 0.4, 0);
            // Sea wave data
            const basePos = sea.geometry.attributes.position.array.slice();
            const pos = sea.geometry.attributes.position;
            return (data, time) => {
                const t = time * 0.001;
                const bass = data[8] / 255;
                const mid = data[18] / 255;
                // Bobbing and slight roll with audio
                const bob = Math.sin(t * 1.2) * (0.05 + bass * 0.12);
                const roll = Math.sin(t * 0.9) * (0.03 + mid * 0.08);
                group.position.y = bob;
                group.rotation.z = roll;
                group.rotation.y = Math.sin(t * 0.3) * 0.05;
                // Flag flutter
                flag.rotation.y = Math.PI/8 + Math.sin(t*6.0 + bass*4.0)*0.25;
                // Sails subtle billow
                sail1.position.z = Math.sin(t*1.5) * 0.08;
                sail2.position.z = Math.cos(t*1.7) * 0.06;
                // Sea waves (displace Y of plane)
                for (let i = 0; i < pos.count; i++) {
                    const ix = i*3;
                    const x = basePos[ix+0];
                    const z = basePos[ix+2];
                    const w1 = Math.sin(x*0.6 + t*1.8) * 0.15;
                    const w2 = Math.cos(z*0.55 - t*1.3) * 0.12;
                    const w3 = Math.sin((x+z)*0.25 + t*2.2) * (0.08 + bass*0.22 + mid*0.1);
                    pos.array[ix+1] = -1.0 + w1 + w2 + w3;
                }
                pos.needsUpdate = true;
                sea.geometry.computeVertexNormals();
                // Color tint by audio
                const hue = (t*0.06 + bass*0.4) % 1;
                seaMat.color.setHSL(0.55 + 0.05*Math.sin(t*0.2), 0.6, 0.35 + bass*0.15);
                sailMat.emissiveIntensity = 0.08 + bass*0.35;
                hullMat.emissive.setHSL(hue, 0.8, 0.2);
                sun.intensity = 0.8 + bass*0.6;
            };
        }; */

        // Lorenz Attractor (line)
        const sceneLorenz = (scene, camera) => {
            const maxPts = 6000;
            const positions = new Float32Array(maxPts*3);
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.LineBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.9 });
            const line = new THREE.Line(geo, mat);
            scene.add(line);
            camera.position.z = 30;
            let x = 0.01, y = 0.0, z = 0.0;
            const dt = 0.005;
            const sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
            let idx = 0;
            return (audio, time) => {
                const g = (audio[10]+audio[18])/(2.0*255.0);
                for (let k=0;k<10;k++) {
                    const dx = sigma*(y - x);
                    const dy = x*(rho - z) - y;
                    const dz = x*y - beta*z;
                    x += dx*dt;
                    y += dy*dt;
                    z += dz*dt;
                    positions[idx*3+0] = x;
                    positions[idx*3+1] = y;
                    positions[idx*3+2] = z - 20.0;
                    idx = (idx + 1) % maxPts;
                }
                geo.attributes.position.needsUpdate = true;
                line.rotation.y = time*0.0002 + g*0.4;
                line.rotation.x = Math.sin(time*0.0003)*0.2;
                const hue = (time*0.00008 + g*0.4) % 1.0;
                mat.color.setHSL(hue, 1, 0.6);
            };
        };

        // Sierpinski Carpet (instanced)
        const sceneSierpinskiCarpet = (scene, camera) => {
            const depth = 4;
            const squares = [];
            function add(level, cx, cy, size) {
                if (level === 0) { squares.push({ cx, cy, size }); return; }
                const s3 = size/3;
                for (let dx=-1; dx<=1; dx++){
                    for (let dy=-1; dy<=1; dy++){
                        if (dx === 0 && dy === 0) continue;
                        add(level-1, cx + dx*s3, cy + dy*s3, s3);
                    }
                }
            }
            add(depth, 0, 0, 6);
            const geo = new THREE.PlaneGeometry(1, 1);
            const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const inst = new THREE.InstancedMesh(geo, mat, squares.length);
            const dummy = new THREE.Object3D();
            squares.forEach((s, i) => {
                dummy.position.set(s.cx, s.cy, 0);
                dummy.scale.set(s.size*0.95, s.size*0.95, 1);
                dummy.updateMatrix();
                inst.setMatrixAt(i, dummy.matrix);
            });
            inst.instanceMatrix.needsUpdate = true;
            scene.add(inst);
            camera.position.z = 10;
            return (audio, time) => {
                const g = (audio[4]+audio[12]+audio[28])/(3.0*255.0);
                const hue = (time*0.00015 + g*0.4) % 1.0;
                mat.color.setHSL(hue, 1, 0.6);
                inst.rotation.z = Math.sin(time*0.0003)*0.2;
                inst.scale.setScalar(1.0 + g*0.25);
            };
        };

        // --- Emoji Helpers and Scenes ---
        function createEmojiTexture(char) {
            const size = 256;
            const canvas = document.createElement('canvas');
            canvas.width = size; canvas.height = size;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0,0,size,size);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = '200px Apple Color Emoji, Noto Color Emoji, Segoe UI Emoji, EmojiOne Mozilla, Twemoji Mozilla, Segoe UI Symbol, sans-serif';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(char, size/2, size/2);
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            return texture;
        }

        const EMOJIS = ['🎵','✨','🔥','💫','🌈','💥','❤️','🌀','🌟','🤖','🎧','🎉','🎇','🛰️','🪐','⚡️'];

        const sceneEmojiSwarm = (scene, camera) => {
            const sprites = [];
            const velocities = [];
            const count = 240;
            const textures = EMOJIS.map(e => createEmojiTexture(e));
            for(let i=0;i<count;i++){
                const tex = textures[i % textures.length];
                const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
                const spr = new THREE.Sprite(mat);
                spr.position.set((Math.random()-0.5)*18, (Math.random()-0.5)*10, (Math.random()-0.5)*12);
                const s = Math.random()*0.7 + 0.3;
                spr.scale.set(s, s, 1);
                sprites.push(spr);
                velocities.push(new THREE.Vector3((Math.random()-0.5)*0.03, (Math.random()-0.5)*0.03, (Math.random()-0.5)*0.03));
                scene.add(spr);
            }
            camera.position.z = 8;
            return (data, time) => {
                const t = time*0.001;
                const bass = data[6]/255;
                sprites.forEach((s, i) => {
                    const v = velocities[i];
                    // Simple cohesion toward origin + slight orbit
                    const toCenter = s.position.clone().multiplyScalar(-0.002);
                    v.add(toCenter);
                    v.x += Math.sin(t + i*0.1)*0.001;
                    v.y += Math.cos(t*0.8 + i*0.07)*0.001;
                    s.position.add(v);
                    // Wrap bounds
                    if(s.position.x > 10) s.position.x = -10;
                    if(s.position.x < -10) s.position.x = 10;
                    if(s.position.y > 6) s.position.y = -6;
                    if(s.position.y < -6) s.position.y = 6;
                    if(s.position.z > 8) s.position.z = -8;
                    if(s.position.z < -8) s.position.z = 8;
                    // Pulse size with bass
                    const base = 0.3 + (i%5)*0.04;
                    const pulse = base + bass*0.8;
                    s.scale.set(pulse, pulse, 1);
                    s.material.opacity = 0.6 + bass*0.4;
                });
                camera.rotation.z = Math.sin(t*0.4)*0.1;
            };
        };

        /* Removed Emoji Orbits visualizer per request */
        /* const sceneEmojiOrbits = (scene, camera) => {
            const group = new THREE.Group();
            scene.add(group);
            const textures = EMOJIS.map(e => createEmojiTexture(e));
            const rings = [];
            for(let r=0;r<5;r++){
                const ringGroup = new THREE.Group();
                const num = 14 + r*5;
                const radius = 2.2 + r*1.4;
                for(let i=0;i<num;i++){
                    const tex = textures[Math.floor(Math.random()*textures.length)];
                    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
                    const spr = new THREE.Sprite(mat);
                    const a = (i/num)*Math.PI*2;
                    spr.position.set(Math.cos(a)*radius, Math.sin(a)*radius, -r*0.4);
                    const s = 0.22 + r*0.09;
                    spr.scale.set(s, s, 1);
                    ringGroup.add(spr);
                }
                group.add(ringGroup);
                rings.push({ group: ringGroup, radius });
            }
            camera.position.z = 8;
            let lastSwapMs = 0;
            return (data, time) => {
                const bass = data[8]/255;
                const mid = data[24]/255;
                rings.forEach((r, i) => {
                    r.group.rotation.z += 0.0006 + bass*0.006 + i*0.00025;
                    r.group.position.z = Math.sin(time*0.00025 + i)*0.8;
                    r.group.children.forEach((spr, j) => {
                        spr.material.rotation = time*0.0002 + j*0.03;
                        const s0 = spr.scale.x;
                        const pulse = s0 * (0.96 + mid*0.25*Math.sin(time*0.003 + j));
                        spr.scale.set(pulse, pulse, 1);
                    });
                });
                if (time - lastSwapMs > 4000) {
                    lastSwapMs = time;
                    const ring = rings[Math.floor(Math.random()*rings.length)];
                    if (ring && ring.group.children.length > 0) {
                        const idx = Math.floor(Math.random()*ring.group.children.length);
                        const spr = ring.group.children[idx];
                        const tex = textures[Math.floor(Math.random()*textures.length)];
                        spr.material.map = tex;
                        spr.material.needsUpdate = true;
                    }
                }
                group.rotation.x = Math.sin(time*0.0004)*0.2;
                group.rotation.y = Math.cos(time*0.0003)*0.2;
            };
        }; */

        // --- ENGINE: ProjectM (Filtered Presets) ---
        class MilkdropPresetEngine {
            constructor(label, filterFn, options = {}) {
                this.name = `ProjectM: ${label}`;
                this.filterFn = typeof filterFn === 'function' ? filterFn : (() => true);
                this.options = { invert: !!options.invert };
                this.resizeHandler = this.onResize.bind(this);
                this.canvas = null;
                this.transitionSec = visualSettings.transitionSec;
                this.presets = butterchurnPresets.getPresets?.() || {};
                const allKeys = Object.keys(this.presets);
                this.filteredKeys = allKeys.filter(k => {
                    try { return !!this.filterFn(k); } catch { return false; }
                });
                // If too few presets match, fall back to full set so it cycles
                if (this.filteredKeys.length < 2) this.filteredKeys = allKeys;
                this.currentPresetIdx = Math.floor(Math.random() * Math.max(1, this.filteredKeys.length));
            }

            init() {
                container.innerHTML = '';
                this.canvas = document.createElement('canvas');
                this.canvas.width = window.innerWidth;
                this.canvas.height = window.innerHeight;
                // Apply color inversion if requested
                if (this.options.invert) {
                    this.canvas.style.filter = 'invert(1)';
                }
                container.appendChild(this.canvas);
                const pxRatio = Number(visualSettings.pixelRatio) || (window.devicePixelRatio || 1);
                this.visualizer = butterchurn.createVisualizer(state.audioCtx, this.canvas, {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    pixelRatio: pxRatio
                });
                this.visualizer.connectAudio(state.analyserNode);
                this.loadPreset(this.currentPresetIdx);
                window.addEventListener('resize', this.resizeHandler);
                this.animate();
                this.restartCycle();
            }

            loadPreset(idx) {
                if(this.filteredKeys.length === 0) return;
                const key = this.filteredKeys[idx];
                const transition = Number(visualSettings.transitionSec) || this.transitionSec || 2.7;
                this.visualizer.loadPreset(this.presets[key], transition);
                if (state.activeVisualizer === this) {
                    try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
                }
                try { setBottomTextRandomColor(); } catch(e) {}
            }

            nextPreset() {
                if(this.filteredKeys.length === 0) return;
                let next = this.currentPresetIdx;
                if(this.filteredKeys.length > 1) {
                    while(next === this.currentPresetIdx) {
                        next = Math.floor(Math.random() * this.filteredKeys.length);
                    }
                }
                this.currentPresetIdx = next;
                this.loadPreset(this.currentPresetIdx);
                this.restartCycle();
            }

            restartCycle() {
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                const schedule = () => {
                    const minS = Number(visualSettings.shuffleMinSec) || 18;
                    const maxS = Number(visualSettings.shuffleMaxSec) || 36;
                    const lo = Math.min(minS, maxS);
                    const hi = Math.max(minS, maxS);
                    const delay = (lo + Math.random() * (hi - lo)) * 1000;
                    this.cycleTimeout = setTimeout(() => {
                        this.nextPreset();
                        schedule();
                    }, delay);
                };
                schedule();
            }

            onResize() {
                if(this.visualizer && this.canvas) {
                    const w = window.innerWidth;
                    const h = window.innerHeight;
                    this.canvas.width = w;
                    this.canvas.height = h;
                    this.visualizer.setRendererSize(w, h);
                }
            }

            animate() {
                this.animId = requestAnimationFrame(this.animate.bind(this));
                this.visualizer.render();
            }

            destroy() {
                cancelAnimationFrame(this.animId);
                if(this.cycleTimeout) clearTimeout(this.cycleTimeout);
                window.removeEventListener('resize', this.resizeHandler);
                container.innerHTML = '';
            }
        }

        // --- MASTER CONTROL ---
        const modes = [
            new MilkdropEngine(),
            new MilkdropEngineV2(),
            new MilkdropEngineV3(),
            new ThreeEngine("Blank", sceneBlank),
            new ThreeEngine("Audio Bars (Vertical Mirror)", sceneBarsVerticalMirror),
            new ThreeEngine("Audio Bars", sceneBars),
            new DjDecksEngine(),
            new RadioVisualEngine({ name: 'Digital Radio', skin: 'digital', skinLocked: true }),
            new RadioVisualEngine({ name: 'Analogue radio', skin: 'analogue', skinLocked: true }),
            new LogoEngine("Logo Fx", sceneLogoFx),
            new LogoEngine("Logo", sceneLogo),
            new ThreeEngine("Audio Bars (Circle)", sceneBarsCircle),
            new ThreeEngine("Audio Bars 3D", sceneBars3D),
            new ThreeEngine("Audio Bars: Vortex", sceneBarsVortex),
            new ThreeEngine("Neon Sphere", sceneSphere),
            new ThreeEngine("Electro Sphere", sceneElectroSphere),
            new ThreeEngine("Photon Shell", scenePhotonShell),
			new ThreeEngine("Pulse Orb", scenePulseOrb),
            new ThreeEngine("Cyber Tunnel", sceneTunnel),
            new ThreeEngine("Starfield", sceneStarfield),
            new ThreeEngine("Wave Grid", sceneWaveGrid),
            new ThreeEngine("Particles", sceneParticles),
            new ThreeEngine("Infinity Tunnel", sceneInfinityTunnel),
            new ThreeEngine("Neon Tunnel", sceneNeonTunnel),
            new ThreeEngine("Twist Tunnel", sceneTwistTunnel),
            new ThreeEngine("Particle Tunnel", sceneParticleTunnel),
            new ThreeEngine("Galaxy", sceneGalaxy),
            new ThreeEngine("Terrain", sceneTerrain),
            new ThreeEngine("Hex Grid", sceneHexGrid),
            new ThreeEngine("Mandelbrot", sceneMandelbrot),
            new ThreeEngine("Julia Set", sceneJulia),
            new ThreeEngine("Lorenz Attractor", sceneLorenz),
            new ThreeEngine("Sierpinski Carpet", sceneSierpinskiCarpet),
            new ThreeEngine("Emoji Swarm", sceneEmojiSwarm),
            new ThreeEngine("Kaleidoscope Layered", sceneKaleidoLayered),
			new ThreeEngine("Kaleidoscope Zoom", sceneKaleidoZoom),
            // Colorful Butterchurn set (inverted colors) - moved after Kaleidoscope Zoom
            new MilkdropPresetEngine("Rainbow (Invert)", (k) => /rainbow|color|colour|spectrum|palette/i.test(k), { invert: true })
        ];

        function updateSkipPresetButtonVisibility() {
            try {
                const skipBtn = document.getElementById('btn-skip-preset');
                if (!skipBtn) return;
                const av = state.activeVisualizer;
                const mainHasNext = !!(av && typeof av.nextPreset === 'function');
                const deckBProjectM =
                    !!(av &&
                        av.name === 'DJ Decks' &&
                        av.deckBVizMode === 'projectm' &&
                        typeof av.nextDeckBProjectMPreset === 'function');
                const digitalStagingPm =
                    !!(av &&
                        av.name === 'Digital Radio' &&
                        av.skin === 'digital' &&
                        av._digitalStagingView === 'projectm' &&
                        typeof av.nextPreset === 'function');
                if (mainHasNext || deckBProjectM || digitalStagingPm) skipBtn.classList.remove('display-none');
                else skipBtn.classList.add('display-none');
            } catch (_) {}
        }

        function isRadioVisualModeName(name) {
            try {
                if (typeof RadioVisualEngine !== 'undefined' && RadioVisualEngine.isRadioModeName) {
                    return RadioVisualEngine.isRadioModeName(name);
                }
            } catch (_) {}
            return name === 'Analogue radio' || name === 'Digital Radio'
                || name === 'Radio' || name === 'Radio Visual';
        }

        function findAnalogueRadioModeIndex() {
            try {
                if (!Array.isArray(modes)) return -1;
                return modes.findIndex((m) => m && m.name === 'Analogue radio');
            } catch (_) {
                return -1;
            }
        }

        function findDigitalRadioModeIndex() {
            try {
                if (!Array.isArray(modes)) return -1;
                return modes.findIndex((m) => m && m.name === 'Digital Radio');
            } catch (_) {
                return -1;
            }
        }

        function findRadioVisualModeIndex() {
            try {
                if (!Array.isArray(modes)) return -1;
                let idx = findAnalogueRadioModeIndex();
                if (idx >= 0) return idx;
                idx = findDigitalRadioModeIndex();
                if (idx >= 0) return idx;
                return modes.findIndex((m) => m && (m.name === 'Radio' || m.name === 'Radio Visual'));
            } catch (_) {
                return -1;
            }
        }

        function loadRadioVisualMode() {
            let variant = 'analogue';
            try {
                const stored = localStorage.getItem('radioVisual.lastVariant.v1');
                if (stored === 'digital' || stored === 'analogue') variant = stored;
            } catch (_) {}
            const idx = variant === 'digital' ? findDigitalRadioModeIndex() : findAnalogueRadioModeIndex();
            if (idx >= 0) loadMode(idx);
        }

        function activeRadioVariantKey() {
            try {
                const vis = state.activeVisualizer;
                if (!vis) return null;
                if (vis.name === 'Digital Radio' || vis.skin === 'digital') return 'digital';
                if (vis.name === 'Analogue radio' || vis.skin === 'analogue') return 'analogue';
            } catch (_) {}
            return null;
        }

        function toggleRadioVisualVariant() {
            try {
                const cur = activeRadioVariantKey();
                if (cur === 'analogue') {
                    const idx = findDigitalRadioModeIndex();
                    if (idx >= 0) { loadMode(idx); return true; }
                }
                if (cur === 'digital') {
                    const idx = findAnalogueRadioModeIndex();
                    if (idx >= 0) { loadMode(idx); return true; }
                }
            } catch (_) {}
            return false;
        }

        function shouldShowReturnRadioButton() {
            try {
                const visName = state.activeVisualizer && state.activeVisualizer.name;
                if (visName === 'DJ Decks') return true;
                if (!globalThis.__vizLaunchedFromRadioVisual) return false;
                if (!visName || isRadioVisualModeName(visName)) return false;
                return true;
            } catch (_) {
                return false;
            }
        }

        function updateDjDecksShortcutVisibility() {
            try {
                const btn = document.getElementById('btn-dj-decks');
                if (btn) {
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
                    const visName = state.activeVisualizer && state.activeVisualizer.name;
                    const onAnalogueRadio = visName === 'Analogue radio';
                    const onDigitalRadio = visName === 'Digital Radio';
                    const onRadioVisual = onAnalogueRadio || onDigitalRadio
                        || !!(state.activeVisualizer && isRadioVisualModeName(visName));
                    const showReturn = onDj && (deckBVisualActive || deckBTextActive);
                    btn.classList.toggle('display-none', onDj && !showReturn && !onRadioVisual);
                    btn.title = showReturn
                        ? 'Return Deck B controls'
                        : (onRadioVisual ? 'Open DJ Decks' : 'Open DJ Decks visual');
                }
                const btnRadio = document.getElementById('btn-return-radio');
                if (btnRadio) {
                    const visName = state.activeVisualizer && state.activeVisualizer.name;
                    const variant = activeRadioVariantKey();
                    const onAnalogueRadio = variant === 'analogue';
                    const onDigitalRadio = variant === 'digital';
                    const showToggle = onAnalogueRadio || onDigitalRadio;
                    const showReturn = shouldShowReturnRadioButton();
                    btnRadio.classList.toggle('display-none', !showToggle && !showReturn);
                    if (onAnalogueRadio) {
                        btnRadio.title = 'Switch to Digital Radio';
                        btnRadio.setAttribute('aria-label', 'Switch to Digital Radio');
                    } else if (onDigitalRadio) {
                        btnRadio.title = 'Switch to Analogue radio';
                        btnRadio.setAttribute('aria-label', 'Switch to Analogue radio');
                    } else {
                        btnRadio.title = 'Return to Radio Visual';
                        btnRadio.setAttribute('aria-label', 'Return to Radio Visual');
                    }
                }
            } catch (_) {}
        }

        function loadMode(index) {
            if(state.activeVisualizer) state.activeVisualizer.destroy();
            let idx = index;
            if(idx < 0) idx = modes.length - 1;
            if(idx >= modes.length) idx = 0;
            state.currentModeIdx = idx;
            state.activeVisualizer = modes[idx];
            try {
                const n = state.activeVisualizer && state.activeVisualizer.name;
                if (isRadioVisualModeName(n)) {
                    globalThis.__vizLaunchedFromRadioVisual = false;
                    try {
                        if (n === 'Analogue radio') {
                            localStorage.setItem('radioVisual.lastVariant.v1', 'analogue');
                        } else if (n === 'Digital Radio') {
                            localStorage.setItem('radioVisual.lastVariant.v1', 'digital');
                        }
                    } catch (_) {}
                }
            } catch (_) {}
            // Persist last selected visualizer
            try {
                localStorage.setItem('lastModeIndex', String(idx));
                if (state.activeVisualizer && state.activeVisualizer.name) {
                    localStorage.setItem('lastModeName', state.activeVisualizer.name);
                }
            } catch(_) {}
            
            // UI Updates
            document.getElementById('mode-title').innerText = state.activeVisualizer.name;
            const isRadioVis = !!(state.activeVisualizer && (
                isRadioVisualModeName(state.activeVisualizer.name)
            ));
            try { globalThis.updateModeSubStationLine?.(); } catch (_) {}
            
            updateSkipPresetButtonVisibility();

            try { initAudio(); } catch (_) {}
            state.activeVisualizer.init();
            // Randomize bottom text color each time a mode loads
            setBottomTextRandomColor();
            updateDjDecksShortcutVisibility();
        }

        try { globalThis.modes = modes; } catch (_) {}
        try { globalThis.loadMode = loadMode; } catch (_) {}
        try { globalThis.loadRadioVisualMode = loadRadioVisualMode; } catch (_) {}
        try { globalThis.toggleRadioVisualVariant = toggleRadioVisualVariant; } catch (_) {}
        try { globalThis.isRadioVisualModeName = isRadioVisualModeName; } catch (_) {}
        try { globalThis.updateDjDecksShortcutVisibility = updateDjDecksShortcutVisibility; } catch (_) {}
        try { globalThis.updateSkipPresetButtonVisibility = updateSkipPresetButtonVisibility; } catch (_) {}
