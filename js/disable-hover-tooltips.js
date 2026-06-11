/**
 * Controls native browser hover tooltips (title attributes) on Digital Radio buttons.
 * Default: off — titles are stripped and backed up to data-ui-title-backup.
 */
(function disableHoverTooltips() {
    const STORE = 'data-ui-title-backup';
    const STORAGE_KEY = 'radioVisual.buttonInfoOverlays.v1';
    const SCOPE = '#radio-visual-root button, #radio-visual-root [role="button"]';

    function readEnabledFromStorage() {
        try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch (_) {}
        return false;
    }

    function writeEnabledToStorage(enabled) {
        try { localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0'); } catch (_) {}
    }

    function shouldStrip(el) {
        if (!el || el.nodeType !== 1) return false;
        const tag = el.tagName;
        if (tag === 'HTML' || tag === 'TITLE') return false;
        if (tag === 'title' && el.closest('svg')) return false;
        return el.hasAttribute('title');
    }

    function inScope(el) {
        try { return !!(el && el.matches && el.matches(SCOPE)); } catch (_) {}
        return false;
    }

    function stripTitle(el) {
        if (!shouldStrip(el) || !inScope(el)) return;
        const text = el.getAttribute('title');
        if (!text) return;
        if (!el.hasAttribute(STORE)) el.setAttribute(STORE, text);
        el.removeAttribute('title');
        try {
            if (!el.getAttribute('aria-label') && !el.getAttribute('aria-labelledby')) {
                el.setAttribute('aria-label', text);
            }
        } catch (_) {}
    }

    function restoreTitle(el) {
        if (!el || el.nodeType !== 1 || !inScope(el)) return;
        if (!el.hasAttribute(STORE)) return;
        const text = el.getAttribute(STORE);
        if (text) el.setAttribute('title', text);
    }

    function stripScoped(root) {
        const base = root || document;
        try { base.querySelectorAll?.(SCOPE).forEach(stripTitle); } catch (_) {}
    }

    function restoreScoped(root) {
        const base = root || document;
        try { base.querySelectorAll?.(SCOPE).forEach(restoreTitle); } catch (_) {}
    }

    function disconnectObserver() {
        if (!window.__uiTitleTooltipMo) return;
        try { window.__uiTitleTooltipMo.disconnect(); } catch (_) {}
        window.__uiTitleTooltipMo = null;
    }

    function installObserver() {
        if (window.__uiTitleTooltipMo || readEnabledFromStorage()) return;
        window.__uiTitleTooltipMo = new MutationObserver((mutations) => {
            if (readEnabledFromStorage()) return;
            for (const m of mutations) {
                if (m.type === 'attributes' && m.attributeName === 'title' && m.target?.nodeType === 1) {
                    stripTitle(m.target);
                }
                m.addedNodes?.forEach((n) => {
                    if (n.nodeType !== 1) return;
                    if (inScope(n)) stripTitle(n);
                    try { n.querySelectorAll?.(SCOPE).forEach(stripTitle); } catch (_) {}
                });
            }
        });
        window.__uiTitleTooltipMo.observe(document.documentElement, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeFilter: ['title']
        });
    }

    function setButtonInfoOverlaysEnabled(enabled) {
        const on = !!enabled;
        writeEnabledToStorage(on);
        if (on) {
            disconnectObserver();
            restoreScoped(document);
        } else {
            stripScoped(document);
            installObserver();
        }
    }

    function isButtonInfoOverlaysEnabled() {
        return readEnabledFromStorage();
    }

    window.__uiButtonInfoOverlays = {
        setEnabled: setButtonInfoOverlaysEnabled,
        isEnabled: isButtonInfoOverlaysEnabled
    };

    document.addEventListener(
        'pointerover',
        (e) => {
            if (readEnabledFromStorage()) return;
            const el = e.target?.closest?.('[title]');
            if (el) stripTitle(el);
        },
        true
    );

    function boot() {
        if (readEnabledFromStorage()) {
            restoreScoped(document);
        } else {
            stripScoped(document);
            installObserver();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot, { once: true });
    } else {
        boot();
    }
})();
