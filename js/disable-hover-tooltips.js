/**
 * Disables native browser hover tooltips (title attributes) app-wide.
 * Backs up text to data-ui-title-backup and mirrors to aria-label when missing.
 */
(function disableHoverTooltips() {
    const STORE = 'data-ui-title-backup';

    function shouldStrip(el) {
        if (!el || el.nodeType !== 1) return false;
        const tag = el.tagName;
        if (tag === 'HTML' || tag === 'TITLE') return false;
        if (tag === 'title' && el.closest('svg')) return false;
        return el.hasAttribute('title');
    }

    function stripTitle(el) {
        if (!shouldStrip(el)) return;
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

    function stripTree(root) {
        if (!root) return;
        if (root.nodeType === 1 && shouldStrip(root)) stripTitle(root);
        try {
            root.querySelectorAll?.('[title]').forEach(stripTitle);
        } catch (_) {}
    }

    function install() {
        stripTree(document);
        if (window.__uiTitleTooltipMo) return;
        window.__uiTitleTooltipMo = new MutationObserver((mutations) => {
            for (const m of mutations) {
                if (m.type === 'attributes' && m.attributeName === 'title' && m.target?.nodeType === 1) {
                    stripTitle(m.target);
                }
                m.addedNodes?.forEach((n) => stripTree(n));
            }
        });
        window.__uiTitleTooltipMo.observe(document.documentElement, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeFilter: ['title']
        });
    }

    document.addEventListener(
        'pointerover',
        (e) => {
            const el = e.target?.closest?.('[title]');
            if (el) stripTitle(el);
        },
        true
    );

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', install, { once: true });
    } else {
        install();
    }
})();
