(function () {
    const SELECTOR = '[data-feynml-logo]';
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    function normalizeBoolean(value, fallback) {
        if (value === undefined || value === null || value === '') return fallback;
        return value === true || value === 'true';
    }

    function applyOptions(element, options) {
        if (!element) return;

        if (options.size) {
            ['small', 'medium', 'large'].forEach((size) => element.classList.remove(`feynml-logo--${size}`));
            element.classList.add(`feynml-logo--${options.size}`);
        }

        if (options.theme) {
            element.dataset.logoTheme = options.theme;
        }

        if (typeof options.animated !== 'undefined') {
            element.dataset.animated = String(options.animated);
        }

        if (typeof options.hoverEffects !== 'undefined') {
            element.dataset.hoverEffects = String(options.hoverEffects);
        }

        if (typeof options.introAnimation !== 'undefined') {
            element.dataset.intro = String(options.introAnimation);
        }

        const wordmark = element.querySelector('.feynml-logo__wordmark');
        if (wordmark && typeof options.showWordmark !== 'undefined') {
            wordmark.style.display = options.showWordmark ? '' : 'none';
        }
    }

    function mountLogo(element) {
        if (!element || element.dataset.logoMounted === 'true') return element;

        const animated = normalizeBoolean(element.dataset.animated, true);
        const hoverEffects = normalizeBoolean(element.dataset.hoverEffects, true);
        const introAnimation = normalizeBoolean(element.dataset.intro, true);

        if (prefersReducedMotion.matches) {
            element.dataset.animated = 'false';
            element.dataset.hoverEffects = 'false';
            element.dataset.intro = 'false';
        } else {
            element.dataset.animated = String(animated);
            element.dataset.hoverEffects = String(hoverEffects);
            element.dataset.intro = String(introAnimation);
        }

        if (document.hidden) {
            element.classList.add('is-paused');
        }

        element.dataset.logoMounted = 'true';
        return element;
    }

    function mountAll() {
        document.querySelectorAll(SELECTOR).forEach(mountLogo);
    }

    document.addEventListener('visibilitychange', () => {
        document.querySelectorAll(SELECTOR).forEach((element) => {
            element.classList.toggle('is-paused', document.hidden);
        });
    });

    if (typeof prefersReducedMotion.addEventListener === 'function') {
        prefersReducedMotion.addEventListener('change', mountAll);
    } else if (typeof prefersReducedMotion.addListener === 'function') {
        prefersReducedMotion.addListener(mountAll);
    }

    document.addEventListener('DOMContentLoaded', mountAll);

    window.FeynMLLogo = {
        mount(target, options = {}) {
            const element = typeof target === 'string' ? document.querySelector(target) : target;
            if (!element) return null;
            applyOptions(element, options);
            return mountLogo(element);
        },
        mountAll,
    };
})();
