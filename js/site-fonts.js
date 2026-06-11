(function initSiteFonts(global) {
    const SITE_FONT_OPTIONS = [
        { label: 'Orbitron', value: "'Orbitron', sans-serif" },
        { label: 'Orbitron (legacy stack)', value: "'Orbitron', 'Share Tech Mono', ui-monospace, monospace" },
        { label: 'Share Tech Mono', value: "'Share Tech Mono', monospace" },
        { label: 'Audiowide', value: "'Audiowide', cursive" },
        { label: 'Rajdhani', value: "'Rajdhani', sans-serif" },
        { label: 'Quicksand', value: "'Quicksand', sans-serif" },
        { label: 'Electrolize', value: "'Electrolize', sans-serif" },
        { label: 'Oxanium', value: "'Oxanium', cursive" },
        { label: 'Quantico', value: "'Quantico', sans-serif" },
        { label: 'Syncopate', value: "'Syncopate', sans-serif" },
        { label: 'Exo 2', value: "'Exo 2', sans-serif" },
        { label: 'Chakra Petch', value: "'Chakra Petch', sans-serif" },
        { label: 'Teko', value: "'Teko', sans-serif" },
        { label: 'Sarpanch', value: "'Sarpanch', sans-serif" },
        { label: 'Major Mono Display', value: "'Major Mono Display', monospace" },
        { label: 'Monoton', value: "'Monoton', cursive" },
        { label: 'Quicksilver', value: "'Quicksilver', cursive" },
        { label: 'Press Start 2P', value: "'Press Start 2P', cursive" },
        { label: 'VT323', value: "'VT323', monospace" },
        { label: 'Bebas Neue', value: "'Bebas Neue', sans-serif" },
        { label: 'Anton', value: "'Anton', sans-serif" },
        { label: 'Bangers', value: "'Bangers', cursive" },
        { label: 'Black Ops One', value: "'Black Ops One', cursive" },
        { label: 'Bungee', value: "'Bungee', cursive" },
        { label: 'Russo One', value: "'Russo One', sans-serif" },
        { label: 'Righteous', value: "'Righteous', cursive" },
        { label: 'Staatliches', value: "'Staatliches', cursive" },
        { label: 'Tourney', value: "'Tourney', cursive" },
        { label: 'Zen Dots', value: "'Zen Dots', cursive" },
        { label: 'Unbounded', value: "'Unbounded', cursive" },
        { label: 'Michroma', value: "'Michroma', sans-serif" },
        { label: 'Archivo Black', value: "'Archivo Black', sans-serif" },
        { label: 'Alfa Slab One', value: "'Alfa Slab One', cursive" },
        { label: 'Abril Fatface', value: "'Abril Fatface', cursive" },
        { label: 'Roboto', value: "'Roboto', sans-serif" },
        { label: 'Roboto Condensed', value: "'Roboto Condensed', sans-serif" },
        { label: 'Roboto Mono', value: "'Roboto Mono', monospace" },
        { label: 'Roboto Slab', value: "'Roboto Slab', serif" },
        { label: 'Oswald', value: "'Oswald', sans-serif" },
        { label: 'Montserrat', value: "'Montserrat', sans-serif" },
        { label: 'Raleway', value: "'Raleway', sans-serif" },
        { label: 'Poppins', value: "'Poppins', sans-serif" },
        { label: 'Open Sans', value: "'Open Sans', sans-serif" },
        { label: 'Inter', value: "'Inter', sans-serif" },
        { label: 'Lato', value: "'Lato', sans-serif" },
        { label: 'Barlow', value: "'Barlow', sans-serif" },
        { label: 'Ubuntu', value: "'Ubuntu', sans-serif" },
        { label: 'Work Sans', value: "'Work Sans', sans-serif" },
        { label: 'DM Sans', value: "'DM Sans', sans-serif" },
        { label: 'Manrope', value: "'Manrope', sans-serif" },
        { label: 'Mulish', value: "'Mulish', sans-serif" },
        { label: 'Outfit', value: "'Outfit', sans-serif" },
        { label: 'Karla', value: "'Karla', sans-serif" },
        { label: 'Josefin Sans', value: "'Josefin Sans', sans-serif" },
        { label: 'Comfortaa', value: "'Comfortaa', cursive" },
        { label: 'Kanit', value: "'Kanit', sans-serif" },
        { label: 'Titillium Web', value: "'Titillium Web', sans-serif" },
        { label: 'Yanone Kaffeesatz', value: "'Yanone Kaffeesatz', sans-serif" },
        { label: 'Space Grotesk', value: "'Space Grotesk', sans-serif" },
        { label: 'IBM Plex Sans', value: "'IBM Plex Sans', sans-serif" },
        { label: 'Archivo', value: "'Archivo', sans-serif" },
        { label: 'Fira Sans', value: "'Fira Sans', sans-serif" },
        { label: 'Rubik', value: "'Rubik', sans-serif" },
        { label: 'Nunito', value: "'Nunito', sans-serif" },
        { label: 'Playfair Display', value: "'Playfair Display', serif" },
        { label: 'Merriweather', value: "'Merriweather', serif" },
        { label: 'Lora', value: "'Lora', serif" },
        { label: 'Bitter', value: "'Bitter', serif" },
        { label: 'PT Sans', value: "'PT Sans', sans-serif" },
        { label: 'Lobster', value: "'Lobster', cursive" },
        { label: 'Pacifico', value: "'Pacifico', cursive" },
        { label: 'Permanent Marker', value: "'Permanent Marker', cursive" },
        { label: 'Caveat', value: "'Caveat', cursive" },
        { label: 'Dancing Script', value: "'Dancing Script', cursive" },
        { label: 'Fira Mono', value: "'Fira Mono', monospace" },
        { label: 'Inconsolata', value: "'Inconsolata', monospace" },
        { label: 'Source Code Pro', value: "'Source Code Pro', monospace" },
        { label: 'Space Mono', value: "'Space Mono', monospace" },
        { label: 'JetBrains Mono', value: "'JetBrains Mono', monospace" },
        { label: 'IBM Plex Mono', value: "'IBM Plex Mono', monospace" },
        { label: 'System Mono', value: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' },
        { label: 'Segoe UI', value: "'Segoe UI', sans-serif" },
        { label: 'Arial', value: 'Arial, Helvetica, sans-serif' },
        { label: 'Courier New', value: "'Courier New', Courier, monospace" },
        { label: 'Times New Roman', value: "'Times New Roman', Times, serif" },
        { label: 'Georgia', value: "Georgia, 'Times New Roman', serif" },
        { label: 'Trebuchet MS', value: "'Trebuchet MS', sans-serif" }
    ];

    function populateSiteFontSelect(selectEl, selectedValue) {
        if (!selectEl) return;
        const prev = selectedValue != null ? String(selectedValue) : String(selectEl.value || '');
        selectEl.innerHTML = '';
        SITE_FONT_OPTIONS.forEach(({ label, value }) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            selectEl.appendChild(opt);
        });
        if (prev && !SITE_FONT_OPTIONS.some((entry) => entry.value === prev)) {
            const custom = document.createElement('option');
            custom.value = prev;
            custom.textContent = 'Custom';
            selectEl.appendChild(custom);
        }
        if (prev) selectEl.value = prev;
    }

    function populateAllSiteFontSelects(root) {
        const doc = root || document;
        const map = [
            ['opt-digital-font', null],
            ['opt-digital-clock-font', null],
            ['opt-digital-btn-blue-font', null],
            ['opt-digital-btn-purple-font', null],
            ['ti-font', doc.getElementById('ti-font') && doc.getElementById('ti-font').value]
        ];
        map.forEach(([id, selected]) => {
            const el = doc.getElementById(id);
            if (!el) return;
            let value = selected != null ? selected : el.value;
            if (!value && id === 'ti-font') value = "'Lobster', cursive";
            populateSiteFontSelect(el, value);
        });
    }

    global.SITE_FONT_OPTIONS = SITE_FONT_OPTIONS;
    global.populateSiteFontSelect = populateSiteFontSelect;
    global.populateAllSiteFontSelects = populateAllSiteFontSelects;
})(globalThis);
