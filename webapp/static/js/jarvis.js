/* ============================================================
   JARVIS WIDGET LOGIC
   Location: webapp/static/js/jarvis.js

   Design notes:
   - Text input is the primary interface. Mic is progressive
     enhancement — if SpeechRecognition isn't supported, the
     mic button is disabled but the widget stays fully usable.
   - No hardcoded personal name — replies are generic.
   - Single source of truth (no duplicate logic across files).
   - Phase A only: local command handling (open pages, greet,
     time). Phase B will POST to a /api/voice-query endpoint
     for dataset/report-aware answers.
   ============================================================ */

(function () {
    'use strict';

    const widget   = document.getElementById('jarvis-widget');
    const trigger  = document.getElementById('jarvis-trigger');
    const panel    = document.getElementById('jarvis-panel');
    const closeBtn = document.getElementById('jarvis-close');
    const form     = document.getElementById('jarvis-form');
    const input    = document.getElementById('jarvis-input');
    const micBtn   = document.getElementById('jarvis-mic');
    const statusEl = document.getElementById('jarvis-status');
    const logEl    = document.getElementById('jarvis-log');

    if (!widget || !trigger || !panel) return;

    /* ---------------- Panel open/close ---------------- */
    function openPanel() {
        panel.hidden = false;
        trigger.setAttribute('aria-expanded', 'true');
        widget.dataset.state = 'expanded';
        input.focus();
    }

    function closePanel() {
        panel.hidden = true;
        trigger.setAttribute('aria-expanded', 'false');
        widget.dataset.state = 'collapsed';
        trigger.focus();
    }

    trigger.addEventListener('click', () => {
        panel.hidden ? openPanel() : closePanel();
    });

    closeBtn.addEventListener('click', closePanel);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !panel.hidden) closePanel();
    });

    // Close when clicking outside the widget
    document.addEventListener('click', (e) => {
        if (!panel.hidden && !widget.contains(e.target)) closePanel();
    });

    /* ---------------- Logging ---------------- */
    function logEntry(text, kind) {
        const entry = document.createElement('div');
        entry.className = 'jarvis-log-entry jarvis-log-' + kind;
        entry.textContent = text;
        logEl.appendChild(entry);
        logEl.scrollTop = logEl.scrollHeight;
    }

    function setStatus(text) {
        statusEl.textContent = text;
    }

    /* ---------------- Command handling ----------------
       Phase A: minimal local commands. Extend this map as
       Phase B wires in real FeynML dataset/report queries. */
    function handleCommand(raw) {
        const cmd = raw.trim().toLowerCase();
        if (!cmd) return;

        logEntry(raw, 'user');

        let reply;
        if (/^(hi|hello|hey)\b/.test(cmd)) {
            reply = 'Hello — how can I help?';
        } else if (cmd.includes('time')) {
            reply = 'Current time: ' + new Date().toLocaleTimeString();
        } else if (cmd.includes('dashboard')) {
            reply = 'Opening the dashboard…';
            window.location.href = '/analyze';
        } else if (cmd.includes('root cause')) {
            reply = 'Opening Root Cause analysis…';
            window.location.href = '/analysis/root-cause';
        } else {
            reply = "I can't help with that yet — voice-aware report queries are coming soon.";
        }

        logEntry(reply, 'reply');
        setStatus('Ready. Type a command or use the mic.');
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const value = input.value;
        input.value = '';
        handleCommand(value);
    });

    /* ---------------- Speech recognition (optional) ----------------
       Progressive enhancement only. If unsupported, the mic button
       is disabled and clearly labeled — text input remains the
       fully-functional path, so nothing is voice-only. */
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
        micBtn.disabled = true;
        micBtn.setAttribute('aria-label', 'Voice input not supported in this browser');
        micBtn.title = 'Voice input not supported in this browser — use the text field instead.';
    } else {
        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        let listening = false;

        recognition.addEventListener('start', () => {
            listening = true;
            widget.dataset.listening = 'true';
            micBtn.setAttribute('aria-pressed', 'true');
            setStatus('Listening…');
        });

        recognition.addEventListener('end', () => {
            listening = false;
            widget.dataset.listening = 'false';
            micBtn.setAttribute('aria-pressed', 'false');
        });

        recognition.addEventListener('result', (event) => {
            const transcript = event.results[0][0].transcript;
            input.value = transcript;
            handleCommand(transcript);
            input.value = '';
        });

        recognition.addEventListener('error', (event) => {
            if (event.error === 'not-allowed' || event.error === 'permission-denied') {
                setStatus('Microphone access denied — use the text field instead.');
                micBtn.disabled = true;
            } else {
                setStatus('Voice input error — try typing instead.');
            }
        });

        micBtn.addEventListener('click', () => {
            if (listening) {
                recognition.stop();
            } else {
                try {
                    recognition.start();
                } catch (err) {
                    // start() throws if called while already active; ignore.
                }
            }
        });
    }
})();
