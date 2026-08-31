window.addEventListener('DOMContentLoaded', () => {
	if (!window.tsParticles) return;

	tsParticles.load('tsparticles', {
		fpsLimit: 60,
		particles: {
			number: { value: 120, density: { enable: true, area: 800 } },
			color: { value: ['#7ab2ff', '#b3d1ff', '#2b6fff'] },
			shape: { type: 'circle' },
			opacity: { value: 0.85 },
			size: { value: { min: 0.8, max: 2.5 } },
			move: { enable: true, speed: 1.2, direction: 'none', outModes: { default: 'bounce' } }
		},
		interactivity: {
			detectsOn: 'canvas',
			events: { onHover: { enable: true, mode: 'repulse' }, onClick: { enable: true, mode: 'push' } },
			modes: { repulse: { distance: 80 }, push: { quantity: 4 } }
		},
		detectRetina: true
	});
});

// --- Jarvis interactive functionality: speech, commands, and simple intent parsing ---
(() => {
	const speak = (text) => {
		if (!('speechSynthesis' in window)) return;
		const utter = new SpeechSynthesisUtterance(text);
		utter.lang = 'en-US';
		utter.rate = 1;
		utter.pitch = 1;
		window.speechSynthesis.cancel();
		window.speechSynthesis.speak(utter);
	};

	const setMessage = (text) => {
		const el = document.querySelector('.jarvis-message .typewriter');
		if (!el) return;
		// stop any typing animation and set text immediately
		el.style.animation = 'none';
		el.textContent = text;
		// restore caret visually
		setTimeout(() => { el.style.animation = ''; }, 50);
	};

	// simple command handler
	const handleCommand = (cmdRaw) => {
		if (!cmdRaw) return;
		const cmd = cmdRaw.trim();
		setMessage(cmd);
		const lower = cmd.toLowerCase();

		// search intent
		if (lower.startsWith('search for ') || lower.startsWith('search ')) {
			const q = cmd.replace(/search for /i, '').replace(/search /i, '');
			speak('Searching for ' + q);
			window.open('https://www.google.com/search?q=' + encodeURIComponent(q), '_blank');
			return;
		}

		// open website shortcuts
		if (lower.startsWith('open ')) {
			const target = lower.replace(/^open /i, '').trim();
			const map = {
				'youtube': 'https://www.youtube.com',
				'google': 'https://www.google.com',
				'github': 'https://github.com',
				'gmail': 'https://mail.google.com'
			};
			for (const key in map) if (target.includes(key)) { speak('Opening ' + key); window.open(map[key], '_blank'); return; }
			// try open as url
			let url = target;
			if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
			try { speak('Opening ' + target); window.open(url, '_blank'); } catch (e) { speak('Cannot open ' + target); }
			return;
		}

		// time
		if (lower.includes('time')) {
			const now = new Date();
			const t = now.toLocaleTimeString();
			speak('The time is ' + t);
			return;
		}

		// timer
		const timerMatch = lower.match(/(\d+)\s*(second|seconds|minute|minutes|hour|hours)/);
		if (timerMatch) {
			const n = parseInt(timerMatch[1], 10);
			const unit = timerMatch[2];
			let ms = 1000 * n;
			if (unit.startsWith('minute')) ms *= 60;
			if (unit.startsWith('hour')) ms *= 3600;
			speak('Setting a ' + n + ' ' + unit + ' timer');
			setTimeout(() => { speak('Timer finished: ' + n + ' ' + unit); }, ms);
			return;
		}

		// note taking
		if (lower.startsWith('note ') || lower.startsWith('remember ')) {
			const note = cmd.replace(/^(note|remember) /i, '').trim();
			const notes = JSON.parse(localStorage.getItem('jarvis_notes') || '[]');
			notes.push({ text: note, ts: Date.now() });
			localStorage.setItem('jarvis_notes', JSON.stringify(notes));
			speak('Saved note');
			return;
		}

		// jokes
		if (lower.includes('joke')) {
			const jokes = [
				"Why did the developer go broke? Because he used up all his cache.",
				"I would tell you a UDP joke, but you might not get it.",
				"Why do programmers prefer dark mode? Because light attracts bugs."
			];
			const j = jokes[Math.floor(Math.random() * jokes.length)];
			speak(j);
			return;
		}

		// fallback help
		speak("I heard: " + cmd + ". I can search, open websites, set timers, take notes, or tell the time. Try 'search for cats'.");
	};

		// voice-only wake-word: listen continuously for the word "jarvis", then capture a single command
		const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
		if (!SpeechRec) {
			speak('Speech recognition not supported in this browser');
			setMessage('Speech recognition not supported in this browser');
		} else {
			const wakeRec = new SpeechRec();
			wakeRec.lang = 'en-US';
			wakeRec.continuous = true;
			wakeRec.interimResults = true;

			const startWake = () => {
				try { wakeRec.start(); setMessage("Say 'Jarvis' to wake me."); } catch (e) { console.error('wake start failed', e); }
			};

			wakeRec.onresult = (ev) => {
				// accumulate transcripts from the latest results
				let transcript = '';
				for (let i = ev.resultIndex; i < ev.results.length; i++) {
					transcript += ev.results[i][0].transcript;
				}
				const lower = transcript.toLowerCase();
				if (lower.includes('jarvis')) {
					// detected wake word
					try { wakeRec.stop(); } catch (e) { /* ignore */ }
					// Speak the personalized wake response and show it in the UI
					setMessage('Yes, Mudit — how may I help you?');
					speak('Yes, Mudit — how may I help you?');

					// single-shot recognition for the command
					const cmdRec = new SpeechRec();
					cmdRec.lang = 'en-US';
					cmdRec.interimResults = false;
					cmdRec.maxAlternatives = 1;
					cmdRec.onresult = (e2) => {
						const text = e2.results[0][0].transcript;
						setMessage(text);
						handleCommand(text);
					};
					cmdRec.onerror = (e3) => { console.warn('command recog error', e3); speak('I could not understand the command.'); };
					cmdRec.onend = () => {
						// resume wake listener after a short delay
						setTimeout(startWake, 500);
						setMessage("Say 'Jarvis' to wake me.");
					};
					try { cmdRec.start(); } catch (err) { console.error('cmd start failed', err); startWake(); }
				}
			};

			wakeRec.onerror = (e) => { console.warn('wake error', e); setTimeout(startWake, 1000); };
			wakeRec.onend = () => { /* restart automatically */ setTimeout(startWake, 300); };

			// start listening for wake word
			startWake();
		}
})();
