/**
 * Sign Language Multimodal AI — Frontend Client
 * Real-time Webcam Stream -> Flask Backend ML Pipeline -> Multilingual Text & Voice
 * Features:
 *   - Temporal Prediction Stabilization (Raw vs Stable Distinction)
 *   - Configurable Confidence Filtering
 *   - Deduplicated Prediction History
 *   - Multilingual Audio & Speech
 */

// ==========================================================================
// 1. Centralized Stability & Recognition Configuration
// ==========================================================================
const STABILITY_CONFIG = {
    STABILITY_FRAMES: 3,          // 3 frames (~120ms) — snappy, responsive recognition!
    CONFIDENCE_THRESHOLD: 0.45,   // 45% minimum confidence (readily catches hands)
    HISTORY_DEBOUNCE_MS: 3000,    // Cooldown between logging distinct signs
};

// ==========================================================================
// 2. DOM Elements
// ==========================================================================
const video = document.getElementById('webcam-video');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const hiddenCanvas = document.getElementById('hidden-capture-canvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

// Overlays & Badges
const cameraOverlay = document.getElementById('camera-overlay');
const overlayIcon = document.getElementById('overlay-icon');
const overlayTitle = document.getElementById('overlay-title');
const overlayDesc = document.getElementById('overlay-desc');
const cameraStatusBadge = document.getElementById('camera-status-badge');
const cameraStatusText = document.getElementById('camera-status-text');
const handsCountText = document.getElementById('hands-count-text');
const latencyText = document.getElementById('latency-text');

// Diagnostic Banner Elements
const diagHandCount = document.getElementById('diag-hand-count');
const diagStatus = document.getElementById('diag-status');
const diagFeatures = document.getElementById('diag-features');
const diagLatency = document.getElementById('diag-latency');

// Result Card, Stability & Language Controls
const recStatusBadge = document.getElementById('rec-status-badge');
const recStatusText = document.getElementById('rec-status-text');
const signBadge = document.getElementById('sign-badge');
const stabilityPill = document.getElementById('stability-pill');
const stabilityDot = document.getElementById('stability-dot');
const stabilityText = document.getElementById('stability-text');
const selectedTextLangFlag = document.getElementById('selected-text-lang-flag');
const selectedTextLangLabel = document.getElementById('selected-text-lang-label');
const translatedText = document.getElementById('translated-text');
const confidenceValue = document.getElementById('confidence-value');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceStateBadge = document.getElementById('confidence-state-badge');
const confidenceStateIcon = document.getElementById('confidence-state-icon');
const confidenceStateText = document.getElementById('confidence-state-text');
const guidanceText = document.getElementById('guidance-text');
const guidanceIcon = document.getElementById('guidance-icon');
const selectTextLang = document.getElementById('select-text-lang');
const selectAudioLang = document.getElementById('select-audio-lang');

// History Elements
const historyTableBody = document.getElementById('history-table-body');
const historyCountBadge = document.getElementById('history-count-badge');
const btnClearHistory = document.getElementById('btn-clear-history');

// Buttons & Audio
const btnRequestCamera = document.getElementById('btn-request-camera');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const btnSpeak = document.getElementById('btn-speak');
const btnSpeakLabel = document.getElementById('btn-speak-label');
const btnClear = document.getElementById('btn-clear');
const ttsAudio = document.getElementById('tts-audio');

// ==========================================================================
// 3. State Management
// ==========================================================================
let stream = null;
let isCameraRunning = false;
let isProcessingFrame = false;
let captureInterval = null;

let selectedTextLang = 'english';
let selectedAudioLang = 'hindi';

// Available language metadata with BCP-47 locales
const LANGUAGE_META = {
    english:  { name: 'English',  flag: '🇬🇧', bcp47: 'en-US', native: 'English' },
    hindi:    { name: 'Hindi',    flag: '🇮🇳', bcp47: 'hi-IN', native: 'हिन्दी' },
    bhojpuri: { name: 'Bhojpuri', flag: '🇮🇳', bcp47: 'bho-IN', native: 'भोजपुरी' },
    odia:     { name: 'Odia',     flag: '🇮🇳', bcp47: 'or-IN', native: 'ଓଡ଼ିଆ' },
    telugu:   { name: 'Telugu',   flag: '🇮🇳', bcp47: 'te-IN', native: 'తెలుగు' },
    bengali:  { name: 'Bengali',  flag: '🇮🇳', bcp47: 'bn-IN', native: 'বাংলা' },
};

// ── Raw vs Stable Prediction State ─────────────────────────────────────────
// Raw frame state (accumulates across consecutive frames)
let lastRawLabel = null;
let rawConsistencyCount = 0;

// Stable prediction state (the verified, confirmed user gesture)
let stableLabel = null;
let stableTranslations = {};
let stableConfidence = 0.0;
let isStabilized = false;

// History state
const MAX_HISTORY_ENTRIES = 20;
let historyList = [];
let lastRecordedSign = null;
let lastRecordedTime = 0;
let hadZeroHandsSinceLastRecord = true;

// Hand landmark drawing is disabled — clean output only (no skeleton lines or dots)

// ==========================================================================
// 4. Camera Management
// ==========================================================================

async function startWebcam() {
    try {
        setCameraStatus('connecting', 'Camera: Requesting Permission…');
        showCameraOverlay(true, '📷', 'Camera Permission', 'Please click Allow in your browser popup to enable your webcam.');

        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });

        // Hidden canvas: 640x480 for reliable MediaPipe hand detection at any distance
        hiddenCanvas.width = 640;
        hiddenCanvas.height = 480;
        overlayCanvas.width = video.videoWidth || 640;
        overlayCanvas.height = video.videoHeight || 480;

        isCameraRunning = true;
        showCameraOverlay(false);
        setCameraStatus('active', 'Camera: Connected');
        btnToggleCamera.textContent = '⏹ Stop Camera';
        btnToggleCamera.classList.remove('btn-primary');
        btnToggleCamera.classList.add('btn-secondary');

        // Start non-blocking adaptive capture loop
        setTimeout(processCurrentFrame, 40);

    } catch (err) {
        console.error('Camera access error:', err);
        isCameraRunning = false;
        setCameraStatus('error', 'Camera: Unavailable');
        
        let errorMsg = 'Could not access webcam. Please check browser permissions in settings.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg = 'Camera permission was denied. Please allow camera access in your browser address bar.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg = 'No camera device found on this system.';
        }
        showCameraOverlay(true, '⚠️', 'Camera Unavailable', errorMsg, true);
    }
}

function stopWebcam() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    isCameraRunning = false;
    video.srcObject = null;
    clearOverlayCanvas();

    setCameraStatus('stopped', 'Camera: Stopped');
    showCameraOverlay(true, '📷', 'Camera Stopped', 'Click below to restart camera recognition.', true);
    btnToggleCamera.textContent = '▶ Start Camera';
    btnToggleCamera.classList.remove('btn-secondary');
    btnToggleCamera.classList.add('btn-primary');

    handsCountText.textContent = 'Hands detected: 0';
    latencyText.textContent = 'Latency: -- ms';
    recStatusText.textContent = 'Camera Off';
    recStatusBadge.classList.remove('active');
    setStabilityStatus('waiting', 'Camera Off');
    updateConfidenceGauge(0, false);
}

function showCameraOverlay(show, icon = '📷', title = '', desc = '', showButton = false) {
    if (show) {
        cameraOverlay.classList.remove('hidden');
        overlayIcon.textContent = icon;
        overlayTitle.textContent = title;
        overlayDesc.textContent = desc;
        btnRequestCamera.style.display = showButton ? 'inline-flex' : 'none';
        btnRequestCamera.textContent = isCameraRunning ? 'Allow Camera' : 'Start Camera';
    } else {
        cameraOverlay.classList.add('hidden');
    }
}

function setCameraStatus(type, text) {
    cameraStatusText.textContent = text;
    cameraStatusBadge.classList.remove('active');
    if (type === 'active') {
        cameraStatusBadge.classList.add('active');
    }
}

// ==========================================================================
// 5. Frame Processing & Temporal Stabilization Pipeline
// ==========================================================================

async function processCurrentFrame() {
    if (!isCameraRunning || isProcessingFrame || video.readyState !== video.HAVE_ENOUGH_DATA) {
        return;
    }

    isProcessingFrame = true;
    const startTime = performance.now();

    try {
        hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        const imageBase64 = hiddenCanvas.toDataURL('image/jpeg', 0.75);

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageBase64 })
        });

        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);
        latencyText.textContent = `Latency: ${latency} ms`;

        if (data.status === 'success') {
            handlePredictionResponse(data);
        } else {
            handleNoHandsOrError(data.message || 'Detection idle');
        }

    } catch (err) {
        console.debug('Frame processing error:', err);
    } finally {
        isProcessingFrame = false;
        if (isCameraRunning) {
            setTimeout(processCurrentFrame, 35);
        }
    }
}

function handlePredictionResponse(data) {
    const handsDetected = data.hands_detected || 0;
    handsCountText.textContent = `Hands detected: ${handsDetected}`;

    // Update Live Diagnostic Display
    if (diagLatency && data.detection_ms !== undefined) {
        diagLatency.textContent = `MEDIA-PIPE: ${data.detection_ms} ms`;
    }
    if (handsDetected === 0) {
        if (diagHandCount) diagHandCount.innerHTML = `<span style="color: #94a3b8;">✋ HANDS: 0</span>`;
        if (diagStatus) diagStatus.innerHTML = `<span style="color: #f59e0b;">STATUS: SEARCHING</span>`;
        if (diagFeatures) diagFeatures.textContent = `FEATURES: 126 (0 active)`;
    } else if (handsDetected === 1) {
        if (diagHandCount) diagHandCount.innerHTML = `<span style="color: #4ade80;">✋ 1 HAND DETECTED</span>`;
        if (diagStatus) diagStatus.innerHTML = `<span style="color: #4ade80;">STATUS: 1-HAND ACTIVE</span>`;
        if (diagFeatures) diagFeatures.textContent = `FEATURES: 126 (63 active + 63 padding)`;
    } else {
        if (diagHandCount) diagHandCount.innerHTML = `<span style="color: #38bdf8;">✋✋ 2 HANDS DETECTED</span>`;
        if (diagStatus) diagStatus.innerHTML = `<span style="color: #38bdf8;">STATUS: 2-HANDS ACTIVE</span>`;
        if (diagFeatures) diagFeatures.textContent = `FEATURES: 126 (126 active)`;
    }

    // Landmark drawing disabled — keep canvas clean (no skeleton/dot overlay)
    clearOverlayCanvas();

    // ── Case 1: No Hands Detected — clear everything immediately ─────────
    if (handsDetected === 0) {
        hadZeroHandsSinceLastRecord = true;
        lastRawLabel = null;
        rawConsistencyCount = 0;

        // Always reset stable state so old result never lingers
        stableLabel = null;
        stableTranslations = {};
        stableConfidence = 0.0;
        isStabilized = false;

        recStatusText.textContent = 'Searching Hand';
        recStatusBadge.classList.remove('active');

        // Clear result card completely
        signBadge.textContent = '[—]';
        translatedText.textContent = 'Show your hand to the camera';
        setStabilityStatus('waiting', 'Waiting for Hand');
        guidanceText.textContent = 'Show your hand clearly to the camera.';
        guidanceIcon.textContent = '✋';
        updateConfidenceGauge(0, false);
        return;
    }

    // ── Case 2: 1 or 2 Hands Active (Full 17-Class Support) ───────────────
    recStatusText.textContent = handsDetected >= 2 ? '2 Hands Active' : '1 Hand Active';
    recStatusBadge.classList.add('active');

    const rawLabel = data.label;
    const rawConfidence = data.confidence || 0.0;
    const rawTranslations = data.translations || {};

    // Check Confidence Filter
    if (!rawLabel || rawConfidence < STABILITY_CONFIG.CONFIDENCE_THRESHOLD) {
        updateConfidenceGauge(rawConfidence, true);
        setStabilityStatus('low_conf', '⚠ Low Confidence');
        guidanceText.textContent = '⚠ Low Confidence — Please show the sign clearly.';
        guidanceIcon.textContent = '⚠️';
        rawConsistencyCount = Math.max(0, rawConsistencyCount - 1);
        return;
    }

    // Accumulate consecutive matching raw frames
    if (rawLabel === lastRawLabel) {
        rawConsistencyCount++;
    } else {
        lastRawLabel = rawLabel;
        rawConsistencyCount = 1;
    }

    // Check Temporal Stability Threshold
    if (rawConsistencyCount >= STABILITY_CONFIG.STABILITY_FRAMES) {
        const isNewSign = (stableLabel !== rawLabel);

        // Promote raw prediction to confirmed stable prediction
        stableLabel = rawLabel;
        stableTranslations = rawTranslations;
        stableConfidence = rawConfidence;
        isStabilized = true;

        updateResultCard(stableLabel, stableTranslations, stableConfidence);
        setStabilityStatus('stable', '✓ Stable');

        // Record to history ONLY ONCE per deliberate sign (no timer spam)
        const now = Date.now();
        if ((isNewSign || hadZeroHandsSinceLastRecord) && lastRecordedSign !== stableLabel) {
            addHistoryEntry(stableLabel, stableTranslations, stableConfidence);
            lastRecordedSign = stableLabel;
            lastRecordedTime = now;
            hadZeroHandsSinceLastRecord = false;
        }

    } else {
        // Stabilizing transition state — NEVER flash raw unstable guesses
        updateConfidenceGauge(rawConfidence, true);
        setStabilityStatus('stabilizing', `⏳ Holding... (${rawConsistencyCount}/${STABILITY_CONFIG.STABILITY_FRAMES})`);

        if (!stableLabel) {
            signBadge.textContent = '[—]';
            translatedText.textContent = 'Hold gesture steady…';
        }
    }
}

function handleNoHandsOrError(message) {
    handsCountText.textContent = 'Hands detected: 0';
    clearOverlayCanvas();
    recStatusText.textContent = 'Waiting Hand';
    recStatusBadge.classList.remove('active');
    hadZeroHandsSinceLastRecord = true;
    lastRawLabel = null;
    rawConsistencyCount = 0;

    // Clear stable state and result card
    stableLabel = null;
    stableTranslations = {};
    stableConfidence = 0.0;
    isStabilized = false;
    signBadge.textContent = '[—]';
    translatedText.textContent = 'Show your hand to the camera';
    updateConfidenceGauge(0, false);
    setStabilityStatus('waiting', 'Waiting for Hand');
}

// ==========================================================================
// 6. Result Card & Confidence Display
// ==========================================================================

function updateResultCard(label, translations, confidence) {
    signBadge.textContent = `[${label}]`;

    const langMeta = LANGUAGE_META[selectedTextLang] || LANGUAGE_META.english;
    selectedTextLangFlag.textContent = langMeta.flag;
    selectedTextLangLabel.textContent = `${langMeta.name} Translation`;

    const displayText = (translations && translations[selectedTextLang]) 
                        ? translations[selectedTextLang] 
                        : (translations && translations.english ? translations.english : '—');
    translatedText.textContent = displayText;

    updateConfidenceGauge(confidence, true);

    const audioMeta = LANGUAGE_META[selectedAudioLang] || LANGUAGE_META.hindi;
    const pct = Math.round(confidence * 100);

    if (confidence >= 0.80) {
        guidanceText.textContent = `✨ Stable "${label}" (${pct}%) • Displaying in ${langMeta.name} • Ready to speak in ${audioMeta.name}.`;
        guidanceIcon.textContent = '✨';
    } else {
        guidanceText.textContent = `ℹ️ Stable "${label}" (${pct}%) • Displaying in ${langMeta.name} • Ready to speak.`;
        guidanceIcon.textContent = 'ℹ️';
    }
}

function setStabilityStatus(type, text) {
    if (!stabilityPill || !stabilityText) return;

    stabilityPill.classList.remove('status-stable', 'status-stabilizing', 'status-low-conf', 'status-waiting');

    switch (type) {
        case 'stable':
            stabilityPill.classList.add('status-stable');
            stabilityText.textContent = text || '✓ Stable';
            break;
        case 'stabilizing':
            stabilityPill.classList.add('status-stabilizing');
            stabilityText.textContent = text || '⏳ Stabilizing...';
            break;
        case 'low_conf':
            stabilityPill.classList.add('status-low-conf');
            stabilityText.textContent = text || '⚠ Low Confidence';
            break;
        case 'waiting':
        default:
            stabilityPill.classList.add('status-waiting');
            stabilityText.textContent = text || 'Waiting';
            break;
    }
}

/**
 * Confidence Handling with Strict States:
 * High:   >= 80% (✓ High Confidence)
 * Medium: 60% - 79% (⚠ Medium Confidence)
 * Low:    < 60% (⚠ Low Confidence — Please repeat the sign)
 */
function updateConfidenceGauge(confidence, hasHand = true) {
    const pct = Math.round(confidence * 100);
    confidenceValue.textContent = `${pct}%`;
    confidenceFill.style.width = `${pct}%`;

    if (!confidenceStateBadge) return;

    confidenceStateBadge.classList.remove('state-high', 'state-med', 'state-low');

    if (!hasHand || confidence === 0) {
        confidenceValue.style.color = 'var(--text-muted)';
        confidenceFill.style.background = 'var(--bg-badge)';
        confidenceStateIcon.textContent = 'ℹ️';
        confidenceStateText.textContent = 'Waiting for sign';
        return;
    }

    if (pct >= 80) {
        confidenceValue.style.color = 'var(--accent-green)';
        confidenceFill.style.background = 'linear-gradient(90deg, var(--accent-purple), var(--accent-green))';
        confidenceStateBadge.classList.add('state-high');
        confidenceStateIcon.textContent = '✓';
        confidenceStateText.textContent = '✓ High Confidence';
    } else if (pct >= 60) {
        confidenceValue.style.color = 'var(--accent-amber)';
        confidenceFill.style.background = 'var(--accent-amber)';
        confidenceStateBadge.classList.add('state-med');
        confidenceStateIcon.textContent = '⚠';
        confidenceStateText.textContent = '⚠ Medium Confidence';
    } else {
        confidenceValue.style.color = 'var(--accent-red)';
        confidenceFill.style.background = 'var(--accent-red)';
        confidenceStateBadge.classList.add('state-low');
        confidenceStateIcon.textContent = '⚠';
        confidenceStateText.textContent = '⚠ Low Confidence — Please repeat the sign';
    }
}

// drawHandLandmarks removed — skeleton/dot overlay disabled per user request

function clearOverlayCanvas() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ==========================================================================
// 7. Prediction History Management
// ==========================================================================

function addHistoryEntry(label, translations, confidence) {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0]; // "HH:MM:SS"
    const langMeta = LANGUAGE_META[selectedTextLang] || LANGUAGE_META.english;
    const text = (translations && translations[selectedTextLang]) 
                 ? translations[selectedTextLang] 
                 : (translations && translations.english ? translations.english : label);

    const pct = Math.round(confidence * 100);
    const tier = pct >= 80 ? 'high' : (pct >= 60 ? 'med' : 'low');

    const entry = {
        id: Date.now() + Math.random(),
        time: timeStr,
        sign: label,
        langId: selectedTextLang,
        langName: langMeta.name,
        langFlag: langMeta.flag,
        translatedText: text,
        confidencePct: pct,
        tier: tier
    };

    historyList.unshift(entry);
    if (historyList.length > MAX_HISTORY_ENTRIES) {
        historyList = historyList.slice(0, MAX_HISTORY_ENTRIES);
    }

    renderHistoryTable();
}

function renderHistoryTable() {
    if (!historyTableBody || !historyCountBadge) return;

    historyCountBadge.textContent = historyList.length;

    if (historyList.length === 0) {
        historyTableBody.innerHTML = `
            <tr class="history-empty-row" id="history-empty-row">
                <td colspan="5">No recognition history yet. Show a sign to the camera to record.</td>
            </tr>
        `;
        return;
    }

    historyTableBody.innerHTML = historyList.map(item => `
        <tr class="history-row">
            <td class="history-time">${escapeHtml(item.time)}</td>
            <td><span class="history-sign-pill">[${escapeHtml(item.sign)}]</span></td>
            <td><span class="history-lang-badge">${item.langFlag} ${escapeHtml(item.langName)}</span></td>
            <td class="history-trans-text">${escapeHtml(item.translatedText)}</td>
            <td>
                <span class="history-conf-badge conf-${item.tier}">
                    ${item.tier === 'high' ? '✓' : '⚠'} ${item.confidencePct}%
                </span>
            </td>
        </tr>
    `).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function onClearHistory() {
    historyList = [];
    lastRecordedSign = null;
    lastRecordedTime = 0;
    hadZeroHandsSinceLastRecord = true;
    renderHistoryTable();
}

// ==========================================================================
// 8. Language Selectors & Audio Handlers
// ==========================================================================

function onTextLanguageChange() {
    selectedTextLang = selectTextLang.value;
    const langMeta = LANGUAGE_META[selectedTextLang] || LANGUAGE_META.english;
    selectedTextLangFlag.textContent = langMeta.flag;
    selectedTextLangLabel.textContent = `${langMeta.name} Translation`;

    if (stableLabel && stableTranslations) {
        const text = stableTranslations[selectedTextLang] || stableTranslations.english || 'Translation unavailable';
        translatedText.textContent = text;
        guidanceText.textContent = `Text language switched to ${langMeta.name} (${langMeta.flag}).`;
        guidanceIcon.textContent = '📝';
    } else {
        translatedText.textContent = 'Show your hand to the camera';
    }
}

function onAudioLanguageChange() {
    selectedAudioLang = selectAudioLang.value;
    const audioMeta = LANGUAGE_META[selectedAudioLang] || LANGUAGE_META.hindi;
    btnSpeakLabel.textContent = `Speak (${audioMeta.name})`;

    if (selectedAudioLang === 'odia') {
        guidanceText.textContent = `Audio Language: Odia. (Neural Odia voice: or-IN-SubhasiniNeural / Azure Speech).`;
        guidanceIcon.textContent = 'ℹ️';
    } else {
        guidanceText.textContent = `Audio Language: ${audioMeta.name} (${audioMeta.flag}) • Neural Voice Ready.`;
        guidanceIcon.textContent = '🔊';
    }
}

function findBrowserVoice(localeCode) {
    if (!('speechSynthesis' in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    const prefix = localeCode.split('-')[0].toLowerCase();
    return voices.find(v => v.lang.toLowerCase().startsWith(prefix));
}

/**
 * Speak the translation of the current STABLE prediction
 */
async function onSpeak() {
    if (!stableLabel || !stableTranslations) {
        guidanceText.textContent = 'No stable sign confirmed yet. Hold your sign steadily first!';
        guidanceIcon.textContent = '⚠️';
        return;
    }

    const audioMeta = LANGUAGE_META[selectedAudioLang] || LANGUAGE_META.hindi;
    const textToSpeak = stableTranslations[selectedAudioLang] || stableTranslations.english;

    if (!textToSpeak || textToSpeak === '—' || textToSpeak.includes('अनुपलब्ध') || textToSpeak.includes('unavailable')) {
        guidanceText.textContent = `Translation not available to speak in ${audioMeta.name}.`;
        guidanceIcon.textContent = '⚠️';
        return;
    }

    btnSpeak.classList.add('speaking');
    guidanceText.textContent = `🔊 Speaking ${audioMeta.name}...`;
    guidanceIcon.textContent = '🔊';

    try {
        const audioUrl = `/api/tts?text=${encodeURIComponent(textToSpeak)}&lang=${encodeURIComponent(selectedAudioLang)}`;
        const response = await fetch(audioUrl);

        if (response.ok) {
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            ttsAudio.src = blobUrl;
            await ttsAudio.play();

            guidanceText.textContent = `✓ ${audioMeta.name} voice played ("${textToSpeak}")`;
            guidanceIcon.textContent = '✓';
            return;
        }

        if (response.status === 404) {
            const data = await response.json().catch(() => ({}));
            
            // Try browser native voice if available
            const browserVoice = findBrowserVoice(audioMeta.bcp47);
            if (browserVoice) {
                const utterance = new SpeechSynthesisUtterance(textToSpeak);
                utterance.voice = browserVoice;
                utterance.lang = browserVoice.lang;
                window.speechSynthesis.speak(utterance);

                guidanceText.textContent = `✓ ${audioMeta.name} voice played via Browser (${browserVoice.name})`;
                guidanceIcon.textContent = '✓';
                return;
            }

            // Honest error reporting
            guidanceText.textContent = `⚠ ${audioMeta.name} voice unavailable. Displaying "${textToSpeak}" as text.`;
            guidanceIcon.textContent = '⚠';
            return;
        }

        throw new Error(`TTS server error (${response.status})`);

    } catch (err) {
        console.warn('Audio playback notice:', err);
        guidanceText.textContent = `⚠ ${audioMeta.name} voice unavailable on this device.`;
        guidanceIcon.textContent = '⚠';
    } finally {
        setTimeout(() => {
            btnSpeak.classList.remove('speaking');
        }, 1200);
    }
}

function onClear() {
    stableLabel = null;
    stableTranslations = {};
    stableConfidence = 0.0;
    lastRawLabel = null;
    rawConsistencyCount = 0;
    isStabilized = false;

    signBadge.textContent = '[WAITING]';
    translatedText.textContent = 'Show your hand to the camera';
    setStabilityStatus('waiting', 'Waiting');
    updateConfidenceGauge(0, false);

    guidanceText.textContent = 'Recognition cleared. Show a sign to start.';
    guidanceIcon.textContent = '💡';
}

// ==========================================================================
// 9. Event Listeners & Initialization
// ==========================================================================

btnRequestCamera.addEventListener('click', startWebcam);
btnToggleCamera.addEventListener('click', () => {
    if (isCameraRunning) {
        stopWebcam();
    } else {
        startWebcam();
    }
});

selectTextLang.addEventListener('change', onTextLanguageChange);
selectAudioLang.addEventListener('change', onAudioLanguageChange);
btnSpeak.addEventListener('click', onSpeak);
btnClear.addEventListener('click', onClear);
if (btnClearHistory) {
    btnClearHistory.addEventListener('click', onClearHistory);
}

// Auto-start on load
window.addEventListener('DOMContentLoaded', () => {
    onAudioLanguageChange();
    onTextLanguageChange();
    renderHistoryTable();
    setStabilityStatus('waiting', 'Waiting');

    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            const statusPill = document.getElementById('model-status-text');
            if (data.model_loaded) {
                if (statusPill) {
                    statusPill.textContent = `${data.model_title || '17-Class Model Loaded'} (${data.model_badge || '126 Features | 17 Classes'})`;
                }
            } else {
                if (statusPill) {
                    statusPill.textContent = '❌ Model Failed to Load';
                }
            }
            if (data.classes && Array.isArray(data.classes)) {
                const listEl = document.getElementById('supported-signs-list');
                if (listEl) listEl.textContent = data.classes.join(', ');
            }
        })
        .catch(() => {});

    if ('speechSynthesis' in window) {
        window.speechSynthesis.getVoices();
    }

    startWebcam();
});
