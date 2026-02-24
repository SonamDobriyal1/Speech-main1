import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

(() => {
  'use strict';

  const startBtn = document.getElementById('startCameraBtn');
  const stopBtn = document.getElementById('stopCameraBtn');
  const attentionScoreEl = document.getElementById('attentionScore');
  const attentionStatusEl = document.getElementById('attentionStatus');
  const videoEl = document.getElementById('attentionVideo');
  const canvasEl = document.getElementById('attentionCanvas');

  if (!startBtn || !stopBtn || !attentionScoreEl || !attentionStatusEl || !videoEl || !canvasEl) {
    return;
  }

  let faceLandmarker = null;
  let videoStream = null;
  let isRunning = false;
  let lastVideoTime = -1;
  let animationId = null;
  const scoreHistory = [];
  const sessionSamples = [];
  const maxHistory = 8;

  const canvasCtx = canvasEl.getContext('2d');

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function setStatus(message, tone) {
    attentionStatusEl.textContent = message;
    attentionStatusEl.className = `mt-2 text-base font-semibold ${tone || 'text-white'}`;
  }

  function setScore(value) {
    attentionScoreEl.textContent = `${value}%`;
  }

  async function ensureFaceLandmarker() {
    if (faceLandmarker) {
      return faceLandmarker;
    }

    setStatus('Loading attention model...', 'text-cyan-200');
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
    return faceLandmarker;
  }

  function computeAttention(landmarks) {
    const leftEyeOuter = landmarks[33];
    const leftEyeInner = landmarks[133];
    const rightEyeInner = landmarks[362];
    const rightEyeOuter = landmarks[263];
    const leftEyeTop = landmarks[159];
    const leftEyeBottom = landmarks[145];
    const rightEyeTop = landmarks[386];
    const rightEyeBottom = landmarks[374];
    const noseTip = landmarks[1];

    const leftEyeCenterX = (leftEyeOuter.x + leftEyeInner.x) / 2;
    const rightEyeCenterX = (rightEyeOuter.x + rightEyeInner.x) / 2;
    const eyesMidX = (leftEyeCenterX + rightEyeCenterX) / 2;

    const leftEyeCenterY = (leftEyeTop.y + leftEyeBottom.y) / 2;
    const rightEyeCenterY = (rightEyeTop.y + rightEyeBottom.y) / 2;
    const eyesMidY = (leftEyeCenterY + rightEyeCenterY) / 2;

    const yawOffset = noseTip.x - eyesMidX;
    const pitchOffset = noseTip.y - eyesMidY;

    const yawScore = clamp(1 - Math.abs(yawOffset) / 0.08, 0, 1);
    const pitchScore = clamp(1 - Math.abs(pitchOffset) / 0.1, 0, 1);

    const leftOpen = Math.abs(leftEyeTop.y - leftEyeBottom.y);
    const rightOpen = Math.abs(rightEyeTop.y - rightEyeBottom.y);
    const openScore = clamp((leftOpen + rightOpen) / 2 - 0.015, 0, 0.03) / 0.03;

    const blended = 0.55 * yawScore + 0.25 * pitchScore + 0.2 * openScore;
    const score = Math.round(clamp(blended, 0, 1) * 100);

    return {
      score,
      nose: noseTip,
      leftEye: { x: leftEyeCenterX, y: leftEyeCenterY },
      rightEye: { x: rightEyeCenterX, y: rightEyeCenterY },
    };
  }

  function smoothScore(value) {
    scoreHistory.push(value);
    if (scoreHistory.length > maxHistory) {
      scoreHistory.shift();
    }
    const sum = scoreHistory.reduce((acc, item) => acc + item, 0);
    return Math.round(sum / scoreHistory.length);
  }

  function recordSample(value) {
    sessionSamples.push(value);
  }

  function resetSession() {
    sessionSamples.length = 0;
  }

  function getAverage() {
    if (!sessionSamples.length) {
      return null;
    }
    const sum = sessionSamples.reduce((acc, item) => acc + item, 0);
    return Math.round(sum / sessionSamples.length);
  }

  function resizeCanvas() {
    if (!videoEl.videoWidth || !videoEl.videoHeight) {
      return;
    }
    if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
    }
  }

  function drawOverlay(attention) {
    if (!canvasCtx || !attention) {
      return;
    }

    resizeCanvas();
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const nose = attention.nose;
    const leftEye = attention.leftEye;
    const rightEye = attention.rightEye;

    const toPixel = (point) => ({
      x: point.x * canvasEl.width,
      y: point.y * canvasEl.height,
    });

    const nosePx = toPixel(nose);
    const leftEyePx = toPixel(leftEye);
    const rightEyePx = toPixel(rightEye);

    canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
    canvasCtx.lineWidth = 2;
    canvasCtx.beginPath();
    canvasCtx.moveTo(leftEyePx.x, leftEyePx.y);
    canvasCtx.lineTo(rightEyePx.x, rightEyePx.y);
    canvasCtx.stroke();

    canvasCtx.fillStyle = 'rgba(16, 185, 129, 0.9)';
    canvasCtx.beginPath();
    canvasCtx.arc(nosePx.x, nosePx.y, 4, 0, Math.PI * 2);
    canvasCtx.fill();
  }

  function clearOverlay() {
    if (!canvasCtx) {
      return;
    }
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }

  async function startCamera() {
    if (isRunning) {
      return;
    }

    startBtn.disabled = true;
    stopBtn.disabled = false;
    setScore('--');
    resetSession();

    try {
      await ensureFaceLandmarker();
      videoStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          facingMode: 'user',
        },
        audio: false,
      });
      videoEl.srcObject = videoStream;
      await videoEl.play();
      isRunning = true;
      lastVideoTime = -1;
      scoreHistory.length = 0;
      setStatus('Analyzing...', 'text-emerald-200');
      renderLoop();
    } catch (error) {
      startBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus(`Camera unavailable: ${error.message || error}`, 'text-rose-200');
    }
  }

  function stopCamera() {
    if (!isRunning) {
      return;
    }

    isRunning = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop());
      videoStream = null;
    }

    if (videoEl.srcObject) {
      videoEl.srcObject = null;
    }

    clearOverlay();
    setScore('--');
    setStatus('Camera idle', 'text-white');
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  function renderLoop() {
    if (!isRunning) {
      return;
    }

    if (!faceLandmarker || !videoEl.videoWidth) {
      animationId = requestAnimationFrame(renderLoop);
      return;
    }

    if (videoEl.currentTime !== lastVideoTime) {
      lastVideoTime = videoEl.currentTime;
      const results = faceLandmarker.detectForVideo(videoEl, performance.now());

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const attention = computeAttention(results.faceLandmarks[0]);
        const smoothed = smoothScore(attention.score);
        recordSample(smoothed);
        setScore(smoothed);

        if (smoothed >= 65) {
          setStatus('In focus', 'text-emerald-200');
        } else if (smoothed >= 40) {
          setStatus('Glancing away', 'text-amber-200');
        } else {
          setStatus('Looking away', 'text-rose-200');
        }

        drawOverlay(attention);
      } else {
        clearOverlay();
        setScore('--');
        setStatus('No face detected', 'text-slate-200');
      }
    }

    animationId = requestAnimationFrame(renderLoop);
  }

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);

  window.AttentionTracker = {
    reset: resetSession,
    getAverage,
    isRunning: () => isRunning,
  };
})();
