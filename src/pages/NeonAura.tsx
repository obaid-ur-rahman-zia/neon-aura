import { useEffect, useRef, useState, useCallback, type CSSProperties } from "react";

declare global {
  interface Window {
    Hands: new (config: { locateFile: (f: string) => string }) => MediaPipeHands;
    Camera: new (
      video: HTMLVideoElement,
      config: { onFrame: () => Promise<void>; facingMode: string }
    ) => { start: () => void };
    HAND_CONNECTIONS: [number, number][];
    drawConnectors: (
      ctx: CanvasRenderingContext2D,
      landmarks: Landmark[],
      connections: [number, number][],
      options: { color: string; lineWidth: number }
    ) => void;
    drawLandmarks: (
      ctx: CanvasRenderingContext2D,
      landmarks: Landmark[],
      options: { color: string; lineWidth: number }
    ) => void;
  }
}

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface MediaPipeHands {
  setOptions: (opts: {
    maxNumHands: number;
    modelComplexity: number;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
  }) => void;
  onResults: (cb: (results: { multiHandLandmarks?: Landmark[][] }) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
}

interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  size: number;
  color: string;
}

interface Ripple {
  x: number; y: number;
  radius: number;
  maxRadius: number;
  life: number;
  color: string;
}

interface FireParticle {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  maxLife: number;
  size: number;
}

interface Fireball {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  size: number;
}

interface BgEmber {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  size: number;
  hue: number;
}

interface LightsaberState {
  progress: number;
  active: boolean;
}

interface ClashSpark {
  x: number; y: number;
  vx: number; vy: number;
  life: number;
  maxLife: number;
  size: number;
  color: string;
}

interface BladeGeom {
  baseX: number; baseY: number;
  tipX: number; tipY: number;
  progress: number;
}

type ThemeName = "Prism" | "Rapture" | "Blade" | "Ember" | "Tide" | "Cosmos" | "Fire" | "Lightsaber";

const THEMES: Record<string, (t: number, index: number, total: number) => string> = {
  Prism: (t, index, total) => `hsl(${(t * 100 + index * (360 / total)) % 360}, 100%, 60%)`,
  Rapture: (t, index) => `hsl(0, 100%, ${46 + Math.sin(t * 3 + index * 0.9) * 14}%)`,
  Blade: (t, index) => (index % 2 === 0) ? "#ff003c" : "#00f0ff",
  Ember: (t, index) => `hsl(${(10 + (index * 10)) % 40}, 100%, ${50 + Math.sin(t) * 10}%)`,
  Tide: (t, index) => `hsl(${180 + (index * 20)}, 100%, 60%)`,
  Cosmos: (t, index) => `hsl(${260 + Math.sin(t * 2 + index) * 40}, 100%, 65%)`,
};

const BLADE_COLORS = ["#ff003c", "#00f0ff", "#ff003c", "#00f0ff", "#ff003c"];
const FINGER_TIPS = [4, 8, 12, 16, 20];

const FIRE_LUT = (() => {
  const lut: string[] = [];
  for (let i = 0; i < 128; i++) {
    const t = i / 127;
    let r: number, g: number, b: number;
    if (t < 0.25) {
      const s = t / 0.25;
      r = 255; g = Math.floor(255 - s * 60); b = Math.floor(210 - s * 210);
    } else if (t < 0.58) {
      const s = (t - 0.25) / 0.33;
      r = 255; g = Math.floor(195 - s * 145); b = 0;
    } else {
      const s = (t - 0.58) / 0.42;
      r = Math.floor(255 - s * 130); g = Math.floor(50 - s * 50); b = 0;
    }
    lut.push(`rgb(${r},${g},${b})`);
  }
  return lut;
})();

function getDist(p1: { x: number; y: number }, p2: { x: number; y: number }) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function getDist3(p1: Landmark, p2: Landmark): number {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z - p2.z) * 0.5);
}

// Orientation-independent fist detection using 3D wrist-relative distances.
// Works upside-down, sideways, etc.
function detectFist(hand: Landmark[]): boolean {
  const wrist = hand[0];
  const palmCenter = {
    x: (hand[0].x + hand[5].x + hand[9].x + hand[13].x + hand[17].x) / 5,
    y: (hand[0].y + hand[5].y + hand[9].y + hand[13].y + hand[17].y) / 5,
    z: (hand[0].z + hand[5].z + hand[9].z + hand[13].z + hand[17].z) / 5,
  };
  const fingerDefs = [
    { tip: 8,  mcp: 5  },
    { tip: 12, mcp: 9  },
    { tip: 16, mcp: 13 },
    { tip: 20, mcp: 17 },
  ];
  let curled = 0;
  for (const f of fingerDefs) {
    const tipDist = getDist3(hand[f.tip], palmCenter as Landmark);
    const mcpDist = getDist3(hand[f.mcp], wrist);
    // Curled when fingertip is close to palm center relative to how far the knuckle is from wrist
    if (tipDist < mcpDist * 1.25) curled++;
  }
  return curled >= 3;
}

type SaberColorKey = "red" | "blue" | "green";

const SABER_COLORS: Record<SaberColorKey, {
  halo: [number,number,number];
  mid:  [number,number,number];
  core: [number,number,number];
  white:[number,number,number];
  shadow:[number,number,number];
  refl: [number,number,number];
  accent: string;
}> = {
  red: {
    halo:  [255,  0,  0], mid:  [255, 30, 30], core:  [255,110,110],
    white: [255,230,230], shadow:[255, 0,  0], refl:  [255, 40, 40],
    accent: "#ff2222",
  },
  blue: {
    halo:  [ 20, 80,255], mid:  [ 40,110,255], core:  [110,165,255],
    white: [210,225,255], shadow:[ 0, 60,255], refl:  [ 40, 90,255],
    accent: "#2266ff",
  },
  green: {
    halo:  [  0,210, 50], mid:  [ 20,230, 70], core:  [ 80,255,130],
    white: [210,255,225], shadow:[  0,200, 50], refl:  [ 20,230, 70],
    accent: "#00dd44",
  },
};

function rgba([r,g,b]: [number,number,number], a: number) {
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

// Segment–segment intersection; returns 2D hit point or null
function segmentsIntersect(
  ax: number, ay: number, bx: number, by: number,
  cx: number, cy: number, dx: number, dy: number
): { x: number; y: number } | null {
  const denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (Math.abs(denom) < 1e-8) return null;
  const t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom;
  const u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom;
  if (t >= 0.05 && t <= 1 && u >= 0.05 && u <= 1) {
    return { x: ax + t * (bx - ax), y: ay + t * (by - ay) };
  }
  return null;
}

export default function NeonAura() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);

  const [started, setStarted] = useState(false);
  const [currentTheme, setCurrentTheme] = useState<ThemeName>("Prism");
  const [saberColor, setSaberColor] = useState<SaberColorKey>("red");
  const [zoomPct, setZoomPct] = useState(100);
  /** Matches bright/dim slider default (40 ≈ brightness 0.66). */
  const [dimPct, setDimPct] = useState(40);

  const stateRef = useRef({
    width: window.innerWidth,
    height: window.innerHeight,
    time: 0,
    lastTime: performance.now(),
    framesThisSecond: 0,
    lastFpsTime: performance.now(),
    currentHands: [] as Landmark[][],
    handVelocities: 0,
    currentTheme: "Prism" as ThemeName,
    particles: [] as Particle[],
    ripples: [] as Ripple[],
    bgEmbers: [] as BgEmber[],
    fireMode: false,
    saberMode: false,
    fireParticles: [] as FireParticle[],
    fireballs: [] as Fireball[],
    lastFistState: [false, false] as [boolean, boolean],
    handVelDirs: [{ x: 0, y: 1 }, { x: 0, y: 1 }],
    prevHandPalms: [null, null] as (({ x: number; y: number }) | null)[],
    smoothVel: 0,
    smoothFlux: 0,
    lastVel: 0,
    smoothSync: 0,
    lastPinchState: [false, false] as [boolean, boolean],
    // Lightsaber state per hand (max 2)
    sabers: [
      { progress: 0, active: false },
      { progress: 0, active: false },
    ] as LightsaberState[],
    // Clash / duel effects
    clashSparks: [] as ClashSpark[],
    clashFlash: 0,          // 0–1, fades each frame
    lastClashTime: 0,       // timestamp to throttle sounds
    clashPoint: null as { x: number; y: number } | null,
    saberColor: "red" as SaberColorKey,
    audioCtx: null as AudioContext | null,
    humOsc: null as OscillatorNode | null,
    humGain: null as GainNode | null,
  });

  const hudRef = useRef({
    fps: "0",
    vel: "0.00",
    flux: "0.00",
    sync: "——",
    barVel: 0,
    barFlux: 0,
    barSync: 0,
    barFps: 0,
  });

  const hudFpsEl = useRef<HTMLSpanElement>(null);
  const hudVelEl = useRef<HTMLSpanElement>(null);
  const hudFluxEl = useRef<HTMLSpanElement>(null);
  const hudSyncEl = useRef<HTMLSpanElement>(null);
  const barVelEl = useRef<HTMLDivElement>(null);
  const barFluxEl = useRef<HTMLDivElement>(null);
  const barSyncEl = useRef<HTMLDivElement>(null);
  const barFpsEl = useRef<HTMLDivElement>(null);

  const dimSliderRef = useRef<HTMLInputElement>(null);
  const zoomSliderRef = useRef<HTMLInputElement>(null);

  /** Pixels of the video frame as displayed (object-fit: cover), matching MediaPipe’s normalized [0,1] space. */
  function getVideoLayoutRect(cw: number, ch: number) {
    const video = videoRef.current;
    const vw = video?.videoWidth ?? 0;
    const vh = video?.videoHeight ?? 0;
    if (!vw || !vh) {
      return { offsetX: 0, offsetY: 0, drawW: cw, drawH: ch };
    }
    const scale = Math.max(cw / vw, ch / vh);
    const drawW = vw * scale;
    const drawH = vh * scale;
    return {
      offsetX: (cw - drawW) / 2,
      offsetY: (ch - drawH) / 2,
      drawW,
      drawH,
    };
  }

  function mapToCanvas(point: Landmark, cw: number, ch: number) {
    const { offsetX, offsetY, drawW, drawH } = getVideoLayoutRect(cw, ch);
    return {
      x: offsetX + point.x * drawW,
      y: offsetY + point.y * drawH,
    };
  }

  /** MediaPipe drawConnectors assumes landmarks map to the full canvas; remap to the displayed video rect. */
  function drawHandConnectors(
    ctx: CanvasRenderingContext2D,
    hand: Landmark[],
    opts: { color: string; lineWidth: number },
  ) {
    if (!window.drawConnectors || !window.HAND_CONNECTIONS) return;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    const { offsetX, offsetY, drawW, drawH } = getVideoLayoutRect(w, h);
    ctx.save();
    ctx.setTransform(drawW / w, 0, 0, drawH / h, offsetX, offsetY);
    window.drawConnectors(ctx, hand, window.HAND_CONNECTIONS, opts);
    ctx.restore();
  }

  const mirrorZoomTransform = `scaleX(-1) scale(${zoomPct / 100})`;
  const videoBrightness = 1.0 - (dimPct / 100) * 0.85;
  const videoStyle: CSSProperties = {
    transform: mirrorZoomTransform,
    filter: `brightness(${videoBrightness.toFixed(2)}) contrast(1.1)`,
  };

  function createParticle(x: number, y: number, color: string): Particle {
    const angle = Math.random() * Math.PI * 2;
    const speed = 1 + Math.random() * 3;
    return {
      x, y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: 1,
      size: 2 + Math.random() * 4,
      color,
    };
  }

  function createShockwave(x: number, y: number, color: string) {
    const s = stateRef.current;
    s.ripples.push({ x, y, radius: 5, maxRadius: 150, life: 1, color });
    for (let i = 0; i < 12; i++) {
      s.particles.push(createParticle(x, y, color));
    }
  }

  function triggerZap() {
    const s = stateRef.current;
    if (!s.audioCtx) return;
    const osc = s.audioCtx.createOscillator();
    const gain = s.audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(800, s.audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(40, s.audioCtx.currentTime + 0.1);
    gain.gain.setValueAtTime(0.5, s.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, s.audioCtx.currentTime + 0.1);
    osc.connect(gain);
    gain.connect(s.audioCtx.destination);
    osc.start();
    osc.stop(s.audioCtx.currentTime + 0.15);
  }

  function triggerSaberIgnite() {
    const s = stateRef.current;
    if (!s.audioCtx) return;
    const now = s.audioCtx.currentTime;

    // ── Layer 1: initial electrical snap (short noise-like burst) ──
    const snapOsc = s.audioCtx.createOscillator();
    const snapGain = s.audioCtx.createGain();
    snapOsc.type = "sawtooth";
    snapOsc.frequency.setValueAtTime(1800, now);
    snapOsc.frequency.exponentialRampToValueAtTime(300, now + 0.06);
    snapGain.gain.setValueAtTime(0.28, now);
    snapGain.gain.exponentialRampToValueAtTime(0.001, now + 0.07);
    snapOsc.connect(snapGain); snapGain.connect(s.audioCtx.destination);
    snapOsc.start(now); snapOsc.stop(now + 0.08);

    // ── Layer 2: main rising "vwoom" — the classic ignition sweep ──
    const mainOsc = s.audioCtx.createOscillator();
    const mainGain = s.audioCtx.createGain();
    const filter = s.audioCtx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(200, now);
    filter.frequency.exponentialRampToValueAtTime(800, now + 0.55);
    filter.Q.value = 1.8;
    mainOsc.type = "sawtooth";
    mainOsc.frequency.setValueAtTime(55, now);
    mainOsc.frequency.exponentialRampToValueAtTime(140, now + 0.45);
    mainOsc.frequency.exponentialRampToValueAtTime(110, now + 0.75);
    mainGain.gain.setValueAtTime(0, now);
    mainGain.gain.linearRampToValueAtTime(0.45, now + 0.04);
    mainGain.gain.setValueAtTime(0.45, now + 0.35);
    mainGain.gain.exponentialRampToValueAtTime(0.09, now + 0.8);
    mainOsc.connect(filter); filter.connect(mainGain); mainGain.connect(s.audioCtx.destination);
    mainOsc.start(now); mainOsc.stop(now + 0.85);

    // ── Layer 3: sub rumble for body/weight ──
    const subOsc = s.audioCtx.createOscillator();
    const subGain = s.audioCtx.createGain();
    subOsc.type = "sine";
    subOsc.frequency.setValueAtTime(48, now);
    subOsc.frequency.linearRampToValueAtTime(90, now + 0.5);
    subGain.gain.setValueAtTime(0, now);
    subGain.gain.linearRampToValueAtTime(0.22, now + 0.03);
    subGain.gain.exponentialRampToValueAtTime(0.001, now + 0.7);
    subOsc.connect(subGain); subGain.connect(s.audioCtx.destination);
    subOsc.start(now); subOsc.stop(now + 0.75);

    // ── Layer 4: high harmonic shimmer ──
    const shimOsc = s.audioCtx.createOscillator();
    const shimGain = s.audioCtx.createGain();
    shimOsc.type = "square";
    shimOsc.frequency.setValueAtTime(320, now + 0.02);
    shimOsc.frequency.exponentialRampToValueAtTime(680, now + 0.5);
    shimGain.gain.setValueAtTime(0, now + 0.02);
    shimGain.gain.linearRampToValueAtTime(0.07, now + 0.08);
    shimGain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
    shimOsc.connect(shimGain); shimGain.connect(s.audioCtx.destination);
    shimOsc.start(now + 0.02); shimOsc.stop(now + 0.7);
  }

  function triggerSaberRetract() {
    const s = stateRef.current;
    if (!s.audioCtx) return;
    const osc = s.audioCtx.createOscillator();
    const gain = s.audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(180, s.audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(50, s.audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.15, s.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, s.audioCtx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(s.audioCtx.destination);
    osc.start();
    osc.stop(s.audioCtx.currentTime + 0.35);
  }

  function triggerClash() {
    const s = stateRef.current;
    if (!s.audioCtx) return;
    const now = s.audioCtx.currentTime;
    // Crackling high-energy impact burst
    const osc1 = s.audioCtx.createOscillator();
    const gain1 = s.audioCtx.createGain();
    osc1.type = "sawtooth";
    osc1.frequency.setValueAtTime(600 + Math.random() * 300, now);
    osc1.frequency.exponentialRampToValueAtTime(80, now + 0.12);
    gain1.gain.setValueAtTime(0.35, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.14);
    osc1.connect(gain1); gain1.connect(s.audioCtx.destination);
    osc1.start(); osc1.stop(now + 0.15);
    // Secondary crackle layer
    const osc2 = s.audioCtx.createOscillator();
    const gain2 = s.audioCtx.createGain();
    osc2.type = "square";
    osc2.frequency.setValueAtTime(200 + Math.random() * 150, now);
    osc2.frequency.exponentialRampToValueAtTime(40, now + 0.1);
    gain2.gain.setValueAtTime(0.18, now);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
    osc2.connect(gain2); gain2.connect(s.audioCtx.destination);
    osc2.start(); osc2.stop(now + 0.12);
  }

  function spawnClashSparks(cx: number, cy: number) {
    const s = stateRef.current;
    const colors = [
      "rgba(255,255,180,1)", "rgba(255,220,80,1)", "rgba(255,160,40,1)",
      "rgba(255,100,100,1)", "rgba(255,255,255,1)", "rgba(255,200,120,1)",
    ];
    for (let i = 0; i < 32; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 80 + Math.random() * 280;
      const ml = 0.35 + Math.random() * 0.45;
      s.clashSparks.push({
        x: cx + (Math.random() - 0.5) * 8,
        y: cy + (Math.random() - 0.5) * 8,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 60,
        life: ml,
        maxLife: ml,
        size: 1.5 + Math.random() * 3,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }
    s.clashFlash = 1.0;
    s.clashPoint = { x: cx, y: cy };
  }

  function updateClashSparks(ctx: CanvasRenderingContext2D, dt: number) {
    const s = stateRef.current;

    // Fade flash
    s.clashFlash = Math.max(0, s.clashFlash - dt * 5);

    // Draw flash bloom at clash point
    if (s.clashFlash > 0 && s.clashPoint) {
      const flashRadius = 55 * (1 - s.clashFlash * 0.3);
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      const flashGrad = ctx.createRadialGradient(
        s.clashPoint.x, s.clashPoint.y, 0,
        s.clashPoint.x, s.clashPoint.y, flashRadius
      );
      flashGrad.addColorStop(0, `rgba(255,255,230,${s.clashFlash * 0.95})`);
      flashGrad.addColorStop(0.25, `rgba(255,160,60,${s.clashFlash * 0.7})`);
      flashGrad.addColorStop(0.6, `rgba(255,60,60,${s.clashFlash * 0.35})`);
      flashGrad.addColorStop(1, "rgba(255,0,0,0)");
      ctx.beginPath();
      ctx.arc(s.clashPoint.x, s.clashPoint.y, flashRadius, 0, Math.PI * 2);
      ctx.fillStyle = flashGrad;
      ctx.fill();
      ctx.restore();
    }

    // Update and draw sparks
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    for (let i = s.clashSparks.length - 1; i >= 0; i--) {
      const sp = s.clashSparks[i];
      sp.x  += sp.vx * dt;
      sp.y  += sp.vy * dt;
      sp.vy += 420 * dt;   // gravity
      sp.vx *= 1 - dt * 2; // air drag
      sp.life -= dt;
      if (sp.life <= 0) { s.clashSparks.splice(i, 1); continue; }
      const alpha = sp.life / sp.maxLife;
      const r = sp.size * alpha;

      // Trail
      ctx.globalAlpha = alpha * 0.5;
      ctx.strokeStyle = sp.color;
      ctx.lineWidth = r * 0.6;
      ctx.beginPath();
      ctx.moveTo(sp.x - sp.vx * dt * 3, sp.y - sp.vy * dt * 3);
      ctx.lineTo(sp.x, sp.y);
      ctx.stroke();

      // Spark dot
      ctx.globalAlpha = alpha;
      ctx.fillStyle = sp.color;
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2);
      ctx.fill();

      // Mini glow
      if (r > 2) {
        ctx.globalAlpha = alpha * 0.4;
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, r * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = sp.color;
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  function drawBackground() {
    const s = stateRef.current;
    const bgCanvas = bgCanvasRef.current;
    if (!bgCanvas) return;
    const bgCtx = bgCanvas.getContext("2d");
    if (!bgCtx) return;

    bgCtx.clearRect(0, 0, s.width, s.height);

    // Spawn embers
    while (s.bgEmbers.length < 180) {
      s.bgEmbers.push({
        x: Math.random() * s.width,
        y: s.height + 10,
        vx: (Math.random() - 0.5) * 0.4,
        vy: -(0.2 + Math.random() * 0.5),
        life: 1,
        size: 0.5 + Math.random() * 1.5,
        hue: 200 + Math.random() * 120,
      });
    }

    for (let i = s.bgEmbers.length - 1; i >= 0; i--) {
      const e = s.bgEmbers[i];
      e.x += e.vx;
      e.y += e.vy;
      e.life -= 0.003;
      if (e.life <= 0 || e.y < -10) {
        s.bgEmbers.splice(i, 1);
        continue;
      }
      bgCtx.beginPath();
      bgCtx.arc(e.x, e.y, e.size, 0, Math.PI * 2);
      const col = s.fireMode
        ? `hsla(${20 + Math.random() * 20}, 100%, 55%, ${e.life * 0.4})`
        : s.saberMode
          ? `hsla(0, 100%, 45%, ${e.life * 0.35})`
          : `hsla(${e.hue}, 80%, 60%, ${e.life * 0.25})`;
      bgCtx.fillStyle = col;
      bgCtx.fill();
    }
  }

  function drawSineWave(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number,
    x2: number, y2: number,
    amplitude: number,
    frequency: number,
    phase: number,
    maxSteps = 0
  ) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    const nx = -dy / len, ny = dx / len;
    const steps = maxSteps > 0 ? maxSteps : Math.max(40, Math.floor(len / 5));
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const wave = Math.sin(t * Math.PI * 2 * frequency + phase) * amplitude;
      const px = x1 + dx * t + nx * wave;
      const py = y1 + dy * t + ny * wave;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  function drawBladeLines(ctx: CanvasRenderingContext2D, hand: Landmark[], w: number, h: number, t: number) {
    const pairs = [[8, 5], [12, 9], [16, 13], [20, 17], [4, 0]] as [number, number][];
    pairs.forEach(([a, b], idx) => {
      const pa = mapToCanvas(hand[a], w, h);
      const pb = mapToCanvas(hand[b], w, h);
      const col = BLADE_COLORS[idx % BLADE_COLORS.length];
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.shadowBlur = 14;
      ctx.shadowColor = col;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.shadowBlur = 0;
    });
  }

  function spawnHandFire(x: number, y: number, intensity: number, fireParticles: FireParticle[]) {
    const count = Math.floor(intensity * 4);
    for (let i = 0; i < count; i++) {
      const maxLife = 0.5 + Math.random() * 0.8;
      fireParticles.push({
        x: x + (Math.random() - 0.5) * 15,
        y: y + (Math.random() - 0.5) * 10,
        vx: (Math.random() - 0.5) * 2.5,
        vy: -(1.5 + Math.random() * 2.5),
        life: maxLife,
        maxLife,
        size: 3 + Math.random() * 8,
      });
    }
  }

  function updateFireParticles(ctx: CanvasRenderingContext2D, dt: number, fireParticles: FireParticle[]) {
    for (let i = fireParticles.length - 1; i >= 0; i--) {
      const p = fireParticles[i];
      p.x += p.vx;
      p.y += p.vy * (60 * dt);
      p.vy += 0.02;
      p.vx *= 0.97;
      p.life -= dt * 1.2;
      if (p.life <= 0) { fireParticles.splice(i, 1); continue; }
      const ageRatio = 1 - p.life / p.maxLife;
      const lutIdx = Math.min(127, Math.floor(ageRatio * 128));
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * (p.life / p.maxLife), 0, Math.PI * 2);
      ctx.fillStyle = FIRE_LUT[lutIdx];
      ctx.globalAlpha = p.life / p.maxLife * 0.85;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function updateFireballs(ctx: CanvasRenderingContext2D, dt: number, fireballs: Fireball[], fireParticles: FireParticle[]) {
    for (let i = fireballs.length - 1; i >= 0; i--) {
      const fb = fireballs[i];
      fb.x += fb.vx * (60 * dt);
      fb.y += fb.vy * (60 * dt);
      fb.vy += 0.08;
      fb.life -= dt * 0.7;
      if (fb.life <= 0) { fireballs.splice(i, 1); continue; }
      for (let j = 0; j < 3; j++) {
        fireParticles.push({
          x: fb.x + (Math.random() - 0.5) * fb.size,
          y: fb.y + (Math.random() - 0.5) * fb.size,
          vx: (Math.random() - 0.5) * 2,
          vy: -(0.5 + Math.random() * 1.5),
          life: 0.4 + Math.random() * 0.4,
          maxLife: 0.4 + Math.random() * 0.4,
          size: fb.size * 0.5 + Math.random() * fb.size * 0.3,
        });
      }
      const grad = ctx.createRadialGradient(fb.x, fb.y, 0, fb.x, fb.y, fb.size);
      grad.addColorStop(0, "rgba(255,255,200,0.9)");
      grad.addColorStop(0.3, "rgba(255,150,0,0.7)");
      grad.addColorStop(1, "rgba(255,50,0,0)");
      ctx.beginPath();
      ctx.arc(fb.x, fb.y, fb.size, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.globalAlpha = fb.life;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  function drawFireHands(
    ctx: CanvasRenderingContext2D,
    hands: Landmark[],
    w: number, h: number,
    handVelocities: number,
    fireParticles: FireParticle[]
  ) {
    const vel = Math.min(handVelocities * 50, 1);
    const pts = [0, 8, 12, 16, 20];
    pts.forEach((lm) => {
      const pt = mapToCanvas(hands[lm], w, h);
      spawnHandFire(pt.x, pt.y, 0.45 + vel * 0.55, fireParticles);
    });
  }

  function detectFireGestures(
    hands: Landmark[][],
    lastFistState: [boolean, boolean],
    handVelDirs: { x: number; y: number }[],
    fireballs: Fireball[],
    w: number, h: number
  ) {
    hands.forEach((hand, idx) => {
      const isFist = detectFist(hand);
      if (isFist && !lastFistState[idx] && fireballs.length < 6) {
        const palm = mapToCanvas(hand[0], w, h);
        const dir = handVelDirs[idx] || { x: 0, y: -1 };
        const speed = 6 + Math.random() * 3;
        fireballs.push({
          x: palm.x, y: palm.y,
          vx: dir.x * speed, vy: dir.y * speed,
          life: 1,
          size: 25 + Math.random() * 15,
        });
      }
      lastFistState[idx] = isFist;
    });
  }

  function drawLightsabers(
    ctx: CanvasRenderingContext2D,
    hands: Landmark[][],
    sabers: LightsaberState[],
    dt: number,
    w: number, h: number,
    t: number,
    colorKey: SaberColorKey
  ) {
    const s = stateRef.current;
    const pal = SABER_COLORS[colorKey];
    const bladeGeoms: (BladeGeom | null)[] = [];

    hands.forEach((hand, idx) => {
      const isFist = detectFist(hand);
      if (!sabers[idx]) sabers[idx] = { progress: 0, active: false };
      const saber = sabers[idx];
      const wasActive = saber.active;

      if (isFist && !saber.active) {
        saber.active = true;
        if (!wasActive) triggerSaberIgnite();
      } else if (!isFist && saber.active) {
        saber.active = false;
        if (wasActive) triggerSaberRetract();
      }

      if (saber.active) {
        saber.progress = Math.min(1, saber.progress + dt / 0.35);
      } else {
        saber.progress = Math.max(0, saber.progress - dt / 0.25);
      }

      // --- Landmark geometry (always computed so skeleton draws even when blade is retracting) ---
      const wrist    = mapToCanvas(hand[0], w, h);
      const mcp5     = mapToCanvas(hand[5], w, h);   // index knuckle
      const mcp9     = mapToCanvas(hand[9], w, h);   // middle knuckle
      const mcp13    = mapToCanvas(hand[13], w, h);  // ring knuckle
      const mcp17    = mapToCanvas(hand[17], w, h);  // pinky knuckle

      // Knuckle midpoint — the blade emitter sits here
      const knuckleMidX = (mcp5.x + mcp9.x + mcp13.x + mcp17.x) / 4;
      const knuckleMidY = (mcp5.y + mcp9.y + mcp13.y + mcp17.y) / 4;

      // Palm axis: wrist → knuckle midpoint
      const pdx = knuckleMidX - wrist.x;
      const pdy = knuckleMidY - wrist.y;
      const palmLen = Math.hypot(pdx, pdy) || 1;
      const dirX = pdx / palmLen;
      const dirY = pdy / palmLen;
      const perpX = -dirY;
      const perpY = dirX;

      // Hilt width proportional to the actual hand width (index–pinky knuckle span)
      const knuckleSpan = Math.hypot(mcp5.x - mcp17.x, mcp5.y - mcp17.y);
      const hiltHalfW = Math.max(6, Math.min(14, knuckleSpan / 3.5));

      // The hilt occupies the exact fist volume: wrist → knuckle midpoint
      const hiltBaseX = wrist.x;
      const hiltBaseY = wrist.y;
      const emitterX  = knuckleMidX;
      const emitterY  = knuckleMidY;

      // Blade starts just past the emitter, extends in palm direction
      const BLADE_FULL_LENGTH = 380;
      const currentLength = BLADE_FULL_LENGTH * saber.progress;
      const bladeBaseX = emitterX + dirX * 4;
      const bladeBaseY = emitterY + dirY * 4;
      const bladeTipX  = bladeBaseX + dirX * currentLength;
      const bladeTipY  = bladeBaseY + dirY * currentLength;

      // Store geometry for collision detection after all blades are drawn
      bladeGeoms[idx] = saber.progress > 0.25
        ? { baseX: bladeBaseX, baseY: bladeBaseY, tipX: bladeTipX, tipY: bladeTipY, progress: saber.progress }
        : null;

      const showAlpha = Math.min(1, saber.progress * 3 + (isFist ? 0.4 : 0));

      // ── Hand skeleton overlay (always visible so you can see your grip) ──
      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      if (window.drawConnectors && window.HAND_CONNECTIONS) {
        // Ghost connectors in dim crimson matching the saber colour
        ctx.globalAlpha = 0.45;
        drawHandConnectors(ctx, hand, {
          color: "rgba(220, 40, 40, 0.7)",
          lineWidth: 1.5,
        });
      }
      // Fingertip dots — dim when open, brighter as fist closes
      const tipAlphaSkel = isFist ? 0.55 : 0.25;
      ctx.globalAlpha = tipAlphaSkel;
      FINGER_TIPS.forEach((tipIdx) => {
        const pt = mapToCanvas(hand[tipIdx], w, h);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = isFist ? "rgba(255,120,120,1)" : "rgba(200,60,60,1)";
        ctx.fill();
      });
      ctx.restore();

      if (saber.progress <= 0.005) return;

      const saberFlicker = 0.93 + Math.sin(t * 58 + idx * 97) * 0.07;

      // ── Hilt ──────────────────────────────────────────────────────────────
      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = showAlpha;

      // Main body — tapered quad running wrist → emitter
      const hiltGrad = ctx.createLinearGradient(
        hiltBaseX + perpX * hiltHalfW, hiltBaseY + perpY * hiltHalfW,
        hiltBaseX - perpX * hiltHalfW, hiltBaseY - perpY * hiltHalfW
      );
      hiltGrad.addColorStop(0,   "rgba(55,55,65,1)");
      hiltGrad.addColorStop(0.28,"rgba(195,195,210,1)");
      hiltGrad.addColorStop(0.55,"rgba(110,110,125,1)");
      hiltGrad.addColorStop(1,   "rgba(40,40,48,1)");

      const hiltNarrowW = hiltHalfW * 0.78;
      ctx.beginPath();
      ctx.moveTo(hiltBaseX + perpX * hiltNarrowW,   hiltBaseY + perpY * hiltNarrowW);
      ctx.lineTo(emitterX  + perpX * hiltHalfW,     emitterY  + perpY * hiltHalfW);
      ctx.lineTo(emitterX  - perpX * hiltHalfW,     emitterY  - perpY * hiltHalfW);
      ctx.lineTo(hiltBaseX - perpX * hiltNarrowW,   hiltBaseY - perpY * hiltNarrowW);
      ctx.closePath();
      ctx.fillStyle = hiltGrad;
      ctx.fill();

      // Pommel cap at wrist
      ctx.beginPath();
      ctx.arc(hiltBaseX, hiltBaseY, hiltHalfW * 0.85, 0, Math.PI * 2);
      const pommelGrad = ctx.createRadialGradient(
        hiltBaseX - perpX * 2, hiltBaseY - perpY * 2, 0,
        hiltBaseX, hiltBaseY, hiltHalfW * 0.85
      );
      pommelGrad.addColorStop(0, "rgba(180,180,195,1)");
      pommelGrad.addColorStop(1, "rgba(40,40,50,1)");
      ctx.fillStyle = pommelGrad;
      ctx.fill();

      // Grip rings (3 evenly spaced)
      ctx.strokeStyle = "rgba(90,90,100,1)";
      ctx.lineWidth = 1.5;
      for (let r = 0; r < 3; r++) {
        const t2 = (r + 1) / 4;
        const rx = hiltBaseX + pdx * t2;
        const ry = hiltBaseY + pdy * t2;
        ctx.beginPath();
        ctx.moveTo(rx + perpX * hiltHalfW * (0.78 + t2 * 0.22),
                   ry + perpY * hiltHalfW * (0.78 + t2 * 0.22));
        ctx.lineTo(rx - perpX * hiltHalfW * (0.78 + t2 * 0.22),
                   ry - perpY * hiltHalfW * (0.78 + t2 * 0.22));
        ctx.stroke();
      }

      // Activation button (small jewel on the side of hilt)
      const btnT = 0.55;
      const btnX = hiltBaseX + pdx * btnT + perpX * hiltHalfW;
      const btnY = hiltBaseY + pdy * btnT + perpY * hiltHalfW;
      ctx.beginPath();
      ctx.arc(btnX, btnY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = saber.active ? "rgba(255,60,60,1)" : "rgba(100,20,20,1)";
      if (saber.active) {
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(255,0,0,1)";
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      // Emitter collar — glows red when active
      ctx.beginPath();
      ctx.arc(emitterX, emitterY, hiltHalfW + 2, 0, Math.PI * 2);
      ctx.fillStyle = saber.active
        ? `rgba(80,10,10,${saberFlicker})`
        : "rgba(40,40,48,1)";
      ctx.fill();
      ctx.strokeStyle = saber.active
        ? `rgba(255,60,60,${0.8 * saberFlicker})`
        : "rgba(80,80,90,1)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      if (saber.active) {
        ctx.shadowBlur = 14;
        ctx.shadowColor = "rgba(255,0,0,0.9)";
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      ctx.restore();

      if (saber.progress <= 0.01) return;

      // ── Blade ─────────────────────────────────────────────────────────────
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ctx.lineCap = "round";

      // Pulsing secondary flicker layered on top of base flicker
      const pulseFlicker = saberFlicker * (0.88 + Math.sin(t * 14 + idx * 3.7) * 0.12);
      const pf = pulseFlicker;

      // Pass 0 — enormous atmospheric halo (very wide, very soft)
      ctx.globalAlpha = 0.13 * pf;
      ctx.strokeStyle = rgba(pal.halo, 1);
      ctx.lineWidth = 110;
      ctx.shadowBlur = 0;
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();

      // Pass 1 — wide outer halo
      ctx.globalAlpha = 0.30 * pf;
      ctx.strokeStyle = rgba(pal.halo, 1);
      ctx.lineWidth = 68;
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();

      // Pass 2 — mid bloom
      ctx.globalAlpha = 0.55 * pf;
      ctx.strokeStyle = rgba(pal.mid, 1);
      ctx.lineWidth = 32;
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();

      // Pass 3 — inner vivid with strong shadow
      ctx.globalAlpha = 0.88 * pf;
      ctx.strokeStyle = rgba(pal.core, 1);
      ctx.lineWidth = 13;
      ctx.shadowBlur = 44;
      ctx.shadowColor = rgba(pal.shadow, 1);
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();

      // Pass 4 — bright saturated core
      ctx.globalAlpha = pf;
      ctx.strokeStyle = rgba(pal.core, 1);
      ctx.lineWidth = 6;
      ctx.shadowBlur = 32;
      ctx.shadowColor = rgba(pal.mid, 1);
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();

      // Pass 5 — hot white centre line
      ctx.globalAlpha = pf;
      ctx.strokeStyle = rgba(pal.white, 1);
      ctx.lineWidth = 2.5;
      ctx.shadowBlur = 22;
      ctx.shadowColor = rgba(pal.core, 1);
      ctx.beginPath(); ctx.moveTo(bladeBaseX, bladeBaseY); ctx.lineTo(bladeTipX, bladeTipY); ctx.stroke();
      ctx.shadowBlur = 0;

      // Base corona where blade meets emitter
      const baseGlow = ctx.createRadialGradient(bladeBaseX, bladeBaseY, 0, bladeBaseX, bladeBaseY, hiltHalfW * 3.6);
      baseGlow.addColorStop(0,    rgba(pal.white,  pf));
      baseGlow.addColorStop(0.25, rgba(pal.core,   0.9 * pf));
      baseGlow.addColorStop(0.55, rgba(pal.halo,   0.5 * pf));
      baseGlow.addColorStop(1,    rgba(pal.halo,   0));
      ctx.beginPath();
      ctx.arc(bladeBaseX, bladeBaseY, hiltHalfW * 3.6, 0, Math.PI * 2);
      ctx.fillStyle = baseGlow;
      ctx.globalAlpha = pf;
      ctx.fill();

      // Tip flare at full extension
      if (saber.progress > 0.85) {
        const tipFade = (saber.progress - 0.85) / 0.15;
        const tipGrad = ctx.createRadialGradient(bladeTipX, bladeTipY, 0, bladeTipX, bladeTipY, 40);
        tipGrad.addColorStop(0,    rgba(pal.white,  tipFade));
        tipGrad.addColorStop(0.28, rgba(pal.core,   0.8 * tipFade));
        tipGrad.addColorStop(0.6,  rgba(pal.halo,   0.4 * tipFade));
        tipGrad.addColorStop(1,    rgba(pal.halo,   0));
        ctx.beginPath();
        ctx.arc(bladeTipX, bladeTipY, 40, 0, Math.PI * 2);
        ctx.fillStyle = tipGrad;
        ctx.globalAlpha = tipFade * pf;
        ctx.fill();
      }

      ctx.restore();

      // Surface reflection glow near screen bottom
      if (bladeTipY > h * 0.78) {
        const reflA = ((bladeTipY - h * 0.78) / (h * 0.22)) * 0.38 * saber.progress;
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        const reflGrad = ctx.createRadialGradient(bladeTipX, h, 0, bladeTipX, h, 110);
        reflGrad.addColorStop(0, rgba(pal.refl, reflA));
        reflGrad.addColorStop(1, rgba(pal.refl, 0));
        ctx.beginPath();
        ctx.arc(bladeTipX, h, 110, 0, Math.PI * 2);
        ctx.fillStyle = reflGrad;
        ctx.fill();
        ctx.restore();
      }
    });

    // ── Blade vs Blade collision (duel clash) ─────────────────────────────
    const g0 = bladeGeoms[0];
    const g1 = bladeGeoms[1];
    if (g0 && g1) {
      const hit = segmentsIntersect(
        g0.baseX, g0.baseY, g0.tipX, g0.tipY,
        g1.baseX, g1.baseY, g1.tipX, g1.tipY
      );
      if (hit) {
        // Ongoing clash — draw continuous crackle at impact point
        s.clashPoint = hit;

        // Crackle arcs between the two blades at the contact point
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        for (let arc = 0; arc < 4; arc++) {
          const spread = 18 + Math.random() * 22;
          ctx.beginPath();
          ctx.moveTo(hit.x, hit.y);
          const mx = hit.x + (Math.random() - 0.5) * spread;
          const my = hit.y + (Math.random() - 0.5) * spread;
          ctx.lineTo(mx, my);
          ctx.strokeStyle = arc % 2 === 0 ? "rgba(255,255,180,0.9)" : "rgba(255,120,60,0.8)";
          ctx.lineWidth = 1.2 + Math.random() * 1.5;
          ctx.shadowBlur = 8;
          ctx.shadowColor = "rgba(255,200,80,1)";
          ctx.stroke();
        }
        ctx.shadowBlur = 0;

        // Persistent glow at clash point
        const clashGlow = ctx.createRadialGradient(hit.x, hit.y, 0, hit.x, hit.y, 32);
        clashGlow.addColorStop(0, "rgba(255,255,200,0.85)");
        clashGlow.addColorStop(0.4, "rgba(255,140,40,0.5)");
        clashGlow.addColorStop(1, "rgba(255,0,0,0)");
        ctx.beginPath();
        ctx.arc(hit.x, hit.y, 32, 0, Math.PI * 2);
        ctx.fillStyle = clashGlow;
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.restore();

        // Throttled spark burst + sound (at most every 120ms)
        const nowMs = performance.now();
        if (nowMs - s.lastClashTime > 120) {
          s.lastClashTime = nowMs;
          spawnClashSparks(hit.x, hit.y);
          triggerClash();
        }
      } else {
        // Blades separated — let flash and sparks decay naturally
        if (s.clashFlash === 0) s.clashPoint = null;
      }
    }
  }

  function detectGestures(
    hands: Landmark[][],
    lastPinchState: [boolean, boolean],
    particles: Particle[],
    ripples: Ripple[],
    w: number, h: number,
    time: number,
    theme: ThemeName
  ) {
    hands.forEach((hand, idx) => {
      const thumb = hand[4], index = hand[8];
      const isPinching = getDist(thumb, index) < 0.05;
      if (isPinching && !lastPinchState[idx]) {
        const cx = mapToCanvas({ x: (thumb.x + index.x) / 2, y: (thumb.y + index.y) / 2, z: 0 }, w, h);
        const col = THEMES[theme] ? THEMES[theme](time, 1, 1) : "#fff";
        createShockwave(cx.x, cx.y, col);
        triggerZap();
      }
      lastPinchState[idx] = isPinching;
    });
  }

  const renderLoopRef = useRef<number>(0);

  function renderLoop(timestamp: number) {
    renderLoopRef.current = requestAnimationFrame(renderLoop);
    const s = stateRef.current;

    const dt = Math.min((timestamp - s.lastTime) / 1000, 0.05);
    s.lastTime = timestamp;
    s.time += dt;

    // FPS
    s.framesThisSecond++;
    if (timestamp > s.lastFpsTime + 1000) {
      const fps = s.framesThisSecond;
      if (hudFpsEl.current) hudFpsEl.current.innerText = String(fps);
      if (barFpsEl.current) barFpsEl.current.style.width = Math.min(fps / 60 * 100, 100) + "%";
      s.framesThisSecond = 0;
      s.lastFpsTime = timestamp;
    }

    // Motion stats
    const rawVel = Math.min(s.handVelocities * 40, 1);
    s.smoothVel = s.smoothVel * 0.85 + rawVel * 0.15;
    const rawFlux = Math.abs(rawVel - s.lastVel);
    s.smoothFlux = s.smoothFlux * 0.80 + rawFlux * 0.20;
    s.lastVel = rawVel;

    if (hudVelEl.current) hudVelEl.current.innerText = s.smoothVel.toFixed(2);
    if (hudFluxEl.current) hudFluxEl.current.innerText = s.smoothFlux.toFixed(2);
    if (barVelEl.current) barVelEl.current.style.width = (s.smoothVel * 100).toFixed(1) + "%";
    if (barFluxEl.current) barFluxEl.current.style.width = Math.min(s.smoothFlux * 400, 100).toFixed(1) + "%";

    if (s.currentHands.length >= 2) {
      const p1 = s.currentHands[0][8], p2 = s.currentHands[1][8];
      const rawSync = Math.max(0, 1 - getDist(p1, p2) * 3);
      s.smoothSync = s.smoothSync * 0.88 + rawSync * 0.12;
      if (hudSyncEl.current) hudSyncEl.current.innerText = (s.smoothSync * 100).toFixed(0) + "%";
      if (barSyncEl.current) barSyncEl.current.style.width = (s.smoothSync * 100).toFixed(1) + "%";
    } else {
      s.smoothSync = s.smoothSync * 0.95;
      if (hudSyncEl.current) hudSyncEl.current.innerText = "——";
      if (barSyncEl.current) barSyncEl.current.style.width = "0%";
    }

    drawBackground();

    const mainCanvas = mainCanvasRef.current;
    if (!mainCanvas) return;
    const ctx = mainCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, s.width, s.height);
    const { width: w, height: h } = s;

    if (s.saberMode) {
      // Lightsaber mode
      ctx.globalCompositeOperation = "source-over";
      if (s.currentHands.length > 0) {
        drawLightsabers(ctx, s.currentHands, s.sabers, dt, w, h, s.time, s.saberColor);
      }
      // Clash sparks + flash — always update so they decay even when no hands visible
      updateClashSparks(ctx, dt);
      ctx.globalCompositeOperation = "source-over";

    } else if (s.fireMode) {
      // Fire mode
      ctx.globalCompositeOperation = "screen";
      if (s.currentHands.length > 0) {
        s.currentHands.forEach((hand) => drawFireHands(ctx, hand, w, h, s.handVelocities, s.fireParticles));
        detectFireGestures(s.currentHands, s.lastFistState, s.handVelDirs, s.fireballs, w, h);
      }
      updateFireParticles(ctx, dt, s.fireParticles);
      updateFireballs(ctx, dt, s.fireballs, s.fireParticles);
      ctx.globalCompositeOperation = "source-over";

    } else {
      // Neon mode
      ctx.globalCompositeOperation = "screen";

      // Update particles and ripples
      for (let i = s.particles.length - 1; i >= 0; i--) {
        const p = s.particles[i];
        p.x += p.vx; p.y += p.vy;
        p.life -= 0.02; p.vy += 0.1;
        if (p.life <= 0) { s.particles.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.life;
        ctx.fill();
      }
      for (let i = s.ripples.length - 1; i >= 0; i--) {
        const r = s.ripples[i];
        r.radius += (r.maxRadius - r.radius) * 0.20;
        r.life -= 0.055;
        if (r.life <= 0) { s.ripples.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
        ctx.strokeStyle = r.color;
        ctx.lineWidth = 4 * r.life;
        ctx.globalAlpha = r.life;
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;

      if (s.currentHands.length > 0) {
        s.currentHands.forEach((hand, handIndex) => {
          if (s.currentTheme === "Blade") {
            if (window.drawConnectors && window.HAND_CONNECTIONS) {
              drawHandConnectors(ctx, hand, { color: "rgba(255,255,255,0.15)", lineWidth: 1 });
            }
            drawBladeLines(ctx, hand, w, h, s.time);
          } else {
            const glowColor = THEMES[s.currentTheme]?.(s.time, handIndex, 2) ?? "#fff";
            if (window.drawConnectors && window.HAND_CONNECTIONS) {
              drawHandConnectors(ctx, hand, { color: glowColor, lineWidth: 2 });
            }
            ctx.shadowBlur = 15;
            ctx.shadowColor = glowColor;
            FINGER_TIPS.forEach((tipIndex) => {
              const pt = mapToCanvas(hand[tipIndex], w, h);
              ctx.beginPath();
              ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
              ctx.fillStyle = "#fff";
              ctx.globalAlpha = 1;
              ctx.fill();
            });
            ctx.shadowBlur = 0;
          }
        });

        if (s.currentHands.length >= 2) {
          const h1 = s.currentHands[0];
          const h2 = s.currentHands[1];

          FINGER_TIPS.forEach((tipIndex, idx) => {
            const pt1 = mapToCanvas(h1[tipIndex], w, h);
            const pt2 = mapToCanvas(h2[tipIndex], w, h);
            const dist = getDist(pt1, pt2);
            const col = THEMES[s.currentTheme]?.(s.time, idx, FINGER_TIPS.length) ?? "#fff";

            if (s.currentTheme !== "Blade" && dist < 150 && Math.random() > 0.5) {
              ctx.beginPath();
              ctx.moveTo(pt1.x, pt1.y);
              const midX = (pt1.x + pt2.x) / 2 + (Math.random() - 0.5) * 50;
              const midY = (pt1.y + pt2.y) / 2 + (Math.random() - 0.5) * 50;
              ctx.lineTo(midX, midY);
              ctx.lineTo(pt2.x, pt2.y);
              ctx.strokeStyle = "#ffffff";
              ctx.shadowBlur = 20;
              ctx.shadowColor = col;
              ctx.lineWidth = 3;
              ctx.stroke();
              ctx.shadowBlur = 0;
            }

            if (s.currentTheme === "Prism" || s.currentTheme === "Rapture") {
              const amplitude = 20 + Math.sin(s.time * 5.5 + idx * 1.1) * 14;
              const frequency = 3.2 + idx * 0.32;
              const phase = s.time * 11.0 + idx * Math.PI * 0.5;
              const grad = ctx.createLinearGradient(pt1.x, pt1.y, pt2.x, pt2.y);
              grad.addColorStop(0, THEMES[s.currentTheme]?.(s.time, idx, 5) ?? "#fff");
              grad.addColorStop(0.5, THEMES[s.currentTheme]?.(s.time, idx + 1, 5) ?? "#fff");
              grad.addColorStop(1, THEMES[s.currentTheme]?.(s.time, idx + 2, 5) ?? "#fff");
              ctx.strokeStyle = col;
              ctx.lineWidth = 26; ctx.globalAlpha = 0.18; ctx.shadowBlur = 0;
              drawSineWave(ctx, pt1.x, pt1.y, pt2.x, pt2.y, amplitude, frequency, phase, 24);
              ctx.strokeStyle = col; ctx.lineWidth = 11; ctx.globalAlpha = 0.42;
              drawSineWave(ctx, pt1.x, pt1.y, pt2.x, pt2.y, amplitude, frequency, phase, 24);
              ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.globalAlpha = 1; ctx.shadowBlur = 10; ctx.shadowColor = col;
              drawSineWave(ctx, pt1.x, pt1.y, pt2.x, pt2.y, amplitude, frequency, phase);
              ctx.shadowBlur = 0;
            } else if (s.currentTheme === "Blade") {
              const bcol = BLADE_COLORS[idx];
              const ldx = pt2.x - pt1.x, ldy = pt2.y - pt1.y;
              const llen = Math.hypot(ldx, ldy) || 1;
              const px = -ldy / llen, py = ldx / llen;
              const offsets = [-5, -1.7, 1.7, 5];
              ctx.strokeStyle = bcol; ctx.lineWidth = 12; ctx.globalAlpha = 0.16; ctx.shadowBlur = 0;
              ctx.beginPath();
              offsets.forEach(o => { ctx.moveTo(pt1.x + px * o, pt1.y + py * o); ctx.lineTo(pt2.x + px * o, pt2.y + py * o); });
              ctx.stroke();
              ctx.lineWidth = 5; ctx.globalAlpha = 0.40;
              ctx.beginPath();
              offsets.forEach(o => { ctx.moveTo(pt1.x + px * o, pt1.y + py * o); ctx.lineTo(pt2.x + px * o, pt2.y + py * o); });
              ctx.stroke();
              ctx.lineWidth = 1.2; ctx.globalAlpha = 1; ctx.shadowBlur = 14; ctx.shadowColor = bcol;
              ctx.beginPath();
              offsets.forEach(o => { ctx.moveTo(pt1.x + px * o, pt1.y + py * o); ctx.lineTo(pt2.x + px * o, pt2.y + py * o); });
              ctx.stroke();
              ctx.shadowBlur = 0;
            } else {
              const grad = ctx.createLinearGradient(pt1.x, pt1.y, pt2.x, pt2.y);
              grad.addColorStop(0, THEMES[s.currentTheme]?.(s.time, idx, 5) ?? "#fff");
              grad.addColorStop(0.5, THEMES[s.currentTheme]?.(s.time, idx + 1, 5) ?? "#fff");
              grad.addColorStop(1, THEMES[s.currentTheme]?.(s.time, idx + 2, 5) ?? "#fff");
              ctx.strokeStyle = grad; ctx.lineWidth = 3; ctx.globalAlpha = 1; ctx.shadowBlur = 10; ctx.shadowColor = col;
              ctx.beginPath();
              ctx.moveTo(pt1.x, pt1.y);
              ctx.lineTo(pt2.x, pt2.y);
              ctx.stroke();
              ctx.shadowBlur = 0;
            }
            ctx.globalAlpha = 1;
          });

          if (s.currentTheme !== "Blade") {
            const allTips = FINGER_TIPS.map(t => mapToCanvas(h1[t], w, h))
              .concat(FINGER_TIPS.map(t => mapToCanvas(h2[t], w, h)));
            ctx.save();
            const cx = allTips.reduce((s, p) => s + p.x, 0) / 10;
            const cy = allTips.reduce((s, p) => s + p.y, 0) / 10;
            ctx.translate(cx, cy);
            ctx.rotate(s.time * 0.5);
            ctx.beginPath();
            for (let i = 0; i < 10; i++) {
              ctx.moveTo(allTips[i].x - cx, allTips[i].y - cy);
              ctx.lineTo(allTips[(i + 3) % 10].x - cx, allTips[(i + 3) % 10].y - cy);
            }
            ctx.strokeStyle = "rgba(255,255,255,0.14)";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
          }
        }

        detectGestures(s.currentHands, s.lastPinchState, s.particles, s.ripples, w, h, s.time, s.currentTheme);
      }
      ctx.globalCompositeOperation = "source-over";
    }
  }

  function handleResize() {
    const s = stateRef.current;
    /* Use the canvas's actual displayed CSS box (respects safe-area padding on .ar-root)
       and fall back to window dimensions before mount. */
    const ref = mainCanvasRef.current ?? bgCanvasRef.current;
    const w = ref?.clientWidth || window.innerWidth;
    const h = ref?.clientHeight || window.innerHeight;
    s.width = w;
    s.height = h;
    if (bgCanvasRef.current) {
      bgCanvasRef.current.width = w;
      bgCanvasRef.current.height = h;
    }
    if (mainCanvasRef.current) {
      mainCanvasRef.current.width = w;
      mainCanvasRef.current.height = h;
    }
  }

  function initAudio() {
    const s = stateRef.current;
    try {
      s.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      s.humOsc = s.audioCtx.createOscillator();
      s.humGain = s.audioCtx.createGain();
      s.humOsc.type = "sine";
      s.humOsc.frequency.value = 100;
      s.humGain.gain.value = 0;
      s.humOsc.connect(s.humGain);
      s.humGain.connect(s.audioCtx.destination);
      s.humOsc.start();
    } catch (e) {
      console.error("Audio init failed", e);
    }
  }

  function updateHum(hands: Landmark[][]) {
    const s = stateRef.current;
    if (!s.audioCtx || !s.humGain || !s.humOsc) return;
    if (hands.length < 2) {
      s.humGain.gain.setTargetAtTime(0, s.audioCtx.currentTime, 0.1);
      return;
    }
    const p1 = hands[0][8], p2 = hands[1][8];
    const dx = p1.x - p2.x, dy = p1.y - p2.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    s.humOsc.frequency.setTargetAtTime(100 + (1 - Math.min(dist, 1)) * 300, s.audioCtx.currentTime, 0.1);
    s.humGain.gain.setTargetAtTime(0.05 + (1 - Math.min(dist, 1)) * 0.15, s.audioCtx.currentTime, 0.1);
  }

  function initMediaPipe() {
    const video = videoRef.current;
    if (!video || !window.Hands || !window.Camera) return;

    const hands = new window.Hands({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults((results) => {
      const s = stateRef.current;
      const newHands = results.multiHandLandmarks || [];
      if (s.currentHands.length > 0 && newHands.length > 0) {
        const oldP = s.currentHands[0][8];
        const newP = newHands[0][8];
        if (oldP && newP) s.handVelocities = getDist(oldP, newP);
      } else {
        s.handVelocities = 0;
      }
      newHands.forEach((hand, idx) => {
        const prev = s.prevHandPalms[idx];
        const curr = hand[0];
        if (prev && curr) {
          const dx = curr.x - prev.x, dy = curr.y - prev.y;
          const len = Math.hypot(dx, dy);
          if (len > 0.002) s.handVelDirs[idx] = { x: dx / len, y: dy / len };
        }
        s.prevHandPalms[idx] = { x: curr.x, y: curr.y };
      });
      if (newHands.length < 2) s.prevHandPalms[1] = null;
      s.currentHands = newHands;
      if (s.audioCtx) updateHum(newHands);
    });

    const camera = new window.Camera(video, {
      onFrame: async () => { await hands.send({ image: video }); },
      facingMode: "user",
    });
    camera.start();
  }

  const handleStart = useCallback(() => {
    handleResize();
    setStarted(true);
    setTimeout(() => {
      initAudio();
      initMediaPipe();
      renderLoopRef.current = requestAnimationFrame(renderLoop);
    }, 100);
  }, []);

  useEffect(() => {
    let raf = 0;
    const onResize = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(handleResize);
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("orientationchange", onResize);
    /* iOS Safari URL bar resize fires on visualViewport, not window */
    window.visualViewport?.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("orientationchange", onResize);
      window.visualViewport?.removeEventListener("resize", onResize);
      cancelAnimationFrame(renderLoopRef.current);
    };
  }, []);

  const handleSaberColor = (c: SaberColorKey) => {
    stateRef.current.saberColor = c;
    setSaberColor(c);
    if (stateRef.current.saberMode) {
      document.documentElement.style.setProperty("--accent", SABER_COLORS[c].accent);
    }
  };

  const handleThemeChange = (theme: ThemeName) => {
    const s = stateRef.current;
    if (theme === "Fire") {
      s.fireMode = true;
      s.saberMode = false;
      s.fireParticles = [];
      s.fireballs = [];
      document.documentElement.style.setProperty("--accent", "#ff6622");
    } else if (theme === "Lightsaber") {
      s.saberMode = true;
      s.fireMode = false;
      s.fireParticles = [];
      s.fireballs = [];
      s.sabers = [{ progress: 0, active: false }, { progress: 0, active: false }];
      document.documentElement.style.setProperty("--accent", SABER_COLORS[s.saberColor].accent);
    } else {
      s.fireMode = false;
      s.saberMode = false;
      s.currentTheme = theme;
      s.fireParticles = [];
      s.fireballs = [];
      s.sabers = [{ progress: 0, active: false }, { progress: 0, active: false }];
      document.documentElement.style.setProperty("--accent", THEMES[theme]?.(0, 1, 1) ?? "#00ffcc");
    }
    setCurrentTheme(theme);
  };

  const themeButtons: { name: ThemeName; label: string; className?: string }[] = [
    { name: "Prism", label: "Prism" },
    { name: "Rapture", label: "Rapture" },
    { name: "Blade", label: "Blade" },
    { name: "Ember", label: "Ember" },
    { name: "Tide", label: "Tide" },
    { name: "Cosmos", label: "Cosmos" },
  ];

  return (
    <div className="ar-root">
      <video
        ref={videoRef}
        className="input-video"
        style={videoStyle}
        autoPlay
        playsInline
      />
      <canvas
        ref={bgCanvasRef}
        className="bg-canvas"
        style={{ transform: mirrorZoomTransform }}
      />
      <canvas
        ref={mainCanvasRef}
        className="main-canvas"
        style={{ transform: mirrorZoomTransform }}
      />

      {/* Start Overlay */}
      {!started && (
        <div className="start-overlay">
          <h1>CAMERA WAVES</h1>
          <p>Grant camera permissions and click to begin</p>
          <button className="start-btn" onClick={handleStart}>
            Enter Experience
          </button>
        </div>
      )}

      {/* HUD */}
      {started && (
        <div className="hud">
          <div className="glass-panel">
            <div className="hud-title">Motion Array</div>
            <div className="hud-row">
              <span className="hud-lbl">VEL</span>
              <div className="hud-bar-track"><div className="hud-bar-fill" ref={barVelEl} /></div>
              <span className="hud-val" ref={hudVelEl}>0.00</span>
            </div>
            <div className="hud-row">
              <span className="hud-lbl">FLUX</span>
              <div className="hud-bar-track"><div className="hud-bar-fill" ref={barFluxEl} /></div>
              <span className="hud-val" ref={hudFluxEl}>0.00</span>
            </div>
            <div className="hud-row">
              <span className="hud-lbl">SYNC</span>
              <div className="hud-bar-track"><div className="hud-bar-fill" ref={barSyncEl} /></div>
              <span className="hud-val" ref={hudSyncEl}>——</span>
            </div>
            <hr className="hud-divider" />
            <div className="hud-row">
              <span className="hud-lbl">FPS</span>
              <div className="hud-bar-track"><div className="hud-bar-fill" ref={barFpsEl} /></div>
              <span className="hud-val" ref={hudFpsEl}>0</span>
            </div>
          </div>
        </div>
      )}

      {/* Lightsaber hint */}
      {started && currentTheme === "Lightsaber" && (
        <div className="saber-hint">
          <div className="saber-hint-title">⚔ Lightsaber</div>
          <div className="saber-hint-text">Make a fist to ignite</div>
        </div>
      )}

      {/* Theme Bar */}
      {started && (
        <div className="theme-bar">
          {themeButtons.map((btn) => (
            <button
              key={btn.name}
              className={`theme-btn${currentTheme === btn.name ? " active" : ""}`}
              onClick={() => handleThemeChange(btn.name)}
            >
              {btn.label}
            </button>
          ))}
          <span className="theme-sep" />
          <button
            className={`theme-btn fire-btn${currentTheme === "Fire" ? " active" : ""}`}
            onClick={() => handleThemeChange("Fire")}
          >
            Fire
          </button>
          <span className="theme-sep" />
          <button
            className={`theme-btn saber-btn${currentTheme === "Lightsaber" ? " active" : ""}`}
            onClick={() => handleThemeChange("Lightsaber")}
          >
            ⚔ Lightsaber
          </button>
        </div>
      )}

      {/* Saber colour picker — right side, only in Lightsaber mode */}
      {started && currentTheme === "Lightsaber" && (
        <div className="saber-color-panel">
          <div className="saber-color-label">Saber</div>
          {(["red","blue","green"] as SaberColorKey[]).map((c) => (
            <button
              key={c}
              className={`saber-color-btn saber-color-${c}${saberColor === c ? " active" : ""}`}
              onClick={() => handleSaberColor(c)}
              title={c.charAt(0).toUpperCase() + c.slice(1)}
            />
          ))}
        </div>
      )}

      {/* Dim Slider */}
      {started && (
        <div className="slider-panel slider-panel--right">
          <label>Bright</label>
          <input
            type="range"
            className="v-slider"
            min={0} max={100}
            value={dimPct}
            ref={dimSliderRef}
            onChange={(e) => setDimPct(Number(e.target.value))}
          />
          <label>Dim</label>
        </div>
      )}

      {/* Zoom Slider */}
      {started && (
        <div className="slider-panel slider-panel--left">
          <label>+</label>
          <input
            type="range"
            className="v-slider"
            min={100} max={250}
            value={zoomPct}
            ref={zoomSliderRef}
            onChange={(e) => setZoomPct(Number(e.target.value))}
          />
          <label>Zoom</label>
        </div>
      )}
    </div>
  );
}
