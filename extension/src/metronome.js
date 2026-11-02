/* Metronome bằng Web Audio, đồng bộ với result.beats/downbeats theo video.currentTime và playbackRate.
 * Phách mạnh cao và to hơn; âm lượng = CT.settings.metronomeVolume; bật/tắt = CT.settings.metronome.
 * Lên lịch trước (look-ahead scheduler), không phát khi video dừng, không phát dồn khi tua.
 * CT.metronome = { start(), stop() } ; tự nghe "time", "seek", "settings", "song:ready".
 *
 * Đồng bộ: giữ một mốc (anchor) {a, v: video t, rate}, trong đó `a` là thời điểm (theo đồng hồ AudioContext)
 * của mẫu âm thanh đang ra loa lúc nhận "time" (`heardNow`: getOutputTimestamp(), hoặc currentTime trừ
 * outputLatency + baseLatency nếu trình duyệt chưa có số đó). Video đang phát t ở thời điểm ra loa x là
 *   v(x) = anchor.v + (x - anchor.a) * rate
 * nên phách b được lên lịch ở x = anchor.a + (b - anchor.v) / rate, và tiếng gõ ra loa cùng lúc với tiếng
 * của video (Chrome tính video.currentTime theo tiếng đang ra loa). Vậy độ trễ đầu ra đã được bù.
 * Sự kiện "time" (mỗi khung hình) chỉ chỉnh mốc khi lệch quá JITTER, vì currentTime của video đọc trong
 * requestAnimationFrame dao động vài ms; lệch lớn (tua, video khựng) thì đặt lại mốc và lên lịch lại.
 */
(() => {
  const LOOKAHEAD = 0.1;      // giây: lên lịch trước bao xa
  const TICK_MS = 25;         // chu kỳ bộ lên lịch
  const JITTER = 0.012;       // lệch dưới mức này thì giữ mốc (chỉ kéo nhẹ)
  const RESYNC = 0.05;        // lệch trên mức này thì coi như tua: hủy tiếng đã lên lịch
  const LATE = 0.005;         // phách trễ hơn mức này so với audio hiện tại thì bỏ, không phát dồn
  const STALE_MS = 300;       // không có "time" lâu hơn (tab ẩn, rAF dừng) thì đọc thẳng video

  const SOUNDS = {
    // Woodblock tổng hợp: sine có pitch rơi nhanh + họa âm 2.7x tắt sớm, đuôi ~45 ms. Ấm, không chói.
    wood(ctx, out, when, strong) {
      const f = strong ? 1250 : 880;
      const peak = strong ? 1.0 : 0.55;
      const dur = strong ? 0.06 : 0.045;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, when);
      g.gain.exponentialRampToValueAtTime(peak, when + 0.0015);
      g.gain.exponentialRampToValueAtTime(0.0001, when + dur);
      g.connect(out);
      const o1 = ctx.createOscillator();
      o1.type = "sine";
      o1.frequency.setValueAtTime(f * 1.5, when);
      o1.frequency.exponentialRampToValueAtTime(f, when + 0.008);
      o1.connect(g);
      const g2 = ctx.createGain();
      g2.gain.setValueAtTime(0.35, when);
      g2.gain.exponentialRampToValueAtTime(0.001, when + 0.015);
      g2.connect(g);
      const o2 = ctx.createOscillator();
      o2.type = "sine";
      o2.frequency.setValueAtTime(f * 2.7, when);
      o2.connect(g2);
      return [o1, o2].map((o) => { o.start(when); o.stop(when + dur + 0.01); return o; });
    },
    // Sine có envelope ngắn (giống mockup), mềm hơn nhưng ít "gõ" hơn.
    sine(ctx, out, when, strong) {
      const dur = 0.06;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, when);
      g.gain.exponentialRampToValueAtTime(strong ? 0.9 : 0.5, when + 0.002);
      g.gain.exponentialRampToValueAtTime(0.0001, when + dur);
      g.connect(out);
      const o = ctx.createOscillator();
      o.frequency.value = strong ? 1760 : 1175;
      o.connect(g);
      o.start(when);
      o.stop(when + dur + 0.01);
      return [o];
    },
  };

  let ctx = null, master = null, timer = null;
  let on = false;
  let beats = [], downs = new Set();
  let anchor = null;          // {a, v, rate}
  let paused = true;
  let lastTimeAt = 0;         // performance.now() của sự kiện "time" gần nhất
  let nextIdx = 0;
  let pending = [];           // [{when, nodes}] tiếng đã lên lịch, chưa phát xong

  const volume = () => {
    const v = Number(CT.settings.metronomeVolume);
    return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.6;
  };
  const latency = () => (ctx ? (ctx.outputLatency || 0) + (ctx.baseLatency || 0) : 0);

  /** Thời điểm (đồng hồ ctx) của mẫu âm thanh đang ra loa ngay lúc này. */
  function heardNow() {
    if (ctx.getOutputTimestamp) {
      const ts = ctx.getOutputTimestamp();
      if (ts.contextTime > 0 && ts.performanceTime > 0) {
        return ts.contextTime + (performance.now() - ts.performanceTime) / 1000;
      }
    }
    return ctx.currentTime - latency();
  }

  function ensureContext() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC({ latencyHint: "interactive" });
      master = ctx.createGain();
      master.gain.value = volume();
      master.connect(ctx.destination);
    }
    if (ctx.state === "suspended") {
      ctx.resume().catch(() => {});
      // Bật lúc tải trang (cài đặt đã lưu) thì chưa có thao tác người dùng: chờ cú bấm đầu tiên.
      if (ctx.state === "suspended") armGestureResume();
    }
    return true;
  }

  let gestureArmed = false;
  function armGestureResume() {
    if (gestureArmed) return;
    gestureArmed = true;
    const resume = () => {
      gestureArmed = false;
      window.removeEventListener("pointerdown", resume, true);
      window.removeEventListener("keydown", resume, true);
      if (on && ctx) ctx.resume().catch(() => {});
    };
    window.addEventListener("pointerdown", resume, true);
    window.addEventListener("keydown", resume, true);
  }

  function cancelPending() {
    for (const p of pending) {
      for (const n of p.nodes) { try { n.stop(); n.disconnect(); } catch {} }
    }
    pending = [];
  }

  /** Chỉ số phách đầu tiên có thời điểm >= t (tìm nhị phân). */
  function indexAt(t) {
    let lo = 0, hi = beats.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (beats[m] < t - 1e-4) lo = m + 1; else hi = m; }
    return lo;
  }

  function setAnchor(t, rate) {
    anchor = { a: heardNow(), v: t, rate: rate > 0 ? rate : 1 };
  }

  /** Đặt lại mốc, hủy tiếng đã lên lịch, bắt đầu từ phách >= t. */
  function resync(t, rate) {
    cancelPending();
    if (!ctx) { anchor = null; return; }
    setAnchor(t, rate);
    nextIdx = indexAt(t);
  }

  const audioTimeOf = (b) => anchor.a + (b - anchor.v) / anchor.rate;
  const videoTimeAt = (x) => anchor.v + (x - anchor.a) * anchor.rate;

  function onTime({ t, rate, paused: p }) {
    lastTimeAt = performance.now();
    rate = rate > 0 ? rate : 1;
    const wasPaused = paused;
    paused = !!p;
    if (!on || !ctx) return;
    if (paused) { if (!wasPaused) cancelPending(); anchor = null; return; }
    if (!anchor || wasPaused || Math.abs(rate - anchor.rate) > 1e-6) { resync(t, rate); return; }
    const err = t - videoTimeAt(heardNow());
    if (Math.abs(err) > RESYNC) { resync(t, rate); return; }
    // Kéo mốc dần về số đo mới: lọc nhiễu của currentTime nhưng vẫn theo kịp trôi đồng hồ.
    if (Math.abs(err) > JITTER) anchor.v += err * 0.5;
    else anchor.v += err * 0.05;
  }

  function tick() {
    if (!on || !ctx || ctx.state !== "running" || !beats.length) return;
    // Tab ẩn thì rAF (nguồn của "time") dừng: đọc thẳng phần tử video để biết dừng/tua/đổi tốc độ.
    const video = CT.state.video;
    if (video && performance.now() - lastTimeAt > STALE_MS) {
      onTime({ t: video.currentTime, rate: video.playbackRate, paused: video.paused || video.ended });
    }
    if (paused || !anchor) return;
    const now = ctx.currentTime;
    pending = pending.filter((p) => p.when > now - 0.2);
    const horizon = now + LOOKAHEAD;
    while (nextIdx < beats.length) {
      const b = beats[nextIdx];
      const when = audioTimeOf(b);
      if (when > horizon) break;
      nextIdx++;
      if (when < now - LATE) continue;           // trễ rồi: bỏ, không phát dồn
      const strong = downs.has(Math.round(b * 1000));
      const at = Math.max(when, now);
      const nodes = (SOUNDS[CT.metronome.sound] || SOUNDS.wood)(ctx, master, at, strong);
      pending.push({ when: at, nodes });
      if (CT.metronome.onSchedule) CT.metronome.onSchedule({ beat: b, when: at, strong, latency: latency() });
    }
  }

  function setResult(result) {
    cancelPending();
    beats = Array.isArray(result?.beats) ? result.beats.map(Number).filter(Number.isFinite).sort((a, b) => a - b) : [];
    downs = new Set((result?.downbeats || []).map((d) => Math.round(Number(d) * 1000)));
    anchor = null;
    nextIdx = 0;
  }

  function start() {
    on = true;
    if (!ensureContext()) return;
    master.gain.setTargetAtTime(volume(), ctx.currentTime, 0.01);
    anchor = null;               // "time" kế tiếp sẽ đặt mốc
    const video = CT.state.video;
    if (video) onTime({ t: video.currentTime, rate: video.playbackRate, paused: video.paused || video.ended });
    if (!timer) timer = setInterval(tick, TICK_MS);
  }

  function stop() {
    on = false;
    cancelPending();
    anchor = null;
    if (timer) { clearInterval(timer); timer = null; }
    if (ctx && ctx.state === "running") ctx.suspend().catch(() => {});
  }

  CT.metronome = {
    start, stop,
    sound: "wood",                          // "wood" | "sine"
    onSchedule: null,                       // hook cho trang thử (dev/metronome_test.html)
    get context() { return ctx; },
    get running() { return on; },
    /** Gõ thử một tiếng ngay (để nghe kiểu tiếng). */
    preview(strong = true) {
      if (!ensureContext()) return;
      SOUNDS[CT.metronome.sound]?.(ctx, master, ctx.currentTime + 0.01, strong);
    },
  };

  CT.bus.on("time", onTime);
  CT.bus.on("seek", ({ t }) => {
    if (!on || !ctx) return;
    const video = CT.state.video;
    const rate = video?.playbackRate || anchor?.rate || 1;
    if (paused) { cancelPending(); anchor = null; nextIdx = indexAt(t); return; }
    resync(t, rate);
  });
  CT.bus.on("song:ready", ({ result }) => setResult(result));
  CT.bus.on("song:none", () => setResult(null));
  CT.bus.on("settings", (s) => {
    if (s.metronome && !on) start();
    else if (!s.metronome && on) stop();
    if (ctx && master) master.gain.setTargetAtTime(volume(), ctx.currentTime, 0.01);
  });
  if (CT.state.result) setResult(CT.state.result);
})();
