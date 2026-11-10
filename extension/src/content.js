/* ChordTrack content: điều phối. Nhận videoId khi YouTube điều hướng SPA, gọi backend, phát song:*.
 *
 * Chỉ phân tích khi người dùng bấm (nút trên thanh điều khiển player hoặc Alt+Shift+C):
 *   - Mở video: chỉ GET /result/{videoId} (đọc cache, không tải audio, không dùng GPU) để biết bài đã có kết quả
 *     chưa. Không gọi POST /analyze, không hiện panel.
 *   - Bấm: hiện panel, POST /analyze, theo dõi tiến độ. Bấm lần nữa: ẩn panel và lớp phủ; job đang chạy vẫn
 *     chạy tiếp (vẫn theo dõi để nút hiện vòng tiến độ), xong thì bấm lại là có ngay.
 *   - Nhớ theo video trong phiên (sessionStorage của tab): quay lại video đã bật thì tự hiện lại.
 *   - CT.settings.autoShowCached: bài đã có kết quả thì hiện luôn. Không bao giờ tự phân tích bài mới.
 *
 * CT.app.reload(force)   phân tích lại video hiện tại (nút "Thử lại" của panel dùng)
 * CT.app.toggle()        bật/tắt ChordTrack cho video hiện tại
 * Sự kiện "app" {videoId, visible, status, cached, progress} cho nút trên thanh điều khiển.
 * Server tắt: khi đang hiện thì tự thử lại mỗi 5 giây cho tới khi kết nối được (chỉ khi tab đang hiện).
 *
 * Giai điệu (docs/melody_api.md): job riêng, chỉ chạy khi người dùng bấm "Dò giai điệu" (CT.app.detectMelody).
 *   Khi hợp âm của bài đã hiện, GET /melody/{id} một lần: đã dò rồi thì hiện luôn. Phát sự kiện "melody".
 */
(() => {
  const RETRY_MS = 5000;
  const SESSION_KEY = "chordtrack.activeVideos";
  let cur = null;           // video đang mở: {id, visible, status, cached, result, job, error, ctrl, userHid}
  let retryTimer = 0;

  /* ----- nhớ video đã bật trong phiên ----- */
  const active = new Set();
  try { for (const id of JSON.parse(sessionStorage.getItem(SESSION_KEY) || "[]")) active.add(id); } catch { /* bỏ qua */ }
  function remember(id, on) {
    active.delete(id);      // xóa rồi thêm: đưa video vừa bật về cuối (cắt danh sách theo lần dùng gần nhất)
    if (on) active.add(id);
    try { sessionStorage.setItem(SESSION_KEY, JSON.stringify([...active].slice(-200))); } catch { /* bỏ qua */ }
  }

  function notify() {
    const c = cur;
    CT.bus.emit("app", c ? { videoId: c.id, visible: c.visible, status: c.status, cached: !!c.cached,
      serverDown: !!c.serverDown, progress: c.job ? c.job.progress || 0 : 0 } : { videoId: null, visible: false, status: "off", cached: false, progress: 0 });
  }

  function stopRetry() { clearTimeout(retryTimer); retryTimer = 0; }

  /* ----- giai điệu ----- */
  const newMel = () => ({ status: "unknown", melody: null, stale: false, job: null, error: null, ctrl: null });

  function emitMelody(c) {
    if (c !== cur) return;
    const m = c.mel;
    // Đang dò lại hoặc dò lại lỗi: vẫn hiện dòng nốt cũ (nếu có) cho tới khi có kết quả mới.
    CT.state.melody = m.melody;
    CT.bus.emit("melody", { videoId: c.id, status: m.status, melody: CT.state.melody, stale: m.stale, job: m.job, error: m.error });
  }

  /** GET /melody/{id}: chỉ đọc kết quả đã có. Server tắt thì để "unknown" để lần hợp âm hiện sau hỏi lại. */
  async function probeMelody(c) {
    if (c.mel.status !== "unknown" || c.mel.probing) return;
    c.mel.probing = true;
    let r;
    try { r = await CT.api.getMelody(c.id); } finally { c.mel.probing = false; }
    if (cur !== c || c.mel.status !== "unknown") return;    // người dùng đã bấm dò trong lúc chờ
    if (r.melody) Object.assign(c.mel, { status: "ready", melody: r.melody, stale: r.stale });
    else if (r.status === 0) return;
    else c.mel.status = "none";                              // 404 (chưa dò) hoặc server chưa có /melody
    emitMelody(c);
  }

  async function detectMelody(c, force = false) {
    const m = c.mel;
    if (m.status === "loading" && !force) return;
    m.ctrl?.abort();
    const my = (m.ctrl = new AbortController());
    Object.assign(m, { status: "loading", error: null,
      job: { status: "queued", progress: 0, step: null, message: "Đang gửi yêu cầu dò giai điệu", eta_seconds: null } });
    emitMelody(c);
    try {
      const melody = await CT.api.loadMelody(c.id, { force, signal: my.signal, onProgress: (job) => {
        if (m.ctrl !== my) return;
        m.job = job;
        emitMelody(c);
      } });
      if (m.ctrl !== my) return;
      Object.assign(m, { ctrl: null, status: "ready", melody, stale: false, job: null });
      emitMelody(c);
      // Lần dò đầu (không phải ↻ dò lại) mà dòng nốt đang tắt: bật để thấy kết quả vừa bấm dò.
      if (!force && CT.settings.showMelody === false) CT.setSettings({ showMelody: true });
      return;
    } catch (e) {
      if (m.ctrl !== my || e.code === "aborted") return;
      Object.assign(m, { ctrl: null, status: "error", job: null, error: { code: e.code || "unknown", message: e.message } });
    }
    emitMelody(c);
  }

  /** Gửi trạng thái hiện tại của video ra panel/dải/lớp phủ (chỉ khi đang hiện). */
  function emitSong(c) {
    if (!c.visible || c !== cur) return;
    if (c.status === "ready") {
      CT.state.result = c.result;
      CT.bus.emit("song:ready", { videoId: c.id, result: c.result });
      emitMelody(c);
      probeMelody(c);
    }
    else if (c.status === "loading") CT.bus.emit("song:loading", { videoId: c.id, job: c.job });
    else if (c.status === "error") CT.bus.emit("song:error", { videoId: c.id, ...c.error });
  }

  /** silent: lần tự thử lại khi server tắt, giữ nguyên thông báo lỗi thay vì nháy sang "đang kết nối". */
  async function load(c, force = false, silent = false) {
    c.ctrl?.abort();
    stopRetry();
    const my = (c.ctrl = new AbortController());
    if (!silent) {
      c.status = "loading";
      c.job = { status: "queued", progress: 0, step: null, message: "Đang kết nối server", eta_seconds: null };
      CT.state.result = null;
      emitSong(c);
      notify();
    }
    try {
      const result = await CT.api.loadSong(c.id, {
        force, signal: my.signal,
        onProgress: (job) => {
          if (c.ctrl !== my) return;
          c.status = "loading";
          c.job = job;
          emitSong(c);
          if (c === cur) notify();
        },
      });
      if (c.ctrl !== my) return;
      c.ctrl = null;
      c.status = "ready";
      c.result = result;
      c.cached = true;
      c.serverDown = false;
      emitSong(c);
      if (c === cur) notify();
    } catch (e) {
      if (c.ctrl !== my || e.code === "aborted") return;
      c.ctrl = null;
      c.status = "error";
      c.error = { message: e.message, code: e.code || "unknown" };
      emitSong(c);
      if (c === cur) notify();
      if (e.code === "server_down" && c.visible) {
        retryTimer = setTimeout(function again() {
          if (cur !== c || !c.visible || c.status !== "error") return;
          if (document.hidden) { retryTimer = setTimeout(again, RETRY_MS); return; }
          load(c, false, true);
        }, RETRY_MS);
      }
    }
  }

  function show(c, { user = false } = {}) {
    if (c.visible || c !== cur) return;
    c.visible = true;
    c.userHid = false;
    if (user) remember(c.id, true);
    CT.ui.layout.mount();
    CT.sync.attach();
    if (c.status === "ready" || c.status === "loading") emitSong(c);
    else load(c);          // "off" hoặc "error": bây giờ mới POST /analyze
    notify();
  }

  function hide(c) {
    if (!c.visible) return;
    c.visible = false;
    c.userHid = true;
    remember(c.id, false);
    stopRetry();
    if (c.status === "error") {        // bấm lại thì thử lại từ đầu; bỏ lần tự thử lại đang chạy
      c.ctrl?.abort();
      c.ctrl = null;
      c.status = "off";
    }
    CT.state.result = null;
    CT.bus.emit("song:none", {});
    CT.sync.detach();
    CT.ui.layout.unmount();
    notify();      // job đang chạy (nếu có) vẫn chạy và vẫn được theo dõi
  }

  /** GET /result: chỉ đọc cache của server, không tạo job. Kết quả cũ (stale) không tính là "có ngay". */
  async function probe(c) {
    const res = await CT.api.request(`/result/${encodeURIComponent(c.id)}`);
    if (cur !== c) return;
    // Server tắt: nút báo "chưa kết nối được server", hỏi lại mỗi 5 giây (chỉ khi tab đang hiện).
    c.serverDown = res.status === 0 && res.error !== "extension_reloaded";
    clearTimeout(c.probeTimer);
    if (c.serverDown) {
      c.probeTimer = setTimeout(function again() {
        if (cur !== c) return;
        if (document.hidden) { c.probeTimer = setTimeout(again, RETRY_MS); return; }
        probe(c);
      }, RETRY_MS);
    }
    if (res.ok && res.data && !res.stale) {
      c.cached = true;
      if (c.status === "off") { c.status = "ready"; c.result = res.data; }
    } else if (res.status === 404 || res.ok) {
      c.cached = false;
    }
    notify();
    maybeAutoShow(c);
  }

  function maybeAutoShow(c) {
    if (c === cur && !c.visible && !c.userHid && c.cached && CT.settings.autoShowCached) show(c);
  }

  function leave() {
    const c = cur;
    if (!c) return;
    c.ctrl?.abort();      // chỉ ngừng theo dõi; job trên server vẫn chạy và lưu cache
    c.mel.ctrl?.abort();
    clearTimeout(c.probeTimer);
    stopRetry();
    cur = null;
    if (c.visible) {
      CT.state.result = null;
      CT.bus.emit("song:none", {});
      CT.sync.detach();
      CT.ui.layout.unmount();
    }
  }

  function check() {
    const id = CT.settings.enabled ? CT.sync.videoIdFromUrl(location.href) : null;
    if (id === (cur && cur.id)) { if (id) CT.ui.ctlButton.ensure(); return; }
    leave();
    CT.state.videoId = id;
    CT.state.result = null;
    CT.state.melody = null;
    CT.bus.emit("melody", { videoId: id, status: "unknown", melody: null });
    if (!id) {
      CT.ui.ctlButton.unmount();
      CT.bus.emit("song:none", {});
      notify();
      return;
    }
    const c = (cur = { id, visible: false, status: "off", cached: null, result: null, job: null, error: null, ctrl: null, userHid: false,
      mel: newMel() });
    CT.ui.ctlButton.mount();
    notify();
    if (active.has(id)) show(c);
    probe(c);
  }

  function toggle() {
    const c = cur;
    if (!c) return;
    if (c.visible) hide(c);
    else show(c, { user: true });
  }

  CT.app = {
    reload(force = false) {
      if (!cur) return;
      if (!cur.visible) { show(cur, { user: true }); if (force) load(cur, true); return; }   // show() tự tải khi cần
      load(cur, force);
    },
    toggle,
    check,
    detectMelody(force = false) { if (cur) detectMelody(cur, force); },
  };

  async function start() {
    await CT.loadSettings();
    let lastAuto = !!CT.settings.autoShowCached;
    CT.bus.on("settings", (s) => {
      check();
      const auto = !!s.autoShowCached;
      if (auto && !lastAuto && cur) maybeAutoShow(cur);
      lastAuto = auto;
    });
    window.addEventListener("yt-navigate-finish", check);
    window.addEventListener("popstate", check);
    // Dự phòng: một số kiểu điều hướng của YouTube không phát yt-navigate-finish.
    setInterval(check, 1000);
    check();
  }

  // Chỉ để phát triển (README, mục "Nạp lại khi phát triển"): trang gửi
  // window.postMessage({type: "ct:dev-reload"}, "*") thì nhờ service worker nạp lại extension.
  window.addEventListener("message", (e) => {
    if (e.source !== window || !e.data || e.data.type !== "ct:dev-reload") return;
    try { chrome.runtime.sendMessage({ type: "ct:dev-reload" }, () => void chrome.runtime.lastError); } catch { /* đã mất kết nối */ }
  });

  start().catch((e) => console.error("[ChordTrack] khởi động lỗi", e));
})();
