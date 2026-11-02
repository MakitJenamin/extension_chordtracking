/* ChordTrack content: điều phối. Nhận videoId khi YouTube điều hướng SPA, gọi backend, phát song:*.
 *
 * CT.app.reload(force)   phân tích lại video hiện tại (nút "Thử lại" của panel dùng)
 * Server tắt: tự thử lại mỗi 5 giây cho tới khi kết nối được (chỉ khi tab đang hiện).
 */
(() => {
  const RETRY_MS = 5000;
  let current = null;       // videoId đang hiện
  let ctrl = null;          // AbortController của lần tải hiện tại
  let retryTimer = 0;

  function cancel() {
    ctrl?.abort();
    ctrl = null;
    clearTimeout(retryTimer);
    retryTimer = 0;
  }

  /** silent: lần tự thử lại khi server tắt, giữ nguyên thông báo lỗi thay vì nháy sang "đang kết nối". */
  async function load(videoId, force = false, silent = false) {
    cancel();
    const my = (ctrl = new AbortController());
    CT.state.result = null;
    if (!silent) CT.bus.emit("song:loading", { videoId, job: { status: "queued", progress: 0, step: null, message: "Đang kết nối server", eta_seconds: null } });
    try {
      const result = await CT.api.loadSong(videoId, {
        force, signal: my.signal,
        onProgress: (job) => { if (ctrl === my) CT.bus.emit("song:loading", { videoId, job }); },
      });
      if (ctrl !== my || current !== videoId) return;
      CT.state.result = result;
      CT.bus.emit("song:ready", { videoId, result });
    } catch (e) {
      if (ctrl !== my || current !== videoId || e.code === "aborted") return;
      CT.bus.emit("song:error", { videoId, message: e.message, code: e.code || "unknown" });
      if (e.code === "server_down") {
        retryTimer = setTimeout(function again() {
          if (ctrl !== my) return;
          if (document.hidden) { retryTimer = setTimeout(again, RETRY_MS); return; }
          load(videoId, false, true);
        }, RETRY_MS);
      }
    }
  }

  function check() {
    const id = CT.settings.enabled ? CT.sync.videoIdFromUrl(location.href) : null;
    if (id === current) return;
    current = id;
    CT.state.videoId = id;
    if (!id) {
      cancel();
      CT.sync.detach();
      CT.state.result = null;
      CT.bus.emit("song:none", {});
      CT.ui.layout.unmount();
      return;
    }
    CT.ui.layout.mount();
    CT.sync.attach();
    load(id);
  }

  CT.app = {
    reload(force = false) { if (current) load(current, force); },
    check,
  };

  async function start() {
    await CT.loadSettings();
    CT.bus.on("settings", check);
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
