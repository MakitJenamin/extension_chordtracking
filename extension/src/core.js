/* ChordTrack: namespace chung, bus sự kiện, cài đặt người dùng.
 *
 * HỢP ĐỒNG giữa các module (đừng đổi tên khi chưa báo coordinator):
 *   window.CT                        namespace duy nhất, mọi module gắn vào đây
 *   CT.bus.on(evt, fn) / emit(evt, payload) / off(evt, fn)
 *   Sự kiện:
 *     "song:loading"  {videoId, job}         đang phân tích (job: {status, progress, step, eta_seconds})
 *     "song:ready"    {videoId, result}      có JSON kết quả (dạng xem chordtrack/pipeline.py build_result)
 *     "song:error"    {videoId, message}
 *     "song:none"     {}                     trang không phải video
 *     "time"          {t, rate, paused}      mỗi khung hình (requestAnimationFrame) khi có video
 *     "seek"          {t}                    người dùng tua
 *     "settings"      settings               cài đặt đổi
 *     "fullscreen"    {on, container}        YouTube vào/ra toàn màn hình
 *   CT.settings                      cài đặt hiện tại (xem DEFAULTS); CT.setSettings(patch) lưu chrome.storage.sync
 *   CT.seekTo(t)                     tua video (do sync.js cài)
 *   CT.state = {videoId, result, video}
 *
 * BỔ SUNG (ext-core, chỉ thêm, không đổi phần trên):
 *   "time" có thêm `ad` (true khi YouTube đang chạy quảng cáo; lúc đó paused = true)
 *   "song:error" có thêm `code`: "server_down" | "youtube_blocked" | "job_error" | "http" | "extension_reloaded"
 *   CT.settings.panelCollapsed       khung hợp âm đang thu gọn (không có trong DEFAULTS, mặc định falsy)
 *   CT.api      gọi backend qua background.js (src/api.js)
 *   CT.theory   dịch giọng, nhãn theo chế độ, nhịp (src/theory.js)
 *   CT.sync     attach/detach vòng rAF, tick, videoIdFromUrl (src/sync.js)
 *   CT.ui       strip, panel, sections, fullscreen, layout (src/ui/*)
 *   CT.app      reload(force), check() (src/content.js)
 */
(() => {
  const listeners = {};
  const bus = {
    on(evt, fn) { (listeners[evt] ||= new Set()).add(fn); return () => bus.off(evt, fn); },
    off(evt, fn) { listeners[evt]?.delete(fn); },
    emit(evt, payload) {
      for (const fn of listeners[evt] || []) {
        try { fn(payload); } catch (e) { console.error("[ChordTrack]", evt, e); }
      }
    },
  };

  const DEFAULTS = {
    enabled: true,
    mode: "white",            // "orig" | "white" | "roman"  (mặc định phím trắng: người dùng chơi piano)
    transpose: 0,             // chỉ dùng ở mode "orig", -6..+6
    hideInversions: false,
    simplify: false,
    metronome: false,
    metronomeVolume: 0.6,     // 0..1
    pxPerSecond: 46,          // độ "zoom" của dải hợp âm
    modulationWarnSeconds: 8,
    serverUrl: "http://127.0.0.1:8765",
  };

  const CT = (window.CT = {
    bus,
    DEFAULTS,
    settings: { ...DEFAULTS },
    state: { videoId: null, result: null, video: null },
    seekTo: () => {},
    log: (...a) => console.debug("[ChordTrack]", ...a),
    async loadSettings() {
      try {
        const s = await chrome.storage.sync.get("settings");
        CT.settings = { ...DEFAULTS, ...(s.settings || {}) };
      } catch { CT.settings = { ...DEFAULTS }; }
      bus.emit("settings", CT.settings);
    },
    async setSettings(patch) {
      CT.settings = { ...CT.settings, ...patch };
      bus.emit("settings", CT.settings);
      try { await chrome.storage.sync.set({ settings: CT.settings }); } catch {}
    },
  });

  try {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "sync" && changes.settings) {
        CT.settings = { ...DEFAULTS, ...changes.settings.newValue };
        bus.emit("settings", CT.settings);
      }
    });
  } catch {}
})();
