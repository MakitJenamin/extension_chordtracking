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
 *   CT.settings.snapToBeats          kéo mép hợp âm về phách khi hiển thị (mặc định bật)
 *   CT.settings.autoShowCached       tự hiện cho bài đã có kết quả (mặc định tắt; chỉ phân tích khi người dùng bấm)
 *   "app" {videoId, visible, status, cached, progress}   trạng thái bật/tắt của video hiện tại (content.js phát,
 *                                    nút trên thanh điều khiển player nghe)
 *   CT.settings.panelCollapsed       khung hợp âm đang thu gọn (không có trong DEFAULTS, mặc định falsy)
 *   CT.api      gọi backend qua background.js (src/api.js)
 *   CT.theory   dịch giọng, nhãn theo chế độ, nhịp (src/theory.js)
 *   CT.sync     attach/detach vòng rAF, tick, videoIdFromUrl (src/sync.js)
 *   CT.ui       strip, panel, sections, fullscreen, layout (src/ui/*)
 *   CT.app      reload(force), toggle(), check() (src/content.js)
 *
 * GIAI ĐIỆU (docs/melody_api.md):
 *   "melody" {videoId, model, status, melody, stale, job, error}   trạng thái dòng giai điệu của video hiện tại,
 *                    biến thể `model` đang chọn (CT.settings.melodyModel: "combo" | "yourmt3", nhớ lựa chọn)
 *            status: "unknown" (chưa hỏi server) | "none" (chưa dò) | "loading" (job đang chạy, có job)
 *                    | "ready" (có melody) | "error" (có error {code, message})
 *   CT.state.melody                  <Melody> đang hiện (hoặc null)
 *   CT.settings.showMelody           hiện dòng giai điệu dưới dải hợp âm (mặc định bật, nhớ lựa chọn)
 *   CT.settings.melodyVocals         dò giai điệu bản tách giọng (cờ `vocals` của POST /melody, mặc định tắt)
 *   CT.settings.melodySnap           căn nốt giai điệu về lưới phách khi hiển thị (mặc định tắt); melodySnapDiv 4 | 2
 *   CT.app.detectMelody(force)       bấm "Dò giai điệu": POST /melody và theo dõi job
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
    snapToBeats: true,        // kéo mép hợp âm về phách gần nhất (CT.theory.snapChords), chỉ để hiển thị
    showMelody: true,         // dòng giai điệu dưới dải hợp âm (khi bài đã dò giai điệu)
    melodyModel: "yourmt3",   // biến thể dò giai điệu: "yourmt3" (chỉ YourMT3+, mặc định) | "combo" (ghép YourMT3+ + MuScriptor)
    melodyVocals: false,      // tách giọng (BS-Roformer) trước khi dò: đoạn có giọng lấy YourMT3+ trên stem giọng
    melodySnap: false,        // căn đầu/cuối nốt giai điệu về lưới phách (CT.theory.snapNotes), chỉ để hiển thị
    melodySnapDiv: 4,         // số phần chia mỗi phách khi căn nốt: 4 (nốt 1/16 ở nhịp x/4) | 2 (nốt 1/8)
    settingsRev: 3,           // phiên bản cài đặt, dùng để chuyển đổi một lần
    autoShowCached: false,    // bài đã có kết quả thì hiện luôn, không cần bấm nút (không bao giờ tự phân tích bài mới)
    serverUrl: "http://127.0.0.1:8765",
  };

  const CT = (window.CT = {
    bus,
    DEFAULTS,
    settings: { ...DEFAULTS },
    state: { videoId: null, result: null, video: null, melody: null },
    seekTo: () => {},
    log: (...a) => console.debug("[ChordTrack]", ...a),
    async loadSettings() {
      try {
        const s = await chrome.storage.sync.get("settings");
        CT.settings = { ...DEFAULTS, ...(s.settings || {}) };
        // Một lần (settingsRev < 2): bật lại "Khớp phách" cho người đã lỡ tắt, theo yêu cầu mặc định ON.
        const rev = CT.settings.settingsRev || 0;
        if (rev < 3) {
          if (rev < 2) CT.settings = { ...CT.settings, snapToBeats: true };
          // Một lần (settingsRev < 3): mặc định giai điệu đổi sang "yourmt3" (người dùng chốt 2026-11-11). setSettings
          // lưu cả object nên người đã đổi cài đặt bất kỳ đều có sẵn "combo" cũ, ghi đè mặc định mới.
          CT.settings = { ...CT.settings, melodyModel: "yourmt3", settingsRev: 3 };
          try { await chrome.storage.sync.set({ settings: CT.settings }); } catch {}
        }
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
