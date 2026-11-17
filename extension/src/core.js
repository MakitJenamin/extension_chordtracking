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
 *   CT.settings.fx                   hiệu ứng ánh sáng của dải (mặc định bật; html.ct-fx-on, strip.js)
 *   Tách giọng (`vocals: true` của POST /melody) và căn nốt về phách (theory.snapNotes + autoSnapDiv) luôn bật,
 *   không có công tắc (người dùng chốt 2026-11-11). Server thiếu gói tách giọng thì tự dùng bản không tách giọng
 *   (CT.theory.melodyEffective theo `melody_models` của GET /health).
 *   CT.app.detectMelody(force)       bấm "Dò giai điệu": POST /melody và theo dõi job
 *
 * GIỌNG NGƯỜI DÙNG CHỌN (khi bộ đoán giọng sai):
 *   CT.keyOverrides                  {videoId: {chỉ số vùng giọng: "D major"}}, lưu chrome.storage.local "keyOverrides"
 *   CT.setKeyOverride(id, i, text)   chọn giọng cho vùng i của video id, dời cả bài theo (text null: về giọng tự đoán), phát
 *   "keyOverride" {videoId}          content.js áp vào kết quả (CT.theory.withKeyOverride) và phát lại song:ready
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
    transpose: 0,             // chỉ dùng ở mode "orig", -6..+5 (+6 trùng tông với -6)
    hideInversions: false,
    simplify: false,
    metronome: false,
    metronomeVolume: 0.6,     // 0..1
    pxPerSecond: 46,          // độ "zoom" của dải hợp âm
    modulationWarnSeconds: 8,
    snapToBeats: true,        // kéo mép hợp âm về phách gần nhất (CT.theory.snapChords), chỉ để hiển thị
    showMelody: true,         // dòng giai điệu dưới dải hợp âm (khi bài đã dò giai điệu)
    fx: true,                 // hiệu ứng ánh sáng của dải (vạch đỏ sáng theo phách, hợp âm/nốt sáng dần khi tới gần)
    melodyModel: "yourmt3",   // cách dò giai điệu: "yourmt3" ("Bình thường", mặc định) | "combo" ("Kỹ": YourMT3+ ghép MuScriptor)
    settingsRev: 4,           // phiên bản cài đặt, dùng để chuyển đổi một lần
    autoShowCached: false,    // bài đã có kết quả thì hiện luôn, không cần bấm nút (không bao giờ tự phân tích bài mới)
    serverUrl: "http://127.0.0.1:8765",
  };

  const CT = (window.CT = {
    bus,
    DEFAULTS,
    settings: { ...DEFAULTS },
    state: { videoId: null, result: null, video: null, melody: null },
    keyOverrides: {},
    seekTo: () => {},
    log: (...a) => console.debug("[ChordTrack]", ...a),
    async loadSettings() {
      try {
        const s = await chrome.storage.sync.get("settings");
        CT.settings = { ...DEFAULTS, ...(s.settings || {}) };
        // Một lần (settingsRev < 2): bật lại "Khớp phách" cho người đã lỡ tắt, theo yêu cầu mặc định ON.
        const rev = CT.settings.settingsRev || 0;
        if (rev < 4) {
          if (rev < 2) CT.settings = { ...CT.settings, snapToBeats: true };
          // Một lần (settingsRev < 3): mặc định giai điệu đổi sang "yourmt3" (người dùng chốt 2026-11-11). setSettings
          // lưu cả object nên người đã đổi cài đặt bất kỳ đều có sẵn "combo" cũ, ghi đè mặc định mới.
          if (rev < 3) CT.settings = { ...CT.settings, melodyModel: "yourmt3" };
          // settingsRev 4: tách giọng và nốt khớp phách luôn bật, bỏ hai cài đặt cũ.
          CT.settings = { ...CT.settings, settingsRev: 4 };
          delete CT.settings.melodyVocals;
          delete CT.settings.melodySnap;
          try { await chrome.storage.sync.set({ settings: CT.settings }); } catch {}
        }
      } catch { CT.settings = { ...DEFAULTS }; }
      // Bản cũ cho lên tới +6: đổi thành −6 (cùng tông).
      if ((CT.settings.transpose | 0) > 5) CT.settings = { ...CT.settings, transpose: (CT.settings.transpose | 0) - 12 };
      try {
        const o = await chrome.storage.local.get("keyOverrides");
        CT.keyOverrides = (o.keyOverrides && typeof o.keyOverrides === "object") ? o.keyOverrides : {};
      } catch { CT.keyOverrides = {}; }
      bus.emit("settings", CT.settings);
    },
    async setKeyOverride(videoId, index, text) {
      if (!videoId) return;
      // Một mục mỗi video: giọng chọn cho vùng `index` dời cả bài (CT.theory.withKeyOverride). null: về giọng tự đoán.
      const cur = text ? { [index]: text } : {};
      const all = { ...CT.keyOverrides };
      delete all[videoId];                                   // xóa rồi thêm: video vừa sửa về cuối
      if (Object.keys(cur).length) all[videoId] = cur;
      const ids = Object.keys(all);
      for (const id of ids.slice(0, Math.max(0, ids.length - 500))) delete all[id];   // giữ 500 video gần nhất
      CT.keyOverrides = all;
      bus.emit("keyOverride", { videoId });
      try { await chrome.storage.local.set({ keyOverrides: all }); } catch {}
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
      // Tab khác chọn giọng: áp luôn cho tab này (cùng video thì content.js dựng lại bài).
      if (area === "local" && changes.keyOverrides) {
        const old = CT.keyOverrides;
        CT.keyOverrides = changes.keyOverrides.newValue || {};
        for (const id of new Set([...Object.keys(old), ...Object.keys(CT.keyOverrides)])) {
          if (JSON.stringify(old[id]) !== JSON.stringify(CT.keyOverrides[id])) bus.emit("keyOverride", { videoId: id });
        }
      }
    });
  } catch {}
})();
