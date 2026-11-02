/* ChordTrack API: gọi backend local qua service worker (background.js, tin nhắn "ct:fetch").
 *
 * CT.api.request(path, {method, body}) -> {ok, status, data | error}
 * CT.api.loadSong(videoId, {onProgress, signal}) -> result JSON
 *   POST /analyze; có kết quả thì trả ngay, chưa có thì poll GET /jobs/{id} mỗi ~1 s
 *   (onProgress(job) mỗi lần), xong thì GET /result/{song_id}.
 *   Lỗi ném ra là CT.api.ApiError với `code`: "server_down" | "youtube_blocked" | "job_error" | "http" | "aborted".
 */
(() => {
  const POLL_MS = 1000;

  class ApiError extends Error {
    constructor(code, message, extra = {}) { super(message); this.code = code; Object.assign(this, extra); }
  }

  function request(path, { method = "GET", body } = {}) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "ct:fetch", path, method, body }, (res) => {
          const err = chrome.runtime.lastError;
          if (err || !res) {
            // Extension vừa được nạp lại: content script cũ mất kết nối với service worker.
            resolve({ ok: false, status: 0, error: "extension_reloaded",
                      message: (err && err.message) || "Không có phản hồi từ service worker" });
          } else resolve(res);
        });
      } catch (e) {
        resolve({ ok: false, status: 0, error: "extension_reloaded", message: String(e) });
      }
    });
  }

  function toError(res) {
    if (res.status === 0) {
      if (res.error === "extension_reloaded") {
        return new ApiError("extension_reloaded", "Extension vừa được cập nhật. Tải lại trang (F5) để tiếp tục.");
      }
      return new ApiError("server_down", res.message || "Không kết nối được server", { server: res.server });
    }
    return new ApiError("http", res.error || `HTTP ${res.status}`, { status: res.status });
  }

  /** Lỗi job: nhận ra trường hợp YouTube chặn (cần cookie) để UI hướng dẫn riêng. */
  function jobError(message) {
    const low = (message || "").toLowerCase();
    const blocked = low.includes("cookies") || low.includes("không phải bot") || low.includes("not a bot")
      || low.includes("đăng nhập") || low.includes("429");
    return new ApiError(blocked ? "youtube_blocked" : "job_error", message || "Phân tích thất bại");
  }

  const sleep = (ms, signal) => new Promise((resolve, reject) => {
    const onAbort = () => { clearTimeout(id); reject(new ApiError("aborted", "Đã hủy")); };
    const id = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(); }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

  async function loadSong(videoId, { onProgress, signal, force = false } = {}) {
    const check = () => { if (signal?.aborted) throw new ApiError("aborted", "Đã hủy"); };
    const res = await request("/analyze", { method: "POST", body: { source: videoId, force } });
    check();
    if (!res.ok) throw toError(res);
    const first = res.data || {};
    if (first.status === "done" && first.result) return first.result;
    if (!first.job_id) throw new ApiError("http", "Phản hồi /analyze không có job_id");

    let job = { status: first.status, progress: 0, step: null, message: "Đang chờ", eta_seconds: null };
    onProgress?.(job);
    let failures = 0;
    for (;;) {
      await sleep(POLL_MS, signal);
      const r = await request(`/jobs/${encodeURIComponent(first.job_id)}`);
      check();
      if (!r.ok) {
        // Server tắt giữa chừng: thử lại vài lần trước khi báo lỗi.
        if (r.status === 0 && ++failures < 5) continue;
        throw toError(r);
      }
      if (!r.data || typeof r.data !== "object") { // 200 nhưng không phải JSON (proxy, server đang khởi động lại)
        if (++failures < 5) continue;
        throw new ApiError("http", "Server trả về dữ liệu job không đọc được");
      }
      failures = 0;
      job = r.data;
      if (job.status === "error") throw jobError(job.error || job.message);
      if (job.status === "done") break;
      onProgress?.(job);
    }
    const songId = job.song_id || first.song_id;
    const rr = await request(`/result/${encodeURIComponent(songId)}`);
    check();
    if (!rr.ok) throw toError(rr);
    return rr.data;
  }

  CT.api = { request, loadSong, ApiError };
})();
