/* ChordTrack API: gọi backend local qua service worker (background.js, tin nhắn "ct:fetch").
 *
 * CT.api.request(path, {method, body}) -> {ok, status, data | error}
 * CT.api.loadSong(videoId, {onProgress, signal}) -> result JSON
 *   POST /analyze; có kết quả thì trả ngay, chưa có thì poll GET /jobs/{id} mỗi ~1 s
 *   (onProgress(job) mỗi lần), xong thì GET /result/{song_id}.
 *   Lỗi ném ra là CT.api.ApiError với `code`: "server_down" | "youtube_blocked" | "job_error" | "http" | "aborted".
 * CT.api.getMelody(id) -> {melody, stale} | {melody: null, status}   GET /melody/{id} (không tạo job)
 * CT.api.loadMelody(videoId, {onProgress, signal, force}) -> <Melody>
 *   POST /melody, theo dõi GET /jobs/{id}, xong thì GET /melody/{song_id} (docs/melody_api.md).
 *   Lỗi có message tiếng Việt; thêm code "unsupported" khi server chưa có /melody.
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

  /** Theo dõi job (GET /jobs/{id} mỗi ~1 s) tới khi xong; trả job cuối (status "done"). `fail(msg)` tạo lỗi job. */
  async function waitJob(first, { onProgress, signal, fail = jobError } = {}) {
    const check = () => { if (signal?.aborted) throw new ApiError("aborted", "Đã hủy"); };
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
      if (job.status === "error") throw fail(job.error || job.message);
      if (job.status === "done") return job;
      onProgress?.(job);
    }
  }

  async function loadSong(videoId, { onProgress, signal, force = false } = {}) {
    const check = () => { if (signal?.aborted) throw new ApiError("aborted", "Đã hủy"); };
    const res = await request("/analyze", { method: "POST", body: { source: videoId, force } });
    check();
    if (!res.ok) throw toError(res);
    const first = res.data || {};
    if (first.status === "done" && first.result) return first.result;
    if (!first.job_id) throw new ApiError("http", "Phản hồi /analyze không có job_id");
    const job = await waitJob(first, { onProgress, signal });
    const songId = job.song_id || first.song_id;
    const rr = await request(`/result/${encodeURIComponent(songId)}`);
    check();
    if (!rr.ok) throw toError(rr);
    return rr.data;
  }

  /* ---------------- Giai điệu (docs/melody_api.md) ---------------- */

  /** Lỗi của phần giai điệu, câu tiếng Việt cho người dùng (giữ `code` như lỗi hợp âm). */
  function melodyError(e) {
    if (!(e instanceof ApiError)) return new ApiError("http", "Không dò được giai điệu: " + String(e && e.message || e));
    if (e.code === "server_down") return new ApiError("server_down", "Chưa kết nối được server ChordTrack. Bật server rồi bấm lại.");
    if (e.code === "http" && e.post && (e.status === 404 || e.status === 405)) {   // chỉ khi chính POST /melody 404/405
      return new ApiError("unsupported", "Server chưa hỗ trợ dò giai điệu. Cập nhật mã server (git pull) rồi khởi động lại.", { status: e.status });
    }
    if (e.code === "youtube_blocked") return new ApiError("youtube_blocked", "YouTube chặn tải audio, cần file cookie (xem hướng dẫn khi phân tích hợp âm).");
    if (e.code === "job_error") return new ApiError("job_error", "Không dò được giai điệu: " + e.message);
    if (e.code === "http") return new ApiError("http", "Không dò được giai điệu: " + e.message, { status: e.status });
    return e;
  }

  /** <Melody> hợp lệ tối thiểu: có mảng notes. */
  const isMelody = (d) => !!d && typeof d === "object" && Array.isArray(d.notes);

  /** GET /melody/{id}: kết quả đã có, không tạo job. Trả {melody, stale} hoặc null (404, server tắt…). */
  async function getMelody(songId) {
    const r = await request(`/melody/${encodeURIComponent(songId)}`);
    if (r.ok && isMelody(r.data)) return { melody: r.data, stale: !!r.stale };
    return { melody: null, status: r.status, error: r.error };
  }

  /** POST /melody rồi theo dõi job, xong thì GET /melody/{song_id}. Ném ApiError (message tiếng Việt). */
  async function loadMelody(videoId, { onProgress, signal, force = false } = {}) {
    const check = () => { if (signal?.aborted) throw new ApiError("aborted", "Đã hủy"); };
    try {
      const res = await request("/melody", { method: "POST", body: { source: videoId, force } });
      check();
      if (!res.ok) throw Object.assign(toError(res), { post: true });
      const first = res.data || {};
      if (first.status === "done" && isMelody(first.melody)) return first.melody;
      if (!first.job_id) throw new ApiError("http", "phản hồi /melody không có job_id");
      const job = await waitJob(first, { onProgress, signal, fail: (m) => {
        const e = jobError(m);
        return e.code === "youtube_blocked" ? e : new ApiError("job_error", m || "lỗi không rõ");
      } });
      const rr = await request(`/melody/${encodeURIComponent(job.song_id || first.song_id || videoId)}`);
      check();
      if (!rr.ok) throw new ApiError("http", rr.status === 404 ? "job đã xong nhưng server không có kết quả giai điệu" : (rr.error || `HTTP ${rr.status}`), { status: rr.status });
      if (!isMelody(rr.data)) throw new ApiError("http", "kết quả giai điệu không đọc được");
      return rr.data;
    } catch (e) {
      if (e && e.code === "aborted") throw e;
      if (e && e.code === "extension_reloaded") throw e;
      throw melodyError(e);
    }
  }

  CT.api = { request, loadSong, waitJob, getMelody, loadMelody, ApiError };
})();
