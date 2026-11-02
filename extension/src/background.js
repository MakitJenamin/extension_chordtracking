/* ChordTrack service worker: chuyển tiếp yêu cầu HTTP từ content script tới backend local.
 *
 * Content script chạy trong trang https://www.youtube.com nên không gọi thẳng http://127.0.0.1 được
 * (mixed content, CORS). Nó gửi tin nhắn:
 *   {type: "ct:fetch", path: "/analyze", method: "POST", body: {...}}
 * và nhận lại:
 *   {ok: true,  status: 200, data: <JSON>}
 *   {ok: false, status: 404, error: "<thông báo>", data?: <JSON lỗi của server>}
 *   {ok: false, status: 0,   error: "server_down", message: "<chi tiết>"}   server không chạy
 */
const DEFAULT_SERVER = "http://127.0.0.1:8765";
const TIMEOUT_MS = 15000;

async function serverUrl() {
  try {
    const s = await chrome.storage.sync.get("settings");
    const url = (s.settings && s.settings.serverUrl) || DEFAULT_SERVER;
    return url.replace(/\/+$/, "");
  } catch {
    return DEFAULT_SERVER;
  }
}

async function forward({ path, method = "GET", body }) {
  if (typeof path !== "string" || !path.startsWith("/")) {
    return { ok: false, status: 0, error: "bad_request", message: "path phải bắt đầu bằng /" };
  }
  const base = await serverUrl();
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch(base + path, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    const timeout = e && e.name === "AbortError";
    return {
      ok: false, status: 0, error: timeout ? "timeout" : "server_down",
      message: timeout ? `Server ${base} không trả lời sau ${TIMEOUT_MS / 1000} s`
                       : `Không kết nối được ${base}: ${e && e.message}`,
      server: base,
    };
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await res.json(); } catch { /* phản hồi không phải JSON */ }
  if (res.ok) return { ok: true, status: res.status, data, stale: res.headers.get("X-Chordtrack-Stale") === "1" };
  const detail = data && (data.detail || data.error || data.message);
  return { ok: false, status: res.status, error: typeof detail === "string" ? detail : `HTTP ${res.status}`, data };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.type !== "ct:fetch") return false;
  forward(msg).then(sendResponse, (e) =>
    sendResponse({ ok: false, status: 0, error: "internal", message: String(e) }));
  return true; // trả lời bất đồng bộ
});
