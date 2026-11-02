/* Popup của ChordTrack: trạng thái backend, bật/tắt trên YouTube, cài đặt nhanh, bài đã phân tích.
 *
 * Cài đặt dùng chung CT.settings của src/core.js (chrome.storage.sync, khóa "settings"), nên content script
 * đang mở nhận thay đổi ngay qua chrome.storage.onChanged.
 * Gọi backend qua service worker (tin nhắn "ct:fetch", xem src/background.js). Không có service worker
 * (mở popup.html như trang thường khi phát triển) thì fetch thẳng CT.settings.serverUrl.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const START_CMD = "powershell -ExecutionPolicy Bypass -File scripts\\start_server.ps1";
  const YT_ID = /^[A-Za-z0-9_-]{11}$/;
  const DIRECT_TIMEOUT_MS = 5000;
  const hasExt = typeof chrome !== "undefined" && !!chrome.runtime?.id;

  // ---------- gọi backend
  function viaWorker(path) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage({ type: "ct:fetch", path, method: "GET" }, (res) => {
          resolve(chrome.runtime.lastError || !res ? null : res);
        });
      } catch { resolve(null); }
    });
  }

  async function direct(path) {
    const base = serverBase();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), DIRECT_TIMEOUT_MS);
    try {
      const r = await fetch(base + path, { signal: ctrl.signal });
      let data = null;
      try { data = await r.json(); } catch {}
      return r.ok ? { ok: true, status: r.status, data } : { ok: false, status: r.status, error: `HTTP ${r.status}`, data };
    } catch (e) {
      return { ok: false, status: 0, error: "server_down", message: `Không kết nối được ${base}`, server: base };
    } finally {
      clearTimeout(timer);
    }
  }

  async function api(path) {
    if (hasExt) {
      const res = await viaWorker(path);
      if (res) return res;
    }
    return direct(path);
  }

  // ---------- trạng thái server
  let checking = 0;
  async function checkHealth() {
    const my = ++checking;
    const st = $("status");
    st.dataset.state = "loading";
    $("stTitle").textContent = "Đang kiểm tra server…";
    $("refresh").classList.remove("spin"); void $("refresh").offsetWidth; $("refresh").classList.add("spin");
    const res = await api("/health");
    if (my !== checking) return;
    const up = res.ok && res.data && res.data.status === "ok";
    st.dataset.state = up ? "ok" : "down";
    $("down").hidden = up;
    $("chips").replaceChildren();
    if (up) {
      const h = res.data;
      $("stTitle").textContent = "Đang chạy";
      const notes = [];
      if (!h.worker_alive) notes.push("Worker phân tích đã dừng, khởi động lại server.");
      if (!h.ffmpeg) notes.push("Thiếu ffmpeg trong PATH.");
      $("stSub").textContent = notes.length ? notes.join(" ") : `Phiên bản ${h.version}`;
      $("stSub").classList.toggle("warn", notes.length > 0);
      addChip("GPU", h.gpu ? h.gpu.replace(/^NVIDIA (GeForce )?/, "") : "Không có (chạy CPU)", h.gpu ? "" : "warn", true);
      addChip("Hàng đợi", h.queue ? `${h.queue} bài đang chờ` : "Trống");
      addChip("Deno", h.deno ? "Có" : "Thiếu", h.deno ? "ok" : "warn");
      addChip("ffmpeg", h.ffmpeg ? "Có" : "Thiếu", h.ffmpeg ? "ok" : "warn");
      addChip("Cookie YouTube", h.cookies ? "Có" : "Chưa có", h.cookies ? "ok" : "", false,
              "Chỉ cần khi YouTube đòi xác minh. Xem README, mục cookie.");
      loadSongs();
    } else {
      $("stTitle").textContent = "Chưa chạy";
      $("stSub").textContent = res.status
        ? `${serverBase()} trả lỗi ${res.status}. Có thể đây không phải server ChordTrack.`
        : `Không kết nối được ${serverBase()}.`;
      showSongsMessage("Bật server để xem danh sách bài.");
    }
  }

  function addChip(k, v, cls = "", wide = false, tip = "") {
    const li = document.createElement("li");
    if (wide) li.className = "wide";
    li.innerHTML = `<span class="k"></span><span class="v"></span>`;
    li.querySelector(".k").textContent = k;
    const vEl = li.querySelector(".v");
    vEl.textContent = v;
    li.title = tip || v;
    if (cls) vEl.classList.add(cls);
    $("chips").append(li);
  }

  const serverBase = () => String(CT.settings.serverUrl || CT.DEFAULTS.serverUrl).replace(/\/+$/, "");

  // ---------- danh sách bài
  const fmtDur = (s) => {
    if (!Number.isFinite(s)) return "";
    s = Math.round(s);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const fmtKey = (k) => {
    if (!k) return "Chưa rõ giọng";
    const [tonic, mode] = k.split(" ");
    return `${tonic} ${mode === "minor" ? "thứ" : mode === "major" ? "trưởng" : mode || ""}`.trim();
  };

  function showSongsMessage(msg) {
    songsReq++;
    $("songs").replaceChildren();
    $("songCount").textContent = "";
    $("songsEmpty").hidden = false;
    $("songsEmpty").textContent = msg;
  }

  let songsReq = 0;
  async function loadSongs() {
    const my = ++songsReq;
    const res = await api("/songs");
    if (my !== songsReq) return;     // đã có lần tải mới hơn (bấm ↻ hoặc đổi server)
    if (!res.ok) { showSongsMessage("Không tải được danh sách bài."); return; }
    const songs = (res.data && res.data.songs) || [];
    if (!songs.length) { showSongsMessage("Chưa có bài nào. Mở một video YouTube để phân tích."); return; }
    $("songsEmpty").hidden = true;
    $("songCount").textContent = String(songs.length);
    $("songs").replaceChildren(...songs.map(songItem));
  }

  function songItem(s) {
    const li = document.createElement("li");
    const isYt = YT_ID.test(s.song_id);
    const row = document.createElement(isYt ? "a" : "div");
    const title = s.title || s.song_id;
    row.innerHTML = `<span class="t"></span><span class="m"></span><span class="d"></span>`;
    row.querySelector(".t").textContent = title;
    row.querySelector(".m").textContent = fmtKey(s.key);
    row.querySelector(".d").textContent = fmtDur(s.duration);
    if (isYt) {
      const url = `https://www.youtube.com/watch?v=${s.song_id}`;
      row.href = url;
      row.target = "_blank";
      row.rel = "noopener";
      row.title = `Mở trên YouTube: ${title}`;
      const yt = document.createElement("span");
      yt.className = "yt";
      yt.textContent = "▶";
      yt.setAttribute("aria-hidden", "true");
      row.querySelector(".t").prepend(yt);
    } else {
      row.title = `${title} (file local, không có trên YouTube)`;
    }
    li.append(row);
    return li;
  }

  // ---------- cài đặt
  function renderSettings() {
    const s = CT.settings;
    $("enabled").checked = !!s.enabled;
    $("enabledText").textContent = s.enabled ? "Đang bật trên YouTube" : "Đã tắt trên YouTube";
    document.body.classList.toggle("off", !s.enabled);
    for (const b of $("mode").querySelectorAll("button")) {
      const on = b.dataset.mode === s.mode;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    const vol = Math.round((Number(s.metronomeVolume) || 0) * 100);
    if (document.activeElement !== $("vol")) $("vol").value = vol;
    $("volOut").textContent = `${vol}%`;
    if (document.activeElement !== $("server")) $("server").value = s.serverUrl;
  }

  $("enabled").addEventListener("change", (e) => CT.setSettings({ enabled: e.target.checked }));

  const modes = [...$("mode").querySelectorAll("button")];
  for (const b of modes) b.addEventListener("click", () => CT.setSettings({ mode: b.dataset.mode }));
  $("mode").addEventListener("keydown", (e) => {
    const i = modes.findIndex((b) => b.dataset.mode === CT.settings.mode);
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const next = modes[(i + d + modes.length) % modes.length];
    CT.setSettings({ mode: next.dataset.mode });
    next.focus();
  });

  // chrome.storage.sync giới hạn số lần ghi mỗi phút: lúc kéo thanh trượt chỉ ghi sau 200 ms đứng yên.
  let volTimer = 0;
  $("vol").addEventListener("input", (e) => {
    const v = Number(e.target.value);
    $("volOut").textContent = `${v}%`;
    clearTimeout(volTimer);
    volTimer = setTimeout(() => CT.setSettings({ metronomeVolume: v / 100 }), 200);
  });
  $("vol").addEventListener("change", (e) => {
    clearTimeout(volTimer);
    CT.setSettings({ metronomeVolume: Number(e.target.value) / 100 });
  });

  async function saveServer(url) {
    url = url.trim().replace(/\/+$/, "");
    // Server chỉ nhận Host 127.0.0.1/localhost (chống DNS rebinding) và manifest chỉ xin quyền 127.0.0.1:
    // địa chỉ khác lưu được cũng không gọi được, nên chỉ cho đổi cổng.
    let ok = false;
    try {
      const u = new URL(url);
      ok = u.protocol === "http:" && (u.hostname === "127.0.0.1" || u.hostname === "localhost");
    } catch {}
    $("server").setAttribute("aria-invalid", String(!ok));
    $("serverErr").hidden = ok;
    if (!ok) return;
    $("server").value = url;
    if (url !== CT.settings.serverUrl) await CT.setSettings({ serverUrl: url });
    checkHealth();
  }
  $("server").addEventListener("change", (e) => saveServer(e.target.value));
  $("server").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); });
  $("serverReset").addEventListener("click", () => saveServer(CT.DEFAULTS.serverUrl));

  $("refresh").addEventListener("click", checkHealth);

  $("cmd").textContent = START_CMD;
  $("copy").addEventListener("click", async () => {
    let ok = false;
    try { await navigator.clipboard.writeText(START_CMD); ok = true; } catch {
      const r = document.createRange();
      r.selectNodeContents($("cmd"));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      try { ok = document.execCommand("copy"); } catch {}
      sel.removeAllRanges();
    }
    $("copy").textContent = ok ? "Đã chép ✓" : "Chép thủ công";
    $("copy").classList.toggle("done", ok);
    setTimeout(() => { $("copy").textContent = "Sao chép"; $("copy").classList.remove("done"); }, 1600);
  });

  CT.bus.on("settings", renderSettings);

  (async () => {
    // Trang phát triển (không phải extension): cho đặt server qua ?server=... để thử cả hai trạng thái.
    const devServer = !hasExt && new URLSearchParams(location.search).get("server");
    await CT.loadSettings();
    if (devServer) CT.settings.serverUrl = devServer;
    renderSettings();
    checkHealth();
  })();
})();
