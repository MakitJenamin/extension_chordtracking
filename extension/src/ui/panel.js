/* ChordTrack panel: khung hợp âm ngay dưới player (trên tiêu đề video).
 *
 * CT.ui.panel        {el, show(state)}: trạng thái "hidden" | "loading" | "error" | "ready"
 * CT.ui.BeatDots     chấm phách 1-2-3-4 (dùng chung với lớp phủ toàn màn hình)
 * CT.ui.info(song,t) thông tin theo thời điểm: giọng, transpose, cảnh báo sắp đổi giọng
 * CT.ui.layout       mount()/unmount(): gắn panel và bảng cấu trúc vào DOM YouTube, gắn lại khi YouTube vẽ lại
 * CT.ui.shortcuts    danh sách phím tắt (Alt+Shift+...), không trùng phím của YouTube
 *
 * Panel tự nghe "song:*", "time", "settings". Gọi CT.app.reload(force) để thử lại (content.js cài).
 */
(() => {
  const T = () => CT.theory;
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids) if (kid != null) e.append(kid);
    return e;
  };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const SERVER_CMD = "powershell -ExecutionPolicy Bypass -File scripts\\start_server.ps1";

  /* ---------------- Phím tắt ---------------- */
  const SHORTCUTS = [
    { code: "KeyC", key: "C", label: "Bật/tắt ChordTrack cho video này", run: () => CT.app && CT.app.toggle() },
    { code: "Digit1", key: "1", label: "Chế độ Gốc", run: () => CT.setSettings({ mode: "orig" }) },
    { code: "Digit2", key: "2", label: "Chế độ Phím trắng", run: () => CT.setSettings({ mode: "white" }) },
    { code: "Digit3", key: "3", label: "Chế độ Bậc La Mã", run: () => CT.setSettings({ mode: "roman" }) },
    { code: "ArrowUp", key: "↑", label: "Tông +1 (chế độ Gốc)", run: () => stepTranspose(1) },
    { code: "ArrowDown", key: "↓", label: "Tông −1 (chế độ Gốc)", run: () => stepTranspose(-1) },
    { code: "KeyM", key: "M", label: "Bật/tắt metronome", run: () => CT.setSettings({ metronome: !CT.settings.metronome }) },
    { code: "KeyH", key: "H", label: "Thu gọn/mở khung hợp âm", run: () => CT.setSettings({ panelCollapsed: !CT.settings.panelCollapsed }) },
    { code: "KeyG", key: "G", label: "Ẩn/hiện dòng giai điệu", run: () => CT.setSettings({ showMelody: CT.settings.showMelody === false }) },
  ];
  // Bước của job dò giai điệu (docs/melody_api.md).
  const MELODY_STEPS = { fetch: "Tải audio", melody_voice: "Dò giọng hát", melody_fill: "Dò nhạc cụ", melody_line: "Ghép dòng nốt" };
  const etaText = (eta) => (eta == null ? "" : eta < 60 ? `còn khoảng ${Math.max(5, Math.round(eta / 5) * 5)} giây`
    : `còn khoảng ${Math.round(eta / 60)} phút`);
  const sc = (code) => "Alt+Shift+" + SHORTCUTS.find((s) => s.code === code).key;
  function stepTranspose(d) {
    const v = Math.max(-6, Math.min(6, (CT.settings.transpose | 0) + d));
    CT.setSettings({ transpose: v, mode: CT.settings.mode === "white" ? "orig" : CT.settings.mode });
  }
  window.addEventListener("keydown", (e) => {
    if (!e.altKey || !e.shiftKey || e.ctrlKey || e.metaKey) return;
    if (!CT.state.videoId || !CT.settings.enabled) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const s = SHORTCUTS.find((x) => x.code === e.code);
    if (!s) return;
    e.preventDefault();
    e.stopPropagation();
    s.run();
  }, true);

  /* ---------------- Chấm phách ---------------- */
  class BeatDots {
    constructor() { this.el = h("span", { class: "ct-beats", title: "Phách trong ô nhịp (phách 1 màu đỏ)" }); this.n = 0; this.pos = -1; }
    setMeter(n) {
      if (n === this.n) return;
      this.n = n;
      this.el.textContent = "";
      for (let i = 1; i <= n; i++) this.el.append(h("i", { class: i === 1 ? "ct-first" : null, text: String(i) }));
      this.pos = -1;
    }
    set(pos) {
      if (pos === this.pos) return;
      const kids = this.el.children;
      if (this.pos > 0 && kids[this.pos - 1]) kids[this.pos - 1].classList.remove("ct-on");
      if (pos > 0 && kids[pos - 1]) kids[pos - 1].classList.add("ct-on");
      this.pos = pos;
    }
  }

  /** Thông tin theo thời điểm t, dùng chung cho panel và lớp phủ. */
  function info(song, t) {
    const ki = T().keyIndexAt(song, t);
    const key = song.keys[ki] || null;
    const next = T().nextModulation(song, t);
    const warn = CT.settings.modulationWarnSeconds || 8;
    const soonIn = next && next.time - t <= warn ? next.time - t : null;
    const bi = T().lastLE(song.beats, t);
    return { ki, key, next, soonIn, beatPos: bi >= 0 ? song.beatPos[bi] : 0, beatIndex: bi };
  }
  function soonText(i, withTranspose) {
    if (i.soonIn == null) return "";
    const to = T().keyName(T().parseKeyText(i.next.to));
    let s = `Sắp đổi giọng sau ${Math.max(1, Math.ceil(i.soonIn))} s → ${to}`;
    if (withTranspose) s += ` · bấm ${T().transposeLabel(i.next.transpose_from)} → ${T().transposeLabel(i.next.transpose_to)}`;
    return s;
  }

  /** Chữ ký các cài đặt quyết định nhãn hợp âm và việc hiện bảng (dùng chung với sections.js). */
  const labelSig = (s) => [s.mode, s.transpose | 0, !!s.hideInversions, !!s.simplify, !!s.enabled, !!s.panelCollapsed].join("|");

  /* ---------------- Panel ---------------- */
  const panel = {
    el: null, state: "hidden", song: null, last: {},
    build() {
      const modeBtn = (m, text, tip, icon) => h("button", { "data-m": m, type: "button", title: tip, onclick: () => CT.setSettings({ mode: m }) }, icon, text);
      this.dots = new BeatDots();
      this.keyEl = h("b", { class: "ct-key", text: "–" });
      this.tempoEl = h("span", { class: "ct-chip", title: "Tốc độ (phách mỗi phút)" });
      this.meterEl = h("span", { class: "ct-chip", title: "Số chỉ nhịp: số phách trong một ô nhịp" });
      this.soonEl = h("span", { class: "ct-chip ct-warn", hidden: true });
      this.metroBtn = h("button", { type: "button", class: "ct-btn", title: `Bật/tắt tiếng phách theo nhịp bài (${sc("KeyM")})`,
        onclick: () => CT.setSettings({ metronome: !CT.settings.metronome }) },
        CT.icons.metronome(16), "Metronome");
      this.pendEl = this.metroBtn.querySelector(".ct-ic-pend");
      // Kéo thanh âm lượng: gộp các lần đổi (chrome.storage.sync giới hạn ~120 lần ghi mỗi phút).
      let volTimer = 0;
      this.volEl = h("input", { type: "range", min: 0, max: 100, class: "ct-vol", title: "Âm lượng tiếng phách",
        oninput: (e) => {
          clearTimeout(volTimer);
          const v = e.target.value / 100;
          volTimer = setTimeout(() => CT.setSettings({ metronomeVolume: v }), 150);
        } });
      this.seg = h("div", { class: "ct-seg", role: "group", "aria-label": "Cách hiện hợp âm" },
        modeBtn("orig", "Gốc", `Hợp âm đúng tông bài, có thể dịch tông (${sc("Digit1")})`),
        modeBtn("white", "Phím trắng", `Chơi trên phím trắng, đặt transpose của đàn theo ô bên dưới (${sc("Digit2")})`, CT.icons.keys(16)),
        modeBtn("roman", "Bậc La Mã", `Bậc hợp âm trong giọng (I, IV, V, vi…) (${sc("Digit3")})`));
      this.trEl = h("b", { class: "ct-tr" });
      this.stepper = h("span", { class: "ct-stepper", title: "Dịch tông hợp âm hiển thị" }, "Tông",
        h("button", { type: "button", class: "ct-btn ct-sq", title: `Hạ nửa cung (${sc("ArrowDown")})`, onclick: () => stepTranspose(-1) }, "−"),
        this.trEl,
        h("button", { type: "button", class: "ct-btn ct-sq", title: `Nâng nửa cung (${sc("ArrowUp")})`, onclick: () => stepTranspose(1) }, "+"));
      this.collapseBtn = h("button", { type: "button", class: "ct-btn ct-icon", onclick: () => CT.setSettings({ panelCollapsed: !CT.settings.panelCollapsed }) });
      this.miniEl = h("span", { class: "ct-mini" });
      this.redoBtn = h("button", { type: "button", class: "ct-btn ct-icon ct-redo-btn",
        title: "Phân tích lại bài này (bỏ kết quả đã lưu, mất khoảng 5–20 giây)",
        onclick: () => CT.app && CT.app.reload(true) }, "↻");
      this.helpBtn = h("button", { type: "button", class: "ct-btn ct-icon ct-help-btn", title: "Chú thích và phím tắt",
        "aria-expanded": "false", onclick: () => this.toggleHelp() }, "?");

      this.head = h("div", { class: "ct-head" },
        h("span", { class: "ct-brand", title: "ChordTrack: hợp âm chạy theo video" }, CT.icons.avatar(18), "ChordTrack"),
        this.miniEl,
        h("span", { class: "ct-keywrap" }, "Giọng ", this.keyEl),
        this.tempoEl, this.meterEl, this.dots.el, this.soonEl,
        h("span", { class: "ct-grow" }),
        h("span", { class: "ct-metro" }, this.metroBtn, this.volEl),
        this.redoBtn, this.helpBtn, this.collapseBtn);

      this.tval = h("span", { class: "ct-tval" });
      this.tkey = h("span", { class: "ct-tkey" });
      this.tnext = h("span", { class: "ct-tnext" });
      this.tbox = h("div", { class: "ct-tbox", title: "Giữ nút Transpose của đàn và bấm phím này, rồi chơi hợp âm trên phím trắng. Số trong ngoặc là mức dịch thật của đàn" },
        h("span", {}, "Giữ Transpose, bấm"), this.tval, this.tkey, this.tnext);

      this.slot = h("div", { class: "ct-strip-slot" });
      this.fsNote = h("div", { class: "ct-fs-note", text: "Dải hợp âm đang hiện trên video (toàn màn hình)." });
      const check = (key, text, tip) => h("label", { title: tip },
        h("input", { type: "checkbox", role: "switch", "data-k": key, onchange: (e) => CT.setSettings({ [key]: e.target.checked }) }), text);
      this.melBox = h("span", { class: "ct-melbox" });
      // Chọn biến thể dò giai điệu (CT.settings.melodyModel): nút phân đoạn cùng kiểu Gốc/Phím trắng/Bậc La Mã.
      // Dựng một lần, không nằm trong melBox (melBox dựng lại mỗi lần job báo tiến độ).
      const MM = T().MELODY_MODELS;
      this.melSel = h("div", { class: "ct-seg ct-seg-sm", role: "group", "aria-label": "Model dò giai điệu" },
        h("span", { class: "ct-seg-lbl", "aria-hidden": "true" }, "♪"),
        ...Object.entries(MM).map(([k, v]) => h("button", { type: "button", "data-v": k, title: `${v.label}. ${v.tip}`,
          onclick: () => CT.setSettings({ melodyModel: k }) }, v.short)));
      this.opts = h("div", { class: "ct-controls" },
        this.seg, this.stepper, h("span", { class: "ct-melgrp" }, this.melSel, this.melBox), h("span", { class: "ct-grow" }),
        h("span", { class: "ct-opts" },
          check("hideInversions", "Ẩn hợp âm đảo", "Bỏ nốt bass: C/G hiện là C"),
          check("simplify", "Đơn giản hóa", "Chỉ hiện hợp âm ba: Am7 → Am, Csus4 → C"),
          check("snapToBeats", "Khớp phách", "Kéo đầu/cuối hợp âm về phách gần nhất (mặc định bật). Model nhận hợp âm theo khung ~0,09 s nên mép ô hay lệch vạch phách")));
      // Chú thích và phím tắt: gom vào nút "?" (trước đây là một dòng chữ dài dưới dải).
      const item = (sw, text) => h("li", {}, h("span", { class: "ct-sw " + sw }), text);
      this.help = h("div", { class: "ct-help", hidden: true },
        h("ul", { class: "ct-help-list" },
          item("ct-sw-cur", "Ô có viền sáng: hợp âm đang chơi; thanh đỏ ở đáy ô là tiến độ"),
          item("ct-sw-bar", "Vạch cao kèm số: phách 1 của ô nhịp"),
          item("ct-sw-mod", "Vạch vàng: đổi giọng"),
          item("ct-sw-unc", "⚠ Model không chắc, nên nghe lại"),
          item("ct-sw-mel", "Dòng nốt dưới hợp âm: giai điệu (bấm “Dò giai điệu”). ↑/↓ cao/thấp hơn nốt trước; nốt mờ là model không chắc. Chế độ Bậc La Mã hiện số 1–7 (1 = phím C khi chơi phím trắng)"),
          h("li", {}, h("span", { class: "ct-sw" }), "Bấm vào hợp âm hoặc đoạn để tua tới đó")),
        h("ul", { class: "ct-help-keys" },
          ...SHORTCUTS.map((x) => h("li", {}, h("kbd", {}, "Alt+Shift+" + x.key), x.label))));
      this.body = h("div", { class: "ct-body" }, this.opts, this.tbox, this.help, this.slot, this.fsNote);
      this.status = h("div", { class: "ct-status" });
      this.el = h("div", { id: "ct-panel", class: "ct-root ct-panel", hidden: true }, this.head, this.status, this.body);
      this.slot.append(CT.ui.strip.el);
      this.applySettings(CT.settings);
    },

    /** Nút "Dò giai điệu" / tiến độ job / lỗi / công tắc "Giai điệu" theo sự kiện "melody" mới nhất (this.mel). */
    renderMelody() {
      const box = this.melBox;
      if (!box) return;
      const m = this.mel || { status: "unknown" };
      const variant = T().melodyVariant(m.model || CT.settings.melodyModel);
      const fast = variant === "yourmt3";
      const run = (force) => () => CT.app && CT.app.detectMelody(force);
      const btn = (kids, tip, onclick, cls = "ct-btn") => h("button", { type: "button", class: cls, title: tip, onclick }, ...kids);
      let kids;
      if (m.status === "loading") {
        const j = m.job || {};
        const pct = Math.round((j.progress || 0) * 100);
        const queued = j.status === "queued";
        const what = j.message || MELODY_STEPS[j.step] || (queued ? "Đang chờ tới lượt" : "Đang dò");
        const text = queued ? `Giai điệu: ${what}` : `Giai điệu ${pct}% · ${what}`;
        const eta = etaText(j.eta_seconds);
        kids = [h("span", { class: "ct-chip ct-mel-prog", title: [MELODY_STEPS[j.step], j.message, eta].filter(Boolean).join(" · "),
          role: "status" },
          h("span", { class: "ct-spinner" }), h("span", { class: "ct-mel-txt", text: eta ? `${text} · ${eta}` : text }),
          h("span", { class: "ct-mel-bar" }, h("i", { style: `width:${queued ? 0 : pct}%` })))];
      } else if (m.status === "error") {
        const msg = (m.error && m.error.message) || "Không dò được giai điệu";
        kids = [h("span", { class: "ct-chip ct-warn ct-mel-err", title: msg, text: "⚠ " + msg }),
          btn(["Thử lại"], "Gửi lại yêu cầu dò giai điệu", run(false))];
      } else if (m.status === "ready" && m.melody) {
        const on = CT.settings.showMelody !== false;
        kids = [h("label", { class: "ct-mel-sw", title: `Hiện/ẩn dòng nốt giai điệu dưới dải hợp âm (${sc("KeyG")})` },
          h("input", { type: "checkbox", role: "switch", checked: on, onchange: (e) => CT.setSettings({ showMelody: e.target.checked }) }),
          "Hiện nốt"),
        // Bản cũ: gửi không `force`, server dựng lại từ đầu ra thô của model đã lưu (thường khoảng 1 giây), chỉ
        // chạy lại model nếu đầu ra thô không còn dùng được. Bản mới: ↻ là dò lại thật (`force`, chạy lại model).
        btn(["↻"], m.stale ? "Giai điệu dò bằng mã cũ: cập nhật (dùng lại kết quả model đã lưu nếu được, thường vài giây)"
          : `Dò lại giai điệu (bỏ kết quả đã lưu, chạy lại model, mất ${fast ? "khoảng 1 phút" : "vài phút"})`,
          run(!m.stale), "ct-btn ct-icon ct-mel-redo")];
        if (m.stale) kids.push(h("span", { class: "ct-chip", title: "Kết quả dò bằng phiên bản mã cũ, bấm ↻ để cập nhật", text: "bản cũ" }));
      } else {
        kids = [btn(["♪ Dò giai điệu"],
          `Dò dòng nốt giai điệu bằng AI trên máy (${T().MELODY_MODELS[variant].label}; chạy riêng, mất ${fast ? "khoảng 1 phút" : "vài phút"}). Kết quả hiện dưới dải hợp âm`, run(false))];
      }
      box.replaceChildren(...kids);
    },

    toggleHelp(force) {
      const open = force ?? this.help.hidden;
      this.help.hidden = !open;
      this.helpBtn.classList.toggle("ct-on", open);
      this.helpBtn.setAttribute("aria-expanded", String(open));
    },

    show(state) {
      this.state = state;
      this.el.hidden = state === "hidden" || !CT.settings.enabled;
      this.el.dataset.state = state;
    },

    setSong(song) {
      this.sawSeparation = false;
      this.labelSig = labelSig(CT.settings);
      this.song = song;
      this.last = {};
      const r = song.result;
      this.tempoEl.textContent = r.tempo ? `${Math.round(r.tempo)} BPM` : "";
      this.tempoEl.hidden = !r.tempo;
      this.meterEl.textContent = `Nhịp ${song.meter}/4`;
      this.meterEl.hidden = !song.beats.length;
      this.dots.setMeter(song.beats.length ? song.meter : 0);
      CT.ui.strip.setSong(song);
      this.show("ready");
      this.onTime(CT.ui.lastTime || { t: 0 });
    },

    applySettings(s) {
      if (!this.el) return;
      for (const b of this.seg.children) b.classList.toggle("ct-on", b.dataset.m === s.mode);
      this.el.dataset.mode = s.mode;
      this.trEl.textContent = T().signed(s.transpose | 0);
      this.stepper.hidden = s.mode === "white";
      this.metroBtn.classList.toggle("ct-on", !!s.metronome);
      this.volEl.value = Math.round((s.metronomeVolume ?? 0.6) * 100);
      for (const c of this.opts.querySelectorAll("input[data-k]")) c.checked = !!s[c.dataset.k];
      const col = !!s.panelCollapsed;
      this.el.classList.toggle("ct-collapsed", col);
      this.collapseBtn.textContent = col ? "▾" : "▴";
      this.collapseBtn.title = (col ? "Mở khung hợp âm" : "Thu gọn khung hợp âm") + ` (${sc("KeyH")})`;
      if (this.state !== "hidden") this.show(this.state);
      this.last = {};
      if (this.melOn !== (s.showMelody !== false)) { this.melOn = s.showMelody !== false; this.renderMelody(); }
      const mv = T().melodyVariant(s.melodyModel);
      for (const b of this.melSel.querySelectorAll("button")) {
        const on = b.dataset.v === mv;
        b.classList.toggle("ct-on", on);
        b.setAttribute("aria-pressed", String(on));
      }
      // Chỉ đổi chữ trên dải khi cài đặt ảnh hưởng tới nhãn (không phải mỗi lần đổi âm lượng).
      const sig = labelSig(s);
      if (this.song && sig !== this.labelSig) {
        this.labelSig = sig;
        CT.ui.strip.relabel();
      }
      if (this.song) {
        this.onTime(CT.ui.lastTime || { t: 0 });
      }
    },

    /** Quả lắc trên nút Metronome: chạm biên đúng lúc có phách, đổi bên mỗi phách (khớp tiếng tích).
     * Metronome tắt thì đứng giữa. Chỉ đặt transform khi góc đổi. */
    swing(s, t, k) {
      let a = 0;
      if (CT.settings.metronome && k >= 0 && k + 1 < s.beats.length) {
        const f = (t - s.beats[k]) / Math.max(1e-3, s.beats[k + 1] - s.beats[k]);
        a = Math.round(24 * Math.cos(Math.PI * (k + f)) * 10) / 10;
      }
      if (a !== this.pendA) { this.pendA = a; this.pendEl.style.transform = a ? `rotate(${a}deg)` : ""; }
    },

    onTime({ t, ad }) {
      const s = this.song;
      if (!s || this.state !== "ready") return;
      if (!CT.ui.fullscreenOn) CT.ui.strip.update(t);
      this.el.classList.toggle("ct-ad", !!ad);
      const i = info(s, t);
      this.dots.set(i.beatPos);
      this.swing(s, t, i.beatIndex);
      const L = this.last;
      if (L.ki !== i.ki || L.mode !== CT.settings.mode) {
        L.ki = i.ki; L.mode = CT.settings.mode;
        const k = i.key;
        this.keyEl.textContent = T().keyName(k);
        const w = k ? k.white_key_transpose | 0 : 0;
        this.tval.textContent = T().transposeKey(w);      // phím cần bấm khi giữ Transpose
        this.tkey.textContent = `(${T().signed(w)})`;     // mức dịch thật của đàn (giọng đã ghi ở hàng trên)
        this.tkey.title = k ? `Chơi như ${k.mode === "minor" ? "A thứ" : "C trưởng"} trên phím trắng` : "";
        this.tbox.hidden = !k;
      }
      // Chế độ phím trắng: cảnh báo nằm trong ô transpose (đếm ngược), không lặp lại ở hàng đầu.
      const white = CT.settings.mode === "white";
      const nextText = !i.next ? (s.modulations.length ? "Không còn chỗ đổi giọng" : "")
        : i.soonIn != null
          ? `Đổi sau ${Math.max(1, Math.ceil(i.soonIn))} s → ${T().keyName(T().parseKeyText(i.next.to))}: bấm ${T().transposeLabel(i.next.transpose_to)}`
          : `Kế tiếp ${fmt(i.next.time)}: bấm ${T().transposeLabel(i.next.transpose_to)}`;
      if (L.next !== nextText) { L.next = nextText; this.tnext.textContent = nextText; }
      const soon = white ? "" : soonText(i, false);
      if (L.soon !== soon) {
        L.soon = soon;
        this.soonEl.textContent = soon;
        this.soonEl.hidden = !soon;
      }
      const isSoon = i.soonIn != null;
      if (L.isSoon !== isSoon) { L.isSoon = isSoon; this.tbox.classList.toggle("ct-soon", isSoon); }
      // Hợp âm hiện tại khi thu gọn.
      const ci = CT.ui.strip.idx;
      if (L.ci !== ci) {
        L.ci = ci;
        const c = s.chords[ci];
        this.miniEl.textContent = c ? T().chordLabels(s, c, CT.settings).main : "";
      }
    },

    /* ----- trạng thái đang phân tích / lỗi ----- */
    showLoading(job) {
      this.song = null;
      CT.ui.strip.setSong(null);
      // Tách nguồn mặc định tắt: chỉ hiện bước này khi server thật sự chạy nó.
      if (job.step === "separation") this.sawSeparation = true;
      const STEPS = [["fetch", "Tải audio"], ...(this.sawSeparation ? [["separation", "Tách nguồn"]] : []),
        ["beats", "Tìm phách"], ["chords", "Nhận hợp âm"], ["fusion", "Ghép kết quả"], ["analysis", "Giọng và đoạn"]];
      const pct = Math.round((job.progress || 0) * 100);
      const cur = STEPS.findIndex(([k]) => k === job.step);
      const eta = job.eta_seconds;
      const etaText = eta == null ? "" : eta < 60 ? `còn khoảng ${Math.max(5, Math.round(eta / 5) * 5)} giây`
        : `còn khoảng ${Math.round(eta / 60)} phút`;
      const queued = job.status === "queued";
      this.status.replaceChildren(
        h("div", { class: "ct-st-title" },
          h("span", { class: "ct-spinner" }), queued ? (job.message || "Đang chờ tới lượt") : "Đang phân tích hợp âm bài này",
          h("span", { class: "ct-st-pct", text: queued ? "" : `${pct}%` })),
        h("div", { class: "ct-progress", role: "progressbar", "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 },
          h("div", { style: `width:${queued ? 2 : Math.max(2, pct)}%` })),
        h("ol", { class: "ct-steps" }, ...STEPS.map(([k, label], i) =>
          h("li", { class: cur < 0 ? (queued ? "" : "") : i < cur ? "ct-done" : i === cur ? "ct-now" : "" },
            (cur >= 0 && i < cur ? "✓ " : "") + label))),
        h("div", { class: "ct-st-note" },
          [etaText && etaText[0].toUpperCase() + etaText.slice(1), "Video vẫn xem bình thường, hợp âm sẽ hiện khi xong. Lần sau mở lại bài này là có ngay."]
            .filter(Boolean).join(" · ")));
      this.show("loading");
    },

    showError(err) {
      const key = `${err.code}|${err.message}`;
      if (this.state === "error" && this.errKey === key) return; // lần tự thử lại ra cùng lỗi: không vẽ lại
      this.errKey = key;
      this.song = null;
      CT.ui.strip.setSong(null);
      const retry = (force, text = "Thử lại") => h("button", { type: "button", class: "ct-btn ct-primary",
        title: force ? "Bỏ kết quả cũ, phân tích lại từ đầu" : "Gọi lại server", onclick: () => CT.app && CT.app.reload(force) }, text);
      const code = (cmd) => {
        const btn = h("button", { type: "button", class: "ct-btn", title: "Chép lệnh vào clipboard" }, "Chép lệnh");
        btn.addEventListener("click", async () => {
          try { await navigator.clipboard.writeText(cmd); btn.textContent = "Đã chép ✓"; } catch { btn.textContent = "Không chép được"; }
          setTimeout(() => { btn.textContent = "Chép lệnh"; }, 1500);
        });
        return h("div", { class: "ct-code" }, h("code", { text: cmd }), btn);
      };
      let kids;
      if (err.code === "server_down") {
        kids = [
          h("div", { class: "ct-st-title" }, "⚠ Chưa kết nối được server ChordTrack"),
          h("p", {}, `Extension cần backend chạy trên máy (${CT.settings.serverUrl}). Mở PowerShell ở thư mục repo và chạy:`),
          code(SERVER_CMD),
          h("div", { class: "ct-st-actions" }, retry(false), h("span", { class: "ct-st-note", text: "Tự thử lại mỗi 5 giây." })),
        ];
      } else if (err.code === "youtube_blocked") {
        kids = [
          h("div", { class: "ct-st-title" }, "⚠ YouTube chặn tải audio"),
          h("p", { text: err.message }),
          h("ol", { class: "ct-howto" },
            h("li", {}, "Cài extension “Get cookies.txt LOCALLY”, mở youtube.com (đã đăng nhập), xuất cookie ra file cookies.txt."),
            h("li", {}, "Tắt server rồi chạy lại kèm file cookie:")),
          code(SERVER_CMD + " -Cookies C:\\duong\\dan\\cookies.txt"),
          h("div", { class: "ct-st-actions" }, retry(false)),
        ];
      } else if (err.code === "extension_reloaded") {
        kids = [h("div", { class: "ct-st-title" }, "Extension vừa được cập nhật"),
          h("div", { class: "ct-st-actions" }, h("button", { type: "button", class: "ct-btn ct-primary", onclick: () => location.reload() }, "Tải lại trang"))];
      } else {
        kids = [
          h("div", { class: "ct-st-title" }, "⚠ Không phân tích được bài này"),
          h("p", { text: err.message || "Lỗi không rõ" }),
          h("div", { class: "ct-st-actions" }, retry(false), retry(true, "Phân tích lại")),
        ];
      }
      this.status.replaceChildren(...kids);
      this.show("error");
    },
  };

  /* ---------------- Gắn vào DOM YouTube ---------------- */
  const layout = {
    mo: null, timer: 0, mounted: false,
    flexy: () => document.querySelector("ytd-watch-flexy"),
    /** Bảng cấu trúc ở cột phải khi trang hai cột và không ở chế độ rạp hát. */
    sideBySide() {
      const f = this.flexy();
      if (!f) return true;
      const theater = f.hasAttribute("theater") || f.hasAttribute("theater-requested_");
      const twoCols = f.hasAttribute("is-two-columns_");
      const secondary = document.querySelector("#secondary");
      return twoCols && !theater && !!secondary && secondary.offsetWidth > 0;
    },
    place() {
      this.timer = 0;
      if (!this.mounted) return;
      this.watchFlexy();
      const p = panel.el, secs = CT.ui.sections && CT.ui.sections.el;
      const below = document.querySelector("ytd-watch-flexy #below") || document.querySelector("#below");
      if (below) {
        const meta = [...below.children].find((c) => c.tagName === "YTD-WATCH-METADATA");
        if (p.parentNode !== below || (meta ? p.nextSibling !== meta && p.nextSibling !== secs : below.firstChild !== p)) {
          below.insertBefore(p, meta || below.firstChild);
        }
      }
      if (!secs) return;
      const side = this.sideBySide();
      secs.classList.toggle("ct-side", side);
      if (side) {
        const host = document.querySelector("#secondary #secondary-inner") || document.querySelector("#secondary");
        if (host && host.firstChild !== secs) host.insertBefore(secs, host.firstChild);
      } else if (p.parentNode && p.nextSibling !== secs) {
        p.parentNode.insertBefore(secs, p.nextSibling);
      }
    },
    schedule() { if (!this.timer) this.timer = setTimeout(() => this.place(), 300); },
    /** Theo dõi thuộc tính rạp hát/hai cột. ytd-watch-flexy có thể chưa có lúc mount nên thử lại mỗi lần place(). */
    watchFlexy() {
      const f = this.flexy();
      if (!f || f === this.flexyEl) return;
      this.mo2?.disconnect();
      this.flexyEl = f;
      this.mo2 = new MutationObserver(() => this.schedule());
      this.mo2.observe(f, { attributes: true, attributeFilter: ["theater", "is-two-columns_", "fullscreen"] });
    },
    mount() {
      if (!panel.el) panel.build();
      this.mounted = true;
      this.place();
      if (!this.mo) {
        this.mo = new MutationObserver((muts) => {
          for (const m of muts) {
            const n = m.target;
            if (n.closest && n.closest(".ct-root")) continue; // thay đổi bên trong khung của ta
            this.schedule();
            return;
          }
        });
        this.mo.observe(document.body, { childList: true, subtree: true });
        this.watchFlexy();
        window.addEventListener("resize", this.onResize = () => this.schedule());
      }
    },
    unmount() {
      this.mounted = false;
      this.mo?.disconnect(); this.mo = null;
      this.mo2?.disconnect(); this.mo2 = null; this.flexyEl = null;
      if (this.onResize) window.removeEventListener("resize", this.onResize);
      panel.el?.remove();
      CT.ui.sections?.el?.remove();
    },
  };

  CT.ui = CT.ui || {};
  Object.assign(CT.ui, { panel, layout, BeatDots, info, soonText, labelSig, shortcuts: SHORTCUTS, lastTime: null, fullscreenOn: false });

  CT.bus.on("settings", (s) => panel.applySettings(s));
  // Bật/tắt "Khớp phách": dựng lại bài từ kết quả đang có (dải, panel, toàn màn hình).
  let lastSnap = CT.settings.snapToBeats !== false;
  CT.bus.on("settings", (st) => {
    const snap = st.snapToBeats !== false;
    if (snap === lastSnap) return;
    lastSnap = snap;
    if (CT.ui.song && CT.state.result) CT.bus.emit("song:ready", { videoId: CT.state.videoId, result: CT.state.result });
  });
  CT.bus.on("time", (p) => { CT.ui.lastTime = p; panel.onTime(p); });
  // CT.ui.song chỉ là bài của video hiện tại: xóa ngay khi bắt đầu tải hoặc lỗi (fullscreen.js, sections.js đọc nó).
  CT.bus.on("song:loading", ({ job }) => { CT.ui.song = null; if (panel.el) panel.showLoading(job || {}); });
  CT.bus.on("song:error", (e) => { CT.ui.song = null; if (panel.el) panel.showError(e); });
  CT.bus.on("song:ready", ({ result }) => {
    if (!panel.el) return;
    try {
      CT.ui.song = T().buildSong(result, { snapToBeats: CT.settings.snapToBeats !== false });
      panel.setSong(CT.ui.song);
    } catch (e) {
      CT.ui.song = null;
      console.error("[ChordTrack] result lỗi", e);
      panel.showError({ code: "bad_result", message: "Kết quả từ server không đọc được: " + e.message });
    }
  });
  // Giai điệu: dải giữ <Melody> (vẽ khi có bài), panel hiện nút/tiến độ/công tắc.
  CT.bus.on("melody", (m) => {
    panel.mel = m;
    CT.ui.strip.setMelody(m.melody || null);
    panel.renderMelody();
    refitOverlay();
  });
  /** Dải cao/thấp đi khi làn nốt hiện/ẩn: lớp phủ toàn màn hình đặt lại vị trí phụ đề theo chiều cao mới. */
  function refitOverlay() { if (CT.ui.fullscreen && CT.ui.fullscreen.on) CT.ui.fullscreen.syncControls(); }
  let lastShowMel = CT.settings.showMelody !== false;
  CT.bus.on("settings", (st) => {
    const on = st.showMelody !== false;
    if (on === lastShowMel) return;
    lastShowMel = on;
    CT.ui.strip.buildMelody(true);
    refitOverlay();
  });
  CT.bus.on("song:none", () => { CT.ui.song = null; if (panel.el) { panel.song = null; CT.ui.strip.setSong(null); panel.show("hidden"); } });
})();
