/* ChordTrack sections: bảng "Cấu trúc bài" (vòng hợp âm từng đoạn, giọng, transpose phím trắng).
 * Đặt đầu cột phải (#secondary); chế độ rạp hát hoặc cột hẹp thì xếp dưới panel (CT.ui.layout lo việc này).
 * Bấm vào đoạn để tua tới đó. Chỉ đổi class đoạn hiện tại khi chỉ số đổi.
 */
(() => {
  const T = () => CT.theory;
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const COLORS = { intro: "#7e57c2", verse: "#26a69a", chorus: "#ef5350", bridge: "#ffa726", outro: "#78909c",
    prechorus: "#5c6bc0", "pre-chorus": "#5c6bc0", instrumental: "#8d6e63", solo: "#8d6e63" };
  const PALETTE = ["#42a5f5", "#ab47bc", "#8d6e63", "#9ccc65", "#ec407a", "#26c6da"];
  const LABEL_VI = { intro: "Intro", verse: "Verse", chorus: "Điệp khúc", bridge: "Bridge", outro: "Outro",
    instrumental: "Nhạc dạo", solo: "Solo" };
  const isNamed = (l) => !!(l && (LABEL_VI[l] || COLORS[l] || l.length > 2));

  /** Tên và màu dễ hiểu cho từng đoạn.
   * - Đoạn có tên (verse, chorus…): tên tiếng Việt, thêm số thứ tự nếu lặp: "Verse 1", "Verse 2".
   * - Nhãn chữ cái của backend (A, C, D… là cụm đoạn nhạc giống nhau, chưa đặt tên): đánh số theo thứ
   *   tự xuất hiện "Đoạn 1", "Đoạn 2"; lặp lại thì thêm "· lần 2". Mỗi cụm một màu riêng trong PALETTE,
   *   nên cùng màu = cùng đoạn nhạc. */
  function describe(secs) {
    const total = {}, seen = {}, order = {};
    let nOther = 0;
    for (const x of secs) { const l = x.label || "?"; total[l] = (total[l] || 0) + 1; }
    return secs.map((x) => {
      const l = x.label || "?";
      const occ = (seen[l] = (seen[l] || 0) + 1);
      if (isNamed(l)) {
        const base = LABEL_VI[l] || l.charAt(0).toUpperCase() + l.slice(1);
        return { name: total[l] > 1 ? `${base} ${occ}` : base, color: COLORS[l] || PALETTE[0], tip: base };
      }
      if (!(l in order)) order[l] = ++nOther;
      const k = order[l];
      return { name: `Đoạn ${k}` + (total[l] > 1 ? ` · lần ${occ}` : ""),
        color: PALETTE[(k - 1) % PALETTE.length], tip: `Đoạn chưa đặt tên (nhãn tự động ${l})` };
    });
  }

  const sections = {
    el: null, list: null, idx: -1, song: null,
    build() {
      this.list = document.createElement("div");
      this.list.className = "ct-secs";
      const title = document.createElement("div");
      title.className = "ct-sec-title";
      title.textContent = "Cấu trúc bài";
      const legend = document.createElement("div");
      legend.className = "ct-sec-legend";
      legend.textContent = "Cùng màu là cùng một đoạn nhạc lặp lại. Bấm vào đoạn để tua tới đó.";
      this.el = document.createElement("div");
      this.el.id = "ct-sections";
      this.el.className = "ct-root ct-card";
      this.el.hidden = true;
      this.el.append(title, this.list, legend);
      this.list.addEventListener("click", (e) => {
        const s = e.target.closest(".ct-sec");
        if (s && this.song) CT.seekTo(this.song.sections[+s.dataset.i].start + 0.01);
      });
    },
    setSong(song) {
      this.song = song;
      this.names = song ? describe(song.sections) : [];
      this.sig = CT.ui.labelSig(CT.settings);
      this.idx = -1;
      this.el.hidden = !this.visible();
      this.render();
    },
    /** Ẩn khi không có bài, tắt extension, hoặc người dùng thu gọn khung hợp âm. */
    visible() {
      const s = this.song;
      return !!(s && s.sections.length && CT.settings.enabled && !CT.settings.panelCollapsed);
    },
    render() {
      const s = this.song;
      this.list.textContent = "";
      if (!s) return;
      const st = CT.settings;
      const multiKey = s.keys.length > 1;
      s.sections.forEach((sec, i) => {
        const L = T().sectionLabels(s, sec, st);
        const row = document.createElement("div");
        row.className = "ct-sec";
        row.dataset.i = i;
        const nm = this.names[i];
        row.style.setProperty("--ct-sec-color", nm.color);
        row.title = `${nm.tip} · ${fmt(sec.start)}–${fmt(sec.end)}. Bấm để tua tới đây`;
        const top = document.createElement("div");
        top.className = "ct-sec-top";
        const lab = document.createElement("span");
        lab.className = "ct-sec-lab";
        lab.textContent = nm.name + (sec.semitones ? ` (${T().signed(sec.semitones)})` : "");
        const time = document.createElement("span");
        time.className = "ct-sec-t";
        time.textContent = fmt(sec.start);
        top.append(lab, time);
        const prog = document.createElement("div");
        prog.className = "ct-sec-prog";
        prog.textContent = L.main.join(" – ");
        const sub = document.createElement("div");
        sub.className = "ct-sec-sub";
        sub.textContent = L.sub.join(" – ");
        row.append(top, prog, sub);
        if (L.key && (multiKey || st.mode === "white")) {
          const k = document.createElement("div");
          k.className = "ct-sec-key";
          k.textContent = T().keyName(L.key) + (st.mode === "white" ? ` · transpose ${T().signed(L.white)}` : "");
          row.append(k);
        }
        this.list.append(row);
      });
      this.idx = -1;
      if (CT.ui.lastTime) this.onTime(CT.ui.lastTime);
    },
    onTime({ t }) {
      const s = this.song;
      if (!s) return;
      const j = T().lastLE(s.sections, t, (x) => x.start);
      const cur = j >= 0 && t < s.sections[j].end ? j : -1;
      if (cur === this.idx) return;
      const kids = this.list.children;
      if (this.idx >= 0 && kids[this.idx]) kids[this.idx].classList.remove("ct-cur");
      if (cur >= 0 && kids[cur]) {
        kids[cur].classList.add("ct-cur");
        // Cột phải có thanh cuộn riêng: giữ đoạn hiện tại trong tầm nhìn (không cuộn cả trang).
        const L = this.list, k = kids[cur];
        if (L.scrollHeight > L.clientHeight) {
          const top = k.offsetTop - L.offsetTop;
          if (top < L.scrollTop || top + k.offsetHeight > L.scrollTop + L.clientHeight) {
            L.scrollTo({ top: Math.max(0, top - 8), behavior: "smooth" });
          }
        }
      }
      this.idx = cur;
    },
  };

  sections.build();
  CT.ui.sections = sections;
  CT.bus.on("song:ready", () => sections.setSong(CT.ui.song || null));
  for (const evt of ["song:loading", "song:error", "song:none"]) CT.bus.on(evt, () => sections.setSong(null));
  CT.bus.on("settings", (s) => {
    if (!sections.song) return;
    const sig = CT.ui.labelSig(s);
    if (sig === sections.sig) return; // ví dụ chỉ đổi âm lượng metronome
    sections.sig = sig;
    sections.el.hidden = !sections.visible();
    sections.render();
  });
  CT.bus.on("time", (p) => sections.onTime(p));
})();
