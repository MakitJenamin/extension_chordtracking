/* ChordTrack strip: dải hợp âm trượt ngang kiểu karaoke, vạch đỏ cố định ở 28% chiều rộng.
 *
 * Một phần tử duy nhất (CT.ui.strip.el) dùng cho cả panel dưới video và lớp phủ toàn màn hình
 * (fullscreen.js chuyển chính phần tử này qua lại).
 * Hiệu năng: DOM chỉ dựng lại khi có bài mới; đổi chế độ thì chỉ đổi chữ; mỗi khung hình chỉ đặt
 * transform của dải và đổi class khi chỉ số hợp âm hiện tại đổi.
 */
(() => {
  const PLAYHEAD = 0.28;
  const T = () => CT.theory;
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  class Strip {
    constructor() {
      this.el = el("div", "ct-strip-wrap");
      this.track = el("div", "ct-strip");
      this.head = el("div", "ct-playhead");
      this.empty = el("div", "ct-strip-empty", "Đoạn này không có hợp âm");
      this.el.append(this.track, this.head);
      this.song = null;
      this.scale = Strip.BASE_SCALE;
      this.width = 0;
      this.idx = -1;
      this.t = 0;
      this.chordEls = [];
      this.modEls = [];
      this.beatEls = [];
      this.pastCount = 0;
      this.lastX = null;
      this.track.addEventListener("click", (e) => {
        const c = e.target.closest(".ct-chord");
        if (c && this.song) CT.seekTo(this.song.chords[+c.dataset.i].start + 0.01);
      });
      this.ro = new ResizeObserver((entries) => {
        this.width = entries[0].contentRect.width;
        this.lastX = null;
        this.update(this.t, true);
      });
      this.ro.observe(this.el);
    }

    get px() { return (CT.settings.pxPerSecond || 46) * this.scale; }

    setScale(f) {
      if (f === this.scale) return;
      this.scale = f;
      this.layout();
    }

    setSong(song) {
      this.song = song;
      this.track.textContent = "";
      this.chordEls = [];
      this.modEls = [];
      this.beatEls = [];
      this.idx = -1;
      this.pastCount = 0;
      this.lastX = null;
      if (!song) return;
      const frag = document.createDocumentFragment();
      song.beats.forEach((b, i) => {
        const strong = song.beatPos[i] === 1;
        const e = el("div", strong ? "ct-beat ct-strong" : "ct-beat ct-weak");
        if (strong) e.append(el("span", null, String(song.bar[i])));
        frag.append(e);
        this.beatEls.push(e);
      });
      song.chords.forEach((c, i) => {
        const nc = c.label === "N" || c.label === "X" || c.symbol === "N.C." || c.symbol === "?";
        const e = el("div", "ct-chord" + (nc ? " ct-nc" : ""));
        e.dataset.i = i;
        const lbl = el("div", "ct-lbl");
        lbl.append(el("div", "ct-sym"), el("div", "ct-rn"));
        e.append(lbl);
        if (c.uncertain) {
          const u = el("span", "ct-unc", "⚠");
          u.title = `Model không chắc (độ tin cậy ${Math.round((c.confidence || 0) * 100)}%), nên nghe lại`;
          e.append(u);
        }
        frag.append(e);
        this.chordEls.push(e);
      });
      song.modulations.forEach((m) => {
        const e = el("div", "ct-modmark");
        e.append(el("span"));
        frag.append(e);
        this.modEls.push(e);
      });
      this.track.append(frag);
      this.layout();
      this.relabel();
    }

    layout() {
      const s = this.song;
      if (!s) return;
      const px = this.px;
      this.track.style.width = `${s.duration * px}px`;
      s.chords.forEach((c, i) => {
        const e = this.chordEls[i];
        e.style.left = `${c.start * px}px`;
        e.style.width = `${Math.max(4, (c.end - c.start) * px)}px`;
        // Hợp âm quá ngắn: chữ nhỏ lại thay vì bị cắt mất.
        e.classList.toggle("ct-narrow", (c.end - c.start) * px < 34);
      });
      s.beats.forEach((b, i) => { this.beatEls[i].style.left = `${b * px}px`; });
      s.modulations.forEach((m, i) => { this.modEls[i].style.left = `${m.time * px}px`; });
      this.lastX = null;
      this.update(this.t, true);
    }

    relabel() {
      const s = this.song;
      if (!s) return;
      const st = CT.settings;
      s.chords.forEach((c, i) => {
        const e = this.chordEls[i];
        const { main, sub } = T().chordLabels(s, c, st);
        const lbl = e.firstChild;
        lbl.firstChild.textContent = main;
        lbl.lastChild.textContent = sub && sub !== main ? sub : "";
        e.title = `${main}${sub && sub !== main ? " (" + sub + ")" : ""} · ${fmt(c.start)}. Bấm để tua tới đây`;
      });
      s.modulations.forEach((m, i) => {
        const to = T().parseKeyText(m.to);
        const arrow = m.semitones > 0 ? "⬆" : m.semitones < 0 ? "⬇" : "⇄";
        const label = st.mode === "white"
          ? `${arrow} ${T().keyName(to)} · transpose ${T().signed(m.transpose_from)} → ${T().signed(m.transpose_to)}`
          : `${arrow} ${T().keyName(to)}${m.semitones ? ` (${T().signed(m.semitones)})` : ""}`;
        this.modEls[i].firstChild.textContent = label;
        this.modEls[i].title = `Đổi giọng lúc ${fmt(m.time)}: ${T().keyName(T().parseKeyText(m.from))} → ${T().keyName(to)}`;
      });
      this.update(this.t, true); // bề rộng chữ đổi: đo lại khoảng trượt của hợp âm hiện tại
    }

    /** Chỉ số hợp âm chứa t (tìm gần chỉ số cũ trước, rồi mới tìm nhị phân). */
    indexAt(t) {
      const ch = this.song.chords;
      const i = this.idx;
      if (i >= 0 && i < ch.length && t >= ch[i].start && t < ch[i].end) return i;
      if (i + 1 < ch.length && t >= ch[i + 1].start && t < ch[i + 1].end) return i + 1;
      const j = T().lastLE(ch, t, (c) => c.start);
      return j >= 0 && t < ch[j].end ? j : -1;
    }

    update(t, force = false) {
      this.t = t;
      if (!this.song || !this.width) return;
      const x = this.width * PLAYHEAD - t * this.px;
      if (force || x !== this.lastX) {
        this.track.style.transform = `translate3d(${x}px,0,0)`;
        this.lastX = x;
      }
      const j = this.indexAt(t);
      if (j !== this.idx || force) this.setCurrent(j);
      this.followLabel(x);
    }

    /** Chữ của hợp âm hiện tại trượt theo vạch đỏ để vạch không đè lên chữ (chỉ một phần tử mỗi khung hình). */
    followLabel(x) {
      const c = this.curLbl;
      if (!c) return;
      const ch = this.song.chords[this.idx];
      const left = ch.start * this.px + x;                    // mép trái hợp âm trên màn hình
      const off = Math.max(0, Math.min(this.width * PLAYHEAD + 6 - left - 8, this.curRoom));
      if (off !== this.curOff) { c.style.transform = off ? `translateX(${off}px)` : ""; this.curOff = off; }
    }

    setCurrent(j) {
      const els = this.chordEls;
      const old = this.idx;
      if (old >= 0 && els[old]) els[old].classList.remove("ct-cur");
      // Lớp "đã qua": chỉ sửa đoạn nằm giữa chỉ số cũ và mới.
      const ch = this.song.chords;
      const boundary = j >= 0 ? j : T().lastLE(ch, this.t, (c) => c.start) + 1; // số hợp âm đã qua
      const prev = this.pastCount ?? 0;
      if (boundary > prev) for (let k = prev; k < boundary; k++) els[k].classList.add("ct-past");
      else for (let k = boundary; k < prev; k++) els[k].classList.remove("ct-past");
      this.pastCount = boundary;
      if (this.curLbl) { this.curLbl.style.transform = ""; this.curLbl = null; }
      if (j >= 0) {
        els[j].classList.remove("ct-past");
        els[j].classList.add("ct-cur");
        // Khoảng trượt tối đa: bề rộng hợp âm trừ bề rộng chữ (đọc layout một lần khi đổi hợp âm).
        this.curLbl = els[j].firstChild;
        this.curRoom = Math.max(0, els[j].clientWidth - this.curLbl.offsetWidth - 16);
        this.curOff = -1;
      }
      this.idx = j;
      // Cho harness đo độ lệch đồng bộ (t hiển thị và currentTime thật lúc đổi hợp âm).
      const v = CT.state.video;
      CT.ui.debugLastChange = { idx: j, t: this.t, ct: v ? v.currentTime : this.t };
    }
  }

  // Hệ số zoom so với CT.settings.pxPerSecond (46): 46 px/s làm hợp âm 1 giây của nhạc Kajiura bị cắt chữ.
  // Panel ~800 px x 1.5 thì thấy trước ~8 giây (2–4 ô nhịp); toàn màn hình dùng FS_SCALE.
  Strip.BASE_SCALE = 1.5;
  Strip.FS_SCALE = 2;
  CT.ui = CT.ui || {};
  CT.ui.Strip = Strip;
  CT.ui.strip = new Strip();
})();
