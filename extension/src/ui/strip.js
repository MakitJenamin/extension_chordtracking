/* ChordTrack strip: dải hợp âm trượt ngang kiểu karaoke, vạch đỏ cố định ở 28% chiều rộng.
 *
 * Một phần tử duy nhất (CT.ui.strip.el) dùng cho cả panel dưới video và lớp phủ toàn màn hình
 * (fullscreen.js chuyển chính phần tử này qua lại).
 * Hiệu năng: DOM chỉ dựng lại khi có bài mới; đổi chế độ thì chỉ đổi chữ; mỗi khung hình chỉ đặt
 * transform của dải và đổi class khi chỉ số hợp âm hiện tại đổi.
 *
 * Dòng giai điệu (setMelody): một làn nốt nằm TRONG track, dưới các ô hợp âm, nên trượt cùng transform, cùng
 * px/giây và cùng vạch đỏ, và đi theo dải sang lớp phủ toàn màn hình. Kiểu piano-roll: mỗi nốt là một viên
 * nằm ngang, chiều dài = thời lượng, độ cao = cao độ (CT.theory.pitchRange/pitchY), nên nhìn ra giai điệu lên/xuống
 * mà không cần mũi tên. Tên nốt (CT.theory.melodyName, không số quãng tám) luôn nằm trong viên, viên hẹp thì chữ
 * thu nhỏ dần (fitMelody); hẹp quá thì chỉ còn viên (tooltip vẫn đủ). Nốt lặp không ghi lại tên. Nốt uncertain: viên rỗng viền đứt. Nốt đang phát: viên đỏ, phát sáng,
 * và ô "đang hát" ở mép trái dải ghi tên nốt hiện tại bằng chữ to kèm nốt kế tiếp.
 */
(() => {
  const PLAYHEAD = 0.28;
  const PAD = 16; // đệm trái 8 + đệm phải 8 của .ct-chord (styles.css): khe tối thiểu giữa hai nhãn
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
      this.melody = null;   // <Melody> của bài (docs/melody_api.md)
      this.notes = [];      // nốt hợp lệ, đã sắp
      this.noteEls = [];
      this.lane = null;
      this.nIdx = -1;
      this.nPast = 0;
      this.track = el("div", "ct-strip");
      this.head = el("div", "ct-playhead");
      this.empty = el("div", "ct-strip-empty", "Đoạn này không có hợp âm");
      // Thanh mỏng ở đáy, chỉ hiện khi đang kéo/xem lệch: khung đang xem nằm đâu trong cả bài.
      this.bar = el("div", "ct-panbar");
      this.barView = el("div", "ct-panbar-view");
      this.barNow = el("div", "ct-panbar-now");
      this.barTip = el("div", "ct-panbar-tip");
      this.bar.append(this.barView, this.barNow);
      // Ô "đang hát": cố định ở mép trái làn nốt (không trượt theo dải), tên nốt hiện tại chữ to + nốt kế tiếp.
      this.now = el("div", "ct-mel-now");
      this.nowName = el("b");
      this.nowNext = el("span");
      this.now.append(this.nowName, this.nowNext);
      this.nowKey = null;
      this.labels = [];
      this.range = null;
      // Hiệu ứng (CT.settings.fx), phong cách phát sáng dịu: quầng sáng mềm, thở chậm, chuyển sáng từ từ. Ở điểm nốt
      // chạm vạch đỏ có quầng sáng (lúc nốt ngân) và một đóa sáng nở ra rồi tan khi vào nốt mới. Lớp này nằm TRONG
      // làn nốt, đứng trước các viên trong DOM nên được vẽ bên dưới, không che tên nốt (track có will-change nên
      // z-index chỉ có nghĩa trong track); update() dịch nó theo vạch đỏ. Vài đóa xoay vòng để giai điệu nhanh không bị cắt.
      this.hitZone = el("div", "ct-hitzone");
      this.glow = el("div", "ct-glow");
      this.blooms = [];
      for (let k = 0; k < Strip.BLOOMS; k++) this.blooms.push(el("div", "ct-bloom"));
      this.hitZone.append(this.glow, ...this.blooms);
      this.fx = { beat: -1, flip: false, pop: false, soon: null, np: -1, playing: null, bloom: 0 };
      this.el.append(this.track, this.head, this.now, this.bar, this.barTip);
      this.song = null;
      this.scale = Strip.BASE_SCALE;
      this.width = 0;
      this.idx = -1;
      this.stuckIdx = -1;
      this.t = 0;
      this.chordEls = [];
      this.modEls = [];
      this.beatEls = [];
      this.pastCount = 0;
      this.lastX = null;
      this.viewOff = 0;     // độ lệch khung nhìn do người dùng kéo (px, dương = xem lùi về trước)
      this.returnAt = 0;    // performance.now() lúc bắt đầu trượt về vạch đỏ
      this.drag = null;
      this.track.addEventListener("click", (e) => {
        const n = e.target.closest(".ct-note");
        if (n && this.song) {
          this.viewOff = 0;
          const note = this.notes[+n.dataset.i];
          CT.seekTo((note.start0 ?? note.start) + 0.01);   // nốt đã căn phách: tua tới đầu nốt thật
          return;
        }
        const c = e.target.closest(".ct-chord");
        if (!c || !this.song) return;
        this.viewOff = 0; // tua tới hợp âm vừa bấm thì nó về đúng vạch đỏ
        CT.seekTo(this.song.chords[+c.dataset.i].start + 0.01);
      });
      this.bindPan();
      this.ro = new ResizeObserver((entries) => {
        this.width = entries[0].contentRect.width;
        // Dải hẹp: ô "đang hát" (~130 px ở mép trái) sẽ che vạch đỏ ở 28% bề rộng, nên ẩn đi.
        this.el.classList.toggle("ct-mel-narrow", this.width * PLAYHEAD < 150);
        // Lần đầu dải hiện ra (trước đó bề rộng 0 nên chưa đo được chữ), hoặc đổi panel <-> toàn màn hình.
        // Làn nốt chưa đo được chữ (dựng lúc dải đang ẩn) cũng đo lại.
        if (this.width && (!this.fitWidth || this.fitScale !== this.scale || (this.lane && !this.melFit))) this.fitLabels();
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
      this.viewOff = 0;
      this.returnAt = 0;
      this.fx.beat = -1;
      this.fx.soon = null;
      this.notes = this.melodyLine();     // lưới phách của bài mới
      if (!song) { this.buildMelody(false); return; }
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
      this.buildMelody(false);
      this.relabelMelody(false);   // gán nhãn nốt trước khi layout() đo chữ
      this.layout();
      this.relabel();
    }

    /* ---------------- Dòng giai điệu ---------------- */

    /** Đặt <Melody> (hoặc null). Dựng lại làn nốt nếu dải đang có bài. */
    setMelody(melody) {
      if (melody === this.melody) return;
      this.melody = melody || null;
      this.notes = this.melodyLine();
      this.buildMelody(true);
    }

    /** Nốt của <Melody> đang giữ, luôn căn về lưới phách của bài (CT.theory.snapNotes, mức chia tự chọn bằng
     * autoSnapDiv; người dùng chốt luôn bật, 2026-11-11). Bài không có lưới phách thì giữ nguyên. */
    melodyLine() {
      const notes = T().melodyNotes(this.melody);
      if (!this.song) return notes;
      return T().snapNotes(notes, this.song.beats, T().autoSnapDiv(notes, this.song.beats));
    }

    /** Làn nốt có hiện không: có bài, có nốt và người dùng bật "Giai điệu". */
    get melodyOn() { return !!(this.song && this.notes.length && CT.settings.showMelody !== false); }

    /** Dựng (hoặc bỏ) DOM làn nốt. full: đặt luôn vị trí và nhãn (khi không gọi từ setSong). */
    buildMelody(full) {
      this.lane?.remove();
      this.lane = null;
      this.noteEls = [];
      this.nIdx = -1;
      this.nPast = 0;
      this.nowKey = null;
      const on = this.melodyOn;
      this.el.classList.toggle("ct-has-melody", on);
      if (!on) return;
      const lane = (this.lane = el("div", "ct-mel-lane"));
      const frag = document.createDocumentFragment();
      this.range = T().pitchRange(this.notes);
      this.notes.forEach((n, i) => {
        const e = el("div", "ct-note" + (n.uncertain ? " ct-nunc" : ""));
        e.dataset.i = i;
        e.style.setProperty("--y", T().pitchY(n.pitch, this.range).toFixed(3));
        e.append(el("span", "ct-nbar"), el("span", "ct-nl"));
        frag.append(e);
        this.noteEls.push(e);
      });
      lane.append(this.hitZone, frag);   // hitZone trước các viên: vẽ bên dưới
      this.track.append(lane);
      if (full) {
        this.layoutMelody();
        this.relabelMelody();
      }
    }

    layoutMelody() {
      if (!this.lane) return;
      const px = this.px;
      this.notes.forEach((n, i) => {
        const e = this.noteEls[i];
        e.style.left = `${n.start * px}px`;
        e.style.width = `${Math.max(2, (n.end - n.start) * px)}px`;
      });
    }

    relabelMelody(fit = true) {
      if (!this.lane) return;
      const L = (this.labels = T().melodyLabels(this.song, this.notes, CT.settings));
      this.nowKey = null;   // tên nốt đổi theo chế độ: vẽ lại ô "đang hát"
      this.notes.forEach((n, i) => {
        const e = this.noteEls[i];
        const p = i > 0 ? this.notes[i - 1] : null;
        Strip.setNoteName(e.lastChild, L[i].name, p ? T().octaveMark(L[i - 1].name, p.pitch, L[i].name, n.pitch) : "");
        const conf = Number.isFinite(n.confidence) ? ` · độ tin cậy ${Math.round(n.confidence * 100)}%` : "";
        e.title = `${L[i].name}${L[i].arrow ? " " + L[i].arrow : ""} · ${fmt(n.start)}${conf}`
          + `${n.uncertain ? " (model không chắc)" : ""}. Bấm để tua tới đây`;
      });
      if (!fit) return;
      this.fitMelody();
      this.update(this.t, true);
    }

    /** Tên nốt luôn nằm gọn trong viên của chính nốt đó (không tràn sang viên khác). Viên hẹp thì chữ thu nhỏ
     * dần: cỡ thường -> ct-nsm (chữ nhỏ, dấu thăng/giáng thu thành chỉ số trên) -> ct-nxs (nhỏ nhất, sát lề);
     * vẫn không vừa thì chỉ còn viên (ct-nbare, tooltip vẫn đủ). Mỗi nấc đo chữ thật một lượt cho cả bài.
     * Nốt lặp (ct-nrep): cùng tên hiển thị với nốt ngay trước, nghỉ dưới 0,6 s, và tên đó đang hiện ở viên trước
     * (hoặc đầu chuỗi lặp) thì không ghi lại; đầu chuỗi không có tên thì nốt sau vẫn ghi tên nếu vừa.
     * Dải đang ẩn (chữ chưa vẽ) thì bỏ qua; ResizeObserver gọi lại khi dải hiện ra. */
    fitMelody() {
      this.melFit = false;
      if (!this.lane || !this.width) return;
      const els = this.noteEls, ns = this.notes, px = this.px, L = this.labels;
      for (const e of els) e.classList.remove("ct-nbare", "ct-nsm", "ct-nxs", "ct-nrep");
      const room = ns.map((n) => (n.end - n.start) * px - 1);   // .ct-nbar chừa 1 px bên phải
      const w = els.map((e) => e.lastChild.offsetWidth);
      if (!w.some((x) => x > 0)) return;   // chữ chưa được vẽ (dải đang ẩn): đo lại khi dải hiện ra
      this.melFit = true;
      for (const cls of ["ct-nsm", "ct-nxs"]) {
        const over = [];
        els.forEach((e, i) => { if (w[i] > room[i]) over.push(i); });
        if (!over.length) break;
        for (const i of over) els[i].classList.remove("ct-nsm"), els[i].classList.add(cls);
        for (const i of over) w[i] = els[i].lastChild.offsetWidth;
      }
      let shown = false;   // tên của nốt trước (hoặc đầu chuỗi lặp của nó) có đang hiện không
      els.forEach((e, i) => {
        const p = ns[i - 1];
        if (p && shown && p.pitch === ns[i].pitch && L[i]?.name === L[i - 1]?.name && ns[i].start - p.end < 0.6) {
          e.classList.add("ct-nrep");
          if (w[i] > room[i]) e.classList.add("ct-nbare");   // lúc đang phát (hiện tên) cũng không tràn ra ngoài viên
          return;
        }
        shown = w[i] <= room[i];
        if (!shown) e.classList.add("ct-nbare");
      });
    }

    /** Ô "đang hát": tên nốt đang phát (đang nghỉ thì để trống) và nốt kế tiếp có cao độ khác. Chỉ đổi chữ khi nội dung đổi. */
    showNow(j, past) {
      const L = this.labels, ns = this.notes;
      let nx = j >= 0 ? j + 1 : past;
      if (j >= 0) while (nx < ns.length && ns[nx].pitch === ns[j].pitch) nx++;   // bỏ qua nốt lặp: hiện nốt đổi cao độ
      const cur = j >= 0 && L[j] ? L[j].name : "";
      const arrow = j >= 0 && nx < ns.length ? T().melodyArrow(ns[j].pitch, ns[nx].pitch) : "";
      const next = nx < ns.length && L[nx] ? L[nx].name + arrow : "";
      const key = `${cur}|${next}|${j >= 0 && ns[j].uncertain ? 1 : 0}`;
      if (key === this.nowKey) return;
      this.nowKey = key;
      if (cur) this.retrigger(this.now, "pop", "ct-pop-a", "ct-pop-b");
      this.nowName.textContent = cur || "–";
      this.nowNext.textContent = next ? `→ ${next}` : "";
      this.now.classList.toggle("ct-rest", !cur);
      this.now.classList.toggle("ct-nunc", j >= 0 && !!ns[j].uncertain);
    }

    /** Nốt đang phát: lớp ct-cur; các nốt đã qua: ct-past (chỉ sửa đoạn giữa chỉ số cũ và mới). */
    updateMelody(t, force) {
      if (!this.lane) return;
      const ns = this.notes, els = this.noteEls;
      const j = T().noteIndexAt(ns, t, this.nIdx);
      const past = j >= 0 ? j : T().lastLE(ns, t, (n) => n.start) + 1;
      this.noteFill(j, t);
      if (j === this.nIdx && past === this.nPast && !force) return;
      if (this.nIdx >= 0 && els[this.nIdx]) { els[this.nIdx].classList.remove("ct-cur"); els[this.nIdx].style.removeProperty("--np"); }
      const prev = this.nPast;
      if (past > prev) for (let k = prev; k < past; k++) els[k].classList.add("ct-past");
      else for (let k = past; k < prev; k++) els[k].classList.remove("ct-past");
      this.nPast = past;
      if (j >= 0) els[j].classList.add("ct-cur");
      const fresh = j >= 0 && j !== this.nIdx;
      this.nIdx = j;
      this.showNow(j, past);
      this.moveGlow(j, fresh && !force);
    }

    /* ---------------- Hiệu ứng (CT.settings.fx, CSS trong khối "Hiệu ứng" của styles.css) ----------------
     * Chỉ đổi biến CSS / class (không đọc layout) nên không làm chậm vòng rAF. Hoạt ảnh chạy lại bằng cách
     * đổi qua lại hai class có keyframes giống hệt nhau (đổi tên animation thì trình duyệt chạy lại từ đầu). */

    /** Đổi qua lại class a/b trên e để chạy lại hoạt ảnh. key: ô nhớ trạng thái trong this.fx. */
    retrigger(e, key, a, b) {
      this.fx[key] = !this.fx[key];
      e.classList.remove(this.fx[key] ? b : a);
      e.classList.add(this.fx[key] ? a : b);
    }

    /** Tô karaoke cho nốt đang phát: --np (0–1) là phần đã hát của viên. */
    noteFill(j, t) {
      const e = j >= 0 ? this.noteEls[j] : null;
      if (!e) { this.fx.np = -1; return; }
      const n = this.notes[j];
      const p = Math.round(Math.max(0, Math.min(1, (t - n.start) / Math.max(1e-3, n.end - n.start))) * 100) / 100;
      if (p === this.fx.np && j === this.nIdx) return;
      this.fx.np = p;
      e.style.setProperty("--np", p);
    }

    get playing() { const v = CT.state.video; return !!(v && !v.paused); }

    /** Quầng sáng ở điểm nốt j chạm vạch đỏ (nghỉ thì tắt), trượt mềm theo cao độ. hit: vừa vào nốt mới khi đang
     * phát, thì một đóa sáng nở ra từ điểm chạm rồi tan (lấy đóa kế tiếp trong vòng xoay). */
    moveGlow(j, hit) {
      const z = this.hitZone;
      z.classList.toggle("ct-rest", j < 0);
      if (j < 0) return;
      const n = this.notes[j];
      const y = T().pitchY(n.pitch, this.range).toFixed(3);
      z.style.setProperty("--oy", y);
      z.classList.toggle("ct-unc", !!n.uncertain);
      if (!hit || !this.playing || CT.settings.fx === false) return;
      const r = this.blooms[this.fx.bloom = (this.fx.bloom + 1) % this.blooms.length];
      r.style.setProperty("--gy", y);
      r.classList.toggle("ct-long", n.end - n.start >= 0.35);   // nốt ngân dài: đóa nở rộng và chậm hơn
      const on = r.classList.contains("ct-ga");
      r.classList.remove("ct-ga", "ct-gb");
      r.classList.add(on ? "ct-gb" : "ct-ga");
    }

    /** Mỗi khung hình: vạch đỏ "thở" nhẹ theo phách (phách 1 rõ hơn) và trạng thái phát/dừng.
     * Chỉ chạy khi phách tăng đúng 1 (phát bình thường), tua hay kéo thì không. */
    beatFx(t) {
      const playing = this.playing;
      if (playing !== this.fx.playing) { this.fx.playing = playing; this.el.classList.toggle("ct-playing", playing); }
      const s = this.song;
      const b = T().lastLE(s.beats, t);
      if (b === this.fx.beat) return;
      const step = b === this.fx.beat + 1;
      this.fx.beat = b;
      if (!step || !playing || b < 0) return;
      this.el.classList.toggle("ct-down", s.beatPos[b] === 1);
      this.retrigger(this.head, "flip", "ct-ba", "ct-bb");
    }

    /** Hợp âm kế tiếp "chờ vào" khi hợp âm hiện tại đã qua 75%: quầng sáng nhạt hiện dần. */
    soonFx(p) {
      const next = p >= 0.75 && this.idx >= 0 ? this.chordEls[this.idx + 1] || null : null;
      if (next === this.fx.soon) return;
      this.fx.soon?.classList.remove("ct-soon");
      next?.classList.add("ct-soon");
      this.fx.soon = next;
    }

    layout() {
      const s = this.song;
      if (!s) return;
      this.viewOff = 0; // đổi độ phóng thì px cũ không còn đúng nghĩa
      const px = this.px;
      this.track.style.width = `${s.duration * px}px`;
      s.chords.forEach((c, i) => {
        const e = this.chordEls[i];
        e.style.left = `${c.start * px}px`;
        e.style.width = `${Math.max(4, (c.end - c.start) * px)}px`;
      });
      s.beats.forEach((b, i) => { this.beatEls[i].style.left = `${b * px}px`; });
      s.modulations.forEach((m, i) => { this.modEls[i].style.left = `${m.time * px}px`; });
      this.layoutMelody();
      this.fitLabels();
      this.lastX = null;
      this.update(this.t, true);
    }

    /** Một cỡ chữ cho mọi ô (không co chữ, trông lộn xộn). Ô không đủ chỗ cho cả nhãn:
     *  - ct-clip: bỏ dòng bậc, tên hợp âm cắt bằng "…" (tooltip vẫn có tên đầy đủ);
     *  - ct-hide: hẹp tới mức không hiện nổi một ký tự kèm "…" thì ẩn nhãn, chỉ còn ô.
     * Mỗi ô có đệm phải (CSS) nên giữa hai nhãn luôn có khe tối thiểu.
     * Đo chữ thật (một lượt đọc layout cho cả bài) nên đúng với mọi chế độ, cỡ chữ và độ phóng.
     * Dải đang ẩn (bề rộng 0) thì bỏ qua; ResizeObserver gọi lại khi dải hiện ra. */
    fitLabels() {
      const s = this.song;
      if (!s || !this.width) return;
      const els = this.chordEls;
      for (const e of els) e.classList.remove("ct-clip", "ct-hide");
      const lbl = els.map((e) => e.firstChild.offsetWidth);
      const sym = els.map((e) => e.firstChild.firstChild.offsetWidth);
      const minSym = sym.length ? Math.min(...sym.filter((w) => w > 0)) : 0; // ~ bề rộng tên 1 ký tự
      els.forEach((e, i) => {
        const room = (s.chords[i].end - s.chords[i].start) * this.px - PAD;
        if (lbl[i] <= room) return;
        e.classList.add(room < Math.max(18, minSym * 0.9) ? "ct-hide" : "ct-clip");
      });
      this.fitMelody();
      this.fitWidth = this.width;
      this.fitScale = this.scale;
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
        // ↑ ↓ → có trong Roboto; ⬆ ⬇ ⇄ không có nên toàn màn hình hiện thành ký tự lạ.
        const arrow = m.semitones > 0 ? "↑" : m.semitones < 0 ? "↓" : "→";
        const label = st.mode === "white"
          ? `${arrow} ${T().keyName(to)} · ${T().transposeLabel(m.transpose_to)}`
          : `${arrow} ${T().keyName(to)}${m.semitones ? ` (${T().signed(m.semitones)})` : ""}`;
        this.modEls[i].firstChild.textContent = label;
        this.modEls[i].title = `Đổi giọng lúc ${fmt(m.time)}: ${T().keyName(T().parseKeyText(m.from))} → ${T().keyName(to)}`;
      });
      this.relabelMelody(false);
      this.fitLabels(); // bề rộng chữ đổi theo chế độ (đo lại cả nhãn nốt)
      this.update(this.t, true);
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

    /** Kéo ngang (chuột, cảm ứng, bút) hoặc cuộn ngang (trackpad, Shift + lăn chuột) để xem trước/sau.
     * Thả tay RETURN_MS thì dải tự trượt về vạch đỏ. Kéo dưới DRAG_MIN px vẫn tính là bấm (tua). */
    bindPan() {
      const host = this.el;
      // Click ngay sau khi kéo: không tua và không cho nổi lên player YouTube (toàn màn hình nằm trong #movie_player).
      host.addEventListener("click", (e) => {
        if (this.justDragged) { this.justDragged = false; e.stopPropagation(); }
      }, true);
      host.addEventListener("pointerdown", (e) => {
        if (!this.song || (e.pointerType === "mouse" && e.button !== 0)) return;
        this.drag = { id: e.pointerId, x0: e.clientX, off0: this.viewOff, moved: false };
      });
      host.addEventListener("pointermove", (e) => {
        const d = this.drag;
        if (!d || d.id !== e.pointerId) return;
        const dx = e.clientX - d.x0;
        if (!d.moved) {
          if (Math.abs(dx) < Strip.DRAG_MIN) return;
          d.moved = true;
          try { host.setPointerCapture(e.pointerId); } catch (_) { /* con trỏ đã nhả */ }
          host.classList.add("ct-dragging");
        }
        this.setViewOff(d.off0 + dx);
      });
      const end = (e) => {
        const d = this.drag;
        if (!d || d.id !== e.pointerId) return;
        this.drag = null;
        host.classList.remove("ct-dragging");
        if (d.moved) {
          this.justDragged = true;
          setTimeout(() => { this.justDragged = false; }, 0); // click (nếu có) tới ngay sau pointerup
          this.scheduleReturn();
        }
      };
      host.addEventListener("pointerup", end);
      host.addEventListener("pointercancel", end);
      host.addEventListener("wheel", (e) => {
        if (!this.song) return;
        // Chỉ ăn cuộn ngang; lăn dọc thường vẫn cuộn trang YouTube.
        const dx = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
        if (!dx || (!e.shiftKey && Math.abs(e.deltaX) < Math.abs(e.deltaY))) return;
        e.preventDefault();
        this.setViewOff(this.viewOff - dx * (e.deltaMode === 1 ? 16 : 1));
        this.scheduleReturn();
      }, { passive: false });
    }

    /** Giới hạn để vạch đỏ luôn chỉ vào một thời điểm trong bài [0, duration]. */
    setViewOff(v) {
      const s = this.song;
      if (!s) return;
      const px = this.px;
      this.viewOff = Math.max((this.t - s.duration) * px, Math.min(this.t * px, v));
      this.returnAt = 0;
      this.lastReturnTick = 0;
      this.update(this.t, true);
    }

    scheduleReturn() { this.returnAt = performance.now() + Strip.RETURN_MS; }

    /** Đang kéo thì giữ nguyên; hết thời gian chờ thì trượt dần về 0 (theo thời gian, không theo số khung hình). */
    panOffset() {
      if (!this.viewOff || this.drag?.moved || !this.returnAt) return this.viewOff;
      const now = performance.now();
      if (now < this.returnAt) return this.viewOff;
      const dt = Math.min(100, now - (this.lastReturnTick || now));
      this.lastReturnTick = now;
      this.viewOff *= Math.exp(-dt / 90);
      if (Math.abs(this.viewOff) < 0.5) { this.viewOff = 0; this.returnAt = 0; this.lastReturnTick = 0; }
      return this.viewOff;
    }

    update(t, force = false) {
      this.t = t;
      if (!this.song || !this.width) return;
      const x = this.width * PLAYHEAD - t * this.px + this.panOffset();
      if (force || x !== this.lastX) {
        this.track.style.transform = `translate3d(${x}px,0,0)`;
        // Lớp hiệu ứng trong track đứng yên ở vạch đỏ: tọa độ vạch đỏ theo track.
        const ph = `translate3d(${this.width * PLAYHEAD - x}px,0,0)`;
        this.hitZone.style.transform = ph;
        this.lastX = x;
      }
      const j = this.indexAt(t);
      if (j !== this.idx || force) this.setCurrent(j);
      this.updateMelody(t, force);
      this.beatFx(t);
      this.progress(t);
      this.stickLabel(x);
      this.panBar(x);
    }

    /** Thanh mỏng ở đáy khi đang kéo/xem lệch: vệt sáng là khung đang xem trên toàn bài, vạch đỏ là
     * chỗ đang phát. Nhãn: thời điểm nằm ở vị trí vạch đỏ của khung đang xem, lệch bao nhiêu giây. */
    panBar(x) {
      const on = Math.abs(this.viewOff) >= 1 || !!this.drag?.moved;
      if (on !== this.barOn) { this.el.classList.toggle("ct-panning", on); this.barOn = on; }
      if (!on) return;
      const D = Math.max(1e-3, this.song.duration);
      const t0 = -x / this.px;                 // thời điểm ở mép trái khung
      const span = this.width / this.px;       // số giây nhìn thấy
      const pct = (v) => `${(Math.max(0, Math.min(1, v)) * 100).toFixed(2)}%`;
      const a = Math.max(0, t0 / D);
      const b = Math.min(1, (t0 + span) / D);
      this.barView.style.left = pct(a);
      this.barView.style.width = pct(Math.max(0.004, b - a));
      this.barNow.style.left = pct(this.t / D);
      const tv = Math.max(0, Math.min(D, t0 + span * PLAYHEAD));
      const d = Math.round(tv - this.t);
      const text = `${fmt(tv)}${d ? ` · ${d > 0 ? "+" : "−"}${Math.abs(d)} s` : ""}`;
      if (this.barTip.textContent !== text) this.barTip.textContent = text;
      // Nhãn bám giữa khung đang xem nhưng không tràn khỏi mép dải.
      const w = this.barTip.offsetWidth || 60;
      const mid = ((a + b) / 2) * this.width;
      this.barTip.style.left = `${Math.round(Math.max(4, Math.min(this.width - w - 4, mid - w / 2)))}px`;
    }

    /** Hợp âm dài hơn bề rộng nhìn thấy (ví dụ Cm giữ 25 s): nhãn nằm ở mép trái ô sẽ trôi khỏi khung,
     * để lại một ô trống. Đẩy nhãn của ô đang cắt mép trái khung sang phải để luôn đọc được,
     * nhưng không vượt quá mép phải của chính ô đó. */
    stickLabel(x) {
      const ch = this.song.chords;
      const viewLeft = -x;                               // mép trái khung, theo tọa độ track (px)
      const k = T().lastLE(ch, viewLeft / this.px, (c) => c.start);
      if (k !== this.stuckIdx && this.stuckIdx >= 0 && this.chordEls[this.stuckIdx]) {
        this.chordEls[this.stuckIdx].firstChild.style.transform = "";
      }
      this.stuckIdx = k;
      if (k < 0) return;
      const e = this.chordEls[k];
      const c = ch[k];
      const lbl = e && e.firstChild;
      if (!lbl) return;
      const left = c.start * this.px;
      const room = (c.end - c.start) * this.px - (lbl.offsetWidth || 0) - 16;
      const shift = Math.max(0, Math.min(viewLeft - left, room));
      const v = shift > 0 ? `translateX(${Math.round(shift)}px)` : "";
      if (lbl.style.transform !== v) lbl.style.transform = v;
    }

    /** Tiến độ trong hợp âm hiện tại (0–1): thanh đỏ mảnh ở đáy ô hiện tại. */
    progress(t) {
      const e = this.chordEls[this.idx];
      if (!e) return;
      const c = this.song.chords[this.idx];
      const p = Math.round(Math.max(0, Math.min(1, (t - c.start) / Math.max(1e-3, c.end - c.start))) * 200) / 200;
      if (p !== this.curP) { e.style.setProperty("--ct-p", p); this.curP = p; this.soonFx(p); }
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
      if (old >= 0 && els[old]) els[old].style.removeProperty("--ct-p");
      this.curP = -1;
      this.soonFx(0);
      if (j >= 0) {
        els[j].classList.remove("ct-past");
        els[j].classList.add("ct-cur");
      }
      this.idx = j;
      // Cho harness đo độ lệch đồng bộ (t hiển thị và currentTime thật lúc đổi hợp âm).
      const v = CT.state.video;
      CT.ui.debugLastChange = { idx: j, t: this.t, ct: v ? v.currentTime : this.t };
    }
  }

  /** Ghi tên nốt vào nhãn: chữ cái chính và dấu thăng/giáng (b, #, cả dạng bậc "b3") tách riêng để thu nhỏ được. */
  Strip.setNoteName = (lbl, name, mark = "") => {
    const m = /^([b#]?)(.*?)([b#]?)$/.exec(name || "");
    lbl.textContent = "";
    if (mark) lbl.append(el("i", "ct-noct", mark));   // ↑/↓: cùng tên với nốt trước nhưng khác quãng tám
    if (m[1]) lbl.append(el("sup", null, m[1]));
    lbl.append(m[2]);
    if (m[3]) lbl.append(el("sup", null, m[3]));
  };

  // Hệ số zoom so với CT.settings.pxPerSecond (46): 46 px/s làm hợp âm 1 giây của nhạc Kajiura bị cắt chữ.
  // Panel ~800 px x 1.5 thì thấy trước ~8 giây (2–4 ô nhịp); toàn màn hình dùng FS_SCALE.
  Strip.BASE_SCALE = 1.5;
  Strip.FS_SCALE = 2;
  Strip.DRAG_MIN = 5;      // px: kéo ít hơn thì vẫn là bấm để tua
  Strip.RETURN_MS = 2500;  // thả tay bao lâu thì dải trượt về vạch đỏ
  Strip.BLOOMS = 3;        // số đóa sáng dùng xoay vòng (nốt nhanh vẫn mỗi nốt một đóa)
  CT.ui = CT.ui || {};
  CT.ui.Strip = Strip;
  CT.ui.strip = new Strip();
  // Bật/tắt hiệu ứng (công tắc "Hiệu ứng" của panel). Class đặt trên <html> để áp cho cả panel, dải và lớp phủ
  // toàn màn hình; mọi hiệu ứng trong styles.css đều nằm dưới html.ct-fx-on.
  const fxClass = (s) => document.documentElement.classList.toggle("ct-fx-on", s.fx !== false);
  fxClass(CT.settings);
  CT.bus.on("settings", fxClass);
})();
