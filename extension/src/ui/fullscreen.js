/* ChordTrack fullscreen: khi YouTube toàn màn hình, chuyển CHÍNH dải karaoke (CT.ui.strip.el) vào
 * lớp phủ #ct-fs bên trong #movie_player (phần tử nằm trong fullscreen element nên mới hiện được),
 * thoát thì trả dải về panel.
 *
 * Thanh điều khiển YouTube hiện (không có class "ytp-autohide"): lớp phủ nằm ngay trên thanh điều khiển.
 * Thanh điều khiển ẩn: lớp phủ hạ xuống sát đáy và nền trong hơn, nhưng KHÔNG mờ chữ hợp âm.
 * Lý do: người dùng vừa xem vừa đàn, cần đọc hợp âm liên tục; làm mờ hợp âm khi YouTube tự ẩn thanh
 * điều khiển (sau 3 giây không rê chuột) thì đúng lúc đang đàn lại không đọc được.
 * Phụ đề của YouTube được đẩy lên trên lớp phủ (class ct-fs-on trên #movie_player).
 */
(() => {
  const T = () => CT.theory;
  const h = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; return e; };

  const fs = {
    el: null, on: false, container: null, mo: null, last: {},
    build() {
      this.dots = new CT.ui.BeatDots();
      this.keyEl = h("span", "ct-ov-key");
      this.soonEl = h("span", "ct-chip ct-warn");
      this.soonEl.hidden = true;
      const row = h("div", "ct-ov-row");
      row.append(this.dots.el, this.keyEl, this.soonEl);
      this.slot = h("div", "ct-ov-slot");
      this.el = h("div", "ct-root ct-fs");
      this.el.id = "ct-fs";
      this.el.append(row, this.slot);
    },
    enter(container) {
      const song = CT.ui.song;
      if (!container || !song || !CT.settings.enabled) return;
      if (!this.el) this.build();
      this.on = true;
      CT.ui.fullscreenOn = true;
      this.container = container;
      this.last = {};
      this.dots.setMeter(song.beats.length ? song.meter : 0);
      container.append(this.el);
      container.classList.add("ct-fs-on");
      this.slot.append(CT.ui.strip.el);
      CT.ui.strip.setScale(CT.ui.Strip.FS_SCALE);
      CT.ui.panel.el.classList.add("ct-strip-away");
      this.syncControls();
      this.mo = new MutationObserver(() => this.syncControls());
      this.mo.observe(container, { attributes: true, attributeFilter: ["class"] });
      if (CT.ui.lastTime) this.onTime(CT.ui.lastTime);
    },
    exit() {
      if (!this.on) return;
      this.on = false;
      CT.ui.fullscreenOn = false;
      this.mo?.disconnect();
      this.mo = null;
      this.container?.classList.remove("ct-fs-on", "ct-fs-controls");
      this.el.remove();
      CT.ui.panel.slot.append(CT.ui.strip.el);
      CT.ui.panel.el.classList.remove("ct-strip-away");
      CT.ui.strip.setScale(CT.ui.Strip.BASE_SCALE);
      this.container = null;
    },
    /** Đặt lớp phủ ngay trên thanh điều khiển khi nó hiện. */
    syncControls() {
      const c = this.container;
      if (!c) return;
      const hidden = c.classList.contains("ytp-autohide");
      c.classList.toggle("ct-fs-controls", !hidden);
      this.el.classList.toggle("ct-lowered", hidden);
      if (!hidden) {
        const bar = c.querySelector(".ytp-chrome-bottom");
        const hgt = bar ? bar.offsetHeight + (parseFloat(getComputedStyle(bar).bottom) || 0) : 60;
        this.el.style.setProperty("--ct-fs-bottom", `${Math.round(hgt + 8)}px`);
      }
    },
    onTime(p) {
      if (!this.on) return;
      const song = CT.ui.song;
      if (!song) return;
      CT.ui.strip.update(p.t);
      const i = CT.ui.info(song, p.t);
      this.dots.set(i.beatPos);
      const white = CT.settings.mode === "white";
      const k = i.key;
      const keyText = !k ? "" : white
        ? `Transpose ${T().signed(k.white_key_transpose | 0)} · ${T().keyName(k)}`
        : `${T().keyName(k)}${song.result.tempo ? ` · ${Math.round(song.result.tempo)} BPM` : ""} · ${song.meter}/4`;
      if (this.last.key !== keyText) { this.last.key = keyText; this.keyEl.textContent = keyText; }
      const soon = CT.ui.soonText(i, white);
      if (this.last.soon !== soon) { this.last.soon = soon; this.soonEl.textContent = soon; this.soonEl.hidden = !soon; }
    },
  };

  CT.ui.fullscreen = fs;
  CT.bus.on("fullscreen", ({ on, container }) => (on ? fs.enter(container) : fs.exit()));
  CT.bus.on("time", (p) => fs.onTime(p));
  // Bài mới tới khi đang toàn màn hình (ví dụ tự chuyển bài): vào lại lớp phủ nếu đang toàn màn hình.
  CT.bus.on("song:ready", () => {
    const p = CT.sync.findPlayer();
    const fe = document.fullscreenElement;
    if (fe && p && (fe === p || fe.contains(p))) { fs.exit(); fs.enter(p); }
  });
  for (const evt of ["song:loading", "song:error", "song:none"]) CT.bus.on(evt, () => fs.exit());
  CT.bus.on("settings", (s) => {
    if (fs.on) { fs.last = {}; if (!s.enabled) fs.exit(); }
  });
})();
