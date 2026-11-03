/* ChordTrack: nút kích hoạt trên thanh điều khiển của player YouTube (.ytp-right-controls).
 *
 * CT.ui.ctlButton.mount() / unmount()   gắn / gỡ nút (content.js gọi khi đang ở trang video và extension bật)
 * Nút đọc trạng thái từ sự kiện "app" (content.js phát):
 *   {videoId, visible, status: "off"|"loading"|"ready"|"error", cached, progress}
 * Hình thức theo trạng thái (data-state trên nút):
 *   off       icon trắng như nút của YouTube; cached = true thì có chấm nhỏ ("bấm là có ngay")
 *   loading   vòng tròn tiến độ quanh icon
 *   on        đang hiện: icon màu accent
 *   error     chấm đỏ
 * Bấm nút: CT.app.toggle(). Tooltip tự vẽ giống tooltip của YouTube (YouTube không hiện tooltip cho nút chèn thêm).
 * YouTube vẽ lại thanh điều khiển (đổi chế độ rạp hát, toàn màn hình, điều hướng SPA): MutationObserver gắn lại, không chèn trùng.
 */
(() => {
  const NS = "http://www.w3.org/2000/svg";
  // YouTube bật Trusted Types: không dùng innerHTML, dựng SVG bằng createElementNS.
  const svg = (tag, attrs = {}, ...kids) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    for (const k of kids) e.append(k);
    return e;
  };
  // Hai nốt móc đơn nối chùm, vẽ trong khung 24×24 như icon của YouTube.
  const NOTE = "M9 6.6 19 4.5v11.2a2.9 2.9 0 1 1-1.8-2.68V8.6L10.8 9.95v7.75a2.9 2.9 0 1 1-1.8-2.68Z";
  const RING_R = 15;

  const btn = {
    el: null, tip: null, mounted: false, mo: null, timer: 0, state: null,
    build() {
      this.ring = svg("circle", { class: "ct-ctl-ring", cx: 17, cy: 17, r: RING_R, pathLength: 100 });
      const ringSvg = svg("svg", { class: "ct-ctl-ringsvg", viewBox: "0 0 34 34", width: 34, height: 34, "aria-hidden": "true" },
        svg("circle", { class: "ct-ctl-track", cx: 17, cy: 17, r: RING_R }), this.ring);
      const icon = svg("svg", { class: "ct-ctl-icon", viewBox: "0 0 24 24", width: 24, height: 24, fill: "none", "aria-hidden": "true" },
        svg("path", { d: NOTE, fill: "currentColor" }));
      const dot = document.createElement("span");
      dot.className = "ct-ctl-dot";
      this.el = document.createElement("button");
      this.el.className = "ytp-button ct-ctl-btn";
      this.el.type = "button";
      this.el.dataset.state = "off";
      this.el.append(icon, ringSvg, dot);
      this.el.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        CT.app && CT.app.toggle();
      });
      // Không để YouTube coi phím Space/Enter trên nút là phím tắt của player.
      this.el.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") e.stopPropagation(); });
      this.el.addEventListener("mouseenter", () => this.showTip());
      this.el.addEventListener("mouseleave", () => this.hideTip());
      this.el.addEventListener("focus", () => this.showTip());
      this.el.addEventListener("blur", () => this.hideTip());
      this.el.addEventListener("click", () => this.hideTip());

      this.tip = document.createElement("div");
      this.tip.className = "ct-ctl-tip";
      this.tip.hidden = true;
      this.tipText = document.createElement("span");
      this.tipKey = document.createElement("span");
      this.tipKey.className = "ct-ctl-tip-key";
      this.tipKey.textContent = "Alt+Shift+C";
      this.tip.append(this.tipText, this.tipKey);
      this.render(this.state || { status: "off" });
    },

    /** Chỗ gắn: trước nút phụ đề trong nhóm trái của .ytp-right-controls; giao diện cũ thì trước nút cài đặt. */
    anchor() {
      const player = CT.sync.findPlayer();
      const rc = player && player.querySelector(".ytp-right-controls");
      if (!rc) return null;
      const group = rc.querySelector(".ytp-right-controls-left") || rc;
      const before = group.querySelector(":scope > .ytp-subtitles-button") || group.querySelector(":scope > .ytp-settings-button")
        || group.firstElementChild;
      return { player, group, before };
    },
    place() {
      this.timer = 0;
      if (!this.mounted) return;
      const a = this.anchor();
      if (!a) return;
      // Không chèn trùng: xóa nút ChordTrack lạ (ví dụ của bản extension cũ trước khi nạp lại).
      for (const old of document.querySelectorAll(".ct-ctl-btn")) if (old !== this.el) old.remove();
      if (this.el.parentNode !== a.group || (a.before && this.el.nextSibling !== a.before && a.before !== this.el)) {
        a.group.insertBefore(this.el, a.before && a.before !== this.el ? a.before : null);
      }
      if (this.tip.parentNode !== a.player) a.player.append(this.tip);
    },
    schedule() { if (!this.timer) this.timer = setTimeout(() => this.place(), 150); },
    mount() {
      if (!this.el) this.build();
      this.mounted = true;
      this.place();
      if (!this.mo) {
        // Rẻ: chỉ đặt lại khi nút hoặc tooltip bị YouTube gỡ khỏi thanh điều khiển.
        this.mo = new MutationObserver(() => {
          if (!this.el.isConnected || !this.tip.isConnected || !this.el.closest("#movie_player .ytp-right-controls")) this.schedule();
        });
        this.mo.observe(document.body, { childList: true, subtree: true });
      }
    },
    unmount() {
      this.mounted = false;
      this.mo?.disconnect();
      this.mo = null;
      clearTimeout(this.timer);
      this.timer = 0;
      this.el?.remove();
      this.tip?.remove();
    },

    render(st) {
      this.state = st;
      if (!this.el) return;
      const view = st.status === "error" ? "error" : st.status === "loading" ? "loading" : st.visible ? "on" : "off";
      this.el.dataset.state = view;
      this.el.classList.toggle("ct-ctl-cached", !!st.cached && !st.visible && view === "off");
      this.el.classList.toggle("ct-ctl-hidden-run", view === "loading" && !st.visible);
      const pct = Math.max(4, Math.min(100, Math.round((st.progress || 0) * 100)));
      this.ring.style.strokeDasharray = `${pct} 100`;
      let text;
      if (view === "loading") text = `ChordTrack: đang phân tích ${Math.round((st.progress || 0) * 100)}%` + (st.visible ? " (bấm để ẩn)" : " (bấm để xem)");
      else if (view === "error") text = st.visible ? "ChordTrack: lỗi, bấm để ẩn" : "ChordTrack: lỗi, bấm để thử lại";
      else if (view === "on") text = "ChordTrack: ẩn hợp âm";
      else text = st.cached ? "ChordTrack: hiện hợp âm (đã có kết quả)" : "ChordTrack: hiện hợp âm";
      this.el.setAttribute("aria-label", text + " (Alt+Shift+C)");
      this.el.setAttribute("aria-pressed", String(!!st.visible));
      if (this.tipText.textContent !== text) {
        this.tipText.textContent = text;
        if (!this.tip.hidden) this.positionTip();
      }
    },

    showTip() {
      if (!this.tip.isConnected) return;
      this.tip.hidden = false;
      this.positionTip();
    },
    hideTip() { if (this.tip) this.tip.hidden = true; },
    /** Đặt tooltip giữa phía trên nút, không tràn khỏi player (giống tooltip của YouTube). */
    positionTip() {
      const p = this.tip.parentElement;
      if (!p) return;
      const pr = p.getBoundingClientRect(), br = this.el.getBoundingClientRect();
      const bar = p.querySelector(".ytp-chrome-bottom");
      const top = (bar ? bar.getBoundingClientRect().top : br.top) - pr.top;
      const w = this.tip.offsetWidth;
      const left = Math.max(12, Math.min(pr.width - w - 12, br.left - pr.left + br.width / 2 - w / 2));
      this.tip.style.left = `${Math.round(left)}px`;
      this.tip.style.bottom = `${Math.round(pr.height - top + 12)}px`;
    },
  };

  CT.ui = CT.ui || {};
  CT.ui.ctlButton = btn;
  CT.bus.on("app", (st) => btn.render(st));
})();
