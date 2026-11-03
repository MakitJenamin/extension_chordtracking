/* ChordTrack: bộ icon SVG riêng (không dùng emoji hay icon phổ thông), vẽ trong khung 24×24, màu theo currentColor.
 *
 * CT.icons.logo(size)       bàn phím piano cách điệu một màu (phím nghiêng, phím đen tách bằng khe hở). Nút trên player.
 * CT.icons.avatar(size)     avatar của extension (nền đen bo góc, piano hồng đỏ), giống icons/icon*.png. Chữ ChordTrack.
 * CT.icons.metronome(size)  thân metronome và quả lắc (<g class="ct-ic-pend">); panel xoay quả lắc theo phách của bài.
 * CT.icons.keys(size)       ba phím trắng, hai phím đen, nét mảnh.
 *
 * YouTube bật Trusted Types nên dựng bằng createElementNS, không dùng innerHTML.
 */
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, attrs = {}, ...kids) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    for (const k of kids) e.append(k);
    return e;
  };
  const root = (cls, size, ...kids) =>
    s("svg", { class: `ct-ic ${cls}`, viewBox: "0 0 24 24", width: size, height: size, fill: "none", "aria-hidden": "true" }, ...kids);

  // Bàn phím piano cách điệu (cùng hình học với icons/make_icons.py): bốn phím trắng nghiêng như đang lướt,
  // hai "phím đen" tách khỏi phím trắng bằng khe hở (mask). Màu do người gọi chọn.
  let uid = 0;
  const WHITE_X = [3, 7.7, 12.4, 17.1];
  const BLACK_CX = [7.4, 12.1];
  function piano(keyFill, blackFill) {
    const id = `ct-ic-cut-${++uid}`;
    return s("g", { transform: "translate(-2.33 0) skewX(11)" },
      s("defs", {}, s("mask", { id, maskUnits: "userSpaceOnUse", x: -4, y: 0, width: 32, height: 24 },
        s("rect", { x: -4, width: 32, height: 24, fill: "#fff" }),
        ...BLACK_CX.map((c) => s("rect", { x: c - 2.3, y: 1.5, width: 4.6, height: 11.9, rx: 1.9, fill: "#000" })))),
      s("g", { class: "ct-ic-white", fill: keyFill, mask: `url(#${id})` },
        ...WHITE_X.map((x) => s("rect", { x, y: 3.5, width: 4.1, height: 17, rx: 1.5 }))),
      s("g", { class: "ct-ic-head", fill: blackFill },
        ...BLACK_CX.map((c) => s("rect", { x: c - 1.5, y: 3.5, width: 3, height: 9.1, rx: 1.1 }))));
  }

  /** Logo một màu (nút trên player): phím trắng theo currentColor, "phím đen" tô bằng CSS (.ct-ic-head). */
  function logo(size = 24) {
    return root("ct-ic-logo", size, piano("currentColor", "currentColor"));
  }

  /** Avatar của extension (giống icons/icon*.png): nền đen bo góc, phím hồng đỏ chuyển sắc, phím đen trắng sáng. */
  function avatar(size = 18) {
    const id = `ct-av-${++uid}`;
    const lg = (gid, c0, c1) => s("linearGradient", { id: gid, x1: 0, y1: 0, x2: 0, y2: 1 },
      s("stop", { offset: 0, "stop-color": c0 }), s("stop", { offset: 1, "stop-color": c1 }));
    return root("ct-ic-avatar", size,
      s("defs", {}, lg(`${id}-b`, "#343436", "#121213"), lg(`${id}-g`, "#ff4081", "#e80a2c")),
      s("rect", { width: 24, height: 24, rx: 5.4, fill: `url(#${id}-b)` }),
      s("g", { transform: "translate(12 12) scale(0.66) translate(-12.1 -12)" }, piano(`url(#${id}-g)`, "#fff")));
  }

  function metronome(size = 18) {
    return root("ct-ic-metro", size,
      s("path", { d: "M9.3 3.2h5.4a1 1 0 0 1 .97.76l3.86 15.6a1.2 1.2 0 0 1-1.17 1.49H6.64a1.2 1.2 0 0 1-1.17-1.49L9.33 3.96a1 1 0 0 1 .97-.76Z",
        stroke: "currentColor", "stroke-width": 1.7, "stroke-linejoin": "round" }),
      s("path", { d: "M6.6 16.6h10.8", stroke: "currentColor", "stroke-width": 1.7, "stroke-linecap": "round" }),
      s("g", { class: "ct-ic-pend" },
        s("path", { d: "M12 16.6V5.4", stroke: "currentColor", "stroke-width": 1.7, "stroke-linecap": "round" }),
        s("rect", { x: 10, y: 8, width: 4, height: 3, rx: 1, fill: "currentColor" })));
  }

  function keys(size = 18) {
    return root("ct-ic-keys", size,
      s("rect", { x: 2.8, y: 4.8, width: 18.4, height: 14.4, rx: 2.6, stroke: "currentColor", "stroke-width": 1.6 }),
      s("path", { d: "M8.93 13v6.2M15.07 13v6.2", stroke: "currentColor", "stroke-width": 1.4 }),
      s("rect", { x: 7.3, y: 4.8, width: 3.26, height: 8.6, rx: 0.9, fill: "currentColor" }),
      s("rect", { x: 13.44, y: 4.8, width: 3.26, height: 8.6, rx: 0.9, fill: "currentColor" }));
  }

  window.CT = window.CT || {};
  CT.icons = { logo, avatar, metronome, keys };
})();
