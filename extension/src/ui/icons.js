/* ChordTrack: bộ icon SVG riêng (không dùng emoji hay icon phổ thông), vẽ trong khung 24×24, màu theo currentColor.
 *
 * CT.icons.logo(size)       hợp âm ba ghi trên khuông (ba đầu nốt chồng nhau, chung đuôi) và tam giác play đỏ:
 *                           hợp âm chạy theo video. Dùng cho nút trên player và chữ ChordTrack trong panel.
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

  function logo(size = 24) {
    // Hợp âm ba ghi trên khuông: ba đầu nốt chồng nhau chung một đuôi; tam giác play đỏ: chạy theo video.
    const head = (cy) => s("ellipse", { cx: 7.4, cy, rx: 3.1, ry: 2.15, transform: `rotate(-22 7.4 ${cy})` });
    return root("ct-ic-logo", size,
      s("g", { class: "ct-ic-notes", fill: "currentColor" },
        head(5.9), head(11.6), head(17.3),
        s("rect", { x: 9.75, y: 1.6, width: 1.6, height: 15.7, rx: 0.8 })),
      s("path", { class: "ct-ic-head", d: "M14.6 8.7a1 1 0 0 1 1.52-.85l5.9 3.6a1 1 0 0 1 0 1.7l-5.9 3.6a1 1 0 0 1-1.52-.85Z" }));
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
  CT.icons = { logo, metronome, keys };
})();
