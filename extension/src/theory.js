/* ChordTrack theory: dịch giọng theo ký hiệu, đơn giản hóa, ẩn bass, nhãn theo chế độ, nhịp.
 * Hàm thuần, không đụng DOM. CT.theory = {...}; chạy được cả trong Node (extension/dev/theory.test.js).
 *
 * Cách viết tên nốt giống chordtrack/symbols.py (note_name): giọng thăng dùng dấu thăng trừ các bậc
 * mượn bIII, bVI, bVII; giọng giáng dùng dấu giáng; C trưởng/A thứ (không rõ) dùng bảng FLAT_NAMES (giữ F#).
 * Quy ước transpose phím trắng giống chordtrack/transpose.py: transpose = +k thì hợp âm để bấm = gốc dịch -k.
 */
(() => {
  const FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const ALL_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
  // Khác symbols.py một chỗ: Db trưởng (5 dấu giáng) thay vì C# trưởng (7 dấu thăng), vì chỉ dùng khi dịch tông.
  const SHARP_MAJOR_TONICS = new Set([7, 2, 9, 4, 11, 6]);
  const BORROWED_FLAT = new Set([3, 8, 10]);
  const LETTER_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  const NO_CHORD = new Set(["N.C.", "?", "N", "X", ""]);
  const NOTE_RE = /^([A-G])([#b]*)(.*)$/;

  const mod12 = (x) => ((x % 12) + 12) % 12;
  const wrap = (k) => mod12(k + 6) - 6; // về [-6, +5]

  function pcOf(note) {
    const m = NOTE_RE.exec(note);
    if (!m) return null;
    let pc = LETTER_PC[m[1]];
    for (const c of m[2]) pc += c === "#" ? 1 : -1;
    return mod12(pc);
  }

  function sharpKey(key) {
    if (!key || !key.tonic) return null;
    const t = pcOf(key.tonic);
    if (t === null) return null;
    const major = key.mode === "minor" ? mod12(t + 3) : t;
    if (major === 0) return null;
    return SHARP_MAJOR_TONICS.has(major);
  }

  function noteName(pc, key) {
    const sharp = sharpKey(key);
    if (sharp === true) {
      const t = pcOf(key.tonic);
      const major = key.mode === "minor" ? mod12(t + 3) : t;
      return BORROWED_FLAT.has(mod12(pc - major)) ? ALL_FLAT_NAMES[pc] : SHARP_NAMES[pc];
    }
    if (sharp === false) return ALL_FLAT_NAMES[pc];
    return FLAT_NAMES[pc];
  }

  /** Giọng dịch đi k nửa cung (giữ mode). */
  function shiftKey(key, k) {
    if (!key) return null;
    const pc = pcOf(key.tonic);
    if (pc === null) return key;
    // Giọng đích viết tên theo cách thông dụng nhất cho chính nó.
    const t = mod12(pc + k);
    const name = key.mode === "minor"
      ? ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"][t]
      : ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][t];
    return { ...key, tonic: name };
  }

  /** Tách "Am7/G" -> {root:"A", qual:"m7", bass:"G"}. Bass chỉ nhận khi sau "/" là tên nốt (6/9 không phải bass). */
  function parse(sym) {
    if (!sym || NO_CHORD.has(sym)) return null;
    const m = NOTE_RE.exec(sym);
    if (!m) return null;
    let rest = m[3], bass = null;
    const slash = rest.lastIndexOf("/");
    if (slash >= 0 && /^[A-G][#b]*$/.test(rest.slice(slash + 1))) {
      bass = rest.slice(slash + 1);
      rest = rest.slice(0, slash);
    }
    return { root: m[1] + m[2], qual: rest, bass };
  }

  function format(p) { return p.root + p.qual + (p.bass ? "/" + p.bass : ""); }

  /** Dịch ký hiệu hợp âm k nửa cung, viết tên nốt theo `targetKey` (giọng sau khi dịch). */
  function shiftSym(sym, k, targetKey = null) {
    const p = parse(sym);
    if (!p) return sym;
    if (!k && !targetKey) return sym;
    const name = (n) => noteName(mod12(pcOf(n) + k), targetKey);
    return format({ root: name(p.root), qual: p.qual, bass: p.bass ? name(p.bass) : null });
  }

  /** Bỏ nốt bass của hợp âm đảo: "C/G" -> "C", bậc La Mã "I/5" -> "I". */
  function stripBass(sym) {
    if (!sym) return sym;
    const p = parse(sym);
    if (p) return p.root + p.qual;
    return sym.replace(/\/[#b]*\d+$/, ""); // bậc La Mã
  }

  /** Đơn giản hóa về hợp âm ba: trưởng, thứ hoặc giảm. "Am7/G" -> "Am", "Csus4" -> "C", "Bm7b5" -> "Bdim". */
  function simplify(sym) {
    const p = parse(sym);
    if (!p) return sym;
    const q = p.qual;
    let triad = "";
    if (/^(dim|m7b5|°)/.test(q)) triad = "dim";
    else if (/^m(?!aj)/.test(q)) triad = "m";
    return p.root + triad;
  }

  const MODE_VI = { major: "trưởng", minor: "thứ" };
  function keyName(key) {
    if (!key) return "–";
    return `${key.tonic} ${MODE_VI[key.mode] || key.mode}`;
  }
  /** "C minor" (dạng trong modulations) -> {tonic, mode}. */
  function parseKeyText(s) {
    const [tonic, mode] = String(s || "").split(" ");
    return { tonic, mode: mode || "major" };
  }
  const signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v);

  /* ---------------- Tìm kiếm theo thời gian ---------------- */

  /** Chỉ số phần tử cuối cùng có get(arr[i]) <= t (arr đã sắp xếp), -1 nếu không có. */
  function lastLE(arr, t, get = (x) => x) {
    let lo = 0, hi = arr.length - 1, ans = -1;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (get(arr[m]) <= t) { ans = m; lo = m + 1; } else hi = m - 1;
    }
    return ans;
  }

  /* ---------------- Nhịp ---------------- */

  /** Vị trí trong ô nhịp của mỗi phách (1 = phách mạnh; 0 = trước phách mạnh đầu tiên chưa rõ). */
  function beatPositions(beats, downbeats) {
    const out = new Array(beats.length);
    let j = 0, p = 0;
    const eps = 0.02;
    for (let i = 0; i < beats.length; i++) {
      while (j < downbeats.length && downbeats[j] < beats[i] - eps) j++;
      if (j < downbeats.length && Math.abs(downbeats[j] - beats[i]) <= eps) p = 1;
      else p = p ? p + 1 : 0;
      out[i] = p;
    }
    // Các phách trước phách mạnh đầu tiên: đếm lùi để chấm phách vẫn chạy (pickup).
    const first = out.indexOf(1);
    return { pos: out, first };
  }

  /** Số phách mỗi ô nhịp: result.meter nếu có, không thì lấy độ dài ô nhịp phổ biến nhất. */
  function meterOf(result, posInfo) {
    const m = result && result.meter;
    if (typeof m === "number" && m > 0) return Math.round(m);
    if (m && typeof m === "object") {
      const v = m.beats_per_bar ?? m.numerator ?? m.beats;
      if (typeof v === "number" && v > 0) return Math.round(v);
    }
    const pos = posInfo ? posInfo.pos : beatPositions(result.beats || [], result.downbeats || []).pos;
    const cnt = {};
    for (let i = 1; i < pos.length; i++) {
      if (pos[i] === 1 && pos[i - 1] > 0) cnt[pos[i - 1]] = (cnt[pos[i - 1]] || 0) + 1;
    }
    const best = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0];
    return best ? +best[0] : 4;
  }

  /* ---------------- Mô hình bài hát ---------------- */

  /** Chuẩn bị dữ liệu dùng nhiều lần cho một result. */
  function buildSong(result) {
    const beats = result.beats || [];
    const downbeats = result.downbeats || [];
    const posInfo = beatPositions(beats, downbeats);
    const meter = meterOf(result, posInfo);
    // Phách trước phách mạnh đầu tiên: đặt vị trí đếm lùi (ví dụ 3, 4 rồi tới 1).
    const pos = posInfo.pos.slice();
    if (posInfo.first > 0) {
      for (let i = posInfo.first - 1, p = meter; i >= 0; i--, p--) pos[i] = (((p - 1) % meter) + meter) % meter + 1;
    }
    const bar = new Array(beats.length);
    let n = 0;
    for (let i = 0; i < beats.length; i++) { if (pos[i] === 1) n++; bar[i] = n; }
    const keys = (result.keys && result.keys.length) ? result.keys
      : (result.key ? [{ start: 0, end: result.duration, ...result.key,
                         white_key_transpose: whiteOffset(result.key) }] : []);
    return {
      result, beats, downbeats, beatPos: pos, bar, meter, keys,
      chords: result.chords || [], sections: result.sections || [],
      modulations: result.modulations || [], duration: result.duration || 0,
    };
  }

  function whiteOffset(key) {
    const t = pcOf(key.tonic);
    return wrap(t - (key.mode === "minor" ? 9 : 0));
  }

  function keyIndexAt(song, t) {
    const i = lastLE(song.keys, t, (k) => k.start);
    return Math.max(0, i);
  }
  const keyAt = (song, t) => song.keys[keyIndexAt(song, t)] || song.result.key || null;

  function chordKey(song, c) {
    return (c.key_index != null && song.keys[c.key_index]) || song.result.key || null;
  }

  /** Giọng của một chỗ đổi giọng sắp tới (thời điểm > t), hoặc null. */
  function nextModulation(song, t) {
    for (const m of song.modulations) if (m.time > t) return m;
    return null;
  }

  /* ---------------- Nhãn theo chế độ ---------------- */

  /**
   * Nhãn chính và nhãn phụ của một hợp âm.
   * settings: {mode: "orig"|"white"|"roman", transpose, hideInversions, simplify}
   */
  function chordLabels(song, c, s) {
    const key = chordKey(song, c);
    let main, sub;
    if (s.mode === "white") {
      main = c.white_key_symbol
        ?? shiftSym(c.symbol, -((key && key.white_key_transpose) || 0), key && { tonic: key.mode === "minor" ? "A" : "C", mode: key.mode });
      sub = c.roman || "";
    } else if (s.mode === "roman") {
      main = c.roman || c.symbol;
      sub = origSymbol(c.symbol, key, s.transpose);
    } else {
      main = origSymbol(c.symbol, key, s.transpose);
      sub = c.roman || "";
    }
    if (s.hideInversions) { main = stripBass(main); sub = stripBass(sub); }
    if (s.simplify && s.mode !== "roman") main = simplify(main);
    return { main, sub };
  }

  function origSymbol(sym, key, k) {
    k = k | 0;
    if (!k) return sym;
    return shiftSym(sym, k, shiftKey(key, k));
  }

  /** Vòng hợp âm của một đoạn theo chế độ: {main: [..], sub: [..], key, white}. */
  function sectionLabels(song, sec, s) {
    const key = (sec.key_index != null && song.keys[sec.key_index]) || keyAt(song, sec.start + 0.01);
    const w = (key && key.white_key_transpose) || 0;
    const roman = sec.roman || [];
    let main, sub;
    if (s.mode === "white") {
      const ref = key && { tonic: key.mode === "minor" ? "A" : "C", mode: key.mode };
      main = sec.progression.map((x) => shiftSym(x, -w, ref));
      sub = roman;
    } else if (s.mode === "roman") {
      main = roman.length ? roman : sec.progression;
      sub = sec.progression.map((x) => origSymbol(x, key, s.transpose));
    } else {
      main = sec.progression.map((x) => origSymbol(x, key, s.transpose));
      sub = roman;
    }
    if (s.hideInversions) { main = main.map(stripBass); sub = sub.map(stripBass); }
    if (s.simplify && s.mode !== "roman") main = main.map(simplify);
    return { main, sub, key, white: w };
  }

  const theory = {
    FLAT_NAMES, pcOf, noteName, shiftKey, shiftSym, stripBass, simplify, parse,
    keyName, parseKeyText, signed, wrap, lastLE, beatPositions, meterOf, buildSong,
    keyIndexAt, keyAt, chordKey, nextModulation, chordLabels, sectionLabels, whiteOffset,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = theory;
  else window.CT.theory = theory;
})();
