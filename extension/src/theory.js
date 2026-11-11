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
    // Tên giọng có dấu thì theo dấu đó (Eb thứ viết giáng, D# thứ viết thăng), để hợp âm khớp tên giọng.
    if (key.tonic.length > 1) return key.tonic[1] === "#";
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
  /* Phím cần bấm khi giữ nút Transpose của đàn (đàn của người dùng: giữ Transpose rồi bấm phím tên tông
   * trưởng; mức dịch −6..+5). Tên suy thẳng từ mức dịch: 0 C, +1 C#, +2 D, +3 D#, +4 E, +5 F,
   * −6 F#, −5 G, −4 G#, −3 A, −2 A#, −1 B. Giọng thứ dùng tông trưởng song song (Cm → Eb → bấm D#). */
  const TRANSPOSE_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const transposeKey = (w) => TRANSPOSE_KEYS[(((w | 0) % 12) + 12) % 12];
  /** "D# (+3)" */
  const transposeLabel = (w) => `${transposeKey(w)} (${signed(w | 0)})`;

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

  /** Kéo mốc đầu/cuối hợp âm về phách gần nhất (chỉ để hiển thị và tô sáng).
   * Model hợp âm chạy theo khung ~0,093 s, không theo phách, nên mép ô hay lệch vạch phách 0,1–0,35 s.
   * - Mỗi mốc (đầu hoặc cuối hợp âm; mốc dùng chung giữa hai hợp âm liền nhau) về phách gần nhất nếu cách
   *   không quá nửa khoảng phách tại đó; ngoài lưới phách (trước phách đầu, sau phách cuối) giữ nguyên.
   * - Giữ thứ tự: mốc nào sau khi kéo không còn lớn hơn mốc trước ít nhất MIN_LEN thì trả các mốc liên quan
   *   về giá trị gốc, nên không hợp âm nào bị mất (hợp âm ngắn hơn một phách vẫn hiện).
   * Trả mảng hợp âm mới (bản sao nông, thêm start0/end0 là mốc gốc); không đổi mảng vào. */
  function snapChords(chords, beats) {
    if (!chords || !chords.length || !beats || beats.length < 2) return chords || [];
    const MIN_LEN = 0.05;
    const nearest = (t) => {
      const k = lastLE(beats, t);                 // beats[k] <= t < beats[k + 1]
      const n = beats.length;
      if (k < 0) return beats[0] - t <= (beats[1] - beats[0]) / 2 ? beats[0] : t;
      if (k >= n - 1) return t - beats[n - 1] <= (beats[n - 1] - beats[n - 2]) / 2 ? beats[n - 1] : t;
      return t - beats[k] <= beats[k + 1] - t ? beats[k] : beats[k + 1];
    };
    const orig = [...new Set(chords.flatMap((c) => [c.start, c.end]))].sort((a, b) => a - b);
    const snap = orig.map(nearest);
    for (let k = 1; k < snap.length; k++) {
      if (snap[k] >= snap[k - 1] + MIN_LEN) continue;
      snap[k] = orig[k];
      for (let j = k; j > 0 && snap[j] < snap[j - 1] + MIN_LEN; j--) { snap[j - 1] = orig[j - 1]; snap[j] = orig[j]; }
    }
    const map = new Map(orig.map((t, i) => [t, snap[i]]));
    return chords.map((c) => ({ ...c, start: map.get(c.start), end: map.get(c.end), start0: c.start, end0: c.end }));
  }

  /** Chuẩn bị dữ liệu dùng nhiều lần cho một result.
   * opts.snapToBeats: kéo mép hợp âm về phách (snapChords), mặc định bật. */
  function buildSong(result, opts = {}) {
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
    const raw = result.chords || [];
    const snap = opts.snapToBeats !== false;
    const chords = snap ? snapChords(raw, beats) : raw;
    // Chỗ đổi giọng và mốc vùng giọng trùng mép một hợp âm (theo mốc gốc) thì đi theo mép đã kéo về phách,
    // để vạch vàng, cảnh báo "sắp đổi giọng" và giọng hiện tại khớp với ô hợp âm.
    let follow = (t) => t;
    if (snap && chords !== raw) {
      const m = new Map();
      for (const c of chords) { m.set(c.start0, c.start); m.set(c.end0, c.end); }
      const origs = [...m.keys()].sort((a, b) => a - b);
      follow = (t) => {
        const i = lastLE(origs, t);
        let best = t, d = 0.15;
        for (const j of [i, i + 1]) if (j >= 0 && j < origs.length && Math.abs(origs[j] - t) <= d) { d = Math.abs(origs[j] - t); best = m.get(origs[j]); }
        return best;
      };
    }
    return {
      result, beats, downbeats, beatPos: pos, bar, meter,
      keys: snap ? keys.map((k, i) => ({ ...k, start: i ? follow(k.start) : k.start, end: follow(k.end) })) : keys,
      chords,
      sections: result.sections || [],
      modulations: snap ? (result.modulations || []).map((x) => ({ ...x, time: follow(x.time) })) : (result.modulations || []),
      duration: result.duration || 0,
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

  /* ---------------- Giai điệu (docs/melody_api.md) ---------------- */

  /* Bậc của nốt kiểu số (1–7, "do di động"): 1 là chủ âm của giọng trưởng; giọng thứ dùng giọng trưởng
   * song song (A thứ: A = 6), nên số trùng đúng phím trắng của chế độ Phím trắng (C = 1, D = 2 … B = 7)
   * và nốt trong giọng không bao giờ cần dấu. Nốt ngoài giọng: b2 b3 #4 b6 b7. */
  const DEGREES = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"];
  const majorTonic = (key) => {
    const t = key ? pcOf(key.tonic) : null;
    if (t === null) return 0;
    return key.mode === "minor" ? mod12(t + 3) : t;
  };

  /** Tên một nốt giai điệu (số MIDI `pitch`) trong giọng `key` theo chế độ, không có số quãng tám.
   * - "orig": tên nốt của pitch + transpose, viết theo giọng sau khi dịch (như hợp âm).
   * - "white": dịch -white_key_transpose của vùng giọng (như hợp âm; thiếu thì 0), viết theo C trưởng / A thứ.
   * - "roman": bậc kiểu số 1–7 (xem DEGREES); bằng đúng phím trắng phải bấm ở chế độ Phím trắng. */
  function melodyName(pitch, key, s) {
    const mode = s && s.mode;
    if (mode === "white") {
      const w = (key && key.white_key_transpose) || 0;   // như chordLabels (buildSong đã điền cho giọng đơn)
      return noteName(mod12(pitch - w), key && { tonic: key.mode === "minor" ? "A" : "C", mode: key.mode });
    }
    if (mode === "roman") return DEGREES[mod12(pitch - majorTonic(key))];
    const k = (s && s.transpose) | 0;
    return noteName(mod12(pitch + k), k ? shiftKey(key, k) : key);
  }

  /** Mũi tên so với nốt trước: cao hơn "↑", thấp hơn "↓", bằng nhau hoặc không có nốt trước "".
   * So pitch gốc (cả quãng tám), nên dịch giọng hay đổi vùng giọng không đổi mũi tên. */
  function melodyArrow(prev, pitch) {
    if (prev == null || pitch == null) return "";
    return pitch > prev ? "↑" : pitch < prev ? "↓" : "";
  }

  /** Nhãn mọi nốt: [{name, arrow}], giọng của mỗi nốt là vùng giọng tại lúc nốt bắt đầu. */
  function melodyLabels(song, notes, s) {
    const out = new Array(notes.length);
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const key = song ? keyAt(song, n.start) : null;
      out[i] = { name: melodyName(n.pitch, key, s), arrow: melodyArrow(i ? notes[i - 1].pitch : null, n.pitch) };
    }
    return out;
  }

  /** Lọc và sắp các nốt hợp lệ của một <Melody> (bỏ nốt hỏng, sắp theo start). Trả [] nếu không có. */
  function melodyNotes(melody) {
    const raw = melody && Array.isArray(melody.notes) ? melody.notes : [];
    return raw.filter((n) => n && Number.isFinite(n.start) && Number.isFinite(n.end) && n.end > n.start
      && Number.isFinite(n.pitch)).map((n) => (Number.isInteger(n.pitch) ? n : { ...n, pitch: Math.round(n.pitch) }))
      .sort((a, b) => a.start - b.start);
  }

  /** Lưới phách chia đều mỗi phách thành `div` phần (div 4 = nốt 1/16 ở nhịp x/4), kéo dài một phách ở hai đầu bài. */
  function beatSubdivisions(beats, div = 4) {
    const b = [...new Set((beats || []).filter(Number.isFinite))].sort((x, y) => x - y);
    if (b.length < 2) return b;
    const d = Math.max(1, Math.round(div) || 1);
    const ext = [b[0] - (b[1] - b[0]), ...b, b[b.length - 1] + (b[b.length - 1] - b[b.length - 2])];
    const out = [];
    for (let i = 0; i + 1 < ext.length; i++) {
      for (let f = 0; f < d; f++) out.push(ext[i] + (ext[i + 1] - ext[i]) * f / d);
    }
    out.push(ext[ext.length - 1]);
    return out;
  }

  /** Căn đầu và cuối mỗi nốt (đã sắp, không chồng nhau, như melodyNotes) về điểm gần nhất của lưới phách chia `div`.
   * Chỉ để hiển thị; giữ start0/end0 gốc. Nốt bị căn thành độ dài 0 thì kéo cuối tới điểm lưới kế tiếp; chỗ nào
   * căn xong mà nốt quá ngắn (< 30 ms, hoặc ngắn hơn bản gốc nếu gốc đã ngắn hơn) hay chồng lên nốt trước thì
   * trả các mốc đó về thời điểm gốc. Ngoài phạm vi lưới (trước phách đầu / sau phách cuối hơn một phách) giữ nguyên. */
  function snapNotes(notes, beats, div = 4) {
    if (!notes || !notes.length) return notes || [];
    const g = beatSubdivisions(beats, div);
    if (g.length < 2) return notes;
    const MIN = 0.03;
    const nearest = (t) => {
      const k = lastLE(g, t);
      if (k < 0 || k >= g.length - 1) return t;
      return t - g[k] <= g[k + 1] - t ? g[k] : g[k + 1];
    };
    const next = (t) => { const k = lastLE(g, t + 1e-6); return k + 1 < g.length ? g[k + 1] : null; };
    const b = notes.flatMap((n) => [n.start, n.end]);   // mốc gốc: s0 e0 s1 e1 ...
    const q = b.map(nearest);
    for (let k = 1; k < q.length; k += 2) {
      if (q[k] > q[k - 1]) continue;
      const nx = next(q[k - 1]);
      q[k] = nx == null ? b[k] : nx;
    }
    // Khoảng tối thiểu giữa mốc k-1 và k: độ dài nốt (k lẻ) hoặc khe giữa hai nốt (k chẵn, được bằng 0).
    const need = (k) => (k % 2 ? Math.min(MIN, b[k] - b[k - 1]) : Math.min(0, b[k] - b[k - 1]));
    const bad = (k) => q[k] < q[k - 1] + need(k) - 1e-9;
    for (let k = 1; k < q.length; k++) {
      if (!bad(k)) continue;
      q[k] = b[k];
      for (let j = k; j > 0 && bad(j); j--) { q[j - 1] = b[j - 1]; q[j] = b[j]; }
    }
    return notes.map((n, i) => ({ ...n, start: q[2 * i], end: q[2 * i + 1], start0: n.start, end0: n.end }));
  }
  /** Tự chọn mức chia phách khi căn nốt: 4 (nốt 1/16 ở nhịp x/4) hoặc 2 (nốt 1/8), theo tempo và độ dày giai điệu.
   * Bài nhanh (ô 1/16 ngắn hơn 0,1 s) luôn chia 2. Còn lại chia 4 khi từ 20% số nốt trở lên ngắn hơn 3/8 phách
   * (tức giống nốt 1/16 hơn nốt 1/8), ngược lại chia 2 cho dòng nốt gọn, dễ đọc. */
  function autoSnapDiv(notes, beats) {
    const b = (beats || []).filter(Number.isFinite);
    if (b.length < 2 || !notes || !notes.length) return 4;
    const d = [];
    for (let i = 1; i < b.length; i++) if (b[i] > b[i - 1]) d.push(b[i] - b[i - 1]);
    if (!d.length) return 4;
    d.sort((x, y) => x - y);
    const P = d[d.length >> 1];
    if (P / 4 < 0.1) return 2;
    const short = notes.filter((n) => n.end - n.start < 0.375 * P).length;
    return short >= 0.2 * notes.length ? 4 : 2;
  }


  /** Chỉ số nốt đang phát tại t (start <= t < end), -1 nếu đang nghỉ. `hint`: chỉ số lần trước (thử trước). */
  function noteIndexAt(notes, t, hint = -1) {
    if (hint >= 0 && hint < notes.length && t >= notes[hint].start && t < notes[hint].end) return hint;
    const n1 = hint + 1;
    if (n1 > 0 && n1 < notes.length && t >= notes[n1].start && t < notes[n1].end) return n1;
    const j = lastLE(notes, t, (n) => n.start);
    return j >= 0 && t < notes[j].end ? j : -1;
  }

  /** Cách hiện nhãn mỗi nốt khi dòng nốt dày. room[i]: số px từ đầu nốt i tới đầu nốt kế (khe nhãn được dùng),
   * full[i]/name[i]: bề rộng nhãn đầy đủ / chỉ tên. Trả "full" | "name" (bỏ mũi tên) | "bare" (chỉ còn thanh). */
  function noteFit(room, full, name) {
    if (full <= room) return "full";
    if (name <= room) return "name";
    return "bare";
  }

  /* Cách dò giai điệu (`model` của POST /melody, docs/melody_api.md). `model`: giá trị <Melody>.model. Tên hiện cho
   * người dùng là "Bình thường" / "Kỹ" (không dùng tên model, người dùng chốt 2026-11-11); khóa giữ nguyên. */
  // Thứ tự khóa = thứ tự nút trên panel. Mặc định "yourmt3" (người dùng chốt 2026-11-11: nghe thấy tốt hơn).
  const MELODY_MODELS = {
    // `time`: thời gian dò lần đầu một bài 4 phút (gồm tách giọng), đo trên RTX 3060 Ti: Let It Be 243 s, Bình thường 117 s,
    // Kỹ 182 s (2026-11-11, xem README).
    yourmt3: { model: "yourmt3", short: "Bình thường", label: "Bình thường", time: "khoảng 2 phút",
      tip: "Khoảng 2 phút cho bài 4 phút (lần đầu, gồm tách giọng). Tốt nhất với bài có lời" },
    combo: { model: "yourmt3>muscriptor_small", short: "Kỹ", label: "Kỹ", time: "khoảng 3 phút",
      tip: "Chậm hơn khoảng 1 phút (khoảng 3 phút cho bài 4 phút). Bắt giai điệu nhạc cụ ở đoạn không lời tốt hơn (dùng thêm MuScriptor, cần đăng nhập Hugging Face)" },
  };
  const MELODY_DEFAULT = "yourmt3";
  // Biến thể của kết quả cũ không có trường `model` (trước khi có tùy chọn, server chỉ có bản ghép).
  const MELODY_LEGACY = "combo";

  /** Biến thể hợp lệ từ cài đặt (giá trị lạ hoặc thiếu: mặc định "yourmt3"). */
  function melodyVariant(v) { return Object.prototype.hasOwnProperty.call(MELODY_MODELS, v) ? v : MELODY_DEFAULT; }

  /* Tách giọng (cờ `vocals` của POST /melody): mỗi cách dò có thêm bản tách giọng; extension luôn dùng bản tách giọng
   * khi server có (melodyEffective). Khóa đầy đủ của một biến thể = khóa trạng thái trong content.js = khóa
   * `melody_models` của /health. */
  const MELODY_VOCALS_RESULT = { yourmt3: "yourmt3[vocals]", combo: "yourmt3[vocals]>muscriptor_small" };
  /** Khóa biến thể từ (model, vocals): "yourmt3" | "combo" | "yourmt3_vocals" | "combo_vocals". */
  function melodyKey(model, vocals) { return melodyVariant(model) + (vocals ? "_vocals" : ""); }
  /** {model, vocals} của một khóa biến thể (khóa lạ: model mặc định, không tách giọng). */
  function melodyParts(key) {
    const s = String(key || "");
    const vocals = s.endsWith("_vocals");
    const model = vocals ? s.slice(0, -7) : s;
    return Object.prototype.hasOwnProperty.call(MELODY_MODELS, model) ? { model, vocals } : { model: MELODY_DEFAULT, vocals: false };
  }
  const MELODY_KEYS = Object.keys(MELODY_MODELS).flatMap((m) => [m, m + "_vocals"]);

  /** `melody_models` của GET /health có dùng được không (object). null/khác: chưa biết (server tắt, server cũ). */
  const knownModels = (models) => !!models && typeof models === "object" && !Array.isArray(models);

  /** Cách dò nào server chạy được, theo `melody_models` của GET /health. Chưa biết (`models` null) thì coi như có đủ
   * (giữ cách cũ: gửi rồi để server báo lỗi). {known, any, yourmt3, combo, vocals}; `yourmt3`/`combo`: có bản nào
   * (tách giọng hoặc không) của cách đó; `vocals`: có bản tách giọng nào. */
  function melodyAvail(models) {
    if (!knownModels(models)) return { known: false, any: true, yourmt3: true, combo: true, vocals: true };
    const has = (k) => models[k] === true;
    const base = (m) => has(m) || has(m + "_vocals");
    const yourmt3 = base("yourmt3"), combo = base("combo");
    return { known: true, any: yourmt3 || combo, yourmt3, combo, vocals: has("yourmt3_vocals") || has("combo_vocals") };
  }

  /** Câu hướng dẫn (tiếng Việt) khi server chưa chạy được biến thể `key` (theo `melody_models`). Cùng ý với lỗi 409
   * của POST /melody (chordtrack/server.py). */
  const MELODY_SETUP_HINT = "Chạy lại cài đặt: powershell -ExecutionPolicy Bypass -File scripts\\setup.ps1, rồi khởi động lại server.";
  const MELODY_KY_HINT = "Cách dò Kỹ cần MuScriptor small (tài khoản Hugging Face, chỉ dùng phi thương mại): tạo token Read ở huggingface.co/settings/tokens, bấm đồng ý điều khoản ở huggingface.co/MuScriptor/muscriptor-small, chạy .venv\\Scripts\\hf.exe auth login rồi .venv\\Scripts\\python.exe scripts\\fetch_models.py --muscriptor và khởi động lại server. Xem README, mục Dò giai điệu.";
  function melodyMissingText(key, models) {
    const a = melodyAvail(models);
    const p = melodyParts(key);
    if (!a.yourmt3) return "Server chưa cài dò giai điệu. " + MELODY_SETUP_HINT;
    if (p.model === "combo" && !a.combo) return MELODY_KY_HINT;
    if (p.vocals) return "Server chưa cài phần tách giọng. " + MELODY_SETUP_HINT;
    return "Server chưa cài đủ model cho cách dò này. " + MELODY_SETUP_HINT;
  }

  /** Khóa biến thể sẽ dò/hiện: cách dò người dùng chọn (`choice` = CT.settings.melodyModel), luôn tách giọng. Server
   * thiếu thì lùi mà không đổi lựa chọn đã lưu: thiếu gói tách giọng thì bỏ tách giọng, thiếu MuScriptor ("Kỹ") thì
   * dùng "Bình thường". Chưa biết server có gì, hoặc server không có cách nào: giữ đúng lựa chọn (nút dò bị tắt ở panel). */
  function melodyEffective(choice, models) {
    const want = melodyVariant(choice);
    if (!knownModels(models)) return want + "_vocals";
    const order = want === "combo" ? ["combo", "yourmt3"] : ["yourmt3"];
    for (const b of order) {
      if (models[b + "_vocals"] === true) return b + "_vocals";
      if (models[b] === true) return b;
    }
    return want + "_vocals";
  }

  /** <Melody> có đúng là của biến thể `variant` (khóa đầy đủ) không. Server cũ bỏ qua `?model=`/`vocals` và trả bản
   * ghép, nên các bản khác phải có đúng model; bản ghép nhận cả kết quả cũ không có trường model. */
  function melodyModelOk(variant, melody) {
    const p = melodyParts(variant);
    const want = p.vocals ? MELODY_VOCALS_RESULT[p.model] : MELODY_MODELS[p.model].model;
    const got = melody && melody.model;
    return got === want || (!p.vocals && p.model === MELODY_LEGACY && !got);
  }

  const theory = {
    FLAT_NAMES, pcOf, noteName, shiftKey, shiftSym, stripBass, simplify, parse,
    keyName, parseKeyText, signed, transposeKey, transposeLabel, wrap, lastLE, beatPositions, meterOf, buildSong, snapChords,
    keyIndexAt, keyAt, chordKey, nextModulation, chordLabels, sectionLabels, whiteOffset,
    DEGREES, melodyName, melodyArrow, melodyLabels, melodyNotes, noteIndexAt, noteFit, beatSubdivisions, snapNotes, autoSnapDiv,
    MELODY_MODELS, MELODY_DEFAULT, melodyVariant, melodyModelOk, melodyKey, melodyParts, MELODY_KEYS,
    MELODY_VOCALS_RESULT, melodyAvail, melodyEffective, melodyMissingText, MELODY_KY_HINT, MELODY_SETUP_HINT,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = theory;
  else window.CT.theory = theory;
})();
