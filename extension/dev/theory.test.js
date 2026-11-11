// Test cho extension/src/theory.js. Chạy: node --test extension/dev/theory.test.js
// Dùng thêm result.json thật ở data/cache nếu có (bỏ qua nếu máy không có dữ liệu).
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const T = require("../src/theory.js");

const DATA = process.env.CHORDTRACK_DATA || path.join(__dirname, "..", "..", "data");
const loadResult = (id) => {
  const p = path.join(DATA, "cache", id, "result.json");
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
};
const K = (tonic, mode = "major") => ({ tonic, mode });

test("pcOf và noteName theo giọng", () => {
  assert.equal(T.pcOf("C"), 0);
  assert.equal(T.pcOf("Bb"), 10);
  assert.equal(T.pcOf("F#"), 6);
  assert.equal(T.pcOf("Cb"), 11);
  assert.equal(T.noteName(6, K("D")), "F#");
  assert.equal(T.noteName(10, K("D")), "Bb");      // bậc mượn bVI của D... (Bb = bVI) viết giáng
  assert.equal(T.noteName(3, K("Bb")), "Eb");
  assert.equal(T.noteName(1, K("E")), "C#");
  assert.equal(T.noteName(6, null), "F#");
  assert.equal(T.noteName(1, null), "Db");
});

test("shiftSym: dịch giọng giữ loại hợp âm và bass", () => {
  assert.equal(T.shiftSym("C", 2, K("D")), "D");
  assert.equal(T.shiftSym("Am7/G", 2, K("D")), "Bm7/A");
  assert.equal(T.shiftSym("C/E", 1, K("Db")), "Db/F");
  assert.equal(T.shiftSym("F#m7b5", -1, K("F")), "Fm7b5");
  assert.equal(T.shiftSym("C6/9", 2, K("D")), "D6/9");   // 6/9 không phải hợp âm đảo
  assert.equal(T.shiftSym("N.C.", 3), "N.C.");
  assert.equal(T.shiftSym("?", 3), "?");
  assert.equal(T.shiftSym("G", 0), "G");
  // Cm (giọng C thứ) -> phím trắng (transpose +3): dịch -3, viết theo A thứ
  assert.equal(T.shiftSym("Cm", -3, K("A", "minor")), "Am");
  assert.equal(T.shiftSym("Ab", -3, K("A", "minor")), "F");
  assert.equal(T.shiftSym("Bb", -3, K("A", "minor")), "G");
  // Giọng đích có dấu trong tên: hợp âm viết cùng kiểu dấu (C thứ +3 = Eb thứ, không ra D#m)
  const eb = T.shiftKey(K("C", "minor"), 3);
  assert.equal(eb.tonic, "Eb");
  assert.equal(T.shiftSym("Cm", 3, eb), "Ebm");
  assert.equal(T.shiftSym("Ab", 3, eb), "B");
  assert.equal(T.shiftSym("G7", 3, eb), "Bb7");
});

test("stripBass và simplify", () => {
  assert.equal(T.stripBass("C/G"), "C");
  assert.equal(T.stripBass("Am7/G"), "Am7");
  assert.equal(T.stripBass("C6/9"), "C6/9");
  assert.equal(T.stripBass("I/5"), "I");
  assert.equal(T.stripBass("V7/b7"), "V7");
  assert.equal(T.simplify("Am7/G"), "Am");
  assert.equal(T.simplify("Csus4"), "C");
  assert.equal(T.simplify("Cmaj7"), "C");
  assert.equal(T.simplify("Bm7b5"), "Bdim");
  assert.equal(T.simplify("Bdim7"), "Bdim");
  assert.equal(T.simplify("Fm(maj7)"), "Fm");
  assert.equal(T.simplify("N.C."), "N.C.");
});

test("keyName, signed, parseKeyText", () => {
  assert.equal(T.keyName(K("C", "minor")), "C thứ");
  assert.equal(T.keyName(K("D")), "D trưởng");
  assert.equal(T.signed(3), "+3");
  assert.equal(T.signed(-2), "−2");
  assert.equal(T.signed(0), "±0");
  assert.deepEqual(T.parseKeyText("D minor"), { tonic: "D", mode: "minor" });
});

test("nhịp: vị trí phách, số chỉ nhịp suy ra và result.meter", () => {
  const beats = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
  const downbeats = [1.5, 3.5];
  const { pos } = T.beatPositions(beats, downbeats);
  assert.deepEqual(pos, [0, 0, 1, 2, 3, 4, 1, 2, 3, 4]);
  assert.equal(T.meterOf({ beats, downbeats }), 4);
  assert.equal(T.meterOf({ beats, downbeats, meter: 3 }), 3);
  assert.equal(T.meterOf({ beats, downbeats, meter: { beats_per_bar: 6 } }), 6);
  const song = T.buildSong({ beats, downbeats, duration: 6, chords: [], sections: [] });
  assert.deepEqual(song.beatPos, [3, 4, 1, 2, 3, 4, 1, 2, 3, 4]); // pickup đếm lùi
  assert.deepEqual(song.bar, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
  const waltz = T.buildSong({ beats: [0, 1, 2, 3, 4, 5], downbeats: [0, 3], duration: 6 });
  assert.equal(waltz.meter, 3);
});

test("lastLE tìm nhị phân", () => {
  const a = [0, 1, 2, 3];
  assert.equal(T.lastLE(a, -1), -1);
  assert.equal(T.lastLE(a, 0), 0);
  assert.equal(T.lastLE(a, 2.5), 2);
  assert.equal(T.lastLE(a, 9), 3);
});

const magia = loadResult("magia-f18f50a2");
test("Magia: phím trắng dùng white_key_symbol, vùng giọng, chỗ đổi giọng", { skip: !magia && "không có dữ liệu" }, () => {
  const song = T.buildSong(magia);
  assert.equal(song.meter, 4);
  const white = { mode: "white", transpose: 0 };
  for (const c of song.chords) {
    assert.equal(T.chordLabels(song, c, white).main, c.white_key_symbol);
  }
  // Vùng giọng đầu C thứ (+3), sau 50.99 s là D thứ (+5)
  assert.equal(T.keyAt(song, 10).tonic, "C");
  assert.equal(T.keyAt(song, 60).tonic, "D");
  assert.equal(T.keyAt(song, 60).white_key_transpose, 5);
  const m = T.nextModulation(song, 45);
  // Khớp phách (mặc định): chỗ đổi giọng đi theo mép hợp âm đã kéo về phách gần nhất.
  assert.ok(song.beats.includes(m.time) && Math.abs(m.time - 50.991) < 0.15, `mod ${m.time}`);
  assert.equal(m.transpose_to, 5);
  assert.equal(T.nextModulation(T.buildSong(magia, { snapToBeats: false }), 45).time, 50.991);
  // Bảng cấu trúc ở chế độ phím trắng: vòng hợp âm chỉ còn phím trắng (không có # hay b) với hợp âm trong giọng
  for (const sec of song.sections) {
    const L = T.sectionLabels(song, sec, white);
    assert.equal(L.main.length, sec.progression.length);
  }
  const s1 = T.sectionLabels(song, song.sections[1], white);
  assert.deepEqual(s1.main.slice(0, 3), ["Am", "F", "Gm"]); // Cm Ab Bbm dịch -3
  // Chế độ gốc + transpose
  const c = song.chords.find((x) => x.symbol === "Cm");
  assert.equal(T.chordLabels(song, c, { mode: "orig", transpose: 2 }).main, "Dm");
  assert.equal(T.chordLabels(song, c, { mode: "roman", transpose: 0 }).main, "i");
});

const letItBe = loadResult("QDYfEBY9NM4");
test("Let It Be: C – G – Am – F, ẩn hợp âm đảo", { skip: !letItBe && "không có dữ liệu" }, () => {
  const song = T.buildSong(letItBe);
  assert.equal(song.meter, 4);
  const first = T.sectionLabels(song, song.sections[0], { mode: "orig", transpose: 0 });
  assert.deepEqual(first.main.slice(0, 4), ["C", "G", "Am", "F"]);
  assert.deepEqual(first.sub.slice(0, 4), ["I", "V", "vi", "IV"]);
  const inv = song.chords.find((c) => c.symbol.includes("/"));
  if (inv) {
    const L = T.chordLabels(song, inv, { mode: "orig", transpose: 0, hideInversions: true });
    assert.ok(!L.main.includes("/"));
  }
  const up = T.sectionLabels(song, song.sections[0], { mode: "orig", transpose: 2 });
  assert.deepEqual(up.main.slice(0, 4), ["D", "A", "Bm", "G"]);
});

test("snapChords: kéo mép hợp âm về phách, giữ thứ tự, không mất hợp âm", () => {
  const beats = [0, 1, 2, 3, 4, 5, 6];
  const ch = [
    { symbol: "C", start: 0.1, end: 1.6 },   // 0.1 -> 0, 1.6 -> 2 (mốc chung với G)
    { symbol: "G", start: 1.6, end: 2.3 },   // 2.3 -> 2 thì G dài 0: mốc 2.3 giữ gốc, G còn 2–2.3
    { symbol: "Am", start: 2.3, end: 4.45 }, // 4.45 -> 4
    { symbol: "F", start: 4.45, end: 7.2 },  // 7.2: sau phách cuối hơn nửa phách, giữ nguyên
  ];
  const out = T.snapChords(ch, beats);
  assert.equal(out.length, 4);
  assert.deepEqual(out.map((c) => [c.start, c.end]), [[0, 2], [2, 2.3], [2.3, 4], [4, 7.2]]);
  assert.equal(out[0].start0, 0.1);
  for (let i = 1; i < out.length; i++) assert.ok(out[i].start >= out[i - 1].end - 1e-9);
  for (const c of out) assert.ok(c.end > c.start);
  assert.equal(ch[0].start, 0.1); // không đổi mảng vào
  assert.equal(T.snapChords(ch, [1]), ch); // không đủ phách: giữ nguyên
  const raw = T.buildSong({ beats, downbeats: [0, 4], duration: 8, chords: ch }, { snapToBeats: false });
  assert.equal(raw.chords, ch);
});

test("snapChords trên bài thật: mép ô trùng phách", (t) => {
  const r = loadResult("yGcm81aaTHg");
  if (!r) return t.skip("không có data/cache/yGcm81aaTHg");
  const song = T.buildSong(r);
  const off = (c) => Math.min(...song.beats.map((b) => Math.abs(b - c.start)));
  const inGrid = song.chords.filter((c) => c.start >= song.beats[0] && c.start <= song.beats[song.beats.length - 1]);
  const onBeat = inGrid.filter((c) => off(c) < 1e-6).length;
  assert.equal(song.chords.length, r.chords.length);
  assert.ok(onBeat / inGrid.length > 0.9, `chỉ ${onBeat}/${inGrid.length} hợp âm trùng phách`);
});

/* ---------------- Giai điệu ---------------- */

test("melodyName chế độ Gốc: tên theo giọng, không số quãng tám, dịch tông", () => {
  const orig = { mode: "orig", transpose: 0 };
  assert.equal(T.melodyName(60, K("C"), orig), "C");
  assert.equal(T.melodyName(72, K("C"), orig), "C");          // khác quãng tám: cùng tên
  assert.equal(T.melodyName(66, K("D"), orig), "F#");
  assert.equal(T.melodyName(70, K("D"), orig), "Bb");         // bVI mượn: viết giáng như hợp âm
  assert.equal(T.melodyName(63, K("C", "minor"), orig), "Eb");
  assert.equal(T.melodyName(61, null, orig), "Db");           // không rõ giọng: bảng FLAT_NAMES
  assert.equal(T.melodyName(66, null, orig), "F#");
  // Tông +2: C trưởng -> D trưởng
  assert.equal(T.melodyName(64, K("C"), { mode: "orig", transpose: 2 }), "F#");
  assert.equal(T.melodyName(60, K("C"), { mode: "orig", transpose: -1 }), "B");
  // C thứ +3 = Eb thứ: Ab -> B, G -> Bb (cùng kiểu dấu với hợp âm, xem shiftSym)
  assert.equal(T.melodyName(68, K("C", "minor"), { mode: "orig", transpose: 3 }), "B");
  assert.equal(T.melodyName(67, K("C", "minor"), { mode: "orig", transpose: 3 }), "Bb");
});

test("melodyName chế độ Phím trắng: dịch theo white_key_transpose của vùng giọng", () => {
  const white = { mode: "white", transpose: 5 };              // transpose của chế độ Gốc bị bỏ qua
  const cm = { tonic: "C", mode: "minor", white_key_transpose: 3 };
  assert.equal(T.melodyName(60, cm, white), "A");             // C trong C thứ -> A (A thứ)
  assert.equal(T.melodyName(63, cm, white), "C");             // Eb -> C
  assert.equal(T.melodyName(67, cm, white), "E");
  assert.equal(T.melodyName(71, cm, white), "Ab");            // B (bậc 7 nâng) -> G#, viết Ab như hợp âm (FLAT_NAMES)
  const d = { tonic: "D", mode: "major", white_key_transpose: 2 };
  assert.equal(T.melodyName(66, d, white), "E");              // F# -> E
  assert.equal(T.melodyName(68, d, white), "F#");             // G# (#4) -> F#
  assert.equal(T.melodyName(72, d, white), "Bb");             // C (b7) -> Bb
  // Không có white_key_transpose: không dịch, như hợp âm (chordLabels)
  assert.equal(T.melodyName(69, K("A"), white), "A");
  assert.equal(T.melodyName(62, null, white), "D");
});

test("melodyName chế độ Bậc: số 1–7 theo giọng trưởng (giọng thứ dùng giọng trưởng song song)", () => {
  const roman = { mode: "roman", transpose: 0 };
  assert.deepEqual([60, 62, 64, 65, 67, 69, 71].map((p) => T.melodyName(p, K("C"), roman)), ["1", "2", "3", "4", "5", "6", "7"]);
  assert.equal(T.melodyName(66, K("C"), roman), "#4");
  assert.equal(T.melodyName(70, K("C"), roman), "b7");
  assert.equal(T.melodyName(62, K("D"), roman), "1");
  // A thứ: A = 6, C = 1; C thứ: C = 6, Eb = 1
  assert.equal(T.melodyName(69, K("A", "minor"), roman), "6");
  assert.equal(T.melodyName(60, K("C", "minor"), roman), "6");
  assert.equal(T.melodyName(63, K("C", "minor"), roman), "1");
  // Bậc = phím trắng ở chế độ Phím trắng (C = 1 … B = 7) cho mọi nốt trong giọng
  const cm = { tonic: "C", mode: "minor", white_key_transpose: 3 };
  const LETTER = { C: "1", D: "2", E: "3", F: "4", G: "5", A: "6", B: "7" };
  for (const p of [60, 62, 63, 65, 67, 68, 70]) {
    assert.equal(T.melodyName(p, cm, roman), LETTER[T.melodyName(p, cm, { mode: "white" })]);
  }
});

test("melodyArrow và melodyLabels: mũi tên theo pitch gốc, giọng theo vùng", () => {
  assert.equal(T.melodyArrow(null, 60), "");
  assert.equal(T.melodyArrow(60, 62), "↑");
  assert.equal(T.melodyArrow(62, 60), "↓");
  assert.equal(T.melodyArrow(60, 60), "");
  assert.equal(T.melodyArrow(60, 72), "↑");                   // lên quãng tám: cùng tên nhưng có mũi tên
  const song = T.buildSong({
    duration: 20, beats: [], downbeats: [], chords: [],
    keys: [{ start: 0, end: 10, tonic: "C", mode: "major", white_key_transpose: 0 },
           { start: 10, end: 20, tonic: "D", mode: "major", white_key_transpose: 2 }],
  }, { snapToBeats: false });
  const notes = [
    { start: 1, end: 2, pitch: 64 }, { start: 2, end: 3, pitch: 64 }, { start: 3, end: 4, pitch: 62 },
    { start: 11, end: 12, pitch: 66 }, { start: 12, end: 13, pitch: 78 },
  ];
  const L = T.melodyLabels(song, notes, { mode: "white" });
  assert.deepEqual(L.map((x) => x.name), ["E", "E", "D", "E", "E"]);
  assert.deepEqual(L.map((x) => x.arrow), ["", "", "↓", "↑", "↑"]);
  assert.deepEqual(T.melodyLabels(song, notes, { mode: "orig", transpose: 0 }).map((x) => x.name), ["E", "E", "D", "F#", "F#"]);
  assert.deepEqual(T.melodyLabels(song, notes, { mode: "roman" }).map((x) => x.name), ["3", "3", "2", "3", "3"]);
  // Mũi tên không đổi theo chế độ hay transpose
  assert.deepEqual(T.melodyLabels(song, notes, { mode: "orig", transpose: 4 }).map((x) => x.arrow), ["", "", "↓", "↑", "↑"]);
  assert.deepEqual(T.melodyLabels(null, notes.slice(0, 1), { mode: "orig" }), [{ name: "E", arrow: "" }]);
});

test("melodyNotes, noteIndexAt, noteFit", () => {
  const m = { notes: [{ start: 3, end: 4, pitch: 60 }, { start: 1, end: 2, pitch: 62 }, { start: 5, end: 5, pitch: 1 },
                      null, { start: 6, end: 7 }, { start: 2, end: 2.5, pitch: 64 }] };
  const notes = T.melodyNotes(m);
  assert.deepEqual(notes.map((n) => n.start), [1, 2, 3]);
  assert.deepEqual(T.melodyNotes(null), []);
  assert.deepEqual(T.melodyNotes({}), []);
  assert.equal(T.melodyNotes({ notes: [{ start: 0, end: 1, pitch: 60.6 }] })[0].pitch, 61);   // pitch lẻ: làm tròn
  assert.equal(T.noteIndexAt(notes, 0.5), -1);
  assert.equal(T.noteIndexAt(notes, 1), 0);
  assert.equal(T.noteIndexAt(notes, 2.2), 1);
  assert.equal(T.noteIndexAt(notes, 2.7), -1);                // nghỉ giữa hai nốt
  assert.equal(T.noteIndexAt(notes, 3.5, 1), 2);
  assert.equal(T.noteIndexAt(notes, 1.5, 2), 0);              // hint sai vẫn tìm đúng
  assert.equal(T.noteIndexAt(notes, 9), -1);
  assert.equal(T.noteFit(30, 20, 10), "full");
  assert.equal(T.noteFit(15, 20, 10), "name");
  assert.equal(T.noteFit(5, 20, 10), "bare");
});

const magiaMel = (() => {
  const p = path.join(DATA, "cache", "dev_melody", "devmagia_01.json");
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
})();
test("Magia: dòng nốt giả ở chế độ Phím trắng hầu hết là phím trắng", { skip: !(magia && magiaMel) && "không có dữ liệu" }, () => {
  const song = T.buildSong(magia);
  const notes = T.melodyNotes(magiaMel);
  const L = T.melodyLabels(song, notes, { mode: "white" });
  const white = L.filter((x) => /^[A-G]$/.test(x.name)).length;
  assert.ok(white / L.length > 0.8, `chỉ ${white}/${L.length} nốt là phím trắng`);
  for (let i = 1; i < notes.length; i++) assert.ok(notes[i].start >= notes[i - 1].start);
});

test("biến thể dò giai điệu: melodyVariant, melodyModelOk", () => {
  assert.equal(T.melodyVariant("yourmt3"), "yourmt3");
  assert.equal(T.melodyVariant("combo"), "combo");
  for (const bad of [undefined, null, "", "khac", "toString", "__proto__"]) assert.equal(T.melodyVariant(bad), "yourmt3");
  const combo = { model: "yourmt3>muscriptor_small", notes: [] };
  const y3 = { model: "yourmt3", notes: [] };
  assert.ok(T.melodyModelOk("combo", combo));
  assert.ok(T.melodyModelOk("combo", { notes: [] }));          // kết quả cũ không có trường model
  assert.ok(T.melodyModelOk(undefined, y3));                   // mặc định: chỉ YourMT3+
  assert.ok(!T.melodyModelOk(undefined, combo));
  assert.ok(T.melodyModelOk("yourmt3", y3));
  assert.ok(!T.melodyModelOk("yourmt3", combo));               // server cũ bỏ qua ?model= và trả bản ghép
  assert.ok(!T.melodyModelOk("yourmt3", { notes: [] }));
  assert.ok(!T.melodyModelOk("combo", y3));
  assert.deepEqual(Object.keys(T.MELODY_MODELS), ["yourmt3", "combo"]);
});

test("biến thể tách giọng: melodyKey, melodyParts, melodyModelOk", () => {
  assert.equal(T.melodyKey("yourmt3", false), "yourmt3");
  assert.equal(T.melodyKey("yourmt3", true), "yourmt3_vocals");
  assert.equal(T.melodyKey("combo", true), "combo_vocals");
  assert.equal(T.melodyKey("khac", true), "yourmt3_vocals");
  assert.deepEqual(T.melodyParts("combo_vocals"), { model: "combo", vocals: true });
  assert.deepEqual(T.melodyParts("yourmt3"), { model: "yourmt3", vocals: false });
  assert.deepEqual(T.melodyParts("khac_vocals"), { model: "yourmt3", vocals: false });
  assert.deepEqual(T.melodyParts(undefined), { model: "yourmt3", vocals: false });
  assert.deepEqual(T.MELODY_KEYS, ["yourmt3", "yourmt3_vocals", "combo", "combo_vocals"]);
  const yv = { model: "yourmt3[vocals]", notes: [] };
  const cv = { model: "yourmt3[vocals]>muscriptor_small", notes: [] };
  assert.ok(T.melodyModelOk("yourmt3_vocals", yv) && !T.melodyModelOk("yourmt3", yv));
  assert.ok(T.melodyModelOk("combo_vocals", cv) && !T.melodyModelOk("combo", cv));
  assert.ok(!T.melodyModelOk("combo_vocals", { notes: [] }));     // server cũ bỏ qua vocals
  assert.ok(!T.melodyModelOk("combo_vocals", { model: "yourmt3>muscriptor_small", notes: [] }));
});

test("snapNotes: căn nốt về lưới phách chia 4 / 2, không nốt độ dài 0, không chồng nhau", () => {
  const beats = [0, 1, 2, 3, 4];                     // 60 BPM: lưới 1/16 = 0.25 s
  assert.deepEqual(T.beatSubdivisions(beats, 2), [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]);
  const n = (start, end, pitch = 60) => ({ start, end, pitch, confidence: 0.9 });
  const out = T.snapNotes([n(0.27, 0.71), n(0.74, 1.2), n(1.2, 1.3)], beats, 4);
  assert.deepEqual(out.map((x) => [x.start, x.end]), [[0.25, 0.75], [0.75, 1.25], [1.25, 1.5]]);
  assert.equal(out[0].start0, 0.27);
  assert.equal(out[2].confidence, 0.9);
  // Nốt ngắn rơi vào cùng một điểm lưới: kéo cuối tới điểm kế tiếp.
  assert.deepEqual(T.snapNotes([n(2.01, 2.05)], beats, 4).map((x) => [x.start, x.end]), [[2, 2.25]]);
  // Chia 2: lưới 0.5 s.
  assert.deepEqual(T.snapNotes([n(0.3, 0.8)], beats, 2).map((x) => [x.start, x.end]), [[0.5, 1]]);
  // Nốt dày hơn lưới: không được chồng nhau hay độ dài <= 0 (mốc nào không căn được thì giữ gốc).
  const dense = [];
  for (let t = 0.0; t < 3; t += 0.07) dense.push(n(+t.toFixed(3), +(t + 0.06).toFixed(3), 60 + (dense.length % 5)));
  const d = T.snapNotes(dense, beats, 4);
  assert.equal(d.length, dense.length);
  for (let i = 0; i < d.length; i++) {
    assert.ok(d[i].end > d[i].start, `nốt ${i} độ dài > 0`);
    if (i) assert.ok(d[i].start >= d[i - 1].end - 1e-9, `nốt ${i} không chồng nốt trước`);
  }
  // Ngoài lưới (xa hơn một phách sau phách cuối) giữ nguyên; không có phách thì trả lại như cũ.
  assert.deepEqual(T.snapNotes([n(7.1, 7.3)], beats, 4).map((x) => [x.start, x.end]), [[7.1, 7.3]]);
  const raw = [n(1.1, 1.2)];
  assert.equal(T.snapNotes(raw, [], 4), raw);
  assert.deepEqual(T.snapNotes([], beats), []);
});

test("autoSnapDiv: tự chọn mức chia phách theo tempo và độ dày giai điệu", () => {
  const n = (s, e) => ({ start: s, end: e, pitch: 60 });
  const slow = [0, 1, 2, 3, 4, 5, 6, 7, 8];                       // 60 BPM
  const fast = slow.map((t) => t * 0.4);                           // 150 BPM: ô 1/16 = 0.1 s
  const eighths = Array.from({ length: 16 }, (_, i) => n(i * 0.5, i * 0.5 + 0.45));
  const sixteenths = Array.from({ length: 32 }, (_, i) => n(i * 0.25, i * 0.25 + 0.2));
  assert.equal(T.autoSnapDiv(eighths, slow), 2);                  // giai điệu thưa: 1/8
  assert.equal(T.autoSnapDiv(sixteenths, slow), 4);               // nhiều nốt ngắn: 1/16
  assert.equal(T.autoSnapDiv(sixteenths.map((x) => n(x.start * 0.4, x.end * 0.4)), fast.map((t) => t * 0.6)), 2); // quá nhanh
  assert.equal(T.autoSnapDiv([], slow), 4);
  assert.equal(T.autoSnapDiv(eighths, []), 4);
});

test("melodyAvail / melodyEffective: cách dò theo melody_models của /health", () => {
  const all = { combo: true, yourmt3: true, yourmt3_vocals: true, combo_vocals: true };
  const noMus = { combo: false, yourmt3: true, yourmt3_vocals: true, combo_vocals: false };
  const noSep = { combo: true, yourmt3: true, yourmt3_vocals: false, combo_vocals: false };
  const bare = { combo: false, yourmt3: true, yourmt3_vocals: false, combo_vocals: false };
  const none = { combo: false, yourmt3: false, yourmt3_vocals: false, combo_vocals: false };
  const old = { combo: true, yourmt3: true };                    // server trước khi có tách giọng
  // Chưa biết (server tắt / server cũ không có melody_models): coi như có đủ, luôn tách giọng.
  for (const m of [null, undefined, "x", []]) {
    assert.deepEqual(T.melodyAvail(m), { known: false, any: true, yourmt3: true, combo: true, vocals: true });
    assert.equal(T.melodyEffective("yourmt3", m), "yourmt3_vocals");
    assert.equal(T.melodyEffective("combo", m), "combo_vocals");
  }
  assert.deepEqual(T.melodyAvail(all), { known: true, any: true, yourmt3: true, combo: true, vocals: true });
  assert.equal(T.melodyEffective("yourmt3", all), "yourmt3_vocals");
  assert.equal(T.melodyEffective("combo", all), "combo_vocals");
  assert.equal(T.melodyEffective(undefined, all), "yourmt3_vocals");   // cài đặt lạ: mặc định Bình thường
  // Thiếu MuScriptor: "Kỹ" tắt, đang chọn Kỹ thì dùng Bình thường (vẫn tách giọng).
  assert.equal(T.melodyAvail(noMus).combo, false);
  assert.equal(T.melodyEffective("combo", noMus), "yourmt3_vocals");
  // Thiếu gói tách giọng: bỏ tách giọng, giữ cách dò đã chọn.
  assert.equal(T.melodyAvail(noSep).vocals, false);
  assert.equal(T.melodyEffective("yourmt3", noSep), "yourmt3");
  assert.equal(T.melodyEffective("combo", noSep), "combo");
  assert.equal(T.melodyEffective("combo", bare), "yourmt3");
  assert.equal(T.melodyEffective("combo", old), "combo");
  // Không có gì: any = false, giữ lựa chọn (nút dò bị tắt).
  assert.deepEqual(T.melodyAvail(none), { known: true, any: false, yourmt3: false, combo: false, vocals: false });
  assert.equal(T.melodyEffective("combo", none), "combo_vocals");
  assert.equal(T.melodyAvail({}).any, false);
});

test("tên cách dò cho người dùng: Bình thường | Kỹ, không dùng tên model", () => {
  assert.deepEqual(Object.keys(T.MELODY_MODELS), ["yourmt3", "combo"]);
  assert.equal(T.MELODY_MODELS.yourmt3.short, "Bình thường");
  assert.equal(T.MELODY_MODELS.combo.short, "Kỹ");
  for (const v of Object.values(T.MELODY_MODELS)) assert.ok(!/__/.test(v.tip), v.tip);
});
