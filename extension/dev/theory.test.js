// Test cho extension/src/theory.js. Chạy: node --test extension/dev/
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
