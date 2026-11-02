/* Metronome bằng Web Audio, đồng bộ với result.beats/downbeats theo video.currentTime và playbackRate.
 * Phách mạnh cao và to hơn; âm lượng = CT.settings.metronomeVolume; bật/tắt = CT.settings.metronome.
 * Lên lịch trước (look-ahead scheduler), không phát khi video dừng, không phát dồn khi tua.
 * CT.metronome = { start(), stop() } ; tự nghe "time", "seek", "settings", "song:ready".
 * KHUNG: agent ext-extras viết phần thân.
 */
(() => { CT.metronome = { start() {}, stop() {} }; })();
