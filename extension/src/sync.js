/* ChordTrack sync: bám theo <video> của YouTube.
 *
 * CT.sync.attach() / detach()   bắt đầu / dừng vòng requestAnimationFrame
 *   - phát "time" {t, rate, paused, ad} mỗi khung hình (ad = true khi đang chạy quảng cáo, lúc đó paused = true)
 *   - phát "seek" {t} khi người dùng tua (sự kiện "seeked" của video)
 *   - phát "fullscreen" {on, container} khi YouTube vào/ra toàn màn hình (container = #movie_player)
 * CT.seekTo(t)                  tua video
 * CT.sync.videoIdFromUrl(url)   videoId của trang /watch?v=..., null nếu không phải trang video
 *
 * Làm mượt: Chrome cập nhật video.currentTime theo đồng hồ media, nhưng có khung hình giá trị chưa đổi.
 * Khi đang phát mà currentTime đứng yên, ngoại suy theo performance.now() * playbackRate (tối đa 0.25 s)
 * để dải hợp âm trượt đều. Ngay khi currentTime đổi thì dùng lại giá trị thật.
 */
(() => {
  const PLAYER_SEL = "#movie_player";
  let raf = 0, video = null, player = null;
  let lastCT = -1, lastPerf = 0, lastFs = null;

  const findPlayer = () => document.querySelector(PLAYER_SEL);
  const findVideo = () =>
    document.querySelector(`${PLAYER_SEL} video.html5-main-video`) || document.querySelector(`${PLAYER_SEL} video`);

  function videoIdFromUrl(href) {
    try {
      const u = new URL(href);
      if (u.pathname !== "/watch") return null;
      const v = u.searchParams.get("v");
      return v && /^[A-Za-z0-9_-]{11}$/.test(v) ? v : null;
    } catch { return null; }
  }

  function onSeeked() { CT.bus.emit("seek", { t: video ? video.currentTime : 0 }); lastCT = -1; }

  function bindVideo(v) {
    if (video === v) return;
    if (video) video.removeEventListener("seeked", onSeeked);
    video = v;
    CT.state.video = v;
    lastCT = -1;
    if (video) video.addEventListener("seeked", onSeeked);
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    tick(now);
  }

  /** Phần thân của một khung hình (tách riêng để harness đo đồng bộ khi tab bị bóp rAF). */
  function tick(now) {
    if (!video || !video.isConnected) {
      bindVideo(findVideo());
      player = findPlayer();
      if (!video) return;
    }
    const ad = !!(player && player.classList.contains("ad-showing"));
    const paused = video.paused || video.ended || ad || video.readyState < 2;
    const rate = video.playbackRate || 1;
    const ct = video.currentTime;
    let t = ct;
    if (paused || ct !== lastCT) {
      lastCT = ct;
      lastPerf = now;
    } else {
      t = ct + Math.min(0.25, ((now - lastPerf) / 1000) * rate);
    }
    CT.bus.emit("time", { t, rate, paused, ad });
  }

  function fullscreenState() {
    const fe = document.fullscreenElement;
    const p = findPlayer();
    return !!(fe && p && (fe === p || fe.contains(p)));
  }
  function onFullscreen() {
    const on = fullscreenState();
    if (on === lastFs) return;
    lastFs = on;
    CT.bus.emit("fullscreen", { on, container: findPlayer() });
  }

  function attach() {
    if (raf) return;
    player = findPlayer();
    bindVideo(findVideo());
    lastFs = null;
    document.addEventListener("fullscreenchange", onFullscreen);
    onFullscreen();
    raf = requestAnimationFrame(frame);
  }

  function detach() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    bindVideo(null);
    document.removeEventListener("fullscreenchange", onFullscreen);
    if (lastFs) CT.bus.emit("fullscreen", { on: false, container: findPlayer() });
    lastFs = null;
  }

  CT.seekTo = (t) => {
    const v = video || findVideo();
    if (!v || !isFinite(t)) return;
    v.currentTime = Math.max(0, Math.min(t, (v.duration || Infinity) - 0.05));
  };

  CT.sync = { attach, detach, tick, videoIdFromUrl, findPlayer, findVideo, get attached() { return !!raf; } };
})();
