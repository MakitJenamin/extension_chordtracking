"""Dò giai điệu chính. Mỗi model có một module `chordtrack/melody/<tên>.py` với hàm

    transcribe(audio_path: Path) -> types.MelodyModelOutput

chạy trên bản mix (không tách giọng). Code chạy GPU phải nằm trong `gpu_lock()`.
Lưu và đọc kết quả bằng `chordtrack.melody.io`.
"""
