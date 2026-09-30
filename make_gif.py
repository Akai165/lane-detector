"""
Genera media/demo.gif da media/output.mp4
- Ritaglia un tratto interessante del video (es. secondi 2-12)
- Scala a larghezza 800px mantenendo l'aspect ratio
- Ottimizza la palette a 128 colori per un file leggero
- Salva in media/demo.gif
"""

import cv2
from PIL import Image

INPUT   = "./media/output.mp4"
OUTPUT  = "./media/demo.gif"
START_S = 3       # secondo di inizio
END_S   = 11      # secondo di fine
EVERY_N = 3       # prendi 1 frame ogni N (riduce dimensione file)
WIDTH   = 640     # larghezza output in pixel
COLORS  = 96      # colori nella palette GIF
DURATION = 100    # ms per frame (10 fps)

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(START_S * fps)
end_frame   = int(END_S   * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames = []
count  = 0

print(f"FPS sorgente: {fps:.1f}")
print(f"Estraggo frame {start_frame}–{end_frame} (ogni {EVERY_N})…")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos > end_frame:
        break
    if count % EVERY_N == 0:
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Ridimensiona
        ratio  = WIDTH / img.width
        height = int(img.height * ratio)
        img    = img.resize((WIDTH, height), Image.LANCZOS)
        # Quantizza la palette
        img    = img.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT)
        frames.append(img)
    count += 1

cap.release()
print(f"Frame raccolti: {len(frames)}")

if frames:
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=DURATION,
        optimize=True,
    )
    import os
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"✅ GIF salvata in '{OUTPUT}'  ({size_mb:.1f} MB)")
else:
    print("❌ Nessun frame estratto — controlla START_S / END_S.")
