import cv2
import numpy as np


def pipeline_threshold(
    img, s_thresh=(170, 255), sx_thresh=(20, 100), l_thresh=(200, 255)
):
  """Estrae la maschera binaria combinando HLS, LAB e Sobel X."""
  # 1. Spazi colore
  hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
  l_channel = hls[:, :, 1]
  s_channel = hls[:, :, 2]

  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  b_channel = lab[:, :, 2]

  # 2. Sobel X sul canale L
  sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
  abs_sobelx = np.absolute(sobelx)
  max_sobel = np.max(abs_sobelx)

  if max_sobel > 0:
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
  else:
    scaled_sobel = np.zeros_like(l_channel)

  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

  # 3. Soglie colore
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

  l_binary = np.zeros_like(l_channel)
  l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

  b_binary = np.zeros_like(b_channel)
  b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

  # 4. Unione logica
  combined = np.zeros_like(sxbinary)
  combined[
      (sxbinary == 1) | (s_binary == 1) | (l_binary == 1) | (b_binary == 1)
  ] = 1

  return combined


def apply_roi(binary_img):
  """Applica una maschera trapezoidale per isolare la carreggiata ed eliminare cielo,

  alberi e cofano.
  """
  h, w = binary_img.shape[:2]

  # Definiamo i 4 vertici del trapezio in percentuale rispetto alle dimensioni del frame
  # Ordine: [in basso a sx, in alto a sx, in alto a dx, in basso a dx]
  vertices = np.array(
      [[
          (int(w * 0.08), int(h * 0.88)),  # Fondo sx (appena sopra il cofano)
          (int(w * 0.47), int(h * 0.58)),  # Orizzonte sx
          (int(w * 0.53), int(h * 0.58)),  # Orizzonte dx
          (int(w * 0.92), int(h * 0.82)),  # Fondo dx
      ]],
      dtype=np.int32,
  )

  # Maschera nera a canale singolo (tipo uint8 per compatibilità con OpenCV)
  mask = np.zeros_like(binary_img, dtype=np.uint8)

  # Riempiamo il trapezio con il valore 1 (o 255)
  cv2.fillPoly(mask, vertices, 1)

  # Moltiplicazione elemento per elemento (AND logico)
  masked_img = cv2.bitwise_and(binary_img.astype(np.uint8), mask)

  return masked_img, vertices


def get_perspective_transform_matrice(w, h):
    # 4 punti del trapezio calibrati esattamente sulle linee della corsia in rettilineo
    src = np.float32([
        [w * 0.165, h * 0.96],  # Basso sx (~210, 690 px)
        [w * 0.455, h * 0.635], # Alto sx  (~582, 457 px)
        [w * 0.545, h * 0.635], # Alto dx  (~698, 457 px)  <-- Stessa Y di Alto sx!
        [w * 0.865, h * 0.96]   # Basso dx (~1107, 690 px)
    ])

    # Rettangolo di destinazione (proiettato nella vista dall'alto)
    offset = w * 0.25  # Margine del 25% dai bordi per centrare bene la corsia
    dst = np.float32([
        [offset,     h],       # Basso sx
        [offset,     0],       # Alto sx
        [w - offset, 0],       # Alto dx
        [w - offset, h]        # Basso dx
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv, src, dst

def bird_eye_view(binary_img, M):
  h, w = binary_img.shape[:2]
  warped = cv2.warpPerspective(binary_img.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
  return warped




# --- Loop Video Principale ---
cap = cv2.VideoCapture("./media/project_video.mp4")

if not cap.isOpened():
  print("Errore: impossibile aprire il file video.")
  exit()

# Controlla se è disponibile un display (per evitare il crash di Qt in WSL/headless)
import os
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# Se non c'è display, salviamo l'output su file
out = None
OUTPUT_PATH = "./media/output.mp4"
if not HAS_DISPLAY:
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 25.0, (1440, 270))
  print(f"Nessun display trovato (WSL/headless): salvo l'output in '{OUTPUT_PATH}'")


# Calcolo bird eye pov
ret, frame = cap.read()
h, w = frame.shape[:2]
M, Minv, src, dst = get_perspective_transform_matrice(w, h)




frame_count = 0
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame_count += 1

  # 1. Maschera binaria delle linee
  binary_mask = pipeline_threshold(frame)

  # Applico la vista a volo d'uccello
  bev_mask = bird_eye_view(binary_mask, M)
  bev_visual = cv2.cvtColor(bev_mask * 255, cv2.COLOR_GRAY2BGR)

  # 2. Applicazione della ROI poligonale
  roi_mask, vertices = apply_roi(binary_mask)

  # --- Preparazione per visualizzazione ---
  # Visualizziamo i vertici del trapezio in verde sul video originale per tararli
  preview_orig = frame.copy()
  cv2.polylines(
      preview_orig, vertices, isClosed=True, color=(0, 255, 0), thickness=2
  )

  # Convertiamo la ROI da {0, 1} a {0, 255} BGR
  roi_visual = cv2.cvtColor(roi_mask * 255, cv2.COLOR_GRAY2BGR)

  # Ridimensionamento e affiancamento
  preview_orig = cv2.resize(preview_orig, (480, 270))
  roi_visual = cv2.resize(roi_visual, (480, 270))
  bev_visual = cv2.resize(bev_visual, (480, 270))

  combined = np.hstack((preview_orig, roi_visual, bev_visual))

  # Controllo per eventuali macchine senza display, vado a salvare il video 
  # generato nella cartella media
  if HAS_DISPLAY:
    cv2.imshow("Originale con ROI (sx) vs Maschera Filtrata (dx)", combined)
    if cv2.waitKey(20) & 0xFF == ord("q"):
      break
  else:
    out.write(combined)
    if frame_count % 50 == 0:
      print(f"  Frame elaborati: {frame_count}")

cap.release()

if HAS_DISPLAY:
  cv2.destroyAllWindows()
else:
  out.release()
  print(f"Fatto! Video salvato in '{OUTPUT_PATH}'")