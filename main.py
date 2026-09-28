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


# --- Loop Video Principale ---
cap = cv2.VideoCapture("./media/test.mp4")

if not cap.isOpened():
  print("Errore: impossibile aprire il file video.")
  exit()

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # 1. Maschera binaria delle linee
  binary_mask = pipeline_threshold(frame)

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

  # Ridimensionamento e affiancamento a schermo
  preview_orig = cv2.resize(preview_orig, (640, 360))
  roi_visual = cv2.resize(roi_visual, (640, 360))

  combined = np.hstack((preview_orig, roi_visual))
  cv2.imshow("Originale con ROI (sx) vs Maschera Filtrata (dx)", combined)

  if cv2.waitKey(20) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()