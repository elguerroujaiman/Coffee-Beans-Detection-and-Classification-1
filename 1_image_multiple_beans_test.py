import cv2
import numpy as np
import tensorflow as tf

# -------- Config --------
IMAGE_PATH = "multiple beans 2.jfif"
CLASSIFIER_MODEL = "coffee_mobile_net_tuned.h5"
CLASS_NAMES = ["Good", "Bad"]

IMG_SIZE = 224
MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 50000
# ------------------------


def preprocess_for_mobilenet(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def main():
    print("Loading MobileNet classifier...")
    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ Image not found")
        return

    output = img.copy()

    # Counters
    good = 0
    bad = 0

    # ---- Stage 1: Classical bean detection ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("⚠️ No bean-like regions detected")
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # ---- Stage 2: Crop + classify ----
    for cnt in contours:
        area = cv2.contourArea(cnt)
        used_boxes = []

        def overlaps(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)
        
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 🚫 Reject thin / strip-like regions
        aspect_ratio = h / float(w)
        if aspect_ratio < 1.2:   # beans are tall-ish
            continue

        # 🚫 Reject very small regions
        if w < 15 or h < 30:
            continue

        bean_crop = img[y:y+h, x:x+w]

        if bean_crop.size == 0:
            continue

        current_box = (x, y, w, h)

        if any(overlaps(current_box, ub) for ub in used_boxes):
            continue

        used_boxes.append(current_box)

        # MobileNet classification
        input_tensor = preprocess_for_mobilenet(bean_crop)
        pred = classifier.predict(input_tensor, verbose=0)

        # Support sigmoid OR softmax
        if pred.shape[-1] == 1:
            confidence = float(pred[0][0])
            cls = 1 if confidence > 0.5 else 0
            confidence = confidence if cls == 1 else 1 - confidence
        else:
            cls = int(np.argmax(pred))
            confidence = float(pred[0][cls])

        label = CLASS_NAMES[cls]

        if label == "Good":
            good += 1
            color = (0, 255, 0)
        else:
            bad += 1
            color = (0, 0, 255)

        # Draw result
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            f"{label} {confidence*100:.1f}%",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ---- Summary ----
    print("\n===== RESULT =====")
    print(f"Good beans: {good}")
    print(f"Bad beans : {bad}")

    cv2.imshow("Bean Detection + Classification", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
