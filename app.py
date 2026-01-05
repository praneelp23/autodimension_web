from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import math
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_IMAGE = "static/output.png"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# Utility
# -------------------------------------------------
def is_duplicate(val, arr, tol=5):
    return any(abs(val - a) < tol for a in arr)

def line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def endpoints_close(l1, l2, tol=15):
    pts1 = [(l1[0], l1[1]), (l1[2], l1[3])]
    pts2 = [(l2[0], l2[1]), (l2[2], l2[3])]
    for p1 in pts1:
        for p2 in pts2:
            if abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol:
                return p1
    return None

def angle_between(l1, l2):
    v1 = np.array([l1[2] - l1[0], l1[3] - l1[1]])
    v2 = np.array([l2[2] - l2[0], l2[3] - l2[1]])
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = math.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return min(ang, 180 - ang)

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    linear_dims = []
    circle_dims = []
    fillet_dims = []
    angle_dims = []

    if request.method == "GET":
        return render_template(
            "dashboard.html",
            linear_dims=[],
            circle_dims=[],
            fillet_dims=[],
            angle_dims=[]
        )

    # ---------------- INPUT ----------------
    ref_mm = request.form.get("known_length_mm")
    file = request.files.get("image")

    if not ref_mm or not file or file.filename == "":
        return redirect(url_for("dashboard"))

    ref_mm = float(ref_mm)

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    out = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    # -------------------------------------------------
    # LINE DETECTION  (✔ FIXED)
    # -------------------------------------------------
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=200,
        maxLineGap=10
    )

    if lines is None:
        return redirect(url_for("dashboard"))

    clean_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = math.hypot(x2 - x1, y2 - y1)

        # KEEP ALL LONG LINES (inclined included)
        if length > 150:
            clean_lines.append((x1, y1, x2, y2, length))

    longest = max(clean_lines, key=lambda x: x[4])
    mm_per_px = ref_mm / longest[4]

    used_lengths = []

    for x1, y1, x2, y2, lpx in clean_lines:
        lmm = lpx * mm_per_px
        if is_duplicate(lmm, used_lengths):
            continue

        used_lengths.append(lmm)
        linear_dims.append(f"{lmm:.1f} mm")

        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(out, f"{lmm:.1f} mm",
                    (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # -------------------------------------------------
    # ANGLE DETECTION (WORKING)
    # -------------------------------------------------
    used_angles = []
    used_corners = []

    line_segments = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in clean_lines]

    for i in range(len(line_segments)):
        for j in range(i + 1, len(line_segments)):
            l1 = line_segments[i]
            l2 = line_segments[j]

            corner = endpoints_close(l1, l2)
            if corner is None:
                continue

            if any(abs(corner[0] - cx) < 15 and abs(corner[1] - cy) < 15 for cx, cy in used_corners):
                continue

            ang = angle_between(l1, l2)

            if ang < 15 or ang > 165:
                continue
            if abs(ang - 90) < 5:
                continue
            if is_duplicate(ang, used_angles, tol=3):
                continue

            used_angles.append(ang)
            used_corners.append(corner)
            angle_dims.append(f"{ang:.1f}°")

            cv2.putText(
                out,
                f"{ang:.1f}°",
                (corner[0] + 6, corner[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

    # -------------------------------------------------
    # CIRCLE + FILLET DETECTION
    # -------------------------------------------------
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=120,
        param2=45,
        minRadius=30,
        maxRadius=0
    )

    used_centers = []
    used_radii = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for x, y, r in circles[0]:

            if any(abs(x - ux) < 40 and abs(y - uy) < 40 for ux, uy in used_centers):
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 2)
            edge_pixels = cv2.bitwise_and(edges, edges, mask=mask)
            coverage = np.count_nonzero(edge_pixels) / (2 * math.pi * r)

            if coverage > 0.65:
                used_centers.append((x, y))
                dia_mm = 2 * r * mm_per_px
                circle_dims.append(f"Ø {dia_mm:.1f} mm")

                cv2.circle(out, (x, y), r, (0, 255, 0), 2)
                cv2.putText(out, f"Ø {dia_mm:.1f} mm",
                            (x - r, y - r - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            elif 0.25 < coverage <= 0.65:
                r_mm = r * mm_per_px
                if is_duplicate(r_mm, used_radii):
                    continue

                used_radii.append(r_mm)
                fillet_dims.append(f"R {r_mm:.1f} mm")

                cv2.circle(out, (x, y), r, (255, 0, 0), 2)
                cv2.putText(out, f"R {r_mm:.1f} mm",
                            (x + 5, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # -------------------------------------------------
    cv2.imwrite(OUTPUT_IMAGE, out)

    return render_template(
        "dashboard.html",
        linear_dims=linear_dims,
        circle_dims=circle_dims,
        fillet_dims=fillet_dims,
        angle_dims=angle_dims
    )

# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
