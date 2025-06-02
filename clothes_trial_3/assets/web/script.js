const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let detector, shirtImg;

/* ---------- 1. Camera ---------- */
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise((res) => (video.onloadedmetadata = res));
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

/* ---------- 2. Load shirt PNG (RGBA) ---------- */
async function loadShirt() {
  shirtImg = new Image();
  shirtImg.src = "assets/shirt.png"; // transparent PNG
  await new Promise((res) => (shirtImg.onload = res));
}

/* ---------- 3. Pose detector ---------- */
async function loadDetector() {
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );
}

/* ---------- 4. Main render loop ---------- */
async function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const poses = await detector.estimatePoses(video, { flipHorizontal: false });
  if (poses.length) {
    const kp = poses[0].keypoints;
    const L = kp.find((k) => k.name === "left_shoulder");
    const R = kp.find((k) => k.name === "right_shoulder");
    const LH = kp.find((k) => k.name === "left_hip");
    const RH = kp.find((k) => k.name === "right_hip");

    if (L.score > 0.5 && R.score > 0.5 && LH.score > 0.5 && RH.score > 0.5) {
      // pixel coords
      const x1 = L.x,
        y1 = L.y,
        x2 = R.x,
        y2 = R.y,
        hipY = (LH.y + RH.y) / 2;
      const shoulderDist = Math.hypot(x2 - x1, y2 - y1);
      const neckY = (y1 + y2) / 2;
      const torsoLen = hipY - neckY;

      // dimensions
      const shirtW = 2 * shoulderDist;
      const shirtH = 1.5 * torsoLen;
      if (shirtW > 20 && shirtH > 20) {
        // center + angle
        const centerX = (x1 + x2) / 2;
        const centerY = neckY;
        const angle = Math.atan2(y2 - y1, x2 - x1) + Math.PI; // face‑out correction

        // draw
        ctx.save();
        ctx.translate(centerX, centerY);
        const shear = (0.2 * (y2 - y1)) / (x2 - x1);
        ctx.transform(1, 0, shear, 1, 0, 0);
        ctx.drawImage(shirtImg, -shirtW / 2, -shirtH * 0.15, shirtW, shirtH);
        ctx.restore();

        ctx.restore();
      }
    }
  }
  requestAnimationFrame(render);
}

/* ---------- 5. Boot ---------- */
(async function () {
  await setupCamera();
  await loadShirt();
  await loadDetector();
  render();
})();
