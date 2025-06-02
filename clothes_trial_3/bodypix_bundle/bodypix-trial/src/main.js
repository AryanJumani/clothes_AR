import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';

// Register backend before using bodyPix
await tf.setBackend('webgl'); // or 'cpu' if you want
await tf.ready();             // ensure backend is ready

const canvas = document.createElement('canvas');
canvas.id = 'canvas';
canvas.style.display = 'none';
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');

let net = null;
async function loadModel() {
  net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2
  });
}
await loadModel();

window.processImage = async function(base64Img) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    const segmentation = await net.segmentPersonParts(img, {
      internalResolution: 'medium',
      segmentationThreshold: 0.7
    });

    const torsoParts = [11, 12, 13, 14, 15, 16, 23, 24];
    const mask = bodyPix.toMask(
      segmentation,
      { r: 0, g: 255, b: 0, a: 255 },
      { r: 0, g: 0, b: 0, a: 0 },
      torsoParts
    );

    ctx.putImageData(mask, 0, 0);
    const segmentedBase64 = canvas.toDataURL('image/png');
    window.prompt(segmentedBase64, 'IMAGE_DATA');
  };
  img.src = base64Img;
};
