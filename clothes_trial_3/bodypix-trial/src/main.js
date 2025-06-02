import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';
window.tf = tf;
import * as bodyPix from '@tensorflow-models/body-pix';
window.bodyPix = bodyPix;

window.processImage = async function(base64Input) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = base64Input;

  await new Promise((resolve) => {
    img.onload = resolve;
  });

  await tf.setBackend('webgl');
  await tf.ready();

  const net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2
  });

  const segmentation = await net.segmentPerson(img, {
    internalResolution: 'medium',
    segmentationThreshold: 0.7
  });

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixelData = imageData.data;

  for (let i = 0; i < pixelData.length; i += 4) {
    const index = i / 4;
    const y = Math.floor(index / canvas.width);
    const isTopHalf = y < canvas.height / 2;
    const isPerson = segmentation.data[index];

    if (!(isTopHalf && isPerson)) {
      pixelData[i + 3] = 0; // Make transparent
    }
  }

  ctx.putImageData(imageData, 0, 0);

  const resultBase64 = canvas.toDataURL('image/png');
  if (window.flutter_inappwebview) {
    window.flutter_inappwebview.callHandler('returnImage', resultBase64);
  } else if (window.prompt) {
    window.prompt(resultBase64, 'IMAGE_DATA');
  }
};
