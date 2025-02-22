import os
import cv2
import shutil
import subprocess

def process_image(path):
  if not os.path.exists(path):
    print("File Not Found!!")
    return
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  resize = cv2.resize(image, (1024, 1024))
  output = "pifuhd/sample_images/input.jpg"
  cv2.imwrite(output, resize)
  print("Converted image... starting model gen")
  return output
def generate_keypoints(input_folder):
  print("Generating keypoints using OpenPose...")
  command = [
    "python", "pifuhd/apps/batch_openpose.py",
    "-d", OPENPOSE_PATH,
    "-i", input_folder,
    "-o", input_folder
  ]
  subprocess.run(command, cwd="pifuhd")

def gen3D(image, output = "output"):
  img = process_image(image)
  generate_keypoints("pifuhd/input_images")
  print("Running PifuHD to generate .obj file from given image")
  command = [
    "python", "apps/recon.py",
    "--dataroot", "input_images",
    "--results_path", output,
    "--loadSize", "1024",
    "--resolution", "512",
    "--load_netMR_checkpoint_path", "checkpoints/pifuhd.pt",
    "--start_id", "0",
    "--end_id", "-1"
  ]
  subprocess.run(command, cwd="pifuhd")
  return "success"
img = "input.png"
output = gen3D(img)
print(output)
