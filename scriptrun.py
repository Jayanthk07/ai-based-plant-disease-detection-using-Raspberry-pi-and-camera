import onnxruntime as ort
import numpy as np
from PIL import Image
from datetime import datetime
import os

# === CLASS LIST ===
used_classes = [
 'Apple_black_rot', 'Applehealthy', 'Applerust', 'Apple_scab',
 'Cassava_bacterial_blight', 'Cassavabrown_streak_disease', 'Cassavagreen_mottle', 'Cassava_healthy',
 'Cassava_mosaic_disease', 'Cherryhealthy', 'Cherrypowdery_mildew', 'Chili_healthy',
 'Chili_leaf curl', 'Chilileaf spot', 'Chiliwhitefly', 'Chili_yellowish',
 'Coffee_cercospora_leaf_spot', 'Coffeehealthy', 'Coffeered_spider_mite', 'Coffee_rust',
 'Corn_common_rust', 'Corngray_leaf_spot', 'Cornhealthy', 'Corn_northern_leaf_blight',
 'Cucumber_diseased', 'Cucumberhealthy', 'Gauvadiseased', 'Gauva_healthy',
 'Grape_black_measles', 'Grapeblack_rot', 'Grapehealthy', 'Grapeleaf_blight(isariopsis_leaf_spot)',
 'Jamun_diseased', 'Jamunhealthy', 'Lemondiseased', 'Lemon_healthy',
 'Mango_diseased', 'Mangohealthy', 'Peachbacterial_spot', 'Peach_healthy',
 'Pepper_bell_bacterial_spot', 'Pepper_bellhealthy', 'Pomegranatediseased', 'Pomegranate_healthy',
 'Potato_early_blight', 'Potatohealthy', 'Potatolate_blight', 'Rice_brown_spot',
 'Rice_healthy', 'Ricehispa', 'Riceleaf_blast', 'Rice_neck_blast',
 'Soybean_bacterial_blight', 'Soybeancaterpillar', 'Soybean_diabrotica_speciosa',
 'Soybean_downy_mildew', 'Soybeanhealthy', 'Soybean_mosaic_virus',
 'Soybean_powdery_mildew', 'Soybeanrust', 'Soybean_southern_blight',
 'Strawberry__leaf_scorch', 'Strawberryhealthy', 'Sugarcane_bacterial_blight',
 'Sugarcane_healthy', 'Sugarcanered_rot', 'Sugarcane_red_stripe',
 'Sugarcane_rust', 'Teaalgal_leaf', 'Teaanthracnose', 'Tea_bird_eye_spot',
 'Tea_brown_blight', 'Teahealthy', 'Teared_leaf_spot', 'Tomato_bacterial_spot',
 'Tomato_early_blight', 'Tomatohealthy', 'Tomatolate_blight', 'Tomato_leaf_mold',
 'Tomato_mosaic_virus', 'Tomatoseptoria_leaf_spot', 'Tomatospider_mites(two_spotted_spider_mite)',
 'Tomato_target_spot', 'Tomatoyellow_leaf_curl_virus', 'Wheat_brown_rust',
 'Wheat_healthy', 'Wheatseptoria', 'Wheat_yellow_rust'
]

# === LOAD MODEL ===
model_path = "ensemble_model_v1.0.0.onnx"
session = ort.InferenceSession(model_path)

# === LOAD AND PREPROCESS IMAGE ===
img_path = "image.jpg"
img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === RUN INFERENCE ===
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: img_array})[0]

# === GET TOP 3 PREDICTIONS ===
top_indices = np.argsort(outputs[0])[::-1][:3]
preds = [used_classes[idx] for idx in top_indices]

# === PRINT TO CONSOLE ===
print("\n=== TOP 3 PREDICTIONS ===")
for i, p in enumerate(preds, 1):
    print(f"{i}. {p}")

# === PREPARE HEADER AND LINE ===
date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%H:%M:%S")
line = f"{date:<15}{time:<20}{preds[0]:<30}{preds[1]:<30}{preds[2]:<30}\n"

# === APPEND TO logs.txt WITH HEADER IF NEW ===
log_file = "logs.txt"
if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
    header = f"{'DATE':<15}{'TIME':<20}{'1st PREDICTION':<30}{'2nd PREDICTION':<30}{'3rd PREDICTION':<30}\n"
    with open(log_file, "w") as f:
        f.write(header)

with open(log_file, "a") as f:
    f.write(line)
