# تشخیص خواب‌آلودگی با استفاده از پردازش تصویر

این ریپازیتوری حاوی کدی برای تشخیص خواب‌آلودگی در افراد به وسیله‌ی تشخیص بسته شدن چشم‌ها است. این سیستم می‌تواند به خصوص برای رانندگانی که ممکن است حین رانندگی دچار خواب‌آلودگی شوند، مفید باشد.

## ویژگی‌ها
- تشخیص خودکار چهره و چشم‌ها به وسیله‌ی OpenCV.
- محاسبه‌ی نسبت جنبه‌ی چشم (EAR) برای تعیین وضعیت چشم‌ها (باز یا بسته).
- پخش آلارم در صورت بسته شدن چشم‌ها به مدت زمان مشخص.

## پیش‌نیازها
برای اجرای این کد، شما نیاز به نصب برخی کتابخانه‌ها دارید:
- Python 3.x
- OpenCV
- dlib
- imutils
- scipy
- numpy
- shape_predictor_68_face_landmarks

حتما به صورت دستی فایل shape_predictor_68_face_landmarks به پوشه فایل اضافه بشه

 نسخه جدید:


markdown
Copy code
# Drowsiness Detection System

This repository contains the code for a drowsiness detection system, which uses a camera to monitor a user's eyelid movements and detect signs of drowsiness based on the eye aspect ratio (EAR).

## Features

- Real-time drowsiness detection using webcam feed.
- Adjustable sensitivity to accommodate different user thresholds for drowsiness detection.

## Usage

To run the drowsiness detection system:

1. Ensure you have Python installed along with the required libraries: cv2, dlib, numpy, etc.
2. Open the terminal and navigate to the directory containing the script.
3. Run the script with an optional sensitivity argument:
   ```bash
   python improvesleep.py --sensitivity [1-10]
Where a lower number increases sensitivity to drowsiness detection.

Configuration
The launch.json file is configured for use with VS Code, allowing for easy debugging with predefined arguments.

Adjusting Sensitivity
You can adjust the sensitivity of the detection by changing the --sensitivity parameter when running the script, where 1 is the most sensitive and 10 is the least sensitive.

Dependencies
Python 3
OpenCV
dlib
numpy
License
This project is licensed under the MIT License - see the LICENSE file for details.

sql
Copy code

### Creating the README.md
- In your project directory, create a file named `README.md`.
- Copy and paste the above content into the file.
- Add and commit this file to Git:
  ```bash
  git add README.md
  git commit -m "Add README"
  git push
This will set up your GitHub repository with your files and a comprehensive README to explain the project. If you need further customization or additional details in the README, feel free to adjust the content as needed!
