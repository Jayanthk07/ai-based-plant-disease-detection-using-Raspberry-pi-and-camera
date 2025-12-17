cd Dekstop/project
rm image.jpg

rpicam-still -o ~/Desktop/project/image.jpg

cd ..
source venv/bin/activate

cd project
python3 scriptrun.py
