rm frames/*.ppm
./light2DFrame 480 480 10000 5 0.0001
ffmpeg -framerate 60 -i frames/frame_%04d.ppm -c:v ffv1 output.mkv
