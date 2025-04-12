final project

cd frame

g++ -std=c++17 frame_layer.cpp ../cnpy/cnpy.cpp ../utils/load_ckpt.cpp -I../utils  -I../cnpy -lz -o frame_layer