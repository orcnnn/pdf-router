CUDA_VISIBLE_DEVICES=0 marker_server --port 8001 > logs/marker_server_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 marker_server --port 8002 > logs/marker_server_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 marker_server --port 8003 > logs/marker_server_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 marker_server --port 8004 > logs/marker_server_4.log 2>&1 &
