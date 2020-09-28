python unpack.py husky_ouster.bag \
                 --config calibration-kalibr.yml \
                 --left_image_topic /left/image_raw \
                 --right_image_topic /right/image_raw \
                 --lidar_topic /os1_cloud_node/points \
                 --atlans_topic /atlans_odom \
                 --encoding bgr8 \
                 --outdir ./output \
                 --max_delay 0.05
