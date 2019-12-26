# I3D-Feature-Extractor
I3D feature extractor


### Usage

1. Download checkpoints from this [link](https://github.com/deepmind/kinetics-i3d/tree/master/data/checkpoints/rgb_imagenet). This code is heavily borrowed from Deepmind's Kinetics project code.
2. Extract video frames (e.g., fps=16).
``` shell
ffmpeg -i [video_input_path] -r 16 [video_save_dir]/%d.jpg
```
3. Run feature extractor.
``` bash
python feature_extractor_frm.py
```
4. Generate feature h5 file.
``` bash
python get_i3d_h5.py
```