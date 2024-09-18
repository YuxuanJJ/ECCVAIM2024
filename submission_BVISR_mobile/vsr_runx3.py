# (c) Meta Platforms, Inc. and affiliates.
import argparse
import os

import torch
import torchvision
import imageio

from iopath.common.file_io import g_pathmgr

import importlib
import av
import numpy as np
import torch.nn.functional as F
import subprocess
import model_RTVSRx3


def parse_args():
    parser = argparse.ArgumentParser(description='VSR Testing')
    parser.add_argument('--net', type=str, default='./model_RTVSRx3.py')
    parser.add_argument('--checkpoint', type=str, default='./model_RTVSRx3.pth')
    parser.add_argument('--input', default='Sparks-Sunrise_3840x2160_60fps_8b_420_3x_crf39.mp4', help='input video')
    parser.add_argument('--output', default='./RTVSRx3', help='output path')
    parser.add_argument('--ffmpeg', default='C:/Program Files/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe', help='path to the ffmpeg command')
    args = parser.parse_args()
    return args

def main():
    """ Script for testing RTVSR models.
    """

    args = parse_args()
    os.environ["IMAGEIO_FFMPEG_EXE"] = (args.ffmpeg)

    # prepare input data
    # input_video = torchvision.io.read_video(args.input)
    # normalized_frames = input_video[0].permute(0, 3, 1, 2) # THWC to TCHW
    # normalized_frames = torchvision.transforms.functional.convert_image_dtype(
    #         normalized_frames, torch.float32)
    # input_data = normalized_frames.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_video = torchvision.io.read_video(args.input)

    container = av.open(args.input)
    y_frames = []
    u_frames = []
    v_frames = []
    for frame in container.decode(video=0):

        width = frame.width
        height = frame.height
        y_plane = frame.planes[0]
        y_array = np.frombuffer(y_plane, dtype=np.uint8).reshape((height, y_plane.line_size))

        u_plane = frame.planes[1]
        u_array = np.frombuffer(u_plane, dtype=np.uint8).reshape((height // 2, u_plane.line_size))

        v_plane = frame.planes[2]
        v_array = np.frombuffer(v_plane, dtype=np.uint8).reshape((height // 2, v_plane.line_size))

        y = y_array[:, :width]
        u = u_array[:, :width // 2]
        v = v_array[:, :width // 2]


        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis=0).repeat(2, axis=1)

        y_frames.append(torch.tensor(y, dtype=torch.float32))
        u_frames.append(torch.tensor(u, dtype=torch.float32))
        v_frames.append(torch.tensor(v, dtype=torch.float32))

    y_frames = torch.stack(y_frames)
    u_frames = torch.stack(u_frames)
    v_frames = torch.stack(v_frames)

    input_data = torch.stack([y_frames, u_frames, v_frames], dim=1) / (2**8-1)
    input_data = input_data.unsqueeze(0)

    ################### for x3 track #####################
    model = model_RTVSRx3.Generator()
    model = model.to(device)
    # model.eval()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict_G'])

    frames_per_batch = 16
    num_frames = input_data.shape[1]
    mode = 'w'

    for iFrame in range(0, num_frames, frames_per_batch):
        current_batch_size = min(frames_per_batch, num_frames - iFrame)
        batch_frames = input_data[0, iFrame:iFrame + current_batch_size, :, :, :]
        with torch.no_grad():
            out = model(batch_frames.cuda()) # [B,C,H,W]
        out = chroma_sub_420(out)
        # save to YUV
        for j in range(current_batch_size):
            write_frame_yuv('./SRtempx3.yuv', out[j].clamp(0.0, 1.0).detach().cpu().numpy(), 8, mode)
            mode = 'a'


    if args.input.lower().endswith('.mp4'):
        file_name = os.path.basename(args.input)
        file_name = file_name[:-4] + '_RTVSR.mp4'
        output_file = os.path.join(args.output, file_name)


    ffmpeg_command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-video_size', f'{width*3}x{height*3}',
        '-r', str(input_video[2]['video_fps']),
        '-pix_fmt', 'yuv420p',
        '-i', './SRtempx3.yuv',
        '-c:v', 'libx264',
        '-crf', '0',
        '-preset', 'veryfast',
        '-strict', '2',
        output_file
    ]

    # print(ffmpeg_command)

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"Conversion completed: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    if os.path.isfile('./SRtempx3.yuv'):
        os.remove('./SRtempx3.yuv')






def chroma_sub_420(pixels):
    "pixels = tensor of pixels in [B, C, H, W]"
    "assumes channels = 3 and first channel is Y"

    pixels_Y = pixels[:,0:1,:,:]
    pixels_UV = pixels[:,1:,:,:]

    pixels_UV_down = F.interpolate(pixels_UV, scale_factor=0.5, mode="bilinear", align_corners=True, recompute_scale_factor=False)
    pixels_UV_sub = F.interpolate(pixels_UV_down, scale_factor=2, mode='nearest', recompute_scale_factor=False)

    pixels_sub = torch.cat((pixels_Y, pixels_UV_sub), dim = 1)

    return pixels_sub

def write_frame_yuv(filename, frame, bit_depth, mode):
    "writes YUV numpy frame (frame) to filename"
    "does not interpolate, indexes with step to subsample UV"
    "interpolate before input if desired"

    with open(filename, mode) as stream:
        # 4:2:0 subsampling
        subsample_x = 2
        subsample_y = 2

        height = frame.shape[1]
        width = frame.shape[2]

        frame_size = height*width

        # U,V smaller dimensions than Y due to subsampling
        frame_size_colour = int(frame_size/(subsample_x*subsample_y))

        # expects floating point values in range [0, 1]
        if bit_depth == 10:
            frame = np.around(frame*(1023))
            datatype = np.uint16
        else:
            frame = np.around(frame*(255))
            datatype = np.uint8

        # preps Y values
        frame_Y = frame[0,:,:].reshape(frame_size).astype(datatype)

        # takes subsampled UV values
        frame_U = frame[1,0:height:subsample_y,0:width:subsample_x].reshape(frame_size_colour).astype(datatype)
        frame_V = frame[2,0:height:subsample_y,0:width:subsample_x].reshape(frame_size_colour).astype(datatype)

        # writes/appends bytes to file
        frame_Y.tofile(stream)
        frame_U.tofile(stream)
        frame_V.tofile(stream)

if __name__ == '__main__':
    main()
