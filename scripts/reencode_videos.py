import os
import subprocess
from glob import glob
from multiprocessing import Pool
from pathlib import Path
import random

from tqdm import tqdm

NUM_WORKERS = 32
V_FPS = 25
MIN_SIDE = 256
A_FPS = 16000
# ORIG_PATH = Path('./data/preliminary_eval/temp/')
ORIG_PATH = Path('/saltpool0/data/datasets/avsync/data/v5/videos/')
VCODEC = 'h264'
CRF = 10
PIX_FMT = 'yuv420p'
ACODEC = 'aac'


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def get_new_path(path, vcodec, acodec, v_fps, min_side, a_fps, orig_path, prefix='video') -> Path:
    new_folder_name = f'{vcodec}_{prefix}_{v_fps}fps_{min_side}side_{a_fps}hz_{acodec}'
    if 'vggsound' in str(orig_path) or 'prelim' in str(orig_path):
        new_folder_path = orig_path.parent / new_folder_name
    elif 'mjpeg' in str(orig_path) or 'lrs3' in str(orig_path):
        new_folder_path = Path(str(path.parent).replace(orig_path.name, f'/{new_folder_name}/'))
    elif 'avsync' in str(orig_path):
        new_folder_path = Path(f"{str(path.parent.parent.parent)}/{str(path.parent.parent.name)}_at_25_fps-test/{str(path.parent.name)}/")
    else:
        raise NotImplementedError
    os.makedirs(new_folder_path, exist_ok=True)
    new_path = new_folder_path / path.name
    return new_path


def reencode_video(path):
    new_path = get_new_path(path, VCODEC, ACODEC, V_FPS, MIN_SIDE, A_FPS, ORIG_PATH)
    # reencode the original mp4: rescale, resample video and resample audio
    cmd = f'{which_ffmpeg()}'
    # no info/error printing
    cmd += ' -hide_banner -loglevel error' # panic'
    cmd += f' -i {path}'
    # Solve error by increasing memory?
    cmd += f' -max_muxing_queue_size 2048'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={V_FPS},scale=iw*{MIN_SIDE}/'min(iw,ih)':ih*{MIN_SIDE}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -vcodec {VCODEC} -pix_fmt {PIX_FMT} -crf {CRF}"
    cmd += f' -acodec {ACODEC} -ar {A_FPS} -ac 1'
    cmd += f' {new_path}'
    if new_path.exists():
        print('already exists', new_path)
    else:
        if subprocess.call(cmd.split()) != 0:
            print(f"Failed on {path.name}")


def main():
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'

    if 'vggsound' in str(ORIG_PATH):
        paths_glob = str(ORIG_PATH / '*.mp4')
    elif 'prelim' in str(ORIG_PATH):
        paths_glob = str(ORIG_PATH / '*.mkv')
    elif 'lrs3' in str(ORIG_PATH):
        paths_glob = str(ORIG_PATH / '*/*/*.mp4')
    elif 'avsync' in str(ORIG_PATH):
        paths_glob = str(ORIG_PATH / '*/*[!_broken][!_silent].mkv')
    video_paths = [Path(p) for p in sorted(glob(paths_glob))]
    print(f"Found {len(video_paths)} clips")
    assert len(video_paths) > 0

    random.shuffle(video_paths)
    reencode_fn = reencode_video

    # # single thread (slow)
    # for path in tqdm(video_paths):
    #     reencode_fn(path)

    # multiple threads (fast)
    with Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap(reencode_fn, video_paths), total=len(video_paths)))

    print(f'{VCODEC}_video_{V_FPS}fps_{MIN_SIDE}side_{A_FPS}hz_{ACODEC}')


if __name__ == '__main__':
    main()
