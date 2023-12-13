from moviepy.editor import VideoFileClip, concatenate_videoclips
import os


def cct(save_dir):
    res = save_dir[save_dir.rfind('\\') + 1:]
    name = os.path.join(save_dir, res + '.mp4')
    if os.path.isfile(name):
        os.remove(name)

    f_list = os.listdir(save_dir)
    v_list = []
    clip = []
    for f in f_list:
        if not f.endswith('.mp4'):
            continue
        v_list.append(os.path.join(save_dir, f))
        clip.append(VideoFileClip(os.path.join(save_dir, f)))

    video = concatenate_videoclips(clip)
    video.write_videofile(name, audio=False)

    for v in v_list:
        os.remove(v)
