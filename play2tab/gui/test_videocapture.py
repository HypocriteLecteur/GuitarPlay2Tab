    from tqdm import tqdm


    def seq_acc(video_path: str, frame_ids: list) -> list:
        video = cv2.VideoCapture(video_path)
        frames_seq = []
        frame_id_curr = 0
        for frame_id in tqdm(frame_ids):
            while frame_id != frame_id_curr:
                video.grab()
                frame_id_curr += 1
            success, img = video.read()
            assert success
            frames_seq.append(img)
            frame_id_curr = frame_id + 1
        return frames_seq


    def rand_acc_cv(video_path: str, frame_ids: list) -> list:
        video = cv2.VideoCapture(video_path)
        frames_rand = []
        for frame_id in tqdm(frame_ids):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, img = video.read()
            assert success
            frames_rand.append(img)
        return frames_rand


    video_path = "D:\\GitHub\\GuitarPlay2Tab\\video\\video4\\video4.mp4"
    frame_ids = range(130, 140)
    frames_seq = seq_acc(video_path, frame_ids)
    frames_rand = rand_acc_cv(video_path, frame_ids)