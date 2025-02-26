from video_processor import VideoProcessor
from datetime import datetime

video_path = '/Users/willianklein/Projetos/shot_basketball_detection/data/video_shotings_2024-10-24 11:25:35.498089.mp4'
model_path = '/Users/willianklein/Projetos/shot_basketball_detection/models/model.pkl'
scaler_path = '/Users/willianklein/Projetos/shot_basketball_detection/models/scaler.pkl'

processor = VideoProcessor(
    show_video=True,
    init_frame=0,
    last_shot_index=-1,
    scaler_path=scaler_path,
    model_path=model_path,
    video_path=video_path,
    manual_shotting=False,
    frame_processor=False    
)

processor.process_video()
# processor.save_pose_data(f'./data/pose_data_{datetime.now()}.csv')
# processor.save_make_shots(f'./data/make_shots_{datetime.now()}.csv')
# processor.save_video_shoting(f'./data/video_shotings_{datetime.now()}.mp4')
