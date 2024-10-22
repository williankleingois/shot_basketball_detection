import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from pose_estimation import PoseEstimator
from angle_calculator import AngleCalculator
from shot_predictor import ShotPredictor
from ball_detection import BallDetection
from tqdm import tqdm
import os
import warnings
import absl.logging

# Definir o nível de log do absl para FATAL para suprimir os avisos
absl.logging.set_verbosity(absl.logging.FATAL)
# Ignorar todos os warnings
warnings.filterwarnings('ignore')

class VideoProcessor:
    def __init__(self, show_video, init_frame, last_shot_index, video_path, model_path, scaler_path, manual_shotting, frame_processor):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if init_frame > 0:
            # Configurar o frame inicial
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
        self.pose_estimator = PoseEstimator()
        self.angle_calculator = AngleCalculator()
        self.shot_predictor = None
        if not manual_shotting:
            self.shot_predictor = ShotPredictor(model_path, scaler_path)
        self.ball_detector = BallDetection()
        self.manual_shotting = manual_shotting
        self.pose_data = []
        self.make_shots = []
        self.frames_shotting = []
        self.last_shot_index = last_shot_index
        self.shot_count = 0
        self.start_frame = None
        self.frame_processor = frame_processor
        self.show_video = show_video
        self.total_frames = None
    
    def process_video(self):
        shoting = False
        shot_index = self.last_shot_index if self.last_shot_index > -1 else -1
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=self.total_frames, desc="Processing Video", unit="frame") as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                results = self.pose_estimator.process_frame(frame)
                
                if self.show_video:
                    self.ball_detector.process_frame(frame)
                    wait_key = cv2.waitKey(10)

                    if wait_key & 0xFF == ord('a'):
                        shoting = not shoting
                        if shoting:
                            shot_index += 1
                        print(f'Shotting: {shoting}')

                    if wait_key & 0xFF == ord('0'):
                        print('Ultimo arremesso: Fora')
                        self.store_make_shot(shot_index, False)
                    if wait_key & 0xFF == ord('1'):
                        print('Ultimo arremesso: Cesta')
                        self.store_make_shot(shot_index, True)

                if results.pose_landmarks:
                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    landmarks = results.pose_landmarks.landmark
                    angles = self.calculate_angles(landmarks)

                    if self.shot_predictor:
                        shoting = self.shot_predictor.predict_shot(angles)
                        self.store_frame_shotting(shoting, current_frame)
                    
                    self.store_pose_data(landmarks, current_frame, shoting, shot_index)
                    if self.show_video:
                        self.display_frame(frame, results, shoting, angles, current_frame)

                if self.show_video:
                    if wait_key & 0xFF == ord('q'):
                        break

                pbar.update(1)

        self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()

    def calculate_angles(self, landmarks):
        angles = [
            (16, 14, 12),
            (15, 13, 11),
            (12, 24, 26),
            (11, 23, 25),
            (24, 26, 28),
            (23, 25, 27),
            (16, 12, 24),
            (15, 11, 23)
        ]
        angles_results = []
        for angle in angles:
            angles_results.append(self.angle_calculator.calcular_angulo(landmarks[angle[0]], landmarks[angle[1]], landmarks[angle[2]]))
            
        return angles_results

    def store_pose_data(self, landmarks, current_frame, shoting, shot_index):
        frame_data = [coord for lm in landmarks for coord in [lm.x, lm.y, lm.z, lm.visibility]]
        frame_data.extend([current_frame, shoting, shot_index])
        self.pose_data.append(frame_data)
        

    def store_make_shot(self, shot_index, make_shot):
        self.make_shots.append([shot_index, make_shot])

    def store_frame_shotting(self, shoting, current_frame_number):
        if self.frame_processor:
            if shoting:
                self.shot_count += 1
                if self.shot_count > 5:
                    self.start_frame = max(0, current_frame_number - 60) 
            else:
                if self.start_frame:
                    end_frame = current_frame_number + 80
                    if end_frame > self.total_frames:
                        end_frame = self.total_frames
                    self.frames_shotting.append((self.start_frame, end_frame))
                self.shot_count = 0
                self.start_frame = None

    def display_frame(self, frame, results, shoting, angles, current_frame):
        blank_image = np.zeros(frame.shape, dtype=np.uint8)
        img_height = frame.shape[0]
        circle_radius = int(.007 * img_height)
        point_spec = mp.solutions.drawing_utils.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)
        line_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0) if shoting else (0, 0, 255), thickness=3)
        self.ball_detector.drawing(blank_image)

        for i, angle in enumerate(angles, start=1):
            cv2.putText(blank_image, f'Angle {i}: {angle:.2f}', (50, 50 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(blank_image, f'Frame: {current_frame}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(blank_image, f'Shotting: {shoting}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        mp.solutions.drawing_utils.draw_landmarks(
            blank_image, 
            results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS, 
            point_spec, 
            line_spec
        )
        
        combined_frame = np.hstack((frame, blank_image))
        cv2.imshow('Combined Video', combined_frame)

    def save_pose_data(self, output_path):
        columns = [f'{axis}_{i}' for i in range(33) for axis in ['x', 'y', 'z', 'visibility']]
        columns.extend(['frame', 'shotting', 'shot_index'])
        pose_df = pd.DataFrame(self.pose_data, columns=columns)
        pose_df.to_csv(output_path, index=False)

    def save_make_shots(self, output_path):
        columns = ['shot_index', 'make_shot']
        pose_df = pd.DataFrame(self.make_shots, columns=columns)
        pose_df.to_csv(output_path, index=False)

    def save_video_shoting(self, output_path):
        cap = cv2.VideoCapture(self.video_path)
        # Obter informações sobre o vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo

        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        # Cria o novo video apenas com os shotings
        total = len(self.frames_shotting)
        with tqdm(total=total, desc="Processing Video Shottings", unit="frame") as pbar:
            for start_frame, end_frame in self.frames_shotting:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    output_video.write(frame)

                pbar.update(1)

                
