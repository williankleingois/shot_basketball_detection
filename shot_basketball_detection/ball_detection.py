import cv2
import numpy as np

class BallDetection:
    def __init__(self):
        self.real_ball_diameter = 0.24  # Exemplo: 24 cm
        self.focal_length = 800  #
    
    def process_frame(self, frame):
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar blur para suavizar a imagem e reduzir ruídos
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detectar círculos na imagem usando Hough Circle Transform
        # self.circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        #                         param1=100, param2=40, minRadius=15, maxRadius=30)
        
            # Usar a Transformada de Hough para detectar círculos
        self.circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 
            dp=1.2,  # Aumente para reduzir a detecção de círculos
            minDist=100,  # Aumente a distância mínima entre círculos
            param1=200,  # Limite superior do Canny (ajustável)
            param2=53,  # Limite de votos para considerar um círculo (aumente para mais seletividade)
            minRadius=20,  # Defina o tamanho mínimo do círculo
            maxRadius=50)  # Defina o tamanho máximo do círculo
        
    def drawing(self, frame):
        
        if self.circles is not None:
            self.circles = np.round(self.circles[0, :]).astype("int")
            for (x, y, r) in self.circles:
                # Calcular a distância da bola
                distance = (self.real_ball_diameter * self.focal_length) / (2 * r)
                # print(f"Distância estimada da bola: {distance:.2f} metros")

                # Filtrar por uma distância mínima
                # if distance < 6.0:  # Exemplo: manter apenas objetos a menos de 6 metros
                    # Armazenar posição da bola
                    # positions.append((x, y))

                    # Desenhar o círculo detectado
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)