import cv2
import numpy as np

class BallDetection:
    def __init__(self):
        self.real_ball_diameter = 0.24  # Exemplo: 24 cm
        self.focal_length = 800  # Exemplo de distância focal
        self.positions = []  # Lista para armazenar os últimos pontos da bola
        self.max_positions = 20  # Número máximo de posições para manter (últimos 20 frames)
        self.old_shoting = None  # Variável para armazenar o estado anterior de arremesso
        self.x_list = [item for item in range(0, 1300)]  # Lista para os valores de x da parábola

    def process_frame(self, frame):
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar blur para suavizar a imagem e reduzir ruídos
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detectar círculos na imagem usando Hough Circle Transform
        self.circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, 
            param1=200, param2=53, minRadius=20, maxRadius=50
        )

        # Se encontrar círculos, atualizar posições da bola
        if self.circles is not None:
            self.circles = np.round(self.circles[0, :]).astype("int")
            for (x, y, r) in self.circles:
                # Armazenar a posição atual da bola
                self.positions.append((x, y))

                # Limitar o número de posições armazenadas
                if len(self.positions) > self.max_positions:
                    self.positions.pop(0)

    def drawing(self, frame, shoting):
        cv2.rectangle(frame, (570, 340), (626, 353), (0, 165, 255), 2)
        
        if self.circles is not None:
            for (x, y, r) in self.circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        if self.old_shoting and not shoting:
            self.reset_positions()

        self.old_shoting = shoting

        # Se houver pontos, calcular e desenhar a trajetória
        if len(self.positions) > 1:
            points_listX = [pos[0] for pos in self.positions]
            points_listY = [pos[1] for pos in self.positions]
            
            # Calcular a parábola usando os pontos detectados
            if len(points_listX) > 2:
                coeff = np.polyfit(points_listX, points_listY, 2)
                poly = np.poly1d(coeff)

                # Desenhar a parábola
                for x in self.x_list:
                    y = int(poly(x))
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), cv2.FILLED)

                    # Verificar se o ponto está dentro da região da cesta
                    if 340 <= x <= 353 and 570 <= y <= 626:
                        cv2.putText(frame, 'ACERTOU', (560, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        # Desenhar a trajetória da bola
        # for i in range(1, len(self.positions)):
        #     cv2.line(frame, self.positions[i - 1], self.positions[i], (255, 0, 0), 10)

    def reset_positions(self):
        # Limpar as posições quando o arremesso for finalizado
        self.positions.clear()