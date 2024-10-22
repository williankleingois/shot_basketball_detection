import cv2
import numpy as np

def identificar_bola_de_basquete_pelo_formato(frame):
    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar um desfoque para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Usar a Transformada de Hough para detectar círculos
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=100)

    # Se círculos forem encontrados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, radius) in circles:
            # Desenhar o círculo ao redor da bola
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            # Desenhar um ponto no centro do círculo
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    
    return frame

# Função para testar em um vídeo
def processar_video(caminho_video):
    cap = cv2.VideoCapture(caminho_video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chamar a função para identificar a bola pelo formato
        frame_com_bola = identificar_bola_de_basquete_pelo_formato(frame)

        # Mostrar o frame processado
        cv2.imshow('Bola de Basquete Detectada', frame_com_bola)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
processar_video('/Users/willianklein/Projetos/shot_basketball_detection/data/video_shotings_2024-10-15 22:31:04.449809.mp4')

