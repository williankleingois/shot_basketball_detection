import cv2

image_path = './data/bola.jpeg'
image = cv2.imread(image_path)

if image is None:
    print("Erro ao abrir a imagem.")
else:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_value = hsv_image[y, x]
            print(f"HSV Value at ({x}, {y}): {hsv_value}")
    
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', on_mouse_click)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
