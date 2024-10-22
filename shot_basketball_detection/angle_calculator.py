
import numpy as np

class AngleCalculator:
    @staticmethod
    def calcular_angulo(a, b, c):
        ax, ay = a.x, a.y
        bx, by = b.x, b.y
        cx, cy = c.x, c.y
        
        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)