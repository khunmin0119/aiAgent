"""
STM 이미지 처리 모듈
OpenCV를 사용하여 절점과 부재를 자동으로 검출합니다.
"""

import cv2
import numpy as np


class STMImageProcessor:
    """STM 다이어그램 이미지 처리 클래스"""
    
    def __init__(self):
        self.image = None
        self.gray = None
        
    def load_image(self, image_path):
        """이미지 로드"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 그레이스케일 변환
        if len(self.image.shape) == 3:
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = self.image.copy()
        
        print(f"✅ 이미지 로드 완료: {self.image.shape}", flush=True)
        
    def detect_circles(self, min_radius=5, max_radius=50, param2=20):
        """원형 절점 검출"""
        if self.gray is None:
            raise ValueError("이미지가 로드되지 않았습니다")
        
        circles = cv2.HoughCircles(
            self.gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"✅ 검출된 원(절점): {len(circles)}개", flush=True)
            return circles
        
        return None
    
    def detect_lines(self, threshold=50, min_line_length=50, max_line_gap=10):
        """선분(부재) 검출"""
        if self.gray is None:
            raise ValueError("이미지가 로드되지 않았습니다")
        
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        if lines is not None:
            print(f"✅ 검출된 선(부재): {len(lines)}개", flush=True)
            return lines
        
        return None
