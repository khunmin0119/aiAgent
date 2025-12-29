import sys
import os
os.chdir(r'C:\Users\ssos8\Desktop\aiAgent\aiAgent')
sys.path.insert(0, r'C:\Users\ssos8\Desktop\aiAgent\aiAgent')

from stm_image_processor import STMImageProcessor

p = STMImageProcessor()
p.load_image(r'C:\Users\ssos8\Desktop\aiAgent\stm_model.png')
print('Success: Image shape =', p.image.shape)
