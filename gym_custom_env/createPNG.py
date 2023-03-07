'''
mujoco的XML文件生成hfield（高度图）要求使用.png文件
手动生成
'''
from PIL import Image
import numpy as np
arr = np.zeros([6,6])


for i in range(arr.shape[1]):    
    arr[i,:]=[0.,0.,0.,1.,2.,3.,]
    arr[i,:]=arr[i,:]*255/3
    pass

im = Image.fromarray(arr)  #分别是颜色模式、尺寸、颜色
im.show()  #展示，可不要
out = im.convert("RGB")
out.save('abc.png')  #保存路径及名称