'''
mujoco的XML文件生成hfield（高度图）要求使用.png文件
手动生成
python gym_custom_env/createPNG.py

'''
from PIL import Image
import numpy as np
arr = np.zeros([6,6])

def creat_stairs(total_lenth:int, stair_num: int, total_width = 0, height = 255):
    '''
    创造二维地形高度图：阶梯
    param: total_lenth 总长度
    param: total_width 总宽度，默认等于总长度
    param: stair_num   阶梯数
    param: height      总高度

    return: map 二维数组 高度图
    '''
    if total_width==0:
        total_width = total_lenth
    map = np.zeros([total_lenth,total_width],dtype=np.int16)
    stair_height = int(height / 10)
    stair_width  = int(total_lenth/2/10)
    n_h = 0
    b_s = int(total_lenth/2)-1
    for i in range(stair_num ):
        map[:,b_s] = n_h
        map[:,b_s+1:(b_s+stair_width)] = n_h + stair_height
        n_h = n_h + stair_height
        b_s = b_s + stair_width
    return map


#for i in range(arr.shape[1]):    
#    arr[i,:]=[0.,0.,0.,1.,2.,3.,]
#    arr[i,:]=arr[i,:]*255/3
#    pass

map = creat_stairs(500,10)

im = Image.fromarray(map)  #分别是颜色模式、尺寸、颜色
im.show()  #展示，可不要
out = im.convert("RGB")
out.save('gym_custom_env/assets/stairs.png')  #保存路径及名称