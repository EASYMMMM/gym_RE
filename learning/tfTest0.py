'''
tensorflow 安装测试
python learning/tfTest0.py
'''
import tensorflow as tf


tf.compat.v1.disable_eager_execution() #保证sess.run()能够正常运行
hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session() #版本2.0的函数
print(sess.run(hello))

print('hello tf')
 


'''
import pybullet as p
from time import sleep
from pybullet_envs.bullet import CartPoleBulletEnv

cid = p.connect(p.DIRECT)
env = CartPoleBulletEnv(renders=True, discrete_actions=False)

env.render()
env.reset()

for _ in range(10000):
    sleep(1 / 60)
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

p.disconnect(cid)
'''