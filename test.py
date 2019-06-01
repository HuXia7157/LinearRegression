import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.name_scope('original_data'):
    X = np.arange(-1, 1, 0.02, dtype=np.float32).reshape((100, 1))
    Y = np.tan(X) + np.random.normal(0, 0.4, 100).reshape((100, 1))#概率分布均值为0，标准差为0.1，一共100个数据

with tf.name_scope('parameters'):
    #初始化权重和偏差
    with tf.name_scope('weights'):
        w1=tf.Variable(tf.random_normal([1,1],mean=0,stddev=0))
        tf.summary.histogram('weights',w1)
    with tf.name_scope('biases'):
        b1 = tf.Variable(0.0)
        tf.summary.histogram('bias', b1)
with tf.name_scope('y_prediction'):
    #建立模型
    Y_prediction=tf.add(tf.matmul(X,w1),b1)

with tf.name_scope('loss'):
    #建立损失函数
    loss=tf.reduce_mean(tf.square(Y-Y_prediction))
    tf.summary.scalar('loss', loss)#对标量数据汇总和记录

with tf.variable_scope("optimizer_train"):
    #利用梯度下降优化损失
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)


with tf.variable_scope("init"):
    init_op=tf.global_variables_initializer()#初始化变量

# 3）合并变量
merge = tf.summary.merge_all()
#建立会话运行程序
with tf.Session() as sess:
    # 定义日志文件
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op)
    print("初始化权重：%f 初始化偏置：%f" % (w1.eval(), b1.eval()))

    #训练模型
    STEPS=5000
    for i in range(STEPS):
        sess.run(train_step)#启动会话运行
        rs=sess.run(merge)
        writer.add_summary(rs,i)
        if i%500==0:
            total_loss=sess.run(loss)
            w = sess.run(w1)
            b = sess.run(b1)
            print("After %d training step(s),loss on all data is %g"%(i,total_loss))

    print("w1:\n",w)
    print("b1:\n", b)
    print("\n")


    plt.scatter(X, Y)
    plt.plot(X,np.add(np.multiply(X, w),b),'r')
    plt.show()


