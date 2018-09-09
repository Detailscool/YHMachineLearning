import tensorflow as tf

greeting = tf.constant('Hello Google Tensorflow')

session = tf.Session()
result = session.run(greeting)
print(result)

session.close()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

linear = tf.add(product, tf.constant(2.0))

with tf.Session() as session:
    result = session.run(linear)
    print(result)