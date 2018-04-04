import tensorflow as tf

def CategoryClassifierModel(features, labels, mode, params):
    
    net = tf.contrib.layers.embed_sequence(features['x'], vocab_size=params['vocab_size'], embed_dim=params['embed_dim']) 
    
    wordcnn = tf.layers.conv1d(net, filters=params['conv_filters'], kernel_size=params['conv_kernelsize'], strides=1, padding='SAME', activation=tf.nn.relu)
    wordcnnshape = wordcnn.get_shape()
    rlayer = tf.reshape(wordcnn,[-1, wordcnnshape[1] * wordcnnshape[2]])
    dropout = tf.nn.dropout(rlayer, params['dropout'])
    logits = tf.layers.dense(inputs=dropout, units=params['n_classes'], activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        predictions = {
            'class': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), predicted_indices),
                           'auroc': tf.metrics.auc(labels, probabilities)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
