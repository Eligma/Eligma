import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil

import dataFeed
import model

TESTSIZE = 0.1
MODELDIR = 'trained_models'
CLEARCHECKPOINTS = True
BATCHSIZE = 32
EPOCHS = 10

# load preprocessed training data
x, y, label_dict, vocab_processor = dataFeed.loadTrainingData('trainingdata.hdf5', 'labels.json', 'vocabulary')

# split to train|test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=TESTSIZE, random_state=123)

# set logging
tf.logging.set_verbosity(tf.logging.INFO)

# restart training
if CLEARCHECKPOINTS:
    shutil.rmtree(MODELDIR, ignore_errors=True)

# max steps
STEPS = int((len(train_x)/BATCHSIZE)*EPOCHS)

# set run config
run_config = tf.estimator.RunConfig(
    log_step_count_steps=10000,
    tf_random_seed=123,
    model_dir=MODELDIR
)
# init estimator
classifier = tf.estimator.Estimator(
    model_fn=model.CategoryClassifierModel,
    config=run_config,
    params={
        'embed_dim': 128,
        'conv_kernelsize': 1,
        'conv_filters': 64,
        'dropout': 0.5,
        'n_classes': len(label_dict),
        'vocab_size': len(vocab_processor.vocabulary_)
    })

# define training spec
train_spec = tf.estimator.TrainSpec(input_fn = lambda: dataFeed.train_input_fn(train_x, train_y, BATCHSIZE, mode = tf.estimator.ModeKeys.TRAIN),max_steps=STEPS,hooks=None)

# define eval spec
eval_spec = tf.estimator.EvalSpec(input_fn = lambda: dataFeed.train_input_fn(test_x, test_y, BATCHSIZE, mode=tf.estimator.ModeKeys.EVAL),exporters=[tf.estimator.LatestExporter(name="output", serving_input_receiver_fn=dataFeed.serving_input_fn, exports_to_keep=1, as_text=True)],steps=None,throttle_secs=60)

# train and evaluate
tf.estimator.train_and_evaluate(
    estimator=classifier,
    train_spec=train_spec, 
    eval_spec=eval_spec
)

# done
print('COMPLETED')

