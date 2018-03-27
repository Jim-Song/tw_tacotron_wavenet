import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback
import threading
import json


from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from text import sequence_to_text
from util import audio, infolog, plot, ValueWindow
from wavenet import WaveNetModel, AudioReader, optimizer_factory
from weight_norm_fbank_10782_20171124_low25_high3700_global_norm.ResCNN_wn_A_softmax_prelu_2deeper_group import ResCNN

log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.histogram('linear_outputs', model.linear_outputs)
    tf.summary.histogram('linear_targets', model.linear_targets)
    tf.summary.histogram('mel_outputs', model.mel_outputs)
    tf.summary.histogram('mel_targets', model.mel_targets)
    tf.summary.scalar('loss_mel', model.mel_loss)
    tf.summary.scalar('loss_linear', model.linear_loss)
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  #input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  #log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  with open(args.wavenet_params, 'r') as f:
    wavenet_params = json.load(f)

  if args.batch_size:
    hparams.batch_size = args.batch_size
  if args.outputs_per_step:
    hparams.outputs_per_step = args.outputs_per_step

  # Multi-GPU settings
  GPUs_id = eval(args.GPUs_id)
  num_GPU = len(GPUs_id)
  hparams.num_GPU = num_GPU
  models = []

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    #feeder = DataFeeder(coord, input_path, hparams)
    print('raw_wav_input')
    print('-----------------------------------------------------------------------------------------------------------')
    input_path = ['./datasets/name_LJSpeech_list.txt', './datasets/name_arctic_list.txt',
                  './datasets/name_Blizzard_list.txt', './datasets/name_VCTK_list.txt']
    feeder = DataFeeder(coord, input_path, hparams)

    inputs = feeder.inputs
    #inputs = tf.split(inputs, hparams.num_GPU, 0)
    input_lengths = feeder.input_lengths
    #input_lengths = tf.split(input_lengths, hparams.num_GPU, 0)
    wav_target = feeder.wav
    mel_targets = feeder.mel_targets
    #mel_targets = tf.split(mel_targets, hparams.num_GPU, 0)
    linear_targets = feeder.linear_targets
    #linear_targets = tf.split(linear_targets, hparams.num_GPU, 0)
    #wavs = feeder.wavs
    #wavs = tf.split(wavs, hparams.num_GPU, 0)


  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  '''
  with tf.variable_scope('model') as scope:
    model_tacotron = create_model(args.model, hparams)
    model_tacotron.initialize(inputs, input_lengths, mel_targets, linear_targets)
    model_tacotron.add_loss()
    model_tacotron.add_optimizer(global_step)
    stats = add_stats(model_tacotron)
  '''

  with tf.variable_scope('model') as scope:
    #optimizer = tf.train.AdamOptimizer(learning_rate, hparams.adam_beta1, hparams.adam_beta2)
    for i, GPU_id in enumerate(GPUs_id):
      print(i)
      with tf.device('/gpu:%d' % GPU_id):
        with tf.name_scope('GPU_%d' % GPU_id):

          #net = ResCNN(data=mel_targets[i], batch_size=hparams.batch_size, hyparam=hparams)
          #net.inference()
          #voice_print_feature = tf.reduce_mean(net.features, 0)

          models.append(None)
          models[i] = create_model(args.model, hparams)
          models[i].initialize(inputs=inputs, input_lengths=input_lengths,
                               mel_targets=mel_targets, linear_targets=linear_targets)
          models[i].add_loss()

          models[i].add_optimizer(global_step)

          #models.alignment

          stats = add_stats(models[i])



          #tf.get_variable_scope().reuse_variables()
          print(tf.get_variable_scope())
          '''
          wavenet = WaveNetModel(
            batch_size=args.batch_size,
            dilations=wavenet_params["dilations"],
            filter_width=wavenet_params["filter_width"],
            residual_channels=wavenet_params["residual_channels"],
            dilation_channels=wavenet_params["dilation_channels"],
            skip_channels=wavenet_params["skip_channels"],
            quantization_channels=wavenet_params["quantization_channels"],
            use_biases=wavenet_params["use_biases"],
            scalar_input=wavenet_params["scalar_input"],
            initial_filter_width=wavenet_params["initial_filter_width"],
            histograms=False,#args.histograms,
            global_condition_channels=None,#args.gc_channels,
            global_condition_cardinality=None,#reader.gc_category_cardinality
          )
        loss_wavenet = wavenet.loss(input_batch=wav_target,
                        global_condition_batch=None,
                        l2_regularization_strength=None#args.l2_regularization_strength)
                        )
        optimizer = optimizer_factory['adam'](
          learning_rate=1e-4,
          momentum=0.9)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss_wavenet, var_list=trainable)
        '''







  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)


  # Train!
  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())


      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)


      #feeder.start_in_session(sess)
      feeder.start_threads(sess, args.preprocess_thread)
      def train_func(global_step, model_tacotron, time_window, loss_window, sess, saver):
        while not coord.should_stop():
          start_time = time.time()
          loss_w = None
          #step, loss, opt, loss_w, opt_wavenet = sess.run([global_step, model_tacotron.loss, model_tacotron.optimize, loss_wavenet, optim])
          step, loss, opt = sess.run([global_step, model_tacotron.loss, model_tacotron.optimize])
          #step, inputs = sess.run([global_step, model_tacotron.inputs])
          #print(step)
          #'''
          time_window.append(time.time() - start_time)
          loss_window.append(loss)
          message = 'Step %-7d [%.03f avg_sec/step,  loss=%.05f, avg_loss=%.05f, lossw=%.05f]' % (
            step, time_window.average,  loss, loss_window.average, loss_w if loss_w else loss)
          log(message, slack=(step % args.checkpoint_interval == 0))

          # if the gradient seems to explode, then restore to the previous step
          if loss > 2 * loss_window.average or math.isnan(loss):
            log('recover to the previous checkpoint')
            # tf.reset_default_graph()
            restore_step = int((step - 10) / args.checkpoint_interval) * args.checkpoint_interval
            restore_path = '%s-%d' % (checkpoint_path, restore_step)
            saver.restore(sess, restore_path)
            continue

          if loss > 100 or math.isnan(loss):
            log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
            raise Exception('Loss Exploded')

          if step % args.summary_interval == 0:
            log('Writing summary at step: %d' % step)
            summary_writer.add_summary(sess.run(stats), step)

          if step % args.checkpoint_interval == 0:
            log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
            saver.save(sess, checkpoint_path, global_step=step)
            log('Saving audio and alignment...')
            input_seq, spectrogram, alignment, wav = sess.run([
              model_tacotron.inputs[0], model_tacotron.linear_outputs[0], model_tacotron.alignments[0], wav_target[0]])
            waveform = audio.inv_spectrogram(spectrogram.T)
            audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
            audio.save_wav(wav, os.path.join(log_dir, 'step-%d-audio2.wav' % step))
            plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
              info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))
            print('alignment.shape: %s' % str(alignment.shape))
            print('input_seq.shape: %s' % str(input_seq.shape))
            print('spectrogram.shape: %s' % str(spectrogram.shape))
            print('wav_target.shape: %s' % str(wav.shape))
            #print('input_seq.shape: %s' % input_seq.shape)
            log('Input: %s' % sequence_to_text(input_seq))
            #'''

      train_threads = []
      for model in models:
        train_threads.append(threading.Thread(target=train_func, args=(global_step, model, time_window, loss_window, sess, saver,)))
      for t in train_threads:
        t.start()
      for t in train_threads:
        t.join()

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./logs/')
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--GPUs_id', default='[0]', help='The GPUs\' id list that will be used. Default is 0')
  parser.add_argument('--preprocess_thread', type=int, default=2, help='preprocess_thread.')
  parser.add_argument('--description', default=None, help='description of the model')
  parser.add_argument('--batch_size', default=None, type=int, help='batch size')
  parser.add_argument('--wavenet_params', type=str, default='./wavenet_params.json',
                      help='JSON file with the network parameters. Default: ' + './wavenet_params.json' + '.')
  parser.add_argument('--outputs_per_step', default=None, type=int, help='outputs_per_step')




  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s-%s' % (run_name, args.description))
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
