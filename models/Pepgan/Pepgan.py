from time import time

from models.Gan import Gan
from models.Pepgan.PepganDataLoader import DataLoader, DisDataloader
from models.Pepgan.PepganDiscriminator import Discriminator
from models.Pepgan.PepganGenerator import Generator
from models.Pepgan.PepganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *


def pre_train_epoch_gen(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, _, _ = trainable_model.pretrain_step(sess, batch, .8)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def generate_samples_gen(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True, train=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


class Pepgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        tf.app.flags.DEFINE_string('d', '', 'kernel')
        tf.app.flags.DEFINE_string('g', '', 'kernel')
        tf.app.flags.DEFINE_string('t', '', 'kernel')
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_boolean('restore', False, 'Training or testing a model')
        flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
        flags.DEFINE_integer('length', 20, 'The length of toy data')
        flags.DEFINE_string('model', "", 'Model NAME')
        self.sequence_length = FLAGS.length
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 256
        self.start_token = 0
        self.dis_embedding_dim = 64
        self.goal_size = 16

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'
        self.test_file_scores = 'save/test_file_scores.txt'


    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
            }
            _, _ = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op], feed)
            self.generator.update_feature_function(self.discriminator)

    def evaluate(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

   

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)

        goal_out_size = sum(self.num_filters)
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=4,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              step_size=4)
        self.set_generator(generator)
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        #[word_index_dict, index_word_dict] = get_dict(word_set)#Original

        
        [word_index_dict, index_word_dict] = [{'b':'0','a':'1','r':'2','n':'3','d':'4','c':'5','q':'6','e':'7','g':'8','h':'9','i':'10','l':'11','k':'12','m':'13','f':'14','p':'15','s':'16','t':'17','w':'18','y':'19','v':'20'},{'0':'b','1':'a','2':'r','3':'n','4':'d','5':'c','6':'q','7':'e','8':'g','9':'h','10':'i','11':'l','12':'k','13':'m','14':'f','15':'p','16':'s','17':'t','18':'w','19':'y','20':'v'}]

        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 200
        self.log = open('experiment-log-pepgan-real.csv', 'w')
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})


        print(list(wi_dict))
        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=8)
        for epoch in range(self.adversarial_epoch_num):
            #for epoch_ in range(30):
            print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess, 1)
                rewards = self.reward.get_reward(samples)
                feed = {
                    self.generator.x: samples,
                    self.generator.reward: rewards,
                    self.generator.drop_out: 1
                }
                _, _, g_loss, w_loss = self.sess.run(
                    [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                     self.generator.worker_loss, ], feed_dict=feed)
                print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(epoch) + '\t time:' + str(end - start))
            with open("save/test_file_scores_ave.txt", 'a') as outfile:
                outfile.write(str(np.mean(np.concatenate(rewards))) + "\n")
            
            generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
            get_real_test_file()
            self.evaluate()
            
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open("save/test_file"+str(epoch)+".txt", 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=iw_dict))

            for _ in range(5):
                self.train_discriminator()

