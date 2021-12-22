import argparse
import os
import sys
import numpy as np
import math
import itertools
from PIL import Image
from pathlib import Path
import glob
import pickle
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
rh = logging.handlers.RotatingFileHandler(
        r'./etlcdb_infogan.log', 
        encoding='utf-8'
        )
rh.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"))
logger.addHandler(rh)

MNIST = False
MODEL_DIR = 'jp_ckpt/'
OUT_DIR = 'jp_images/'
os.makedirs(OUT_DIR + "static/", exist_ok=True)
os.makedirs(OUT_DIR + "varying_c1/", exist_ok=True)
os.makedirs(OUT_DIR + "varying_c2/", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--model_ckpt", type=str, default='', help="model checkpoint")
opt = parser.parse_args()
logger.debug(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# ckptがある場合はload
epoch_on_the_way = 0
history = {
    "d_loss" : [],
    "g_loss" : [],
    "info_loss" : []
}
if opt.model_ckpt != '':
    checkpoint = torch.load(opt.model_ckpt)
    epoch_on_the_way = checkpoint['epoch']
    history = checkpoint['history']
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    logger.debug('model loaded. epoch: %s', epoch_on_the_way)

# MyDataLoader
class EtlCdbDataLoader(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.etl_paths, self.np_labels = self._get_etl_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        # # [img_path, [x, y, w, h], label]
        etl_path = self.etl_paths[index]

        # load image
        img = Image.open(etl_path[0])
        # load label(index)
        label = np.where(self.np_labels == etl_path[2])[0][0]

        if self.transform is not None:
            # 前処理がある場合は行う。
            img = self.transform(img)

        return img, label

    def _get_etl_paths(self, img_dir):
        img_dir = Path(img_dir)

        pickle_paths = glob.glob(os.path.join(img_dir, '*.pickle'), recursive=True)

        # 一部の文字のみ学習
        CHAR_LIST = [
            "/177/", "/178/", "/179/", "/180/", "/181/",
            "/182/", "/183/", "/184/", "/185/", "/186/"
            ]

        # pickleに格納したETLCDBデータをロード
        # [img_path, [x, y, w, h], label]
        etl_paths = []
        temp_labels = []
        for pickle_path in pickle_paths:
            with open(pickle_path, 'rb') as pickle_file:
                contents = pickle.load(pickle_file)
                for content in contents:
                    # infoGAN作業ディレクトリからの相対パスに修正
                    # 出力先親ディレクトリ取得
                    root_dir = Path(content[0]).parents[3]
                    # 親ディレクトリ以下のディレクトリ取得
                    child_dir = content[0].split(str(root_dir))[1]
                    dir = os.path.join(str(img_dir) + str(child_dir))

                    result = False
                    for target in CHAR_LIST:
                        if target in dir:
                            result = True

                    if result == True:
                        # label(Unicode)
                        etl_paths.append( [dir, content[1], ord( content[2] )] )
                        temp_labels.append(ord( content[2] ))

        # IMG_EXTENSIONSに該当するファイルのみ取得
        etl_paths = [
            p for p in etl_paths
            # 拡張子取得
            if '.' + os.path.splitext( os.path.basename(p[0]) )[1][1:]
            in EtlCdbDataLoader.IMG_EXTENSIONS
        ]

        logger.debug('data: %s', etl_paths[0])
        # labelsのcategory数を計算
        np_labels = np.sort( np.unique(np.array( temp_labels )) )
        logger.debug('please set --n_classes: %d', np_labels.size)

        return etl_paths, np_labels

    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.etl_paths)

if not MNIST:
    dataloader = torch.utils.data.DataLoader(
        EtlCdbDataLoader(
            "../../ocr_dataset_create_jp/output",
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
else:
# Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

sampling_index = []
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    if z.shape[0] < static_label.shape[0]:
        # z.shape[0]分ランダムに取得
        sampling_index = random.sample(range(0, static_label.shape[0]), z.shape[0])
        static_sample = generator(z, static_label[sampling_index], static_code[sampling_index])
    else:
        static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, OUT_DIR + "static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))

    if c1.shape[0] < static_label.shape[0]:
        sample1 = generator(static_z[:c1.shape[0]], static_label[:c1.shape[0]], c1)
        sample2 = generator(static_z[:c1.shape[0]], static_label[:c1.shape[0]], c2)
    else:
        sample1 = generator(static_z, static_label, c1)
        sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, OUT_DIR + "varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, OUT_DIR + "varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
best_loss = 10
for epoch in range(epoch_on_the_way, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Save best_accuracy
        # --------------
        if info_loss.item() < best_loss:
            best_loss = info_loss.item()
            torch.save(
                {
                    "epoch": epoch,
                    "history": history,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                },
                MODEL_DIR + "jp_best_model_%d.tar" % epoch
            )
            logger.debug(
                "Save best model [Model Path %s][Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % ( MODEL_DIR + "jp_best_model_%d.tar" % epoch, epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
        # --------------
        # Log Progress
        # --------------
        if i % len(dataloader) == 0:
            # append loss
            if epoch < len(history['d_loss']):
                history['d_loss'][epoch] = d_loss.item()
                history['g_loss'][epoch] = g_loss.item()
                history['info_loss'][epoch] = info_loss.item()
            else:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['info_loss'].append(info_loss.item())

            logger.debug(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
            #sample_image(n_row=opt.n_classes, batches_done=batches_done)
            logger.debug('sample image done batches_done: %d' % batches_done)
    # for i, (imgs, labels) in enumerate(dataloader):

    # --------------
    # Save model
    # --------------
    if epoch % 20 == 0 or epoch == (opt.n_epochs-1):
        # do checkpointing
        torch.save(
            {
                "epoch": epoch,
                "history": history,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            },
            MODEL_DIR + "jp_model_%d.tar" % epoch
        )
        logger.debug('save model. epoch: %d, path: %sjp_model_%d.tar' %(epoch, MODEL_DIR, epoch))

# --------------
# Save Graph
# --------------
plt.figure()
plt.plot(range(0, opt.n_epochs), history['d_loss'], label='D loss')
plt.plot(range(0, opt.n_epochs), history['g_loss'], label='G loss')
plt.plot(range(0, opt.n_epochs), history['info_loss'], label='info loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss.png')