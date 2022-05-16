import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt
import time
import csv

from monai.transforms import NormalizeIntensity, AddChannel, Compose, Resize, ScaleIntensity, ToTensor

class Unet(nn.Module):
    def __init__(self, args):
        super(Unet, self).__init__()
        ## 트레이닝 파라메터 설정하기
        resize = (int(args.resize), int(args.resize), int(args.resize))
        opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]
        cmap = plt.cm.gray

        print("norm: %s" % norm)

        print("learning rate: %.4e" % lr)
        print("batch size: %d" % batch_size)
        print("number of epoch: %d" % num_epoch)

        print("task: %s" % task)
        print("which_model_netG %s" % which_model_netG)
        print("which_model_netD %s" % which_model_netD)
        print("opts: %s" % opts)
        print("image_size:", resize)

        print("network: %s" % network)
        print("learning type: %s" % learning_type)
        print("d_loss: %s" % d_loss)

        print("target: %s" % target)
        print("t1_dir: %s" % t1_dir)
        print("ckpt dir: %s" % ckpt_dir)
        print("log dir: %s" % log_dir)
        print("result dir: %s" % result_dir)

        print("gpus_list: %s" % gpus_list)
        print("device: %s" % device)

        ## 디렉토리 생성하기
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        result_dir_train = os.path.join(result_dir, 'train')

        if not os.path.exists(result_dir_train):
            os.makedirs(os.path.join(result_dir_train, 'png', 'a2b'))
            os.makedirs(os.path.join(result_dir_train, 'png', 'b2a'))

        if not os.path.exists(result_dir_train):
            os.makedirs(os.path.join(result_dir_train, 'npy', 'a2b'))
            os.makedirs(os.path.join(result_dir_train, 'npy', 'b2a'))

        ## 디렉토리 생성하기
        result_dir_test = os.path.join(result_dir, 'test')

        if not os.path.exists(result_dir_test):
            os.makedirs(os.path.join(result_dir_test, 'png', 'a2b'))
            os.makedirs(os.path.join(result_dir_test, 'png', 'b2a'))
            os.makedirs(os.path.join(result_dir_test, 'npy', 'a2b'))
            os.makedirs(os.path.join(result_dir_test, 'npy', 'b2a'))
            # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    def train(self, discovery_train, prisma_train, discovery_train_subjectkeys, prisma_train_subjectkeys):

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

        ## 네트워크 생성하기
        # which_model:  resnet, unet_128, unet_256
        netG = define_G(input_nc=nch, output_nc=nch, ngf=nker, which_model_netG=which_model_netG, norm=norm, nblk=9,
                        use_dropout=False, gpu_ids=gpus_list)

        if d_loss == "BCE":
            netD = define_D(input_nc=nch, ndf=nker, which_model_netD=which_model_netD, n_layers_D=3, norm=norm,
                            use_sigmoid=True, gpu_ids=gpus_list)

        elif d_loss == "wgangp":
            netD = define_D(input_nc=nch, ndf=nker, which_model_netD=which_model_netD, n_layers_D=3, norm=norm,
                            use_sigmoid=False, gpu_ids=gpus_list)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

        ## 손실함수 정의하기
        fn_l1 = nn.L1Loss().to(device)
        fn_gan = nn.BCELoss().to(device)
        fn_cls = nn.BCELoss().to(device)
        fn_ident = nn.L1Loss().to(device)

        ## Optimizer 설정하기
        optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
        optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

        # schedG = get_scheduler(optimG, args)
        # schedD = get_scheduler(optimD, args)

        ## 그밖에 부수적인 functions 설정하기
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # For 3D images
        # fn_denorm = lambda x: (x * STD) + MEAN


        ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
        writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

        ## 네트워크 학습시키기
        st_epoch = 0

        # TRAIN MODE
        if mode == 'train':
            if train_continue == "on":
                netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                            netG=netG, netD=netD, optimG=optimG, optimD=optimD,
                                                            device=device)

            now = time.localtime()
            print("%04d/%02d/%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
            for epoch in range(st_epoch + 1, num_epoch + 1):
                netG.train()
                netD.train()

                loss_G_train = []
                loss_D_train = []

                ref_time = time.time()
                end_time = 0
                for batch, data in enumerate(loader_train, 1):
                    start = time.time()
                    print(f'batch: {batch} IO time: {(start - end_time)}')
                    input_a = data['data_a'].to(device)
                    subjectkeys_a = data['subjectkey_a']
                    input_b = data['data_b'].to(device)
                    subjectkeys_b = data['subjectkey_b']

                    # print('input:',input_a.shape) # (8,1,96,96,96)
                    inputs = torch.cat((input_a, input_b), dim=0)

                    # forward netG
                    outputs = netG(inputs)

                    # print('output:',output_a.shape) # (8,1,96,96,96)

                    recon = fn_l1(inputs, outputs)

                    # backward netD
                    set_requires_grad([netD], True)
                    optimD.zero_grad()

                    pred_real = netD(inputs)
                    pred_fake = netD(outputs.detach())

                    # assume that we load pretrained classifier that can classify input_a as 0, and input_b as 1
                    lambda_cls = 10
                    preds = pretrained_classifier(outputs)  # frozen
                    fake_labels = torch.cat((torch.ones_like(input_a.shape[0])), (torch.zeros_like(input_b.shape[0])))
                    cls_loss = fn_cls(preds, fake_labels)

                    # BCE loss
                    if d_loss == "BCE":
                        loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                        loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                        loss_D = 0.5 * (loss_D_real + loss_D_fake) + recon + lambda_cls * cls_loss
                    elif d_loss == "wgangp":
                        gradient_penalty = calc_gradient_penalty(netD, inputs.data, outputs.data)
                        loss_D = -torch.mean(pred_real) + torch.mean(
                            pred_fake) + gradient_penalty + recon + lambda_cls * cls_loss

                    loss_D.backward()
                    optimD.step()

                    # backward netG
                    set_requires_grad([netD], False)
                    optimG.zero_grad()

                    outputs = netG(inputs)
                    pred_fake = netD(outputs)

                    if d_loss == "wgangp":
                        loss_G_GAN = -torch.mean(pred_fake)

                    # assume that we load pretrained classifier that can classify input_a as 0, and input_b as 1
                    lambda_cls = 10
                    preds = pretrained_classifier(outputs)  # frozen
                    fake_labels = torch.cat((torch.ones_like(input_a.shape[0])), (torch.zeros_like(input_b.shape[0])))
                    cls_loss = fn_cls(preds, fake_labels)

                    loss_G = loss_G_GAN + cls_loss

                    loss_G.backward()
                    optimG.step()

                    # 손실함수 계산
                    loss_G_a2b_train += [loss_G_a2b.item()]
                    loss_G_b2a_train += [loss_G_b2a.item()]

                    loss_D_a_train += [loss_D_a.item()]
                    loss_D_b_train += [loss_D_b.item()]

                    loss_cycle_a_train += [loss_cycle_a.item()]
                    loss_cycle_b_train += [loss_cycle_b.item()]

                    loss_ident_a_train += [loss_ident_a.item()]
                    loss_ident_b_train += [loss_ident_b.item()]

                    end_time = time.time()
                    print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                          "GEN a2b %.4f b2a %.4f | "
                          "DISC a %.4f b %.4f | "
                          "CYCLE a %.4f b %.4f | "
                          "IDENT a %.4f b %.4f | "
                          "Process TIME (except IO) %.2f | " %
                          (epoch, num_epoch, batch, num_batch_train,
                           np.mean(loss_G_a2b_train), np.mean(loss_G_b2a_train),
                           np.mean(loss_D_a_train), np.mean(loss_D_b_train),
                           np.mean(loss_cycle_a_train), np.mean(loss_cycle_b_train),
                           np.mean(loss_ident_a_train), np.mean(loss_ident_b_train), (end_time - start)))

                    if batch % 20 == 0:
                        # Tensorboard 저장하기

                        coronal_plane = int(input_a.shape[3] / 2)
                        input_a = fn_tonumpy(
                            input_a[:, :, :, coronal_plane, :]).squeeze()  # (BATCH,X,Z) #channel is squeezed out
                        input_b = fn_tonumpy(input_b[:, :, :, coronal_plane, :]).squeeze()  # (BATCH,X,Z,)
                        output_a = fn_tonumpy(output_a[:, :, :, coronal_plane, :]).squeeze()  # (BATCH,X,Z)
                        output_b = fn_tonumpy(output_b[:, :, :, coronal_plane, :]).squeeze()  # (BATCH,X,Z)

                        id = num_batch_train * (epoch - 1) + batch

                        # plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', f'{subjectkeys_a[0]}_input_a.png'), np.rot90(input_a[0],3),
                        #            cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', f'{subjectkeys_a[0]}_output_b.png'), np.rot90(output_b[0],3),
                        #            cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', f'{subjectkeys_b[0]}_input_b.png'), np.rot90(input_b[0],3),
                        #            cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', f'{subjectkeys_b[0]}_output_a.png'), np.rot90(output_a[0],3),
                        #            cmap=cmap)

                        writer_train.add_image('input_a', np.rot90(input_a[:, np.newaxis, :, :], 3, axes=(2, 3)), id,
                                               dataformats='NCHW')
                        writer_train.add_image('output_b', np.rot90(output_b[:, np.newaxis, :, :], 3, axes=(2, 3)), id,
                                               dataformats='NCHW')
                        writer_train.add_image('input_b', np.rot90(input_b[:, np.newaxis, :, :], 3, axes=(2, 3)), id,
                                               dataformats='NCHW')
                        writer_train.add_image('output_a', np.rot90(output_a[:, np.newaxis, :, :], 3, axes=(2, 3)), id,
                                               dataformats='NCHW')

                # For scalability test
                # print("Seconds spent for EPOCH %04d / %04d : %.3f \n"
                #       "Samples Per Second : %.2f "
                #       % (epoch, num_epoch, (end_time - ref_time), len(dataset_train) / (end_time - ref_time)))
                #
                #
                # listitem = [epoch, end_time - ref_time, len(dataset_train) / (end_time - ref_time)]
                #
                # with open('/home/connectome/junb/GANBERT/t1_harmonization/3D-cycleGAN/List_HDD_workers0.csv', 'a') as file:
                #      write = csv.writer(file)
                #      write.writerow(listitem)

                writer_train.add_scalar('loss_G_a2b', np.mean(loss_G_a2b_train), epoch)
                writer_train.add_scalar('loss_G_b2a', np.mean(loss_G_b2a_train), epoch)

                writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)
                writer_train.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch)

                writer_train.add_scalar('loss_cycle_a', np.mean(loss_cycle_a_train), epoch)
                writer_train.add_scalar('loss_cycle_b', np.mean(loss_cycle_b_train), epoch)

                writer_train.add_scalar('loss_ident_a', np.mean(loss_ident_a_train), epoch)
                writer_train.add_scalar('loss_ident_b', np.mean(loss_ident_b_train), epoch)

                if epoch % 2 == 0 or epoch == num_epoch:
                    save(ckpt_dir=ckpt_dir, epoch=epoch, netG, netD, optimG=optimG, optimD=optimD)

                # update schduler
                # schedG.step()
                # schedD.step()

            writer_train.close()


    def test(args, discovery_test, prisma_test, discovery_test_subjectkeys, prisma_test_subjectkeys):
        ## 손실함수 정의하기
        fn_cycle = nn.L1Loss().to(device)
        fn_gan = nn.BCELoss().to(device)
        fn_ident = nn.L1Loss().to(device)

        ## Optimizer 설정하기
        optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
        optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999))

        ## 그밖에 부수적인 functions 설정하기
        fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # for 3D image
        # fn_denorm = lambda x: (x * STD) + MEAN

        ## 네트워크 학습시키기
        st_epoch = 0

        # TEST MODE
        if mode == "test":
            netG_a2b, netG_b2a, \
            netD_a, netD_b, \
            optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                                            netD_a=netD_a, netD_b=netD_b,
                                            optimG=optimG, optimD=optimD, device=device)

            print('loading model from epoch %04d' % (st_epoch))
            with torch.no_grad():
                netG_a2b.eval()
                netG_b2a.eval()

                for batch, data in enumerate(loader_test_a, 1):
                    # forward pass
                    input_a = data['data_a'].to(device)
                    subjectkeys_a = data['subjectkey_a']

                    output_b = netG_a2b(input_a)

                    # volume save
                    if test_save == "on":
                        input_a_npy = input_a.to('cpu').detach().numpy().squeeze()
                        output_b_npy = output_b.to('cpu').detach().numpy().squeeze()
                        for j in range(input_a_npy.shape[0]):
                            input_a_ = input_a_npy[j]
                            output_b_ = output_b_npy[j]
                            np.save(os.path.join(result_dir_test, 'npy', 'a2b', f'{subjectkeys_a[j]}_input_a.npy'),
                                    input_a_)
                            np.save(os.path.join(result_dir_test, 'npy', 'a2b', f'{subjectkeys_a[j]}_output_b.npy'),
                                    output_b_)

                    # slice save
                    coronal_plane = int(input_a.shape[3] / 2)

                    # Tensorboard 저장하기
                    input_a = fn_tonumpy(input_a[:, :, :, coronal_plane, :]).squeeze()
                    output_b = fn_tonumpy(output_b[:, :, :, coronal_plane, :]).squeeze()

                    for j in range(input_a.shape[0]):
                        id = batch_size * (batch - 1) + j

                        input_a_ = input_a[j]
                        output_b_ = output_b[j]

                        # plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', f'{subjectkeys_a[j]}_input_a.png'), np.rot90(input_a_,3), cmap='gray')
                        # plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', f'{subjectkeys_a[j]}_output_b.png'), np.rot90(output_b_,3), cmap='gray')

                        print("TEST A: BATCH %04d / %04d | " % (id + 1, num_data_test_a))

                for batch, data in enumerate(loader_test_b, 1):
                    # forward pass
                    input_b = data['data_b'].to(device)
                    subjectkeys_b = data['subjectkey_b']

                    output_a = netG_b2a(input_b)

                    # volume save
                    if test_save == "on":
                        input_b_npy = input_b.to('cpu').detach().numpy().squeeze()
                        output_a_npy = output_a.to('cpu').detach().numpy().squeeze()
                        for j in range(input_b_npy.shape[0]):
                            input_b_ = input_b_npy[j]
                            output_a_ = output_a_npy[j]
                            np.save(os.path.join(result_dir_test, 'npy', 'b2a', f'{subjectkeys_b[j]}_input_b.npy'),
                                    input_b_)
                            np.save(os.path.join(result_dir_test, 'npy', 'b2a', f'{subjectkeys_b[j]}_output_a.npy'),
                                    output_a_)

                    # slice save
                    coronal_plane = int(input_b.shape[3] / 2)

                    # Tensorboard 저장하기
                    input_b = fn_tonumpy(input_b[:, :, :, coronal_plane, :]).squeeze()
                    output_a = fn_tonumpy(output_a[:, :, :, coronal_plane, :]).squeeze()

                    for j in range(input_b.shape[0]):
                        id = batch_size * (batch - 1) + j

                        input_b_ = input_b[j]
                        output_a_ = output_a[j]

                        # plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', f'{subjectkeys_b[j]}_input_b.png'), np.rot90(input_b_,3), cmap = 'gray')
                        # plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', f'{subjectkeys_b[j]}_output_a.png'), np.rot90(output_a_,3), cmap = 'gray')

                        print("TEST B: BATCH %04d / %04d | " % (id + 1, num_data_test_b))