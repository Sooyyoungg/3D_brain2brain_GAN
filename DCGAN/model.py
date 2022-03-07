import os
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import nibabel as nib
from nilearn import plotting
import scipy.io as sio
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator

class GAN_3D(nn.Module):
    def __init__(self, dataset, config):
        super(GAN_3D, self).__init__()
        self.config = config
        self.gpu = config.gpu
        self.device = torch.device('cuda:' + str(config.gpu[0]) if torch.cuda.is_available() else 'cpu')
        self.mode = config.mode
        self.restore = config.restore
        self.tot_epoch = config.epoch
        self.batch_size = config.batch_size

        if len(dataset) > 1:
            print("Training Start!")
            self.train_data = dataset[0]
            self.valid_data = dataset[1]
        else:
            print("Testing Start!")
            self.test_data = dataset

        # init networks
        self.G = Generator()
        self.D = Discriminator()

        self.adv_criterion = torch.nn.BCELoss()
        self.img_criterion = torch.nn.L1Loss()
        # Gradient Penalty loss
        #self.gp_criterion = torch.

        self.set_mode_and_gpu()
        self.restore_from_file()

    ### functions
    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.G.train()
            self.D.train()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()
                    self.adv_criterion.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

        elif self.mode == 'test':
            self.G.eval()
            self.D.eval()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_G = os.path.join(self.config.model_dir, 'G_iter_{:06d}.pth'.format(self.restore))
            assert os.path.exists(ckpt_file_G)
            self.G.load_state_dict(torch.load(ckpt_file_G))

            if self.mode == 'train':
                ckpt_file_D = os.path.join(self.config.model_dir, 'D_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_D)
                self.D.load_state_dict(torch.load(ckpt_file_D))

            self.start_epoch = self.restore + 1
        else:
            self.start_epoch = 1

    def save_log(self, epoch):
        scalar_info = {
            'loss_D': self.loss_D,
            'loss_G': self.loss_G,
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0],
        }
        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, epoch)

    def save_img(self, epoch, save_num=5):
        # for i in range(save_num):
        #     mdict = {'instance': self.fake_dwi[i,0].data.cpu().numpy()}
        #     sio.savemat(os.path.join(self.config.img_dir, '{:06d}_{:02d}.mat'.format(epoch, i)), mdict)
        plt.imsave(os.path.join(self.config.img_dir, 'DCGAN_{:04d}_real.png'.format(epoch)),
                   self.dwi[self.batch_size//2,0,:,:,32].detach().cpu().numpy(), cmap='gray')
        plt.imsave(os.path.join(self.config.img_dir, 'DCGAN_{:04d}_fake.png'.format(epoch)),
                   self.fake_dwi[self.batch_size//2,0,:,:,32].detach().cpu().numpy(), cmap='gray')


    def vis_img(self, real_imgs, fake_imgs):
        # Visualize generated image
        feat = np.squeeze((0.5 * real_imgs[0] + 0.5).detach().cpu().numpy())
        feat = nib.Nifti1Image(feat, affine=np.eye(4))
        plotting.plot_anat(feat, title="DCGAN_Real_imgs", cut_coords=(32, 32, 32))
        plotting.show()

        feat_f = np.squeeze((0.5 * fake_imgs[0] + 0.5).detach().cpu().numpy())
        feat_f = nib.Nifti1Image(feat_f, affine=np.eye(4))
        plotting.plot_anat(feat_f, title="DCGAN_fake_imgs", cut_coords=(32, 32, 32))
        plotting.show()

    def save_model(self, epoch):
        torch.save({key: val.cpu() for key, val in self.G.state_dict().items()}, os.path.join(self.config.model_dir, 'G_iter_{:04d}.pth'.format(epoch)))
        torch.save({key: val.cpu() for key, val in self.D.state_dict().items()}, os.path.join(self.config.model_dir, 'D_iter_{:04d}.pth'.format(epoch)))

    ### Train & Test functions
    def train(self, **kwargs):
        self.writer = SummaryWriter(self.config.log_dir)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config.G_lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config.D_lr, betas=(0.5, 0.999))
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_G, step_size=self.config.epoch, gamma=self.config.gamma)
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=self.config.epoch, gamma=self.config.gamma)

        self.val_loss = 0.0

        start_time = time.time()
        # start training
        for epoch in range(self.start_epoch, 1 + self.tot_epoch):
            epoch_time = time.time()
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for i, (struct, dwi) in enumerate(self.train_data):
                if epoch == 1 and i == 0:
                    print("Training structure mri shape: ", struct.shape)
                    print("Training diffusion-weighted image shape: ", dwi.shape)

                struct = struct.to(self.device).float()
                self.dwi = dwi.to(self.device).float()
                self.fake_dwi = self.G(struct)

                """ Generator """
                D_judge = self.D(self.fake_dwi)   # shape: [batch_size, 1]
                self.G_loss = {'adv_fake': self.adv_criterion(D_judge, torch.ones_like(D_judge))}
                #self.G_loss = {'adv_fake': self.adv_criterion(D_judge, torch.ones_like(D_judge)),
                #               'real_fake': self.img_criterion(self.fake_dwi, dwi)}
                self.loss_G = sum(self.G_loss.values())
                self.opt_G.zero_grad()
                self.loss_G.backward()
                self.opt_G.step()

                """ Discriminator """
                D_j_real = self.D(self.dwi)
                D_j_fake = self.D(self.fake_dwi.detach())
                self.D_loss = {'adv_real': self.adv_criterion(D_j_real, torch.ones_like(D_j_real)),
                               'adv_fake': self.adv_criterion(D_j_fake, torch.zeros_like(D_j_fake))}
                self.loss_D = sum(self.D_loss.values())
                self.opt_D.zero_grad()
                self.loss_D.backward()
                self.opt_D.step()

            print('epoch: {:04d}, loss_D: {:.6f}, loss_G: {:.6f}'.format(epoch, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))
            print('Time for an epoch: ', time.time() - epoch_time)

            self.vis_img(dwi, self.fake_dwi)
            self.save_img(epoch)
            # self.save_model(epoch)

            """ Validation """
            if epoch % 100 == 0 or epoch == 1:
                with torch.no_grad():
                    self.valid(epoch)

            # if epoch % 100 == 0:
            #    self.save_log(epoch)

        print('Finish training !!!')
        print('Total Training Time: ', time.time() - start_time)
        self.writer.close()

    def valid(self, epoch):
        print("Validation Start!")
        with torch.no_grad():
            self.G.eval()
            val_losses = 0.0
            for v_i, (v_str, v_dwi) in enumerate(self.valid_data):
                v_str = v_str.to(self.device).float()
                v_dwi = v_dwi.to(self.device).float()
                v_fake_dwi = self.G(v_str)
                val_losses += self.img_criterion(v_fake_dwi, v_dwi)
            val_avg_loss = val_losses / float(v_i + 1.0)

            # If valid loss has the highest score,
            if val_avg_loss > self.val_loss:
                # save loss value
                self.val_loss = val_avg_loss
                # save model info & image
                #self.save_log(epoch)
                #self.save_img(save_num=3)
                #self.save_model(epoch)

                print("======= The highest validation score! =======")
                print('epoch: {:04d}, loss_valid for Generator: {:.6f}'.format(epoch, self.val_loss))

    def test(self):
        with torch.no_grad():
            for i, (struct, dwi) in enumerate(self.test_data):
                struct = struct.to(self.device).float()
                dwi = dwi.to(self.device).float()
                self.test_fake_dwi = self.G(struct)

                """ Generator """
                test_D_judge = self.D(self.test_fake_dwi)
                self.test_G_loss = {'adv_fake': self.adv_criterion(test_D_judge, torch.ones_like(test_D_judge))}
                # self.test_G_loss = {'adv_fake': self.adv_criterion(test_D_judge, torch.ones_like(test_D_judge)),
                #                     'real_fake': self.img_criterion(self.test_fake_dwi, dwi)}
                self.test_loss_G = sum(self.test_G_loss.values())

                """ Discriminator """
                test_D_j_real = self.D(dwi)
                test_D_j_fake = self.D(self.test_fake_dwi)
                self.test_D_loss = {'adv_real': self.adv_criterion(test_D_j_real, torch.ones_like(test_D_j_real)),
                                    'adv_fake': self.adv_criterion(test_D_j_fake, torch.zeros_like(test_D_j_fake))}
                self.test_loss_D = sum(self.test_D_loss.values())

            print('Test Results : loss_D: {:.6f}, loss_G: {:.6f}'.format(self.test_loss_D.data.cpu().numpy(), self.test_loss_G.data.cpu().numpy()))