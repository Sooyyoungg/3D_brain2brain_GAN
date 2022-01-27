import os
import torch
from torch import nn
import scipy.io as sio
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator

class I2I_cGAN(nn.Module):
    def __init__(self, dataset, config):
        super(I2I_cGAN, self).__init__()
        self.gpu = config.gpu
        self.mode = config.mode
        self.restore = config.restore
        self.config = config
        self.epoch = config.epoch
        print("model in!")

        print(len(dataset))
        if len(dataset) > 1:
            self.train_data = dataset[0]
            self.valid_data = dataset[1]
            train_iter = iter(self.train_data)
            dwi, grad = train_iter.next()
            print(type(dwi))   # <class 'torch.Tensor'>
            print(dwi.size())  #
            print(grad.size()) #
        else:
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

    def save_log(self):
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
            self.writer.add_scalar(tag, value, self.epoch)

    def save_img(self, save_num=5):
        for i in range(save_num):
            mdict = {'instance': self.fake_dwi[i,0].data.cpu().numpy()}
            sio.savemat(os.path.join(self.config.img_dir, '{:06d}_{:02d}.mat'.format(self.epoch, i)), mdict)

    def save_model(self):
        torch.save({key: val.cpu() for key, val in self.G.state_dict().items()}, os.path.join(self.config.model_dir, 'G_iter_{:06d}.pth'.format(self.epoch)))
        torch.save({key: val.cpu() for key, val in self.D.state_dict().items()}, os.path.join(self.config.model_dir, 'D_iter_{:06d}.pth'.format(self.epoch)))

    ### Train & Test functions
    def train(self, **kwargs):
        print("Start Training !!!")
        self.writer = SummaryWriter(self.config.log_dir)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config.G_lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config.D_lr, betas=(0.5, 0.999))
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_G, step_size=self.config.epoch, gamma=self.config.gamma)
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=self.config.epoch, gamma=self.config.gamma)

        self.val_loss = 0.0

        # start training
        for epoch in range(self.start_epoch, 1 + self.epoch):
            self.epoch = epoch
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for i, struct, dwi in enumerate(self.train_data):
                if i == 0:
                    print("Training structure mri shape: ", struct.shape)
                    print("Training diffusion-weighted image shape: ", dwi.shape)

                struct = struct.cuda()
                dwi = dwi.cuda()
                self.fake_dwi = self.G(struct)

                """ Generator """
                D_judge = self.D(self.fake_dwi)
                self.G_loss = {'adv_fake': self.adv_criterion(D_judge, torch.ones_like(D_judge)),
                               'real_fake': self.img_criterion(self.fake_dwi, dwi)}
                self.loss_G = sum(self.G_loss.values())
                self.opt_G.zero_grad()
                self.loss_G .backward()
                self.opt_G.step()

                """ Discriminator """
                D_j_real = self.D(struct)
                D_j_fake = self.D(self.fake_dwi)
                self.D_loss = {'adv_real': self.adv_criterion(D_j_real, torch.ones_like(D_j_real)),
                               'adv_fake': self.adv_criterion(D_j_fake, torch.zeros_like(D_j_fake))}
                self.loss_D = sum(self.D_loss.values())

                self.opt_D.zero_grad()
                self.loss_D.backward()
                self.opt_D.step()

            print('epoch: {:06d}, loss_D: {:.6f}, loss_G: {:.6f}'.format(self.epoch, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            """ Validation """
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.valid(self.valid_data)

            """if self.step % 100 == 0:
                self.save_log()

            if self.step % 1000 == 0:
                self.save_img()
                self.save_model()"""

        print('Finish training !!!')
        self.writer.close()

    def valid(self, valid_data):
        self.G.eval()

        val_losses = 0.0
        v_i = 0
        for v_i, v_str, v_dwi, v_grad in enumerate(valid_data):
            v_fake_dwi = self.G(v_str)
            val_losses += self.img_criterion(v_fake_dwi, v_dwi)
        val_avg_loss = val_losses / float(v_i + 1.0)

        # If valid loss has the highest score,
        if val_avg_loss > self.val_loss:
            # save loss value
            self.val_loss = val_avg_loss
            # save model info & image
            self.save_log()
            self.save_img(save_num=3)
            self.save_model()

            print("======= The highest validation score! =======")
            print('epoch: {:06d}, loss_valid for Generator: {:.6f}'.format(self.epoch, self.val_loss))

    def test(self):
        with torch.no_grad():
            for i, struct, dwi, grad in enumerate(self.test_data):
                struct = struct.cuda()
                dwi = dwi.cuda()
                self.test_fake_dwi = self.G(struct)

                """ Generator """
                test_D_judge = self.D(self.test_fake_dwi)
                self.test_G_loss = {'adv_fake': self.adv_criterion(test_D_judge, torch.ones_like(test_D_judge)),
                                    'real_fake': self.img_criterion(self.test_fake_dwi, dwi)}
                self.test_loss_G = sum(self.test_G_loss.values())

                """ Discriminator """
                test_D_j_real = self.D(struct)
                test_D_j_fake = self.D(self.test_fake_dwi)
                self.test_D_loss = {'adv_real': self.adv_criterion(test_D_j_real, torch.ones_like(test_D_j_real)),
                                    'adv_fake': self.adv_criterion(test_D_j_fake, torch.zeros_like(test_D_j_fake))}
                self.test_loss_D = sum(self.test_D_loss.values())

            print('Test Results : loss_D: {:.6f}, loss_G: {:.6f}'.format(self.test_loss_D.data.cpu().numpy(), self.test_loss_G.data.cpu().numpy()))