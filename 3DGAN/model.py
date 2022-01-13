import os
import torch
import scipy.io as sio
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator

class GAN_3D(object):
    def __init__(self, args, config):
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore
        self.config = config

        # init networks
        self.G = Generator()
        self.D = Discriminator()

        self.adv_criterion = torch.nn.BCELoss()
        self.img_criterion = torch.nn.L1Loss()

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

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

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
            self.writer.add_scalar(tag, value, self.step)

    def save_img(self, save_num=5):
        for i in range(save_num):
            mdict = {
                'instance': self.fake_X[i,0].data.cpu().numpy()
            }
            sio.savemat(os.path.join(self.config.img_dir, '{:06d}_{:02d}.mat'.format(self.step, i)), mdict)

    def save_model(self):
        torch.save({key: val.cpu() for key, val in self.G.state_dict().items()}, os.path.join(self.config.model_dir, 'G_iter_{:06d}.pth'.format(self.step)))
        torch.save({key: val.cpu() for key, val in self.D.state_dict().items()}, os.path.join(self.config.model_dir, 'D_iter_{:06d}.pth'.format(self.step)))

    ### Train & Test functions
    def train(self, dataset, epoch):
        print("Start Training !!!")
        self.writer = SummaryWriter(self.config.log_dir)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config.G_lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config.D_lr, betas=(0.5, 0.999))
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_G, step_size=self.config.step_size, gamma=self.config.gamma)
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=self.config.step_size, gamma=self.config.gamma)

        # start training
        for epoch in range(self.start_step, 1 + epoch):
            self.step = epoch
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for i, (struct, dwi, grad) in enumerate(dataset):
                print(struct.shape)
                print(dwi.shape)

                struct = struct.cuda()
                dwi = dwi.cuda()
                fake_dwi = self.G(struct)

                """ Generator """
                D_judge = self.D(fake_dwi)
                self.G_loss = {'adv_fake': self.adv_criterion(D_judge, torch.ones_like(D_judge)),
                               'real_fake': self.img_criterion(fake_dwi, dwi)}
                self.loss_G = sum(self.G_loss.values())
                self.opt_G.zero_grad()
                self.loss_G .backward()
                self.opt_G.step()

                """ Discriminator """
                D_j_real = self.D(struct)
                D_j_fake = self.D(fake_dwi)
                self.D_loss = {'adv_real': self.adv_criterion(D_j_real, torch.ones_like(D_j_real)),
                               'adv_fake': self.adv_criterion(D_j_fake, torch.zeros_like(D_j_fake))}
                self.loss_D = sum(self.D_loss.values())

                self.opt_D.zero_grad()
                self.loss_D.backward()
                self.opt_D.step()

            print('step: {:06d}, loss_D: {:.6f}, loss_G: {:.6f}'.format(self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            if self.step % 100 == 0:
                self.save_log()

            if self.step % 1000 == 0:
                self.save_img()
                self.save_model()

        print('Finished training!')
        self.writer.close()