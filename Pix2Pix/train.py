import random
import torch
import pandas as pd
import tensorboardX
import cv2
from sklearn.metrics import mean_squared_error
import torchsummary

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix
import networks

def main():
    config = Config()
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(device)

    ## Data Loader
    train_list = pd.read_csv(config.train_list)
    # valid_list = pd.read_csv(config.valid_list)

    train_data = DataSplit(config=config, data_list=train_list)
    valid_data = DataSplit(config=config, data_list=valid_list)

    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    print("Train: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", len(train_list))
    print("Valid: ", len(data_loader_valid), "x", 1,"(batch size) =", len(valid_list))

    ## Start Training
    model = Pix2Pix(config)
    model.to(device)

    torchsummary.summary(model, (2, 140, 140), device='cpu')

    train_writer = tensorboardX.SummaryWriter(config.log_dir)

    print("Start Training!!")
    itr_per_epoch = len(data_loader_train)
    tot_itr = 0
    min_v_rmse = 100
    for epoch in range(config.n_epoch):
        for i, data in enumerate(data_loader_train):
            tot_itr += i
            train_dict = model.train(i, data)

            fake_depth = train_dict['fake_depth']
            real_depth = train_dict['real_dwi']

            if i % 20 == 0:
                print("image save")
                r = random.randint(0, config.batch_size-1)
                # image rescaling & save
                f_image = fake_depth[r, 0, :, :].detach().cpu().numpy()
                r_image = real_depth[r, 0, :, :].detach().cpu().numpy()
                # post-processing
                # f_image = ((f_image + 1) / 2) * 255.0 ...???????????????????????????????????????????????
                # r_image = ((r_image + 1) / 2) * 255.0 ...???????????????????????????????????????????????
                # save
                cv2.imwrite('{}/Train/{}_{}_fake_dwi.png'.format(config.img_dir, epoch+1, i+1), f_image)
                cv2.imwrite('{}/Train/{}_{}_real_dwi.png'.format(config.img_dir, epoch+1, i+1), r_image)

            # save & print loss values
            train_writer.add_scalar('Loss_G_RMSE', train_dict['G_RMSE_loss'], tot_itr)
            train_writer.add_scalar('Loss_G_GAN', train_dict['G_GAN_loss'], tot_itr)
            train_writer.add_scalar('Loss_G_L1', train_dict['G_L1_loss'], tot_itr)
            train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
            train_writer.add_scalar('Loss_D', train_dict['D_loss'], tot_itr)
            # train_writer.add_scalar('Avg_RMSE', avg_rmse, tot_itr)
            print("Epoch: %d/%d | itr: %d/%d | tot_itrs: %d | Loss_G: %.5f | Loss_D: %.5f | Avg RMSE: %.5f"%(epoch+1, config.n_epoch, i+1, itr_per_epoch, tot_itr, train_dict['G_loss'], train_dict['D_loss'], train_dict['G_RMSE_loss']))
            # print("Epoch: %d/%d | itr: %d/%d | tot_itrs: %d | Loss_G: %.5f | RMSE: %.5f"%(epoch+1, config.n_epoch, i+1, itr_per_epoch, tot_itr, train_dict['G_loss'], train_dict['G_RMSE_loss']))

        valid_G_loss = 0
        valid_D_loss = 0
        valid_mse = 0
        v = 0
        r = random.randint(0, config.batch_size - 1)
        for v, v_data in enumerate(data_loader_valid):
            val_dict = model.val(v_data)
            valid_G_loss += val_dict['G_loss']
            valid_D_loss += val_dict['D_loss']
            v_fake_depth = val_dict['fake_depth']
            v_real_depth = val_dict['real_dwi']

            v_f_image = v_fake_depth[0, 0, :, :].detach().cpu().numpy()
            v_r_image = v_real_depth[0, 0, :, :].detach().cpu().numpy()
            # post-processing
            v_f_image = ((v_f_image + 1) / 2) * 255.0
            v_r_image = ((v_r_image + 1) / 2) * 255.0

            # save image
            if v == r:
                # save
                cv2.imwrite('{}/Validation/{}_fake_depth.png'.format(config.img_dir, epoch + 1), v_f_image)
                cv2.imwrite('{}/Validation/{}_real_depth.png'.format(config.img_dir, epoch + 1), v_r_image)

            # mse
            valid_mse += mean_squared_error(v_f_image, v_r_image)

        v_G_avg_loss = float(valid_G_loss / (v+1))
        v_D_avg_loss = float(valid_D_loss / (v+1))
        valid_rmse = (valid_mse / len(data_loader_valid)) ** 0.5

        train_writer.add_scalar('Valid_Avg_RMSE', valid_rmse, epoch)
        print("===> Validation <=== Epoch: %d/%d | Loss_G: %.5f | Loss_D: %.5f | Avg RMSE: %.5f"%(epoch+1, config.n_epoch, v_G_avg_loss, v_D_avg_loss, valid_rmse))
        # print("===> Validation <=== Epoch: %d/%d | Loss_G: %.5f | RMSE: %.5f"%(epoch+1, config.n_epoch, v_G_avg_loss, valid_rmse))

        networks.update_learning_rate(model.G_scheduler, model.optimizer_G)
        networks.update_learning_rate(model.D_scheduler, model.optimizer_D)

        # save model for each 10 epochs
        if epoch % 10 == 0 or epoch == config.n_epoch - 1:
            torch.save(model.state_dict(), config.log_dir+'/{}_epoch{}_itr{}.pt'.format(config.data_name, epoch+1, tot_itr))
            with open(config.log_dir+'/latest_log.txt', 'w') as f:
                f.writelines('%d, %d'%(epoch+1, tot_itr))

        # save best performance model
        if valid_rmse < min_v_rmse:
            min_v_rmse = valid_rmse
            torch.save(model.state_dict(), config.log_dir+'/{}_best_epoch{}_itr{}_rmse{}.pt'.format(config.data_name, epoch+1, tot_itr, min_v_rmse)) # SEM_best_epoch80_itr123124.pt

if __name__ == '__main__':
    main()