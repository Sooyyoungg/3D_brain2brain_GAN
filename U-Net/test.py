


def test(args, discovery_test, prisma_test, discovery_test_subjectkeys, prisma_test_subjectkeys):
    ## 트레이닝 파라메터 설정하기
    NUM_WORKER = args.num_workers
    mode = args.mode
    train_continue = args.train_continue
    test_save = args.test_save

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    # Newly added parameters
    subject_info = args.subject_info
    target = args.target
    t1_dir = args.t1_dir
    resize = (int(args.resize), int(args.resize), int(args.resize))

    # data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    gpus_list = args.gpus_list

    task = args.task
    which_model_netG = args.which_model_netG
    which_model_netD = args.which_model_netD
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    # for test with cpu
    device = torch.device(f'cuda:{gpus_list[0]}' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("test_save: %s" % test_save)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("which_model_netG %s" % which_model_netG)
    print("which_model_netD %s" % which_model_netD)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("target: %s" % target)
    print("t1_dir: %s" % t1_dir)

    # print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("gpus_list: %s" % gpus_list)
    print("device: %s" % device)

    cmap = plt.cm.gray

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png', 'a2b'))
        os.makedirs(os.path.join(result_dir_test, 'png', 'b2a'))
        os.makedirs(os.path.join(result_dir_test, 'npy', 'a2b'))
        os.makedirs(os.path.join(result_dir_test, 'npy', 'b2a'))
        # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == 'test':
        test_transform = Compose(
            [ScaleIntensity(minv=-1.0, maxv=1.0),  # array의 각 원소를 그것의 최댓값으로 나누어주어 0~1로 바꿔주는 것. (MinMax Scaler())
             # NormalizeIntensity(subtrahend=MEAN, divisor=STD, nonzero=False),  # 평균 0.5, 표준편차 0.5로 표준화
             AddChannel(),
             Resize(resize),
             ToTensor()])

        dataset_test_a = ABCDImageDataset(discovery_test, prisma_test, discovery_test_subjectkeys,
                                          prisma_test_subjectkeys, transform=test_transform, task=task,
                                          data_type='a')

        loader_test_a = DataLoader(dataset_test_a, batch_size=batch_size,
                                   shuffle=False, num_workers=NUM_WORKER, drop_last=False)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_a = len(dataset_test_a)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

        dataset_test_b = ABCDImageDataset(discovery_test, prisma_test, discovery_test_subjectkeys,
                                          prisma_test_subjectkeys, transform=test_transform, task=task,
                                          data_type='b')

        loader_test_b = DataLoader(dataset_test_b, batch_size=batch_size,
                                   shuffle=False, num_workers=NUM_WORKER, drop_last=False)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_b = len(dataset_test_b)
        num_batch_test_b = np.ceil(num_data_test_b / batch_size)

    ## 네트워크 생성하기
    if network == "CycleGAN":
        netG_a2b = define_G(input_nc=nch, output_nc=nch, ngf=nker, which_model_netG=which_model_netG, norm=norm, nblk=9,
                            use_dropout=False, gpu_ids=gpus_list)
        netG_b2a = define_G(input_nc=nch, output_nc=nch, ngf=nker, which_model_netG=which_model_netG, norm=norm, nblk=9,
                            use_dropout=False, gpu_ids=gpus_list)

        netD_a = define_D(input_nc=nch, ndf=nker, which_model_netD=which_model_netD, n_layers_D=3, norm=norm,
                          use_sigmoid=False,
                          gpu_ids=gpus_list)
        netD_b = define_D(input_nc=nch, ndf=nker, which_model_netD=which_model_netD, n_layers_D=3, norm=norm,
                          use_sigmoid=False,
                          gpu_ids=gpus_list)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)

        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

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