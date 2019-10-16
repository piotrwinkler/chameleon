## Testy wstępne

V1:

    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(outputs, labels)

V2:

    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(outputs, labels) / output.size(0)
    
V10:
    
    Whole dataset loaded with cifar_dataset_class, entire train Y set is standarized per channel,
    X set is from -0.5 to 0.5 with classic normalization.
    
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_step_scheduler = 1
    lr_step_gamma = 0.9
    step_decay = 0.5
    decay_after_steps = 20
    
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss = criterion(outputs, ab_batch.to(device))

    model - FCN_net1

    Results: 
    
V11:   !!! przy zmianie ab_chosen_normalization na normalization jest coś ciekawego
    
    !!! ogólnie jak się nauczy sieć na zestandaryzowanym ab, a potem wyjściowe ab z sieci będzie się nie destandaryzować,
    a denormalizowac ale ograniczając pixele od -127 do 128 a nawet mniej to są świetne wyniki
    
    Whole dataset loaded with cifar_dataset_class, entire train Y set is standarized per channel,
    X set is from -0.5 to 0.5 with classic normalization and with Gausian Blur with kernel (7, 7) and automatic sigma (0).
    
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    lr_step_scheduler = 1
    lr_step_gamma = 0.9
    step_decay = 0.5
    decay_after_steps = 20
    
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss = criterion(outputs, ab_batch.to(device))

    model - FCN_net1

    Results: Oct14_20-59-38_DESKTOP-K2JRB94
    Wnioski: Loss tragedia, cały czas na tym samym poziomie

V12:
    
    Whole dataset loaded with cifar_dataset_class, entire train Y set is normalized,
    X set is from -0.5 to 0.5 with classic normalization and with Gausian Blur with kernel (7, 7) and automatic sigma (0).
    
    batch_size = 128
    learning_rate = 0.001
    momentum = 0.9
    lr_step_scheduler = 1
    lr_step_gamma = 0.99
    step_decay = 0.5
    decay_after_steps = 20
    
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss = criterion(outputs, ab_batch.to(device))

    model - FCN_net1

    Results:  Oct14_21-50-04_DESKTOP-K2JRB94
    
    Wnioski: Podobnie jak w V13, ale nieba są lekko bardziej niebieskie ale jest też więcej niebieskiech plam

V13:
    
    Whole dataset loaded with cifar_dataset_class, entire train Y set is normalized,
    X set is from -0.5 to 0.5 with classic normalization and with Gausian Blur with kernel (7, 7) and automatic sigma (0).
    
    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9
    lr_step_scheduler = 1
    lr_step_gamma = 0.99
    step_decay = 0.5
    decay_after_steps = 20
    
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss = criterion(outputs, ab_batch.to(device))

    model - FCN_net1

    Results: Oct14_22-43-17_DESKTOP-K2JRB94
    
    Wnioski: Szału nie ma, większość brązowa, ale czasami fragmenty nieba są niebieskawe, loss wygląda spoko, ale też 
    szybko się uczy, na obrazkach są jakieś niebieskie plamy często (przez Gaussian blur)
    
    
## Nowa tura testów:


V14:

        init_epoch = 0
        how_many_epochs = 30
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.01
        momentum = 0.9
        # scheduler był wyłączony
        lr_step_scheduler = 1
        lr_step_gamma = 0.99
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = False
        choose_train_dataset = True
        ab_chosen_normalization = "normalization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        criterion = nn.MSELoss(reduction='mean').cuda()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_scheduler, gamma=lr_step_gamma)
        loss = criterion(outputs, ab_batch.to(device))
        
        Results: Oct15_18-15-07_DESKTOP-K2JRB94, Loss = 2,6e-3
                 Oct15_18-40-52_DESKTOP-K2JRB94, Loss = 2,62e-3
                
        
        Wnioski: Słabo wyszło, praktycznie wszystko brązowe, nawet nie ma prześwitów niebieskiego,
        loss wygląda nawet ok, trochę spadł ale przestał się uczyć po 5 min gdzieś
        Z trickiem obrazy są po prostu trochę jaśniejsze, bardziej żółtawe
        
V15:
    
        which_version = "V15"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = False
        choose_train_dataset = True
        ab_chosen_normalization = "normalization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results:  Oct16_18-36-12_DESKTOP-K2JRB94, Loss = 2,53e-3
                  Oct16_18-41-46_DESKTOP-K2JRB94, Loss = 2,58e-3
        
        Wnioski: Słabo, wszystko brązowawe, nawet niebo i ziemia
        Z trickiem bardziej żółtawo niż brązowawo
        
V16:
        
        To samo co V15 ale:
        learning_rate = 0.3

        Results:  Oct16_19-00-43_DESKTOP-K2JRB94, Loss = 2,64e-3
        
        Wnioski: te same praktycznie co w V15
        


 V17:
        
        which_version = "V17"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.3
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = True
        choose_train_dataset = True
        ab_chosen_normalization = "normalization"
        ab_output_normalization = "normalization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
            
        Results:  Oct16_20-32-37_DESKTOP-K2JRB94, Loss = 2,61e-3
        
        Wnioski: Może trochę lepiej niż bez blura, prawie cały czas wszystko brązowe, czasami niebo jest trochę bielsze
        Z trickiem: Trochę lepiej, ale wszystko jest raczej żółtawe
        
V18:
 
        which_version = "V18"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.2
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = True
        choose_train_dataset = True
        ab_chosen_normalization = "normalization"
        ab_output_normalization = "normalization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct16_20-54-56_DESKTOP-K2JRB94, Loss = 2,59e-3
        
        Wnioski: Zmiana kernel size prawie nic nie zmieniło względem V17
        Z trickiem: Może lekko lepiej niż dla V17
        
V19:
 
        which_version = "V19"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.2
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = False
        choose_train_dataset = True
        ab_chosen_normalization = "standardization"
        ab_output_normalization = "standardization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct16_21-19-44_DESKTOP-K2JRB94, Loss = 0.84
        
        Wnioski: Lepiej niż bez standaryzacji ab czyli lepiej niż np V15, niebo jest zazwyczaj niebieskawe
        Z trickiem: Całkiem nieźle, są kolory, niebo jest wyraźne, ale często też podłoga i jakieś fragmenty obiektów są
        niebieskie, kolory też się często przebijają pomiędzy obiektami 
        
V20:
 
        which_version = "V20"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = True
        choose_train_dataset = True
        ab_chosen_normalization = "standardization"
        ab_output_normalization = "standardization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct16_21-54-50_DESKTOP-K2JRB94, Loss = 0,86
        
        Wnioski: Bez tricku jest tak sobie, podobnie jak w V19, chyba nawet gorzej, niebieskość nieba jest słabsza 
        Z trickiem: V19 ma trochę mniej wycieków kolorów, jeśli chodzi o żywośc kolorów to na zmianę raz są żywsze
        w V19, a raz w V20, ogolnie V19 trochę lepsze niż V20
        
V21:
 
        which_version = "V21"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = True
        choose_train_dataset = True
        ab_chosen_normalization = "standardization"
        ab_output_normalization = "standardization"
        L_chosen_normalization = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct16_22-11-21_DESKTOP-K2JRB94, Loss = 0,85
        
        Wnioski: Bez tricka słabo, brązowawo, niebo niebieskawe ale też nie jakoś super
        Z trickiem: Chyba trochę mniej wycieków kolorów niż w V20 ale w sumie porównywalnie 
     
 V22:
 
        which_version = "V22"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 10
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        do_blur_processing = False
        choose_train_dataset = True
        ab_chosen_normalization = "standardization"
        ab_output_normalization = "standardization"
        L_chosen_normalization = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct16_23-23-02_DESKTOP-K2JRB94, Loss = 0,84
        
        Wnioski: Bez tricka słabo, brązowawo, niebo niebieskawe ale też nie jakoś super
        Z trickiem: Chyba trochę mniej wycieków kolorów niż w V20 ale w sumie porównywalnie 