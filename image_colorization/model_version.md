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
    
V11:   !!! przy zmianie ab_input_processing na normalization jest coś ciekawego
    
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
    
    
## Nowa tura testów: model FCN_net1


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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct16_20-54-56_DESKTOP-K2JRB94, Loss = 2,59e-3
        
        Wnioski: Zmiana kernel size prawie nic nie zmieniło względem V17
        Z trickiem: Może lekko lepiej niż dla V17
        
V19:    (Póki co top 3)
 
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct16_23-23-02_DESKTOP-K2JRB94, Loss = 0,84
        
        Wnioski:  Brązowawo, niebo niebieskawe ale też nie jakoś super, ogólnie bez jakichś super kolorów, jedynie 
        całkiem pokolorowany ale trochę wyblakle jest ulubiony koń
        Z trickiem: Są kolory, podobnie jak V20 i V21 ale trochę lepiej, konkuruje z V19, ale chyba jest gorsze
        
V23:

        which_version = "V23"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct17_13-29-10_DESKTOP-K2JRB94 , Loss = 0,86
        
        Wnioski: Bez tricku słabo, wszstko brązowe, rzadko są prześwity niebieskiego na niebie
        Z trickiem: Lepiej niż V22, ale nie zawsze, mniej wycieków kolorów, ale dalej więcej niż w V19, kolory są czasami 
        lekko zbyt żywe, chyba V22 jednak częściej lepsze
        
V24:

        which_version = "V24"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_13-54-56_DESKTOP-K2JRB94 , Loss = 0,85
        
        Wnioski: bez tricku wszystko brązowawe, ale chyba trochę lepiej niż w V23
        Z trickiem: Praktycznie identycznie jak w V23, może leciutko lepiej
        
V25:

        which_version = "V25"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct17_14-25-47_DESKTOP-K2JRB94 , Loss = 2,57e-3
        
        Wnioski: Sam brąz, nawet nie ma przebłysków innych kolorów
        Z trickiem: Wszystko jest żółtawe, tylko żółtawe, bez przebłysków innych kolorów
        
            
V26:

        which_version = "V26"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net1()
        
        gauss_kernel_size = (7, 7)
        
        Results: Oct17_14-45-53_DESKTOP-K2JRB94 , Loss = 2,59e-3
        
        Wnioski: Podobnie jak w V25, praktycznie bez zmian
        Z trickiem: Podobnie jak w V25, praktycznie bez zmian
        
        
### Wnioski przed kolejną fazą:
Dla channeli ab standardyzacja jest o wiele lepsza niż normalizacja, dla obu bez triku ab wyjściowy jest głównie 
brązowawy, ale z trikiem ab znormalizowane jest głównie żółte, a ab zdestandardyzowane ma całkiem fajne kolory

Wielkość kernela dla Gaussa nie ma aż tak dużego znaczenia, pomiędczy (7, 7), a (5, 5) ciężko zauważyć różnicę, może 
lekko lepiej jest dla (5, 5). Ogólnie ten Gauss czasami coś daje, ale nie zawsze, to trzeba przetestować jeszcze na
kolejnych modelach

Dla channelu L ciężko powiedzieć czy lepiej jest ze standardyzacją czy normalizacją, nie widać specjalnie różnicy, 
czasami lepiej jest dla standardyzacji, czasami dla normalizacji, więc mozna póki co pozostać przy normalizacji


## Legenda kolejnych wersji

Model podstawowy to FCN_net1
Możliwe rozszerzenia:
* A - Dodatkowa warstwa 64-64 z kernel_size = 3 i padding=1
* B - Dodatkowa warstwa 32-32 z kernel_size = 3 i padding=1 bardziej na początku
* C - Dodatkowa warstwa 32-32 z kernel_size = 1 i padding=0 bardziej na końcu

## Kolejna faza, model - FCN_net2 (A)
(1 epoka - 180 sek)
(Najlepszy V32)
V30:    

        which_version = "V30"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net2()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_17-13-44_DESKTOP-K2JRB94 , Loss = 0,82
        
        Bez tricku: Niebo zazwyczaj niebieskie, ale reszta brązowa
        Z trickiem: Dobre wyniki, prawie takie same jak dla V19
        
V31:
        
        !!! bez sensu raczej robić, wyjdzie brązowy i żółty
        
        which_version = "V31"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net2()
        
        gauss_kernel_size = (5, 5)
        
        Results:  , Loss = 
        
        Bez tricku: 
        Z trickiem: 
        
V32:    (top 4)

        which_version = "V32"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net2()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_20-46-19_DESKTOP-K2JRB94 , Loss = 0,82
        
        Bez tricku: Brązowo, niebo niebieskawe
        Z trickiem: Całkiem nieźle, lepiej niż V30, porównywalnie z V40
        
V33:

        which_version = "V33"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net2()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_21-23-44_DESKTOP-K2JRB94 , Loss = 0,84
        
        Bez tricku: Brąz ze słabo niebieskim niebem
        Z trickiem: Trochę gorsze niż V32, więcej wycieków kolorów
        
### Wnioski: 
        
## Kolejna faza, model - FCN_net3 (B)
(1 epoka - 115 sek)
(Najlepszy V40)
V40:    (Póki co top 2)

        which_version = "V40"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net3()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_18-21-14_DESKTOP-K2JRB94 , Loss = 0,82
        
        Bez tricku: Ogólnie brązowawo, ale niebo jest trochę niebieskie 
        Z trickiem: Prawie identycznie jak w V19, ale chyba trochę lepiej żywe kolory, na samolocie mini mini też mega
        
V41:

        which_version = "V41"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net3()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_22-05-57_DESKTOP-K2JRB94 , Loss = 2,56e-3
        
        Bez tricku: Praktycznie tylko brązowy, bardzo mało niebieskiego
        Z trickiem: Żółty znowu

V42:

        which_version = "V42"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net3()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_22-27-54_DESKTOP-K2JRB94 , Loss = 0,82
        
        Bez tricku: Brąz z trochę niebieskim niebem i wodą
        Z trickiem: Nieźle ale chyba lekko gorzej niż V40
        
V43:

        which_version = "V43"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net3()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_22-52-12_DESKTOP-K2JRB94 , Loss = 0,84
        
        Bez tricku: Prawie sam brąz z przebłyskami niebieskiego
        Z trickiem: Ok, ale gorsze niż V40, kolory się przebijają i momentami kolorystyka jest aż sztucznie żywa


### Wnioski: 

## Kolejna faza, model - FCN_net4 (C)
(1 epoka - 95 sek)
(Najlepszy V50, ewentualnie V52)
V50: 

        which_version = "V50"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net4()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_19-03-08_DESKTOP-K2JRB94 , Loss = 0,84
        
        Bez tricku: Brązowo, ale niebo zazwyczaj niebieskawe, bez szału
        Z trickiem: Chyba gorzej niż V40, niebieski zaczął się przebijać tu i ówdzie, 
        
V51:

        which_version = "V51"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net4()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_19-37-33_DESKTOP-K2JRB94 , Loss = 2,52e-3
        
        Bez tricku: Brązowo
        Z trickiem: Żółto

V52:

        which_version = "V52"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net4()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_19-55-38_DESKTOP-K2JRB94 , Loss = 0,84
        
        Bez tricku: Brązowo, niebo całkiem niebieskie
        Z trickiem: Podobnie jak V50, ale trochę żywsze kolory
        
V53:

        which_version = "V53"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net4()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct17_20-18-22_DESKTOP-K2JRB94 , Loss = 0,85
        
        Bez tricku: Brązowo, niebo słabo niebieskie
        Z trickiem: Podobnie jak V50, ale trochę niebieski przecieka
        
## Kolejna faza, model - FCN_net5 (A, B, C)
(1 epoka - 250 sek)
V60:

        which_version = "V60"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net5()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct20_15-44-55_DESKTOP-K2JRB94 , Loss = 0,8
        
        Bez tricku: Jest lekki kolor, ale głównie brązowy i szary
        Z trickiem: Spoko kolory, ale raczej gorzej niż w V70
        
V61:

        which_version = "V61"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net5()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct20_16-57-16_DESKTOP-K2JRB94 , Loss = 2.43e-3
        
        Bez tricku: Ogólnie brązowo, ale niebo czasami niebieskie
        Z trickiem: Żółto, ale ogólnie niebo i woda są niebieskie pod tym żółtym

V62:

        which_version = "V62"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net5()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct20_17-59-08_DESKTOP-K2JRB94 , Loss = 0,809
        
        Bez tricku: Ogólnie brązowo, ale niebo czasami niebieskie
        Z trickiem: W sumie spoko, porównywalnie z V70, ale kolory są miejscami mniej wypełnione
        
V63:

        which_version = "V63"
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
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "?"
        
        chosen_net = FCN_net5()
        
        gauss_kernel_size = (5, 5)
        
        Results:  , Loss = 
        
        Bez tricku: 
        Z trickiem: 
        
## Kolejna faza, model - FCN_net_mega (AA, BB, CC)
(1 epoka - 390 sek)
V70:    (top 1)

        which_version = "V70"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 60
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net_mega()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct18_07-35-12_DESKTOP-K2JRB94 , Loss = 0,726
        
        Po 60 epokach:
        Bez tricku: Całkiem spoko, nawet są kolory, trochę wyblakłe ale są
        Z trickiem: Też są kolory fajne, chyba póki co najlepszy, mało przecieków, mega
        
        Po 10 epokach:
        Bez tricku: Brązowo ale są kolorki, więc nieźle, lepiej niż inne po 10 epokach bez tricku, ale chyba tak samo 
        jak po 60 epokach
        Z trickiem: Bardzo podobnie jak po 60 epokach, praktycznie nie da się wypatrzeć różnicy
        
V71:

        which_version = "V71"
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
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "normalization"
        ab_output_processing = "normalization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net_mega()
        
        gauss_kernel_size = (5, 5)
        
        Results:  , Loss = 
        
        Bez tricku: 
        Z trickiem: 

V72:

        which_version = "V72"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 45
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        L_blur_processing = False
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net_mega()
        
        gauss_kernel_size = (5, 5)
        
        Results: Oct18_16-57-27_DESKTOP-K2JRB94 , Loss = 0,73
        
        Po 45 epokach:
        Bez tricku: Całkiem spoko, podobnie jak V70
        Z trickiem: Ogólnie spoko, w porównaniu do V70 są wady i zalety, jest mniej wycieków kolorów, ale z drugiej
        strony kolory są czasami zbyt przejaskrawione
        
        Po 10 epokach:
        Bez tricku: spoko, ale gorzej niż po 45 epokach
        Z trickiem: Gorzej niż po 45 epokach, ale spoko
        
V73:

        which_version = "V73"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 45
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "standardization"
        
        chosen_net = FCN_net_mega()
        
        gauss_kernel_size = (5, 5)
        
        Results:  , Loss = 
        
        Po 45 epokach:
        Bez tricku: 
        Z trickiem: 
        
        Po 10 epokach:
        Bez tricku: 
        Z trickiem: 
        
        
V74:

        which_version = "V74"
        which_epoch_version = 0
        
        load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
        load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
        load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"
        
        log_file = f"logs/logs_fcn_model{which_version}_train.log"
        
        init_epoch = 0
        how_many_epochs = 45
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.1
        momentum = 0.9
        lr_step_scheduler = 1
        lr_step_gamma = 0.999
        step_decay = 0.5
        decay_after_steps = 20
        
        L_blur_processing = True
        choose_train_dataset = True
        ab_input_processing = "standardization"
        ab_output_processing = "standardization"
        L_input_processing = "normalization"
        
        chosen_net = FCN_net_mega()
        
        gauss_kernel_size = (5, 5)
        
        Results:  , Loss = 
        
        Po 45 epokach:
        Bez tricku: 
        Z trickiem: 
        
        Po 10 epokach:
        Bez tricku: 
        Z trickiem: 

### Wnioski: 
Standardyzacja L chyba sprawia, że jest mniej wycieków kolorów, ale też sprawia, że kolory są momentami zbyt 
przejaskrawione. 