# Ranking sieci 
1. V130
2. V84
3. V70
4. V40
5. V19
6. V32

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
        
V32:    

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
V40:   

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
V70:   

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
        

V70_2:

    Próba odtworzenia V70 w frameworku
    which_version = "V70_2"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct23_19-47-27_DESKTOP-K2JRB94 , Loss = 0.71

      
    Po 60 epokach:
    Bez tricku: 
    Z trickiem: 
    
    Po 10 epokach:
    Bez tricku: 
    Z trickiem: 
    

V71:
        
    chosen_net = FCN_net_mega()
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize",
                            "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTenso",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }

        
        Results: Oct23_18-08-14_DESKTOP-K2JRB94 , Loss = 2,516e-3
        
        Bez tricku: tylko brąz
        Z trickiem: Żółto, ale niebo czasami jest niebieskie, więc chyba lepiej niż inne żółte

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
      chosen_net = FCN_net_mega()

      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {

        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
        
        Results: Oct25_15-55-44_DESKTOP-K2JRB94 , Loss = 0.79
        
        Po 45(final) epokach:
        Bez tricku: Praktycznie identycznie z V70 i V72, może leciutko gorzej
        Z trickiem: Ogólnie bardzo dobrze, prawie jak V70 i V72, ale kolory są momentami zbyt intensywne i czasami 
                    wyciekają
        
        Po 10 epokach:  
        
    

        
        
V74:   

      which_version = "V74"
      
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
        
        Results: Oct25_16-29-15_DESKTOP-K2JRB94 , Loss = 0.74
        
        Po final epokach:
        Bez tricku: Identycznie jak V73
        Z trickiem: Identycznie jak V73
        
    
### Wnioski: 
Standardyzacja L chyba sprawia, że jest mniej wycieków kolorów, ale też sprawia, że kolory są momentami zbyt 
przejaskrawione. 

Gaussian Blur pomaga wyciągnąć kolory i bardziej zwrócić uwagę sieci na główne elementy na obrazku, ale często
powoduje, ze kolory są zbyt intensywne i zbyt mocno wyciekają

#### Wnioski Końcowe mega netu:
Bez triku: najlepsze jest V70, V70_2 i V72, są one lekko kolorowe, kolory są mocno stłumione ale są,
te wersje z blurem (V73 i V74) są o wiele gorsze, wszystko jest szarawe i mniej kolorowe, podobnie V71, czyli słabo 
ogólnie

Ciężko wybrać co jest lepsze, V70 czy V72.

Z trikiem: V70 i V72 (remis między nimi) wyglądają spoko, V73 i V 74 (te są z blurem) już gorzej, są przekolorowane i
kolory są zbyt intensywne, czasami aż przesadnie jaskrawe

## Kolejna faza - testy optimizera i funkcji kosztu dla V70
Bez triku najlepszy V84

Z trikiem najlepszy V84

V80:

    which_version = "V80"
    chosen_net = FCN_net_mega()
    
    zmiana w MSELoss na sum i 
    # loss = criterion(outputs, labels) / output.size(0)
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "sum"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Szkoda słów, loss cały czas nan , Loss = nan

    Czarny obraz, porażka
    
V81:

    which_version = "V81"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct24_21-00-29_DESKTOP-K2JRB94 , Loss = 0.84
    
    Po final epokach:
    Bez tricku: Bez różnicy względem V70
    Z trickiem: Podobnie do V70, ale trochę bardziej jaskrawo

V82:

    which_version = "V82"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.1,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct24_21-21-42_DESKTOP-K2JRB94 , Loss = 0.94
    
    Po final epokach:
    Bez tricku: Słabo, wszystko szare, warto o tym wspomnieć w pracy, że Adagrad słaby
    Z trickiem: Też słabo, wszystko jakieś tęczowe, ale ostro błękitne albo różowe
    
V83:

    which_version = "V83"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.001,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct24_21-48-48_DESKTOP-K2JRB94 , Loss = 0.97
    
    Po final epokach:   Porównanie wszystkiego dalej
    Bez tricku: Gorzej niż V82, szaro
    Z trickiem: Jeszcze gorzej niż V82, nie dość, że przekolorowane to jeszcze jakeis zakłócenia wszędzie


V84:

    which_version = "V84"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct24_22-04-51_DESKTOP-K2JRB94 + Oct24_23-04-02_DESKTOP-K2JRB94 + Oct25_19-54-49_DESKTOP-K2JRB94, Loss = 0.76 ; końcowy Loss = 0.79
    
    Po 60 epokach:
    Bez tricku: Całkiem spoko, chyba lepiej niż V70, są przytułmione kolory
    Z trickiem: Batdzo fajnie, są wyraźne kolory, nawet sensowne, lepiej niż V70, momentami tylko zielony wycieka
    
    Po 310 epokach
    Bez tricku: Może lekko lepiej niż po 60 
    Z trickiem: Bardzo fajnie, ale trawa jest trochę bardziej żółta, względem po 60 epokach, ogólnie kolory są 
    trochę bardziej stłumione się wydaje, więc po 60 epokach chyba jednak lepiej
    
    Po 300 epokach
    Okazało się, że po 300 epokach wszystko jest normalnie fajnie zieone zamiast mieć żółte akcenty, wiec aktualny final
    model to jest właśnie po 300 epokach

V85:

    which_version = "V85"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.001,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct24_22-21-03_DESKTOP-K2JRB94 + Oct24_23-37-26_DESKTOP-K2JRB94 , Loss = 0.72
    
    Po final epokach:
    Bez tricku: Całkiem spoko, ale gorzej niż V84
    Z trickiem: Też spoko, ale gorzej niż V84 i chyba gorzej też niż V70

### Wnioski ze zmiany optimizerów
PO 30 epokach uczenia Adamów

Bez triku: chyba V84 prowadzi, Adam spoko

Z trikiem: Adam chyba dalej trochę lepszy niż V70, kolory sa trochę bardziej dokolorowane

PO 70 epokach uczenia Adamów
Bez triku: Adam V84 świetnie, kolory pełne, bez wycieków prawie, ale czasami część obiektów traci swoje barwy i są szare,
może to zniknie po dłuższym uczeniu

Z trikiem: Adam V84 dalej spoko, kolory pełne, ale bardzo się nauczyła sieć zielonego i często nakłada na różne obiekty
zielony, ale chyba dalej V84 top 1

### Testy funkcji lossu dla V84

CrossEntropyLoss nie działa, jest tylko dla klasyfikatorów

    "criterion":
      {
          "name": "CrossEntropyLoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },

V86:

    which_version = "V86"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "L1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct25_14-46-39_DESKTOP-K2JRB94 , Loss = 0.60
    
    Po final epokach:
    Bez tricku: Słabo, mocno szarawo, nie ma kolorów specjalnie
    Z trickiem: Trochę niebieskawo, trawa zamiast zielona to jest żółtawa albo niebieskawa, gorzej niż V84
    
    
V87:

    which_version = "V87"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize",
                            "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization",
                            "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct25_15-06-49_DESKTOP-K2JRB94 , Loss = 0.30
    
    Po final epokach:
    Bez tricku: Średnio, jest więcej kolorów niż w V86 ale mniej niż w V84
    Z trickiem: Nawet nieźle ogólnie, też niebieskawo, trawa zamiast zielona to jest żółtawa albo niebieskawa, gorzej niż V84, ale lepiej niż
                V86
    
### Wnioski:

Cross entropy Loss w ogóle się nie nadaje, nawet się nie chce odpalić to on powinien mieć format jak do klasyfikacji,
czyli pasująca klasa jako Y.

L1 loss powoduje, że zdjęcia są bardziej niebieskawe, MSELoss lepszy
SmoothL1Loss lepszy niż L1, ale gorszy niż MSELoss, też powoduje, że zdjęcia są trochę niebieskawe, MSELoss lepszy

## Kolejna faza - testy optimizera i funkcji kosztu dla V72
Bez triku najlepszy V91

Z trikiem najlepszy V91

V90:    Adagrad zamiast SGD

    which_version = "V90"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.1,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_16-58-15_DESKTOP-K2JRB94 , Loss = 0.88
    
    Po final epokach:
    Bez tricku: Słabo, szarawo
    Z trickiem: Słabo, żółtawo
    
V91:    Adam zamiast SGD

    which_version = "V91"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":standardization
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_17-19-49_DESKTOP-K2JRB94 , Loss = 0.77
    
    Po final epokach:
    Bez tricku: Bardzo podobnie do V72, ale trochę lepiej, obiekty są bardziej wypełnione kolorami
    Z trickiem: Spoko, kolory lepiej wypełnione niż w V72, ale z drugiej strony częściej wyciekaja
    
V92:    Adam zamiast SGD oraz L1Loss zamiast MSELoss

    which_version = "V92"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "L1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_17-42-06_DESKTOP-K2JRB94 , Loss = 0.62
    
    Po final epokach:
    Bez tricku: Słabo, bardziej żółtawo i mniej kolorow
    Z trickiem: Słabo, żółtawo
    
V93:    Adam zamiast SGD oraz SmoothL1Loss zamiast MSELoss

    which_version = "V93"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_18-04-25_DESKTOP-K2JRB94 , Loss = 0.30
    
    Po final epokach:
    Bez tricku: Nawet nieźlem ale trochę gorzej niż V91
    Z trickiem: Nie jest źle, ale trawa jest czasami wyblakła, V91 lepsze
    
### Wnioski:

Adam lepszy niż Adagrad, L1Loss nagorzej, MSELoss najlepszy, SmoothL1Loss całkiem spoko, ale gorszy lekko niż MSELoss

## Kolejna faza - testy optimizera i funkcji kosztu dla V73
Bez triku najlepszy V101/73

Z trikiem najlepszy V101

V100:   Adagrad zamiast SGD

    which_version = "V100"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.1,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_18-26-41_DESKTOP-K2JRB94 , Loss = 0.91
    
    Po 45 epokach:
    Bez tricku: Słabo, szarawo
    Z trickiem: Słabo, wszystko jest tęczowe i jaskrawe
    
V101:   Adam zamiast SGD

    which_version = "V101"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_18-47-58_DESKTOP-K2JRB94 , Loss = 0.79
    
    Po 45 epokach:
    Bez tricku: Spoko, porobnie jak V73, może leciutko gorzej, ale raczej identyczny
    Z trickiem: Bardzo fajnie, kolorowe, biekty są wypełnione kolorami
    
V102:   Adam i L1Loss

    which_version = "V102"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "L1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_19-10-16_DESKTOP-K2JRB94 , Loss = 0.59
    
    Po 45 epokach:
    Bez tricku: Słabo, szaro
    Z trickiem: Średnio, niby kolorowo, ale kolory nie są jakieś dopasowane i są raczej zbyt jaskrawe
    
V103:   Adam i SmoothL1Loss

    which_version = "V103"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct25_19-32-35_DESKTOP-K2JRB94 , Loss = 0.29
    
    Po 45 epokach:
    Bez tricku: Całkiem spoko, ale lekko gorzej niż V101
    Z trickiem: Średnio, lepiej niż V102, ale gorzej niż V101
    
### Wnioski:
Adam lepszy niż Adagrad, L1Loss nagorzej, MSELoss najlepszy, SmoothL1Loss całkiem spoko, ale gorszy lekko niż MSELoss

## Kolejna faza - testy optimizera i funkcji kosztu dla V74
Bez triku najlepszy V74

Z trikiem najlepszy V74

V110   Adagrad zamiast SGD

    which_version = "V110"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.1,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct26_10-12-46_DESKTOP-K2JRB94 , Loss = 0.86
    
    Po 45 epokach:
    Bez tricku: Słabo, szaro
    Z trickiem: Dziwnie, wszystko tęczowe
    
V111:   Adam zamiast SGD

    which_version = "V111"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct26_10-34-24_DESKTOP-K2JRB94 , Loss = 0.88
    
    Po 45 epokach:
    Bez tricku: Słabo, szaro
    Z trickiem: Słabo, wszystko ostro żółte
    
V112:   Adam i L1Loss

    which_version = "V112"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "L1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct26_10-57-52_DESKTOP-K2JRB94 + Oct26_11-31-33_DESKTOP-K2JRB94 , Loss = 0.59
    
    Po 45 epokach:
    Bez tricku: Ok, podobnie jak V74, ale leciutko gorzej
    Z trickiem: Dziwnie, ogólnie często ok, ale często wszystko niebieskie albo żółte, gorzej niż V74
    
V113:   Adam i SmoothL1Loss

    which_version = "V113"
    chosen_net = FCN_net_mega()
    
    "net": "FCN_net_mega",
    "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
     
    Results: Oct26_11-54-20_DESKTOP-K2JRB94 , Loss = 0.28
    
    Po 45 epokach:
    Bez tricku: Tak sobie, gorzej niż V74 i V112
    Z trickiem: Słabo, żółtawo, albo niebieskawo

### Wnioski:

## Kolejna faza - testy optimizera i funkcji kosztu dla V71
Bez triku najlepszy V123

Z trikiem najlepszy V123

V120:   Adagrad zamiast SGD

    which_version = "V120"
            
    chosen_net = FCN_net_mega()
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adagrad",
          "parameters":
          {
            "lr": 0.1,
            "lr_decay": 0.999
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize", "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }
 
        Results: Oct26_12-59-30_DESKTOP-K2JRB94 , Loss = 2.79e-3
        
        Po 45 epokach
        Bez tricku: Słabo, wszystko szare ale niektóre piksele są tęczowe
        Z trickiem: Słabo, wszystko szare albo żółte ale niektóre piksele są tęczowe
     
V121:   Adam zamiast SGD
   
    which_version = "V121"
            
    chosen_net = FCN_net_mega()
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize", "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }
    
        Results: Oct26_13-21-14_DESKTOP-K2JRB94 , Loss = 2,25e-3
        
        Po 45 epokach
        Bez tricku: Słabo, raczej szarawo, ale lepiej niż V71
        Z trickiem: Lepiej niż V71, dalej momentami żółtawo, ale teraz woda jest zazwyczaj niebieska
        
V122:   Adam zamiast SGD oraz L1Loss zamiast MSELoss
   
    which_version = "V122"            
    chosen_net = FCN_net_mega()
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "L1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize", "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }
        
        Results: Oct26_13-44-09_DESKTOP-K2JRB94 , Loss = 0.03
        Results nowe, żeby nie było na czerwono: Oct26_15-11-20_DESKTOP-K2JRB94 , Loss = 
        
        Po 45 epokach:
        Bez tricku: Słabo, wszystko szare 
        Z trickiem: Dziwnie, wszystko na czerwono 
        
        Po 45 epokach nowe:
        Tak samo, też czerwowo
        
V123:   Adam zamiast SGD oraz SmoothL1Loss zamiast MSELoss
   
    which_version = "V123"            
    chosen_net = FCN_net_mega()
      "net": "FCN_net_mega",
      "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize", "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }
    
        Results: Oct26_14-06-52_DESKTOP-K2JRB94 , Loss = 1.14e-3
        
        Po 45 epokach
        Bez tricku: Podobnie jak V71, może lekko lepiej niż V71 i V121
        Z trickiem: Podobnie jak V121, ale chyba lepiej, czyli lepiej niż V71
        
### Wnioski:

W sumie i Adam i Adagrad słabo, ale Adam chyba lekko lepiej

L1 loss tragedia, ale Smooth L1 loss całkiem nieźle, chyba nawet lepiej niż MSELoss


## Testy zamiany kolejności warst BatchNorm i Relu

V130:

FCN_net_mega w wersji V84, ale z zamienionymi miejscami warstwami BatchNorm i Relu

      which_version = "V130"            
      "net": "FCN_net_mega_V2",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {
    
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
    
        Results: Oct27_17-38-09_DESKTOP-K2JRB94 , Loss = 0.75
        
        Po 60 epokach
        Bez tricku: Tak sobie, podobnie jak V84, ale chyba lekko lepiej
        Z trickiem: Fajnie, lepiej niż V84, 
        
V131:

FCN_net_mega_V2 z V123:

    which_version = "V131"            
    
      "net": "FCN_net_mega_V2",
      "criterion":
      {
          "name": "SmoothL1Loss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"CustomNormalize", "parameters": [0, 255.0]}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "normalization",
        "ab_output_processing": "normalization",
        "L_input_processing": "normalization"
      }
    
        Results: Oct27_18-59-32_DESKTOP-K2JRB94 , Loss = 1.28e-3
        
        Po 60 epokach
        Bez tricku: Słabo, wszystko szare
        Z trickiem: Słabo, głównie żółto i są jakieś losowe plamy kolorów
        
V132:

FCN_net_mega_V2 w wersji V91

    "net": "FCN_net_mega_V2",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":standardization
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct27_19-29-28_DESKTOP-K2JRB94 , Loss = 0.7
    
    Po 60 epokach:
    Bez tricku: Spoko, ale V91 lepiej
    Z trickiem: Spoko, ale V91 lepiej

V133:

FCN_net_mega_V2 w wersji V101

    "net": "FCN_net_mega_V2",
    "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {

      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"Standardization", "parameters": []},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "standardization"
      }
     
    Results: Oct27_19-59-45_DESKTOP-K2JRB94 , Loss = 0.85
    
    Po 60 epokach:
    Bez tricku: Słabo, szaro
    Z trickiem: Słabo, wszystko niebieskie

V134:

FCN_net_mega_V2 w wersji V74
    
      "net": "FCN_net_mega_V2",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "SGD",
          "parameters":
          {
            "lr": 0.1,
            "momentum": 0.9
          }
      },
      "scheduler":
      {
          "name": "StepLR",
          "parameters":
          {
            "step_size": 1,
            "gamma": 0.999
          },
          "scheduler_decay": 0.5,
          "scheduler_decay_period": 20
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]},
                              {"name":"GaussKernel", "parameters": [[5,5]]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": true,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
      
        Results: Oct27_20-29-50_DESKTOP-K2JRB94 , Loss = 0.74
        
        Po 60 epokach:
        Bez tricku: Trochę lepiej niżx V74
        Z trickiem: Spoko, podobnie jak V74, może lekko lepiej
        
### Wnioski

Zamiana kolejności warstw czasmia pomaga, ale nie zawsze, w przypadku V84 pomogło, ale w przypadku V123, V91, V101
pogorszyło rezultaty, a w przypadku V74 tylko leciutko polepszył, ogólnie zamiania warstw miejscami nie robi jakiejś 
wielkiej różnicy

## Test dropout

V140:

V130 ale z dropoutami czyli FCN_net_mega_dropout

      "net": "FCN_net_mega_dropout",
      "criterion":
      {
          "name": "MSELoss",
          "patameters":
          {
            "reduction": "mean"
          }
      },
      "optimizer":
      {
          "name": "Adam",
          "parameters":
          {
            "lr": 0.1,
            "weight_decay": 1e-10
          }
      },
      "scheduler":
      {
    
      },
      "dataset":
      {
        "name": "BasicCifar10Dataset",
        "input_conversions": [{"name":"CustomNormalize", "parameters": [50, 100]}],
        "output_conversions": [{"name":"Standardization", "parameters": []}],
        "transforms": [{"name": "ToTensor",
                        "parameters": []}]
      },
      "additional_params":
      {
        "get_data_to_test": false,
        "choose_train_set": true,
    
        "blur":
        {
          "do_blur": false,
          "kernel_size": [5, 5]
        },
        "ab_input_processing": "standardization",
        "ab_output_processing": "standardization",
        "L_input_processing": "normalization"
      }
    
        Results: Oct27_21-33-32_DESKTOP-K2JRB94 , Loss = 0.77
        
        Po 60 epokach
        Bez tricku: podobnie jak V130, ale barwy bardziej przytłumione co ma sens, bo dropout
        Z trickiem: W sumie z trickiem w V140 jest bardziej kolorowo, ale to trick, więc przemilczmy to

V141:

To samo co V140, ale z dropoput rate = 0.5, model FCN_net_mega_dropout2
    
    Results: Oct27_22-29-04_DESKTOP-K2JRB94, Loss = 
    
    Po 60 epokach
    Bez tricku: Słabo, wszystko szare
    Z trickiem: Słabo, wszystko niebieskie
    
### Wnioski: 

Widać, że dropout duży szkodzi


## Mega wnioski końcowe:

Jeśli mamy normalizację na L i standardyzację na AB (V70) to najlepsza jest wersja V84, czyli Adam i MSELoss

Jeśli mamy normalizację na L i na AB (V71) to najlepsza jest wersja V123 czyli Adam i SmoothL1Loss

Jeśli mamy standardyzację na L i AB (V72) to najlepsza jest wersja V91, czyli Adam i MSELOss

Jeśli mamy standardzyację na L i AB i do tego robimy blur (V73) to najlepsza jest wersja V101, czyli Adam i MSELoss

Jeśli mamy normalizację na L i standardyzację na AB (V74) i do tego robimy blur to najlepsza jest wersja V74, czyli 
SGD i MSELoss

Jeśli zamienimy miejscami warstwy BatchNorm i Relu to zyska na tym V84, czyli normalizacja na L i standardyzacja na 
AB z Adam i MSELoss