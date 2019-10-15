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
    
V11:   !!! przy zmianie ab_chosen_normalization na  normalization jest coś ciekawego
    
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
    Wnioski: Loss tragdnia, cały czas na tym samym poziomie

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
    
V14:

        init_epoch = 0
        how_many_epochs = 30
        do_load_model = False
        
        batch_size = 128
        learning_rate = 0.01
        momentum = 0.9
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
        loss = criterion(outputs, ab_batch.to(device))
        
        Results: Oct15_18-15-07_DESKTOP-K2JRB94, Oct15_18-40-52_DESKTOP-K2JRB94
                
        
        Wnioski: Słabo wyszło, praktycznie wszystko brązowe, nawet nie ma prześwitów niebieskiego,
        loss wygląda nawet ok, trochę spadł ale przestał się uczyć po 5 min gdzieś
