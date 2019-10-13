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



