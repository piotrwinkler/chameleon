V1:

    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(outputs, labels)

V2:

    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(outputs, labels) / output.size(0)



