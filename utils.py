import torch
import torch.nn.init as init
from torch.autograd import Variable
from voxnet_coder_decoder import VoxNetAutoencoder as voxnet


# Entrenamiento
# Entradas: Dataset, modelo, funciondeperdida, optimizador, dispositivo
# Salidas: Error
def entrena(dataloader, model, loss_fn, optimizer, device):
    for i, sample in enumerate(dataloader):
        
        grids = sample['grid']

        # convert images to FloatTensors
        grids = grids.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        grids = Variable(grids)   
        
        grids = grids.to(device)
        #=============forward==================
        output = model(grids)
        loss = loss_fn(output, grids)

        #=============backward=================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss


#Validación
# Entradas: Dataset, modelo, funciondeperdida
# Salidas: Error_promedio, accuracy
def valida(dataloader, model, loss_fn, device):
    test_loss = 0

    for sample in dataloader:
        
        # get sample data: images and ground truth keypoints
        grids = sample['grid']
        
        # convert images to FloatTensors
        grids = grids.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        grids = Variable(grids)    
        grids = grids.to(device)
        
        output = model.forward(grids)
        test_loss += loss_fn(output, grids).item()

    
    return test_loss



# función para inicializar los pesos con las distintas inicializaciones

def initialize_weights(model, **kwargs):
    initialization = kwargs['init']
    a = kwargs['a'] # init with uniform distribution
    b = kwargs['b']
    mean = kwargs['mean'] # init with normal distribution
    std = kwargs['std']
    const = kwargs['const'] #init with constant value
    groups = kwargs['groups'] #init with Dirac delta function
    gain = kwargs['gain'] #init with xavier 
    
    for module in model.modules():
        if hasattr(module, 'weight'):
            if initialization == "xavier_uniform":
                init.xavier_uniform_(module.weight, gain=gain)
            elif initialization == "xavier_normal":
                init.xavier_normal_(module.weight, gain=gain)
            elif initialization == "kaiming_uniform":
                init.kaiming_uniform_(module.weight)
            elif initialization == "kaiming_normal":
                init.kaiming_normal_(module.weight)
            elif initialization == "uniform":
                init.uniform_(module.weight, a=a, b=b)
            elif initialization == "normal":
                init.normal_(module.weight, mean= mean, std= std)
            elif initialization == "constant":
                init.constant_(module.weight, const)
            elif initialization == "dirac":
                init.dirac_(module.weight, groups)
            elif initialization == "trunc_normal":
                init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
            else:
                raise ValueError(f"Unknown initialization method: {initialization}")
            
            if module.bias is not None:
                init.zeros_(module.bias)


# Prueba del modelo en un batch de prueba
def net_sample_output(model, train_loader, device):
    model.eval()
    
    # iterate through the test dataset
    for i, sample in enumerate(train_loader):
        
        if i == 0:
            # get sample data: images and ground truth keypoints
            grids = sample['grid']
            nbvs = sample['nbv_class']

            # convert images to FloatTensors
            grids = grids.type(torch.FloatTensor)

                # wrap them in a torch Variable
            grids = Variable(grids)    
            grids = grids.to(device)

                # forward pass to get net output
            output = model(grids)
            grids = grids.cpu()
            output = output.cpu()
                # break after first image is tested
            
            return grids, output, nbvs


## IMPORTANTE: hay que probarlo
def get_latent_space(grids, device):
    path = 'stuff/experimento_2/weights_nmodelo_vox_16_4.pth'
    model1= voxnet(latent_space = 16).cuda()
    model1.load_state_dict(torch.load(path))
    
    #grids = grids.type(torch.FloatTensor)
        # wrap them in a torch Variable
    grids = Variable(grids)    
    grids = grids.to(device)
        # forward pass to get net coder output
    output = model1.codificador(grids)
    grids = grids.cpu()
    output = output[0].cpu()
    del model1
    del grids
    return output



def entrena_nbv(dataloader, model, loss_fn, optimizer, device):
    for i, sample in enumerate(dataloader):
        
        nbv = sample['nbv_class']
        grid = sample ['grid']
        grid = grid.type(torch.FloatTensor)# no pude transformar en get_l
        grid = get_latent_space(grid, device)
        # convert images to FloatTensors
        nbv = nbv.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        nbv = Variable(nbv)   
        grid = Variable(grid)  
        
        nbv = nbv.to(device)
        grid = grid.to(device)
        #=============forward==================
        output = model(grid)
        loss = loss_fn(output, nbv)

        #=============backward=================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss

def valida_nbv(dataloader, model, loss_fn, device):
    test_loss = 0

    for sample in dataloader:
        
        # get sample data: images and ground truth keypoints
        nbv = sample['nbv_class']
        grid = sample ['grid']
        grid = grid.type(torch.FloatTensor)
        grid = get_latent_space(grid, device)
        
        # convert images to FloatTensors
        nbv = nbv.type(torch.FloatTensor)
        
        # wrap them in a torch Variable
        nbv = Variable(nbv)  
        grid = Variable(grid) 

        nbv = nbv.to(device)
        grid = grid.to(device)
        
        output = model.forward(grid)
        test_loss += loss_fn(output, nbv).item()

    return test_loss

def net_sample_output_nbv(model, train_loader, device):
    model.eval()
    
    # iterate through the test dataset
    for i, sample in enumerate(train_loader):
        
        if i == 0:
            # get sample data: images and ground truth keypoints
            grid = sample['grid']
            nbv = sample['nbv_class']

            grid = grid.type(torch.FloatTensor)# no pude transformar en get_l
            l_s = get_latent_space(grid, device)
            # convert images to FloatTensors
            nbv = nbv.type(torch.FloatTensor)
            # wrap them in a torch Variable
            l_s = Variable(l_s)  
            l_s = l_s.to(device)

            # forward pass to get net output
            output = model(l_s)
            output = output.cpu()
            nbv = nbv.cpu()
                # break after first image is tested
            
            return output, nbv

def net_position_nbv(model, grid, device):
    
    with torch.no_grad():
        grid = grid.type(torch.FloatTensor)# no pude transformar en get_l
        l_s = get_latent_space(grid, device)
        l_s = Variable(l_s)  
        l_s = l_s.to(device)
        # forward pass to get net output
        output = model(l_s)
        output = output.cpu()
    return output