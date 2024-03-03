import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
        print(f"model has {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x

if __name__ == '__main__':
    print('PyTorch Simple Model Training Demo')
    torch.random.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Running on {device}')

    model = MyModel().to(device)
    
    x = torch.randn(10000, requires_grad=False, dtype=torch.float).to(device)
    y = 1234 + 5678 * x + torch.randn(x.shape, dtype=torch.float).to(device)
    
    def loss_fn(output, target):
        return torch.mean(torch.pow(output-target, 2))
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    for t in range(int(1e4)):
        # Sets model to TRAIN mode
        model.train()
        
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)
        if t % 1000 == 1000-1:
            print(f'iter {t: 8d}, loss: {loss.item(): 6.2f}', end='')
            for param in model.state_dict().items():
                print(f', {param[0]}: {param[1].item(): 6.2f}', end='')
            print()
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Demo End')