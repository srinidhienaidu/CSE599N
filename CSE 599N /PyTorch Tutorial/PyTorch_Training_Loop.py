import torch.optim as optim 
import torch.nn as nn

# Construct an optimizer and loss function

# optimizer = optim.SGD(model.parameters(), lr = 0.001)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_fn = nn.MSELoss()

# Main training loop 
for epoch in range(100):
    for batch_idx, batch in enumerate(loader): # Assume our dataset is some iterable oject that returns mini ba
        input, target = batch # These are called mini-batches of data (in our case, this will be a target)
        optimizer.zero_grad() # The optimizer accumulates gradients from each backward pass unless explicitly zero
        output = model(input) # Run our model on the input data and compute the loss
        loss = loss_fn(output, target) 
        loss.backward() # Runs a backward pass, computing gradients of all leaf tensors with respect to the variable loss (including all the weights and biases)
        
        nn.utils.clip_grad_value_(model.parameters(), clip_value = 1.0) # Set the maximum gradient value

        optimizer.step() # Does gradient descent- updates on the leaf tensors given to the optimizer (in this case the parameters in model.parameters())

