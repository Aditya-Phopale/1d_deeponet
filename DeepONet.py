## Learning integration operator on 2nd degree random polynomials

### Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrunkNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dim, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim, output_dim)
        self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = branch_output @ trunk_output.t() + self.bias
        return output

def generate_data(num_samples=1000, num_points=50):
    x = np.linspace(0, 1, num_points)
    data = []
    
    for _ in range(num_samples):
        coefficients = np.random.uniform(-1, 1, size=(3,))
        f = coefficients[0] * x**2 + coefficients[1] * x + coefficients[2]
        F = (coefficients[0] / 3) * x**3 + (coefficients[1] / 2) * x**2 + coefficients[2] * x
        
        data.append((f, F))
    
    # print(data)
    return data

### Generating training data
train_data = generate_data()
train_functions, train_anti_derivatives = zip(*train_data) ### Discretized input and output functions at same points can be different too
train_functions = torch.tensor(np.array(train_functions), dtype=torch.float32)
train_anti_derivatives = torch.tensor(np.array(train_anti_derivatives), dtype=torch.float32)
# print(train_functions.shape, train_anti_derivatives.shape)


input_dim = 50  
hidden_dim = 128 
output_dim = 50 
learning_rate = 0.001
num_epochs = 1000

deeponet = DeepONet(branch_input_dim=input_dim, trunk_input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(deeponet.parameters(), lr=learning_rate)

progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")

for epoch in progress_bar:
    deeponet.train()
    optimizer.zero_grad()

    branch_input = train_functions
    trunk_input = torch.linspace(0, 1, output_dim).unsqueeze(1)

    output = deeponet(branch_input, trunk_input)
    # print(output.shape)
    loss = criterion(output, train_anti_derivatives)
    loss.backward()
    optimizer.step()

    progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    if epoch % 100 == 0:
        tqdm.write(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.6f}")

deeponet.eval()

### Generating evaluation data
test_data = generate_data(num_samples=100)
test_functions, test_anti_derivatives = zip(*test_data)
test_functions = torch.tensor(np.array(test_functions), dtype=torch.float32)
test_anti_derivatives = torch.tensor(np.array(test_anti_derivatives), dtype=torch.float32)

with torch.no_grad():
    branch_input = test_functions
    trunk_input = torch.linspace(0, 1, output_dim).unsqueeze(1)
    
    output = deeponet(branch_input, trunk_input)
    test_loss = criterion(output, test_anti_derivatives)
    print(f"Test Loss: {test_loss.item()}")
